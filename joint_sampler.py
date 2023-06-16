import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from numpy import pi, sqrt
import camb
import healpy as hp
import time
from tools import *
from corner import corner
from scipy.linalg import sqrtm

from tqdm import tqdm

class joint_sampler:
    def __init__(self, load_sim_file, beam_deg, n_cosmo_variable, noise_sigma, l_max, l_min, prev_chain = False, proposal_step = 0.3, mask=0, nside=64):
        self.load_sim_file = load_sim_file
        self.n_cosmo_variable = n_cosmo_variable 
        self.mask = mask
        self.nside = nside
        if n_cosmo_variable == 6:
            # Sample all 6 LCDM paramters
            self.labels = np.array([r'$\Omega_b h^2$', r'$\Omega_{CDM} h^2$', r'$H_0$', r'$\tau$', r'$10^{9} A_s$', r'$n_s$'])
            self.correct_cosmo_values = np.array([0.02237, 0.12, 67.36, 0.0544, 2.1, 0.9649])
            self.cosmo_param_sigma = np.array([0.0002, 0.001, 0.3, 0.002, 0.015, 0.001])
        elif n_cosmo_variable == 3:
            # For low ell_max, it is often good to just sample these parameters
            self.labels = np.array([r'$\tau$', r'$10^{9} A_s$', r'$n_s$'])
            self.correct_cosmo_values = np.array([0.0544, 2.1, 0.9649])
            self.cosmo_param_sigma = np.array([0.0005, 0.02, 0.001])
        else:
            self.labels = np.array([r'$\Omega_b h^2$'])
            self.correct_cosmo_values = np.array([0.02237])
            self.cosmo_param_sigma = np.array([0.0005])
        self.l_max = l_max
        self.l_min = l_min
        self.proposal_step = proposal_step

        if prev_chain:
            # Load a previous chain and make a proposal covariance matrix out of it.
            load_prev_chain = np.load('plot_folder_nside{:.0f}/{:.0f}_{:.0f}_previous_chain.npy'.format(self.nside, self.mask * 180/np.pi, self.l_max))
            print('Loading previous chain:', '{:.0f}_{:.0f}_previous_chain.npy'.format(self.mask * 180/np.pi, self.l_max), load_prev_chain.shape)
            
            m = np.cov(load_prev_chain, rowvar=False)
            self.cosmo_param_cov = m
        else:
            # If not previous chain exists, we use a badly tuned diagonal proposal matrix
            # Tune self.cosmo_param_sigma defined above for optimal acceptance rate (\sim 23%)
            print('No previous chain is used to create a proposal matrix!\
                  You are using a diagonal proposal matrix that needs to be tuned for optimal acceptance rate!')
            self.cosmo_param_cov = np.diag(self.cosmo_param_sigma**2)

        self.b_l = np.zeros((3, l_max+1), dtype=np.complex128)
        pixel_window = np.array(hp.pixwin(nside=self.nside, pol=True, lmax=l_max))

        if type(beam_deg) is str:
            beam = fits.open(beam_deg)[1]
            for l in range(l_max+1):
                self.b_l[0, l] = beam.data['T'][l] * pixel_window[0, l]
                self.b_l[1, l] = beam.data['E'][l] * pixel_window[1, l]
                self.b_l[2, l] = beam.data['B'][l] * pixel_window[1, l]
        else:
            beam = np.array(hp.gauss_beam(fwhm=beam_deg*np.pi/180, pol=False, lmax=l_max))
            for l in range(l_max+1):
                self.b_l[0, l] = beam[l] * pixel_window[0, l]
                self.b_l[1, l] = beam[l] * pixel_window[1, l]
                self.b_l[2, l] = beam[l] * pixel_window[1, l]
        
        cp = camb.set_params(tau=0.0544, ns=0.9649, H0=67.36, ombh2=0.02237, omch2=0.12, As=2.1e-09, lmax=2000)
        camb_results = camb.get_results(cp)
        self.c_l_lcdm = np.array(camb_results.get_cmb_power_spectra(lmax=self.l_max+200, raw_cl=True, CMB_unit='muK')['total'][:self.l_max+1, :4])

        self.N_l = noise_sigma**2 * 4*np.pi / (12*self.nside**2)

        if self.mask != 0:
            print('Mask is applied wit a constant latitude of ', self.mask*180/np.pi)
            N_inv_l_mask_list = get_mask_N_inv_ell(self.mask, l_max) / (noise_sigma**2) * 12*self.nside**2
            N_inv_sqrt_mask = np.zeros((l_max+1, 3*(l_max+1), 3*(l_max+1)), dtype=np.complex128)
            N_inv_mask = np.zeros((l_max+1, 3*(l_max+1), 3*(l_max+1)), dtype=np.complex128)
            for m in tqdm(range(l_max+1)):
                min_element = 2 if m <= 2 else m
                for ell in range(min_element, l_max+1):
                    elm = int(3*ell)
                    for ell_p in range(min_element, l_max+1):
                        elm_p = int(3*ell_p)
                        N_inv_mask[m, elm:elm+3, elm_p:elm_p+3] = np.array(
                            [[N_inv_l_mask_list[m, ell, ell_p], 0, 0],
                            [0, N_inv_l_mask_list[m, ell, ell_p] / 2, 0],
                            [0, 0, N_inv_l_mask_list[m, ell, ell_p] / 2]])
            
            # Solve the equation for each m
            for m in tqdm(range(l_max+1)):
                min_element = 2 if m <= 2 else m
                elm_min = int(min_element*3)
                N_inv = N_inv_mask[m, elm_min:, elm_min:]
                N_inv_sqrt_mask[m, elm_min:, elm_min:] = scipy.linalg.sqrtm(N_inv)

            self.N_inv_l_mask = N_inv_mask
            self.N_inv_sqrt_mask = N_inv_sqrt_mask
            
        self.init_CMB_noise()

    def init_CMB_noise(self):
        c_l_lcdm = self.c_l_lcdm
        l_max = self.l_max

        # Make a random a_lm realization. Good to have for plotting later
        a_lm = hp.synalm(c_l_lcdm.T, lmax=self.l_max, new=True)

        map = hp.read_map(self.load_sim_file, field=(0,1,2))

        # This is the data model
        d_lm = np.array(hp.map2alm(map, lmax=l_max))
        print('Loading CMB file:', d_lm.shape)

        # Creating the beam matrix, A_ell, matrices in correct format
        A_ell_list = np.zeros((3*(l_max+1), 3*(l_max+1)), dtype=np.complex128)
        for ell in range(2, l_max+1):
            A_ell_list[ell*3:ell*3+3, ell*3:ell*3+3] = np.real(np.diagflat([self.b_l[0, ell], self.b_l[1, ell], self.b_l[2, ell]]))
        self.A_ell_list = A_ell_list

        self.d_lm = d_lm
        self.a_lm = a_lm
    
    def plot_spectras(self):
        l = np.arange(self.l_min, self.l_max+1)

        c_l = get_D_l(self.c_l_lcdm)
        d_l = get_D_l(hp.sphtfunc.alm2cl(self.d_lm).T)
        a_l = get_D_l(hp.sphtfunc.alm2cl(self.a_lm).T)
        N_l = np.ones((self.l_max+1, 4)) * self.N_l
        N_l[:, 1:3] *= 2 # Polarization has twice the noise
        D_N_l = get_D_l(N_l)

        fig, axs = plt.subplots(2, 2)

        #CAMB [TT, EE, BB, TE]
        label = ['TT', 'EE', 'BB', 'TE']
        # Healpy = [TT, EE, BB, TE, EB, TB]
        i=0
        for ax in axs.flat:
            ax.set_title(label[i])
            ax.plot(l, c_l[self.l_min:, i], label='LCDM', linewidth=4)

            ax.plot(l, a_l[self.l_min:, i], label='LCDM realization')
            ax.plot(l, d_l[self.l_min:, i], label='Measured Data')

            ax.plot(l, D_N_l[self.l_min:, i], label='White Noise')
            
            ax.set_ylabel(r"$C_{\ell}\ell (\ell + 1)/2\pi$")
            ax.set_xlabel(r"$\ell$")
            if i == 0: ax.legend()
            i += 1
        plt.savefig('plot_folder_nside{:.0f}/{:.0f}_{:.0f}_spectra.pdf'.format(self.nside, self.mask*180/np.pi, self.l_max))

    def get_joint_slm_sample_mask(self, c_l, plot=False):
        d_lm = self.d_lm
        l_max = self.l_max

        s_lm = np.zeros(d_lm.shape, dtype=np.complex128)
        f_lm = np.zeros(d_lm.shape, dtype=np.complex128)

        S_inv_list = np.zeros((3*(l_max+1), 3*(l_max+1)))
        N_inv_list = self.N_inv_l_mask
        N_inv_sqrt_list = self.N_inv_sqrt_mask
        A_ell_list = self.A_ell_list

        #CAMB [TT, EE, BB, TE]
        for ell in range(2, l_max+1):
            S_matrix = np.array([
                [c_l[ell, 0], c_l[ell, 3], 0],
                [c_l[ell, 3], c_l[ell, 1], 0],
                [0,           0,           c_l[ell, 2]]
            ])
            S_inv = np.linalg.inv(S_matrix)
            S_inv_list[ell*3:ell*3+3, ell*3:ell*3+3] = S_inv

        # Solve the equation for each m
        for m in tqdm(range(l_max+1)):
            min_element = 2 if m <= 2 else m
            N_inv = N_inv_list[m, min_element*3:, min_element*3:]
            N_inv_sqrt = N_inv_sqrt_list[m, min_element*3:, min_element*3:]
            S_inv = S_inv_list[min_element*3:, min_element*3:]
            S_inv_sqrt = scipy.linalg.sqrtm(S_inv)

            A = A_ell_list[min_element*3:, min_element*3:]
            A_N_inv = np.dot(A, N_inv)
            A_N_inv_A = np.dot(A_N_inv, A)
            A_N_inv_sqrt = np.dot(A, N_inv_sqrt)

            d_vector = np.zeros(3*(l_max+1)-min_element*3, dtype=np.complex128)
            if m == 0:
                # a_lm when m = 0 are real
                omega_0 = np.random.normal(size=3*(l_max+1)-min_element*3, scale=1)
                omega_1 = np.random.normal(size=3*(l_max+1)-min_element*3, scale=1)
            else:
                omega_0 = np.random.normal(size=(3*(l_max+1)-min_element*3, 2), scale=np.sqrt(2)/2).view(np.complex128)[:, 0]
                omega_1 = np.random.normal(size=(3*(l_max+1)-min_element*3, 2), scale=np.sqrt(2)/2).view(np.complex128)[:, 0]
            
            for ell in range(min_element, l_max+1):
                elm = 3*ell-min_element*3
                d_vector[elm:elm+3] = d_lm[:, hp.Alm.getidx(l_max, ell, m)]

            s_l_m = np.linalg.solve(S_inv + A_N_inv_A, np.dot(A_N_inv, d_vector))
            f_l_m = np.linalg.solve(S_inv + A_N_inv_A, np.dot(S_inv_sqrt, omega_0) + np.dot(A_N_inv_sqrt, omega_1))
            for ell in range(min_element, l_max+1):
                idx = hp.Alm.getidx(l_max, ell, m)
                elm = 3*ell-min_element*3
                s_lm[:, idx] = s_l_m[elm:elm+3]
                f_lm[:, idx] = f_l_m[elm:elm+3]

        if plot:
            plt.figure()
            map = hp.alm2map(s_lm[0, :], nside=256)
            hp.mollview(map)
            plt.savefig('plot_folder_nside{}/{:.0f}_{:.0f}_slm.pdf'.format(self.nside, self.mask*180/np.pi, self.l_max))

            plt.figure()
            map = hp.alm2map(f_lm[0, :], nside=256)
            hp.mollview(map)
            plt.savefig('plot_folder_nside{}/{:.0f}_{:.0f}_.pdf'.format(self.nside, self.mask*180/np.pi, self.l_max))

            plt.figure()
            map = hp.alm2map(s_lm[0, :]+f_lm[0, :], nside=256)
            hp.mollview(map)
            plt.savefig('plot_folder_nside{}/{:.0f}_{:.0f}_slm_flm.pdf'.format(self.nside, self.mask*180/np.pi, self.l_max))

            ell = np.arange(l_max+1)
            plt.figure()
            plt.plot(ell, get_D_ell(hp.alm2cl(s_lm[0, :])), label='s_lm')
            plt.plot(ell, get_D_ell(hp.alm2cl(f_lm[0, :])), label='f_lm')
            plt.plot(ell, get_D_ell(hp.alm2cl(s_lm[0, :]+f_lm[0, :])), label='s_lm+f_lm')
            plt.plot(ell, get_D_ell(hp.alm2cl(self.a_lm[0, :])), label='a_lm')
            plt.legend()
            plt.savefig('plot_folder_nside{}/{:.0f}_{:.0f}_c_l.pdf'.format(self.nside, self.mask*180/np.pi, self.l_max))

        return s_lm, f_lm

    def get_joint_slm_sample(self, c_l):
        N_l = self.N_l
        d_lm = self.d_lm
        l_min = self.l_min
        l_max = self.l_max
        B_l = self.b_l

        s_lm = np.zeros(d_lm.shape, dtype=np.complex128)
        f_lm = np.zeros(d_lm.shape, dtype=np.complex128)

        N_matrix_inv = np.array([
            [1, 0, 0],
            [0, 1/2, 0],
            [0, 0, 1/2]]) / N_l
        N_matrix_inv_sqrt = np.array([
            [1, 0, 0],
            [0, 1/np.sqrt(2), 0],
            [0, 0, 1/np.sqrt(2)]]) / sqrt(N_l)

        for l in range(l_min, l_max+1):
            S_matrix = np.array([
                [c_l[l, 0], c_l[l, 3], 0],
                [c_l[l, 3], c_l[l, 1], 0],
                [0,         0,         c_l[l, 2]]])
            A_l = np.diag(B_l[:, l])
            S_inv = np.linalg.inv(S_matrix)
            S_inv_N_inv = S_inv + np.dot(A_l, np.dot(N_matrix_inv, A_l))

            index = hp.sphtfunc.Alm.getidx(l_max, l, np.arange(l+1))
            s_lm[:, index] = np.linalg.solve(S_inv_N_inv, np.dot(A_l, np.dot(N_matrix_inv, d_lm[:, index])))

            # Only m=0 since it is a_lm = real.
            index_m_0 = hp.sphtfunc.Alm.getidx(l_max, l, 0)
            first_term_random_m_0 = np.random.normal(size=(3), scale = 1, loc = 0)
            second_term_random_m_0 = np.random.normal(size=(3), scale = 1, loc = 0)
            first_term_m_0 = np.dot(sqrtm(S_inv), first_term_random_m_0)
            second_term_m_0 = np.dot(np.dot(A_l, N_matrix_inv_sqrt), second_term_random_m_0)
            f_lm[:, index_m_0] = np.linalg.solve(S_inv_N_inv, first_term_m_0 + second_term_m_0)

            # m != 0 which can be complex
            index = hp.sphtfunc.Alm.getidx(l_max, l, np.arange(1, l+1))
            first_term_random = np.random.normal(size=(3, len(index), 2), scale=np.sqrt(2)/2).view(np.complex128)[:, :, 0]
            second_term_random = np.random.normal(size=(3, len(index), 2), scale=np.sqrt(2)/2).view(np.complex128)[:, :, 0]

            first_term = np.dot(sqrtm(S_inv), first_term_random)
            second_term = np.dot(np.dot(A_l, N_matrix_inv_sqrt), second_term_random)
            f_lm[:, index] = np.linalg.solve(S_inv_N_inv, first_term + second_term)
        return s_lm, f_lm

    def proposal_w(self, old_cosmo_param):
        #new_cosmo_param = old_cosmo_param + np.random.normal(np.zeros(6), self.cosmo_param_sigma)
        proposal_step = self.proposal_step
        if self.n_cosmo_variable == 6:
            new_cosmo_param = old_cosmo_param + np.dot(np.linalg.cholesky(self.cosmo_param_cov), np.random.normal(size=6)) * proposal_step
            cp = camb.set_params(tau=new_cosmo_param[3], ns=new_cosmo_param[5], H0=new_cosmo_param[2], ombh2=new_cosmo_param[0], omch2=new_cosmo_param[1], As=new_cosmo_param[4]*1e-9, lmax=self.l_max+200)
        elif self.n_cosmo_variable == 3:
            #'Only sampling tau, As, ns'
            new_cosmo_param = old_cosmo_param + np.dot(np.linalg.cholesky(self.cosmo_param_cov), np.random.normal(size=3)) * proposal_step
            cp = camb.set_params(tau=new_cosmo_param[0], ns=new_cosmo_param[2], H0=67.36, ombh2=0.02237, omch2=0.12, As=new_cosmo_param[1]*1e-9, lmax=self.l_max+200)
        else:
            new_cosmo_param = old_cosmo_param + np.array([np.random.normal(loc=0, scale=self.cosmo_param_sigma[0])])*proposal_step
            cp = camb.set_params(tau=0.0544, ns=0.9649, H0=67.36, ombh2=new_cosmo_param[0], omch2=0.12, As=2.1*1e-9, lmax=self.l_max+200)
        camb_results = camb.get_results(cp)
        new_c_l_lcdm = np.array(camb_results.get_cmb_power_spectra(lmax=self.l_max+200, raw_cl=True, CMB_unit='muK')['total'][:self.l_max+1, :4])

        return new_cosmo_param, new_c_l_lcdm

    def start_joint_sampler(self, samples=100, burnin=10):
        # Sample only ns for now
        l_min = self.l_min
        l_max = self.l_max
        old_cosmo_param = self.correct_cosmo_values
        old_c_l = self.c_l_lcdm

        if self.mask == 0:
            # If no mask, then sampling is quick and simple
            old_s_lm, old_f_lm = self.get_joint_slm_sample(old_c_l)
        else:
            # If there is a constant latitude mask, then solving the constrained realization is more complicated
            old_s_lm, old_f_lm = get_joint_slm_sample_mask_numba(old_c_l, self.N_inv_l_mask, self.N_inv_sqrt_mask, self.d_lm, self.l_max, self.A_ell_list)

        list_of_cosmo_param = np.zeros((samples+1, self.n_cosmo_variable))
        list_of_cosmo_param[0, :] = old_cosmo_param
        
        accept_rate = 0
        tot = 0

        j = 1
        pbar = tqdm(total=samples)
        while j <= samples:
            proposed_cosmo_param, proposed_c_l = self.proposal_w(old_cosmo_param)
            print('Proposed cosmo param:', proposed_cosmo_param)
            print('Proposal step:', self.proposal_step)
            start_t = time.time()
            if self.mask == 0:
                # If no mask, then sampling is quick and simple
                proposed_s_lm, proposed_f_lm = self.get_joint_slm_sample(proposed_c_l)
            else:
                # If there is a constant latitude mask, then solving the constrained realization is more complicated
                proposed_s_lm, proposed_f_lm = get_joint_slm_sample_mask_numba(proposed_c_l, self.N_inv_l_mask, self.N_inv_sqrt_mask, self.d_lm, self.l_max, self.A_ell_list)
            print('Time to get s_lm f_lm:', time.time() - start_t)
            if j == 1:
                # Make some plots for the initial sample
                plt.figure()
                map = hp.alm2map(proposed_s_lm[0, :], nside=256)
                hp.mollview(map)
                plt.savefig('plot_folder_nside{}/{:.0f}_{:.0f}_slm.pdf'.format(self.nside, self.mask*180/np.pi, self.l_max))

                plt.figure()
                map = hp.alm2map(proposed_f_lm[0, :], nside=256)
                hp.mollview(map)
                plt.savefig('plot_folder_nside{}/{:.0f}_{:.0f}_flm.pdf'.format(self.nside, self.mask*180/np.pi, self.l_max))

                plt.figure()
                map = hp.alm2map(proposed_s_lm[0, :]+proposed_f_lm[0, :], nside=256)
                hp.mollview(map)
                plt.savefig('plot_folder_nside{}/{:.0f}_{:.0f}_slm_flm.pdf'.format(self.nside, self.mask*180/np.pi, self.l_max))

                ell = np.arange(l_max+1)
                plt.figure()
                plt.plot(ell, get_D_ell(hp.alm2cl(proposed_s_lm[0, :])), label='s_lm')
                plt.plot(ell, get_D_ell(hp.alm2cl(proposed_f_lm[0, :])), label='f_lm')
                plt.plot(ell, get_D_ell(hp.alm2cl(proposed_s_lm[0, :]+proposed_f_lm[0, :])), label='s_lm+f_lm')
                plt.plot(ell, get_D_ell(hp.alm2cl(self.a_lm[0, :])), label='a_lm')
                plt.legend()
                plt.savefig('plot_folder_nside{}/{:.0f}_{:.0f}_c_l.pdf'.format(self.nside, self.mask*180/np.pi, self.l_max))

            start_t = time.time()
            scaled_f_lm = get_scaled_flm(l_min, l_max, proposed_c_l, old_c_l, old_f_lm)

            start_t = time.time()
            if self.mask == 0:
                # Getting the acceptance rate is quick when no mask is used
                accepted = acceptance(old_s_lm, proposed_s_lm, old_f_lm, scaled_f_lm, old_c_l, proposed_c_l, l_min, l_max, self.b_l, self.N_l, self.d_lm)
            else:
                # but harder when there is a constant latitude mask.
                accepted = acceptance_masked(old_s_lm, proposed_s_lm, old_f_lm, scaled_f_lm, old_c_l, proposed_c_l, l_min, l_max, self.A_ell_list, self.N_inv_l_mask, self.d_lm)
            print('Time to get acceptance rate:', time.time() - start_t)

            tot += 1
            if accepted:
                old_c_l = proposed_c_l
                old_cosmo_param = proposed_cosmo_param
                old_s_lm = proposed_s_lm
                old_f_lm = proposed_f_lm
                accept_rate += 1
                
            list_of_cosmo_param[j, :] = old_cosmo_param
            print('Accept rate:', accept_rate/tot)

            if j % 10 == 0 or j == samples:
                # Save stuff
                if self.n_cosmo_variable == 1:
                    plt.figure()
                    plt.plot(list_of_cosmo_param[:j, 0])
                    plt.ylabel(self.labels[0])
                    plt.axhline(self.correct_cosmo_values[0], color='black')
                else:
                    fig, axs = plt.subplots(self.n_cosmo_variable)
                    for i in range(self.n_cosmo_variable):
                        axs[i].plot(list_of_cosmo_param[:j, i])
                        axs[i].set_ylabel(self.labels[i])
                        axs[i].axhline(self.correct_cosmo_values[i], color='black')
                plt.savefig('plot_folder_nside{}/{:.0f}_{:.0f}_chain.pdf'.format(self.nside, self.mask*180/np.pi, self.l_max))
                burnin = 5
                if j > 200:
                    figure = corner(
                        list_of_cosmo_param[burnin:j, :],
                        quantiles=[0.16, 0.5, 0.84],
                        labels=self.labels,
                        truths=self.correct_cosmo_values,
                        show_titles=True,
                        title_fmt='.4f'
                    )
                    plt.savefig('plot_folder_nside{}/{:.0f}_{:.0f}_corner.pdf'.format(self.nside, self.mask*180/np.pi, self.l_max))
                np.save('plot_folder_nside{}/{:.0f}_{:.0f}_current_chain.npy'.format(self.nside, self.mask*180/np.pi, self.l_max), list_of_cosmo_param[:j, :])

            elif j == 1:
                l = np.arange(l_min, l_max+1)
                _, axs = plt.subplots(2, 2)
                label = ['TT', 'EE', 'BB', 'TE']
                # Healpy = [TT, EE, BB, TE, EB, TB]
                i=0
                for ax in axs.flat:
                    ax.set_title(label[i])
                    
                    ax.plot(l, get_D_l(hp.sphtfunc.alm2cl(old_s_lm).T)[l_min:, i], label='Sigma_l from s_lm')
                    ax.plot(l, get_D_l(hp.sphtfunc.alm2cl(old_f_lm).T)[l_min:, i], label='Sigma_l from f_lm')
                    ax.plot(l, get_D_l(hp.sphtfunc.alm2cl(old_s_lm+old_f_lm).T)[l_min:, i], label='Sigma_l from s_lm+f_lm', alpha = 0.5)
                    ax.plot(l, get_D_l(hp.sphtfunc.alm2cl(self.a_lm).T)[l_min:, i], label='LCDM realization', alpha=0.5)

                    ax.set_ylabel(r"$C_{\ell}\ell (\ell + 1)/2\pi$")
                    ax.set_xlabel(r"$\ell$")
                    if i == 0: ax.legend()
                    i += 1
                
                plt.savefig('plot_folder_nside{}/{:.0f}_{:.0f}_sigma_l.pdf'.format(self.nside, self.mask*180/np.pi, self.l_max))

                
            j += 1
            pbar.update(1)

        print('Accept rate:', accept_rate/tot)
        print('Avg cosmo param:', np.mean(list_of_cosmo_param[:j, :], axis=0), 'Std cosmo param:', np.std(list_of_cosmo_param[:j, :], axis=0))