import numba as nb
import numpy as np
from scipy.linalg import sqrtm
import scipy
import pyshtools as pysh
import tqdm
import sys
import healpy as hp

def get_D_l(c_l):
    # Get D_ell from C_ell if D_ell has more than just one index (only TT for example)
    D_ell = np.zeros(c_l.shape)
    for l in range(len(c_l[:, 0])):
        D_ell[l, :] = c_l[l, :] * l * (l+1)/(2*np.pi)
    return D_ell

def get_D_ell(c_l):
    # Get D_ell from C_ell if D_ell only one index (only TT for example)
    D_ell = np.zeros(c_l.shape)
    for l in range(len(c_l)):
        D_ell[l] = c_l[l] * l * (l+1)/(2*np.pi)
    return D_ell

@nb.njit
def get_idx(l_max, l, m):
    # From the Healpy library. But we copy it here so that Numba can use it
    return m * (2 * l_max + 1 - m) // 2 + l

def get_scaled_flm(l_min, l_max, proposed_c_l, old_c_l, old_f_lm):
  # Get the scaled f_lm
  scaled_f_lm = np.zeros(old_f_lm.shape, dtype=np.complex128)
  for l in range(l_min, l_max+1):
    prop_S_matrix = np.array([
        [proposed_c_l[l, 0], proposed_c_l[l, 3], 0],
        [proposed_c_l[l, 3], proposed_c_l[l, 1], 0],
        [0,                  0,                  proposed_c_l[l, 2]]])
    old_S_matrix = np.array([
        [old_c_l[l, 0], old_c_l[l, 3], 0],
        [old_c_l[l, 3], old_c_l[l, 1], 0],
        [0,             0,             old_c_l[l, 2]]])
    old_S_matrix_inv = np.linalg.inv(old_S_matrix)

    pre_factor = np.dot(sqrtm(prop_S_matrix), sqrtm(old_S_matrix_inv))

    index = get_idx(l_max, l, np.arange(l+1))
    scaled_f_lm[:, index] = np.dot(pre_factor,  old_f_lm[:, index])
  return scaled_f_lm

def get_N_inv_integral(x_list, sph_harm, l_max):
    N_inv_ell = np.zeros((l_max+1, l_max+1, l_max+1), dtype=np.complex128)

    # Do the actual integration numerically.
    for m in tqdm.tqdm(range(0, l_max+1)):
        for ell in range(2, l_max+1):
            if m > ell:
                continue
            id = get_idx(l_max, ell, m)
            for ell_p in range(ell, l_max+1, 2):
                id_p = get_idx(l_max, ell_p, m)
                I = - scipy.integrate.simpson(sph_harm[:, id] * sph_harm[:, id_p], x_list)
                if ell == ell_p:
                    I += 1/(4*np.pi)
                N_inv_ell[m, ell, ell_p] = I
                if ell != ell_p:
                    N_inv_ell[m, ell_p, ell] = I
    return N_inv_ell

def test_x_num_steps(num_x_list, b, l, m, lp):
    # For numerical integration, we need many points on the x-axis
    x_list = np.linspace(0, np.sin(b), num=int(num_x_list), endpoint=True)
    integrand = np.zeros(len(x_list))
    for i in tqdm.tqdm(range(len(x_list))):
        x = x_list[i]
        # Get all the spherical harmonics without the phase. Look at my notes.
        sph_harm_lm_no_phase = np.real(pysh.expand.spharm_lm(l, m, np.arccos(x), 0, kind = 'complex', degrees = False, normalization = 'ortho', csphase=-1))
        sph_harm_l_pm_p_no_phase = np.real(pysh.expand.spharm_lm(lp, m, np.arccos(x), 0, kind = 'complex', degrees = False, normalization = 'ortho', csphase=-1))

        integrand[i] = - sph_harm_lm_no_phase*sph_harm_l_pm_p_no_phase

    I = scipy.integrate.simpson(integrand, x_list)
    if l == lp:
        I += 1/(4*np.pi)
    return I


def get_mask_N_inv_ell(b, l_max):
    # If const lat mask is applied. Then we need to calcualte N_{ell m, ell' m'}
    N_inv_ell = np.zeros((l_max+1, l_max+1, l_max+1), dtype=np.complex128)
    m = np.arange(0, l_max+1)

    num_x_list = 2000
    while True:
        I = test_x_num_steps(num_x_list, b, l_max, 0, l_max)
        I2 = test_x_num_steps(int(2*num_x_list), b, l_max, 0, l_max)
        rel_diff = (I-I2)/I2
        print(rel_diff, I, I2)

        I = test_x_num_steps(num_x_list, b, l_max, 0, l_max-2)
        I2 = test_x_num_steps(int(2*num_x_list), b, l_max, 0, l_max-2)
        
        rel_diff = (I-I2)/I2
        print(rel_diff, I, I2)

        if rel_diff < 1e-2:
            break
        else:
            num_x_list = int(2 * num_x_list)
    
    x_list = np.linspace(0, np.sin(b), num=num_x_list, endpoint=True)
    sph_harm = np.zeros((len(x_list), hp.Alm.getsize(l_max)))
    for i in tqdm.tqdm(range(len(x_list))):
        x = x_list[i]

        # Get all the spherical harmonics without the phase. Equivalent to Tilde(P)_{ell m} in the paper
        all_sph_harm_no_phase = np.real(pysh.expand.spharm(l_max, np.arccos(x), 0, kind = 'complex', degrees = False, normalization = 'ortho', csphase=-1))
        for l in range(l_max+1):
            id = get_idx(l_max, l, m[0:l+1])
            sph_harm[i, id] = all_sph_harm_no_phase[0, l, m[0:l+1]]
    print('Size of sph_harm:', round(sys.getsizeof(sph_harm) / 1024 / 1024,2), 'MB')

    N_inv_ell = get_N_inv_integral(x_list, sph_harm, l_max)
    print(N_inv_ell[0, :])
    print(np.sum(N_inv_ell))

    print('Size of N_inv:', round(sys.getsizeof(N_inv_ell) / 1024 / 1024,2), 'MB')
    return N_inv_ell

@nb.njit(parallel=True)
def acceptance(old_s_lm, proposed_s_lm, old_f_lm, scaled_f_lm, old_c_l, proposed_c_l, l_min, l_max, B_l, N_l, d_lm):
    # Calculate the acceptance rate when no mask is applied (full sky)

    # Tot_a and tot_b is just the chi^2 for new and old sample, respectively.
    tot_a = np.zeros(3)
    tot_b = np.zeros(3)
    N_l_inv = np.array([[1.0, 0, 0], [0, 1/2, 0], [0, 0, 1/2]], dtype=nb.complex128) / N_l
    for l in nb.prange(l_min, l_max+1):
        a = np.zeros(3)
        b = np.zeros(3)
        A_l = np.diag(B_l[:, l])
        prop_S_matrix_inv = np.linalg.inv(np.array([
            [proposed_c_l[l, 0], proposed_c_l[l, 3], 0],
            [proposed_c_l[l, 3], proposed_c_l[l, 1], 0],
            [0,                  0,                  proposed_c_l[l, 2]]], dtype=nb.complex128))
        old_S_matrix_inv = np.linalg.inv(np.array([
            [old_c_l[l, 0], old_c_l[l, 3], 0],
            [old_c_l[l, 3], old_c_l[l, 1], 0],
            [0,             0,             old_c_l[l, 2]]], dtype=nb.complex128))

        for m in range(l+1):
            index = get_idx(l_max, l, m)
            mul = 1 if m == 0 else 2

            v_ip1 = d_lm[:, index] - np.dot(A_l, proposed_s_lm[:, index])
            v_i = d_lm[:, index] - np.dot(A_l, old_s_lm[:, index])
            a[0] += mul*np.abs(np.dot(np.conjugate(v_ip1), np.dot(N_l_inv, v_ip1)))
            b[0] += mul*np.abs(np.dot(np.conjugate(v_i), np.dot(N_l_inv, v_i)))

            a[1] += mul*np.abs(np.dot(
                np.conj(proposed_s_lm[:, index]), np.dot(prop_S_matrix_inv, proposed_s_lm[:, index])
                ))
            b[1] += mul*np.abs(np.dot(
                np.conj(old_s_lm[:, index]), np.dot(old_S_matrix_inv, old_s_lm[:, index])
                ))

            v_ip1 = np.dot(A_l, scaled_f_lm[:, index])
            v_i = np.dot(A_l, old_f_lm[:, index])
            a[2] += mul*np.abs(np.dot(np.conjugate(v_ip1), np.dot(N_l_inv, v_ip1)))
            b[2] += mul*np.abs(np.dot(np.conjugate(v_i), np.dot(N_l_inv, v_i)))

        tot_a += a
        tot_b += b

    tot = np.sum(tot_a - tot_b)
    A = np.real(np.exp(-1/2 * tot))
    eta = np.random.uniform(0, 1)
    print('log prob:', -tot/2, 'Eta:', eta, 'chisq:', np.sum(tot_a))
    #print('chi^2 for sample i+1:', tot_a)
    #print('chi^2 for sample i:', tot_b)
    #print('Delta chi^2:', tot_a-tot_b)
    
    return eta < A

@nb.njit(parallel=False)
def acceptance_masked(old_s_lm, proposed_s_lm, old_f_lm, scaled_f_lm, old_c_l, proposed_c_l, l_min, l_max, A_ell_list, N_inv_list, d_lm):
    # Calculate the acceptance rate when using a mask

    # Tot_a and tot_b is just the chi^2 for new and old sample, respectively.
    tot_a = np.zeros(3, dtype=np.complex128)
    tot_b = np.zeros(3, dtype=np.complex128)

    for l in nb.prange(l_min, l_max+1):
        prop_S_matrix_inv = np.linalg.inv(np.array([
            [proposed_c_l[l, 0], proposed_c_l[l, 3], 0],
            [proposed_c_l[l, 3], proposed_c_l[l, 1], 0],
            [0,                  0,                  proposed_c_l[l, 2]]], dtype=np.complex128))
        old_S_matrix_inv = np.linalg.inv(np.array([
            [old_c_l[l, 0], old_c_l[l, 3], 0],
            [old_c_l[l, 3], old_c_l[l, 1], 0],
            [0,             0,             old_c_l[l, 2]]], dtype=np.complex128))

        for m in range(l+1):
            index = get_idx(l_max, l, m)
            mul = 1 if m == 0 else 2

            tot_a[1] += mul*np.abs(np.dot(
                np.conj(proposed_s_lm[:, index]), np.dot(prop_S_matrix_inv, proposed_s_lm[:, index])
                ))
            tot_b[1] += mul*np.abs(np.dot(
                np.conj(old_s_lm[:, index]), np.dot(old_S_matrix_inv, old_s_lm[:, index])
                ))

    # For each m, do the matrix equation to get the chi^2 contribution for that m
    for m in nb.prange(l_max+1):
        mul = 1 if m == 0 else 2
        min_element = 2 if m <= 2 else m
        min_element_3 = int(min_element*3)
        
        A = A_ell_list[min_element_3:, min_element_3:]
        N_inv = N_inv_list[m, min_element_3:, min_element_3:]
        A_N_inv = np.dot(A, N_inv)
        A_N_inv_A = np.dot(A_N_inv, A)

        d_vector = np.zeros(3*(l_max+1)-min_element_3, dtype=np.complex128)
        s_hat_p1_vector = np.zeros(3*(l_max+1)-min_element_3, dtype=np.complex128)
        s_hat_vector = np.zeros(3*(l_max+1)-min_element_3, dtype=np.complex128)
        f_hat_p1_vector = np.zeros(3*(l_max+1)-min_element_3, dtype=np.complex128)
        f_hat_vector = np.zeros(3*(l_max+1)-min_element_3, dtype=np.complex128)

        for ell in range(min_element, l_max+1):
            index = get_idx(l_max, ell, m)
            elm = int(3*ell-min_element*3)
            d_vector[elm:elm+3] = d_lm[:, index]

            s_hat_p1_vector[elm:elm+3] = proposed_s_lm[:, index]
            s_hat_vector[elm:elm+3] = old_s_lm[:, index]

            f_hat_p1_vector[elm:elm+3] = scaled_f_lm[:, index]
            f_hat_vector[elm:elm+3] = old_f_lm[:, index]

        d_minus_A_s_hat_p1 = d_vector - np.dot(A, s_hat_p1_vector)
        d_minus_A_s_hat = d_vector - np.dot(A, s_hat_vector)
        tot_a[0] += mul*np.dot(np.conj(d_minus_A_s_hat_p1), np.dot(N_inv, d_minus_A_s_hat_p1))
        tot_b[0] += mul*np.dot(np.conj(d_minus_A_s_hat), np.dot(N_inv, d_minus_A_s_hat))
        
        tot_a[2] += mul*np.dot(np.conj(f_hat_p1_vector), np.dot(A_N_inv_A, f_hat_p1_vector))
        tot_b[2] += mul*np.dot(np.conj(f_hat_vector), np.dot(A_N_inv_A, f_hat_vector))

    tot = np.sum(tot_a - tot_b)
    A = np.real(np.exp(-1/2 * tot))
    eta = np.random.uniform(0, 1)
    print('delta chisq:', np.real(tot), 'Eta:', eta, 'chisq:', np.real(np.sum(tot_a)))
    print('(d-A*hat(s))N^(-1)(d-A*hat(s)) - hat(s)*S^(-1)*hat(s) - hat(f)*A*N^(-1)*A*hat(f)')
    print('new chi^2', np.real(tot_a))
    print('old chi^2', np.real(tot_b))
    
    return eta < A

@nb.njit()
def sqrt_2x2_matrix(M):
    # The matrix must be symmetric
    A = M[0, 0]
    B = M[0, 1]
    C = M[1, 0]
    D = M[1, 1]
    tau = A + D
    delta = A*D - B*C
    s = np.sqrt(delta)
    t = np.sqrt(tau + 2*s)
    return (M+s*np.identity(2)) / t

@nb.njit(parallel = True)
def get_joint_slm_sample_mask_numba(c_l, N_inv_list, N_inv_sqrt_list, d_lm, l_max, A_ell_list):
    # Get s_lm and f_lm when using a mask.

    s_lm = np.zeros(d_lm.shape, dtype=np.complex128)
    f_lm = np.zeros(d_lm.shape, dtype=np.complex128)

    S_inv_list = np.zeros((3*(l_max+1), 3*(l_max+1)), dtype=np.complex128)
    S_inv_sqrt_list = np.zeros((3*(l_max+1), 3*(l_max+1)), dtype=np.complex128)

    # Get CAMB power spectra, and put them into the S matrix [TT, EE, BB, TE]
    for ell in range(2, l_max+1):
        # Sometimes BB spectra is slightly negative probably due to numerical weirdness
        S_matrix = np.array([
            [c_l[ell, 0], c_l[ell, 3], 0],
            [c_l[ell, 3], c_l[ell, 1], 0],
            [0,           0,           c_l[ell, 2] if c_l[ell, 2] > 0 else 1e-10]
        ], dtype=np.complex128)
        S_inv = np.linalg.inv(S_matrix)
        elm = int(3*ell)
        S_inv_list[elm:elm+3, elm:elm+3] = S_inv
        
        S_inv_sqrt_list[elm:elm+2, elm:elm+2] = sqrt_2x2_matrix(S_inv[:2, :2])
        S_inv_sqrt_list[elm+2, elm+2] = np.sqrt(S_inv[2, 2])

    # Solve the equation for each m
    for m in nb.prange(l_max+1):
        min_element = 2 if m <= 2 else m
        elm_min = int(min_element*3)
        N_inv = N_inv_list[m, elm_min:, elm_min:]
        N_sqrt_inv = N_inv_sqrt_list[m, elm_min:, elm_min:]
        S_inv = S_inv_list[elm_min:, elm_min:]
        S_inv_sqrt = S_inv_sqrt_list[elm_min:, elm_min:]

        A = A_ell_list[elm_min:, elm_min:]
        A_N_inv = np.dot(A, N_inv)
        A_N_inv_A = np.dot(A_N_inv, A)
        A_N_inv_sqrt = np.dot(A, N_sqrt_inv)

        num_elements = int(3*(l_max+1-min_element))
        d_vector = np.zeros(num_elements, dtype=np.complex128)
        if m == 0:
            # a_lm when m = 0 are real
            omega_0 = np.zeros(num_elements, dtype=np.complex128)
            omega_1 = np.zeros(num_elements, dtype=np.complex128)
            for i in range(num_elements):
                omega_0[i] = np.random.normal()
                omega_1[i] = np.random.normal()
        else:
            omega_0 = np.zeros(num_elements, dtype=np.complex128)
            omega_1 = np.zeros(num_elements, dtype=np.complex128)
            for i in range(num_elements):
                omega_0[i] = np.random.normal()*np.sqrt(2)/2 + 1j*np.random.normal()*np.sqrt(2)/2
                omega_1[i] = np.random.normal()*np.sqrt(2)/2 + 1j*np.random.normal()*np.sqrt(2)/2
        
        for ell in range(min_element, l_max+1):
            elm = int(3*ell-min_element*3)
            d_vector[elm:elm+3] = d_lm[:, get_idx(l_max, ell, m)]

        s_l_m = np.linalg.solve(S_inv + A_N_inv_A, np.dot(A_N_inv, d_vector))
        f_l_m = np.linalg.solve(S_inv + A_N_inv_A, np.dot(S_inv_sqrt, omega_0) + np.dot(A_N_inv_sqrt, omega_1))
        for ell in range(min_element, l_max+1):
            idx = get_idx(l_max, ell, m)
            elm = int(3*ell-min_element*3)
            s_lm[:, idx] = s_l_m[elm:elm+3]
            f_lm[:, idx] = f_l_m[elm:elm+3]

    return s_lm, f_lm