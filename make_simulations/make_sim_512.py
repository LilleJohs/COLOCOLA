import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
import camb

nside = 512
l_max = nside*3

beam_deg = 0.22 # Roughly equivalent to 70 GHz beam

sigma = 50 # muK / pixel
N_pix = 12 * nside**2
iso_N_lm_map = np.ones([3, N_pix]) * sigma
iso_N_lm_map[1:, :] *= np.sqrt(2)

random = np.random.normal(size=iso_N_lm_map.shape)*iso_N_lm_map

cp = camb.set_params(tau=0.0544, ns=0.9649, H0=67.36, ombh2=0.02237, omch2=0.12, As=2.1e-09, lmax=2000)
camb_results = camb.get_results(cp)
all_cls_th = camb_results.get_cmb_power_spectra(lmax=2000, raw_cl=True, CMB_unit='muK')['total']

alm_cmb = hp.synalm(all_cls_th.transpose(), lmax=l_max, new=True)

noise_cl = hp.anafast(random)
c_l_realization = hp.alm2cl(alm_cmb)

def get_D_l(c_l):
        return np.array([c_l[l] * l * (l+1)/(2*np.pi) for l in range(len(c_l))])
           

pixel_window = np.array(hp.pixwin(nside=nside, pol=True, lmax=l_max))

# If you are using the official NPIPE beam

#beam = fits.open('Bl_TEB_npipe6v19_70GHzx70GHz.fits')[1]
#for i in tqdm(range(len(alm_cmb[0, :]))):
#  l, m = hp.Alm.getlm(lmax=l_max, i=i)
#  alm_cmb[0, i] *= beam.data['T'][l] * pixel_window[0, l]
#  alm_cmb[1, i] *= beam.data['E'][l] * pixel_window[1, l]
#  alm_cmb[2, i] *= beam.data['B'][l] * pixel_window[1, l]

# If you specify the beam yourself

beam = np.array(hp.gauss_beam(fwhm=beam_deg*np.pi/180, pol=False, lmax=l_max))
for i in tqdm(range(len(alm_cmb[0, :]))):
  l, m = hp.Alm.getlm(lmax=l_max, i=i)
  alm_cmb[0, i] *= beam[l] * pixel_window[0, l]
  alm_cmb[1, i] *= beam[l] * pixel_window[1, l]
  alm_cmb[2, i] *= beam[l] * pixel_window[1, l]

labels_spectra = ['TT', 'EE', 'BB', 'TE']
lmin = 2
i=0
fig, ax = plt.subplots(nrows=2, ncols=2)
for row in ax:
    for col in row:
      col.set_title(labels_spectra[i])
      # TT, EE, BB, TE
      col.plot(get_D_l(c_l_realization[i, :l_max])[lmin:], label='Realization')
      col.plot(get_D_l(all_cls_th[:l_max, i])[lmin:], label='CAMB')
      col.plot(get_D_l(noise_cl[i, :l_max])[lmin:], label='Noise')

      i += 1
plt.legend(bbox_to_anchor =(1.15, 1.5))
plt.savefig('realization_spectra_nside{}.pdf'.format(nside), bbox_inches='tight')

cmb_map = hp.sphtfunc.alm2map(alm_cmb, nside=nside, pol=True)
cmb_map += random
hp.mollview(cmb_map[0, :])

plt.savefig('cmb_map_nside{}_iso.pdf'.format(nside))
hp.write_map('isotropic_noise_with_cmb_nside{}.fits'.format(nside), cmb_map, overwrite = True)