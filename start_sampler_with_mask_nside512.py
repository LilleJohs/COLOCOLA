import matplotlib.pyplot as plt
import numpy as np
import time

from joint_sampler import joint_sampler

samples = 100000
l_min = 2
l_max = 1200
noise_sigma = 50 # Noise per pixel for a given nside
nside = 512

# No mask is used. Use this setting first to make everything works.
mask_latitude_rad = 0

# The constant latitude mask. 0.1 here gives the sky fraction removed
# which is equivalent of 6 degrees
#mask_latitude_rad = np.arcsin(0.1)


proposal_step = 0.8 # This has to be fine-tuned
b = joint_sampler(
  # Where the simulations are
  load_sim_file = 'make_simulations/isotropic_noise_with_cmb_nside{}.fits'.format(nside),
  # How many cosmological parameters are you sampling? All 6 LCDM paramters?
  n_cosmo_variable = 6,
  # Beam: String points to a beam transfer file, float gives the FWHM in degrees.
  beam_deg = 0.22,
  noise_sigma = noise_sigma,
  l_min = l_min,
  l_max = l_max,
  # Make a proposal covariance from a previous chain. If False, a badly tuned diagonal covariance matrix is used
  # If True, it looks for a `plot_folder_nside_512/{mask_in_deg}_{lmax}_previous_chain.npy` file
  # If you want to use a mask, make a chain without a mask first as this is faster to run. Then use that as a prev_chain.
  prev_chain = False,
  proposal_step = proposal_step,
  mask = mask_latitude_rad,
  nside = nside
)
b.plot_spectras() # Plot the spectra

start = time.time()
b.start_joint_sampler(samples=samples, burnin = 5)
end = time.time()
tot = end-start
print('{} seconds doing {} samples, which gives {} samples/second'.format(tot, samples, samples/tot ))


