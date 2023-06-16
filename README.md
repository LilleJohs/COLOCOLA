# COLOCOLA - COsmoLOgical parameter estimation with a COnstant LAtitude mask

This code reproduces the Python results in Eskilt et al. (2023) where the cosmological parameters of the CMB are sampled from a simulated map with an applied constant latitude mask. We derive the full inverse noise covariance matrix in Appendix A of Eskilt et al. (2023), allowing for fast and accurate sampling of a large N_side map.

# How Do I Run It?

First, you make the simulations by running the file `make_simulations/mak_sim_512.py`, which creates a map file in that folder. Then you run `start_sampler.py` which samples the 6 LCDM parameters with no applied masks. This should be relatively quick. And you can change the parameters in `start_sampler.py` to your likening. Outputted plots and figures are made to a folder named `plot_folder_nside512`.

Then you can apply a constant latitude mask by changing the `mask_latitude_rad` parameter in `start_sampler.py`. This will be much slower, and you should use a previous run without a mask to create the proposal covariance matrix for the cosmological parameters. This can be done by changing the filename ending of the outputted chain file from `..._current_chain.npy` to `..._previous_chain.npy`.

## Is something not working?
Please let me know if you run into any problems or if anything is unclear!

You can either open a GitHub issue, or contact me on j.r.eskilt@astro.uio.no

## Citation

Feel free to use the code as you see fit, but if you use it for published results, please cite
* J. R. Eskilt et al. (2023) arXiv:2306.XXXXX [astro-ph.CO]
