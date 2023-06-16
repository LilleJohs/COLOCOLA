# COLOCOLA - COsmoLOgical parameter estimation with a COnstant LAtitude mask

This code reproduces the Python results in Eskilt et al. (2023) where the cosmological parameters of the CMB are sampled from a simulated map with an applied constant latitude mask. We derive the full inverse noise covariance matrix in Appendix A of Eskilt et al. (2023), allowing for relatively fast sampling of large N_side maps.

# How Do I Run It?

First, you make the simulations by running the file `make_simulations/mak_sim_512.py`, which creates a map file in that folder. Then you run `start_sampler.py` which samples the 6 LCDM parameters with no applied masks. This should be relatively quick. And you can change the parameters in `start_sampler.py` to your likening. Outputted plots and figures are made to a folder named `plot_folder_nside512`.

To have a more efficient run, you should use a previous chain file which creates an optimal proposal matrix for the cosmological parameters. Do a run with a suboptimal diagonal covariance matrix (by running the code as it is). Then you change the outputted chain file name to  hen change the name of the `..._current_chain.npy` to `..._previous_chain.npy` and then set `prev_chain = True` in `start_sampler.py`. Then run the code again. You will need to tune the `proposal_step` parameter to get an optimal acceptance rate of ~23%.

Then you can apply a constant latitude mask by changing the `mask_latitude_rad` parameter in `start_sampler.py`. This will be much slower, and you should use a previous chain file from the run without a mask to create the proposal covariance matrix for the cosmological parameters.

## Is something not working?
Please let me know if you run into any problems or if anything is unclear!

You can either open a GitHub issue, or contact me on j.r.eskilt@astro.uio.no

## Citation

Feel free to use the code as you see fit, but if you use it for published results, please cite
* J. R. Eskilt et al. (2023) arXiv:2306.XXXXX [astro-ph.CO]
