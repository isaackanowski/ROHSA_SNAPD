"""
This is the main file to run the ROHSA-SNAPD code
"""

# make jax 64-bit
from jax import config
config.update('jax_enable_x64', True)

# general imports
from astropy.io import fits
import numpy as np
from scipy import optimize
from jax import grad

# local imports
from ROHSA_SNAPD_funcs import ROHSA_kernel, ROHSA_SNAPD_bounds, cost

# this global variable stores the minimsier params at each iteration
global params_history
params_history = []

def minimiser_callback(xk):
    '''
    THis function saves the parameter history of the run and is called after an iteration of scipy.optimize.minimize()
    :param xk:
    :return:
    '''
    params_history.append(xk)

def fit(fits_path, lambda_flux, lambda_mu, lambda_sig, raveled_init_param_maps,
        sigma_LSF_channels=0., tol=1e-6, max_its=1e7,
        bounds_flux=(1e-6, 1000.0), bounds_mu=(1e-6, 10.), bounds_sigma=(1e-6, 10.),oversample_factor=4,use_Cauchy_likelihood=False,
        skyline_mask=None,spatial_mask=None,
        save_param_history=False):
    """
    The main fitting code

    :param fits_path: A path to a fits file which contains the hdus:
                        - 'DATA' - contains a flux cube of the observed object.
                        - 'RMS' - contains a cube of the noise rms for all values in 'DATA'.
                        - 'PSF' - a 2D image of the PSF of the data (at the resolution of the scaled data.
                            (The resolution of the PSF should be the same as the scaled data
                            ie. native pixel size/oversample_factor)
    :param lambda_flux: The spatial coherence of the amplitude fields.
    :param lambda_mu: The spatial coherence of the position fields.
    :param lambda_sig: The spatial coherence of the dispersion fields.
    :param raveled_init_param_maps: A ravelled array of three parameter maps to be used as initial conditions,
                        which are the amplitude, mu and sigma maps (in units of wavelength channels) returned
                        by a Gaussian fit to the spectra of each spatial pixel.
                        OR
                        If are using a constant dispersion, the array should be a ravelled array of two parameter maps,
                        amp and mu, with a single dispersion value appended on the end.
    :param sigma_LSF_channels: The standard deviation of the line spread function (LSF) in units of wavelength channels.
    :param tol: ftol = factr*step_size. For more details, see scipy.optimize.minimize() documentation.
                 (https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
    :param max_its: Maximum iterations to complete if ftol is not reached.
    :param bounds_flux: The possible range of amplitude values to search
    :param bounds_mu: The possible range of position values to search (in units of wavelength channels)
    :param bounds_sigma: The possible range of dispersion values to search (in units of wavelength channels)
    :param oversample_factor: The factor by which to oversample the parameter maps when
                        calculating the cost function to account for pixelisation
    :param use_Cauchy_likelihood: Whether to use a negative log Cauchy distribution in the cost function, or if False, chi-sqrd
    :param skyline_mask: A 1D bool array with len(skyline_mask) == n spectral channels, where True is to remove from the L(theta) calculation (or None if no masking).
    :param spatial_mask: A ravelled bool array which when resized is a 2D map of regions to be masked, where True is to remove from the L(theta) calculation (or None if no masking).
    :param save_param_history: If True, the optimiser results are stored for each iteration and are saved at the end.
    """

    # load fits hdul
    hdul = fits.open(fits_path)
    data = hdul['DATA'].data
    rms = hdul['RMS'].data
    psf_im = hdul['PSF'].data
    kernel = ROHSA_kernel()

    # define bounds of optimisation
    bounds = ROHSA_SNAPD_bounds(params=raveled_init_param_maps, data_shape=data.shape,
                          bounds_flux=bounds_flux, bounds_mu=bounds_mu, bounds_sigma=bounds_sigma)

    # callback to save param history if needed
    if save_param_history:
        callback_func = minimiser_callback
    else:
        callback_func = None

    # Use gradient-descent to minimise cost
    print('Starting optimisation')
    opt_output = optimize.minimize(cost, raveled_init_param_maps.astype(np.float64),
                                   args=(
                                       data.astype(np.float64), rms.astype(np.float64),
                                       psf_im.astype(np.float64),
                                       lambda_flux, lambda_mu, lambda_sig,
                                       kernel.astype(np.float64),
                                       sigma_LSF_channels, oversample_factor, use_Cauchy_likelihood,
                                       skyline_mask, spatial_mask
                                   ),
                                   jac = grad(cost),
                                   tol=tol,
                                   bounds=bounds, method='L-BFGS-B',
                                   options={'maxiter': max_its, 'maxfun': 1e100, 'iprint': 50},
                                   callback=callback_func)

    # save parameter history array - if needed
    if save_param_history:
        print('Saving parameter history array.')
        np.save(fits_path[:-5] + '_param_history_array.npy', np.array(params_history))

    # return optimiser output
    return opt_output
