"""
This file contains functions used to run ROHSA_SNAPD.py
"""

# make jax 64-bit
from jax import config
config.update('jax_enable_x64', True)

# general imports
import numpy as np
import jax.numpy as jnp
from jax import vmap
from jax.image import resize
from jax.scipy.signal import fftconvolve
import copy

def ROHSA_kernel():
    """
    This is the Laplacian kernel used when calculating the spatial smoothness of the model.
    This is the same kernel as used in the oringinal ROHSA code (Marchal+2019).

    :return: Laplacian kernel
    """
    return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 4.

def ROHSA_SNAPD_bounds(params, data_shape, bounds_flux, bounds_mu, bounds_sigma):
    """
    This function defines the bounds of the optimisation

    :param params: A ravelled array of three parameter maps to be used as initial conditions,
                    which are the flux, mu and sigma maps (in units of wavelength channels) returned
                    by a Gaussian fit to the spectra of each spatial pixel.
                    OR
                    If are using a constant dispersion, the array should be a ravelled array of two parameter maps,
                    flux and mu, with a single dispersion value appended on the end.
    :param data_shape: The observed galaxy data cube shape.
    :param bounds_flux: The possible range of flux values to search.
    :param bounds_mu: The possible range of mu values to search (in units of wavelength channels).
    :param bounds_sigma: The possible range of dispersion values to search (in units of wavelength channels).
    :return:
    """

    # must first reshape params
    # need to check which type of params have, either flux, mu and sig or just flux and mu with a final constant sig
    # if have all flux, mu and sig
    if len(params) == 3 * data_shape[1] * data_shape[2]:
        params = jnp.reshape(params.astype(jnp.float64), (3, data_shape[1], data_shape[2]))

        bounds_inf = np.zeros(params.shape)
        bounds_sup = np.zeros(params.shape)

        bounds_inf[0, :, :] = bounds_flux[0]
        bounds_inf[1, :, :] = bounds_mu[0]
        bounds_inf[2, :, :] = float(bounds_sigma[0])

        bounds_sup[0, :, :] = bounds_flux[1]
        bounds_sup[1, :, :] = bounds_mu[1]
        bounds_sup[2, :, :] = float(bounds_sigma[1])

        bounds = [(bounds_inf.ravel()[i], bounds_sup.ravel()[i]) for i in np.arange(len(bounds_sup.ravel()))]
    else:
        params = jnp.reshape(params[:-1].astype(jnp.float64), (2, data_shape[1], data_shape[2]))

        bounds_inf = np.zeros(params.shape)
        bounds_sup = np.zeros(params.shape)

        bounds_inf[0, :, :] = bounds_flux[0]
        bounds_inf[1, :, :] = bounds_mu[0]

        bounds_sup[0, :, :] = bounds_flux[1]
        bounds_sup[1, :, :] = bounds_mu[1]

        bounds = [(bounds_inf.ravel()[i], bounds_sup.ravel()[i]) for i in np.arange(len(bounds_sup.ravel()))]

        # then append bounds for sigma
        bounds.append((float(bounds_sigma[0]), float(bounds_sigma[1])))

    return np.array(bounds)

def jnp_gaussian(x, a, mu, sigma):
    """
    A JAX Numpy implementation of gaussian
    This function returns a single Gaussian function if one x value is given,
    or a list of Gaussian functions if a list of x-values are input
    :param x: A float, integer, list or np.ndarray
    :param a: The aplitude of the Gaussian function
    :param mu: The mean value of the Gaussian function
    :param sigma: The standard deviation of the Gaussian function
    :return:
    """
    x = jnp.asarray(x)

    return a * jnp.exp(-((x - mu) ** 2) / (2. * sigma ** 2))

def jnp_model(params, output_cube_shape):
    """
    This JAX Numpy function turns a list of Gaussian parameters into a 3D data cube.
    :param params: A raveled array of Gaussian parameters.
    :param output_cube_shape: The shape of the output data cube.
    :return:
    """

    # need to check which type of params have, either amp, mu and sig or just amp and mu with a final constant sig
    # if have all amp, mu and sig
    if len(params) == 3*output_cube_shape[1]*output_cube_shape[2]:
        params = jnp.reshape(params.astype(jnp.float64), (3, output_cube_shape[1], output_cube_shape[2]))
        # create cube of x-positions of each pixel in each slice by adding two new axes to 1D array
        x_cube = jnp.asarray(np.repeat(np.repeat(np.arange(output_cube_shape[0])[:, np.newaxis], output_cube_shape[1], axis=1)[..., np.newaxis],
                           output_cube_shape[2], axis=2))
        # create a cube with a Gaussian model at each pixel
        model_cube = jnp_gaussian(x_cube, params[0, :, :], params[1, :, :], params[2, :, :])
    else:
        # get constant dispersion value from end of params
        sigma = params[-1]
        params = jnp.reshape(params[:-1].astype(jnp.float64), (2, output_cube_shape[1], output_cube_shape[2]))
        # create cube of x-positions of each pixel in each slice by adding two new axes to 1D array
        x_cube = jnp.asarray(np.repeat(np.repeat(np.arange(output_cube_shape[0])[:, np.newaxis],
                                                 output_cube_shape[1], axis=1)[..., np.newaxis],
                                       output_cube_shape[2], axis=2))
        # create a cube with a Gaussian model at each pixel
        model_cube = jnp_gaussian(x_cube, params[0, :, :], params[1, :, :], sigma)

    return model_cube

def jnp_convolve_kernel(map, kernel):
    """
    A JAX Numpy 2D convolution function. The kernel shape must be odd in both dimensions.
    This function can be applied to 3D data cubes using jax.vmap
    For example, to convolve a 3D data cube with a 2D kernel, use:
    jax.vmap(jnp_convolve_kernel, in_axes=(0, None))(map, kernel)

    :param map:
    :param kernel:
    :return:
    """

    # convert to jnp arrays
    # convert both to array
    map_jnp = jnp.asarray(map, dtype=jnp.float64)
    kernel = jnp.asarray(kernel, dtype=jnp.float64)

    # calculate the size of zero-padding required to match the input size
    padding_height = kernel.shape[0] // 2
    padding_width = kernel.shape[1] // 2

    # pad the map - same as using symmetric convolution method
    padded_map = jnp.pad(map_jnp, ((padding_height, padding_height), (padding_width, padding_width)),
                         mode='symmetric')
    # Multiply the Fourier Transforms element-wise
    convolved_map = fftconvolve(padded_map, kernel, mode='valid')
    return convolved_map


def jnp_bin_data(data, spatial_bin_factor,spectral_bin_factor=1):
    """
    Bins the data by the given factor.
    As per https://stackoverflow.com/questions/14916545/numpy-rebinning-a-2d-array
    and
    Scipy cookbook https://scipy-cookbook.readthedocs.io/items/Rebinning.html

    :param data: A 3D data cube
    :param spatial_bin_factor: The factor to spatially bin the data cube (called oversample factor in the paper)
    :param spectral_bin_factor: The factor to spectrally bin the data cube (not used within the main code)
    :return:

    """
    # Calculate new dimensions after binning
    new_shape = (data.shape[0] // spectral_bin_factor, data.shape[1] // spatial_bin_factor, data.shape[2] // spatial_bin_factor)

    # Reshape data into 2D array with dimensions divisible by bin factor
    reshaped_data = data.reshape(new_shape[0], spectral_bin_factor, new_shape[1], spatial_bin_factor, new_shape[2], spatial_bin_factor)

    # Sum or average the binned pixels
    binned_data = jnp.sum(reshaped_data, axis=(1, 3, 5))

    return binned_data

def jnp_resize(param_map,scale_factor,conserve_flux=True):
    '''

    :param param_map: The kinematic parameter map
    :param scale_factor: The factor by which to resize the map (oversample factor in the paper)
    :param conserve_flux: Whether the resized map should conserve the flux of the original map
    :return:
    '''
    # make the param map into a jnp array
    jnp_param_map = jnp.array(param_map, dtype=jnp.float64)

    # resize does not conserve flux so need to scale the output by the original flux
    original_flux_sum = jnp.sum(jnp_param_map)

    resized_map = resize(jnp_param_map,
                         (jnp_param_map.shape[0] * scale_factor,jnp_param_map.shape[1] * scale_factor),
                         method='bilinear')

    output_resized_map_sum = jnp.sum(resized_map)

    if conserve_flux:
        # normalise the output and then scale by the original flux
        resized_map /= output_resized_map_sum
        resized_map *= original_flux_sum

    return resized_map


def kin_maps_to_convolved_cube(flux_mu_sig_param_maps,sigma_LSF_channels,psf_image,observed_data,oversample_factor):
    '''
    This function creates a convolved cube from the kinematic maps, adding LSF and convolving by the PSF.
    This is included to standardise the convolved cube creation process across the code.

    :param flux_mu_sig_param_maps: The flux, mu and sigma parameter maps.
    :param sigma_LSF_channels: The standard deviation of the line spread function (LSF) in units of wavelength channels.
    :param psf_image: The PSF image to convolve with the model (at the resolution of the scaled model).
    :param observed_data: The observed galaxy data cube.
    :param oversample_factor: The factor by which to oversample the parameter maps when
                        calculating the cost function to account for pixelisation
    :return: 
    '''
    
    param_maps = copy.deepcopy(flux_mu_sig_param_maps).astype(jnp.float64)

    # scale the parameter maps (to create a cube which is scaled spatially but not spectrally)
    scaled_flux = jnp_resize(param_maps[0],oversample_factor,conserve_flux=True)
    # do not want to conserve flux for mu and sigma maps
    scaled_mu = jnp_resize(param_maps[1],oversample_factor,conserve_flux=False)
    scaled_sig = jnp_resize(param_maps[2],oversample_factor,conserve_flux=False)

    scaled_param_maps = jnp.array([scaled_flux,scaled_mu,scaled_sig])

    # add sigma_LSF_channels to sigma map in quadrature
    scaled_param_maps = scaled_param_maps.at[2].set(jnp.sqrt(scaled_param_maps[2] ** 2 + sigma_LSF_channels ** 2))

    # convert flux into amplitude
    # dispersion cannot be zero so don't need to worry about nans
    scaled_param_maps = scaled_param_maps.at[0].set(scaled_param_maps[0] / jnp.sqrt(2. * jnp.pi * scaled_param_maps[2] ** 2))

    # make model cube from parameters (must scale the observed data shape spatially by the scale factor, but not spectrally)
    scaled_model_cube = jnp_model(scaled_param_maps.ravel(),
                                  (observed_data.shape[0],
                                   flux_mu_sig_param_maps.shape[1]*oversample_factor,
                                   flux_mu_sig_param_maps.shape[2]*oversample_factor))

    # convolve psf with scaled model in order to compare with data - use vmap to convolve each slice individually
    if psf_image is None:
        scaled_convolved_model = scaled_model_cube
    else:
        scaled_convolved_model = vmap(jnp_convolve_kernel, in_axes=(0, None))(scaled_model_cube, psf_image)

    # dont't need spectral binning
    descaled_convolved_model = jnp_bin_data(scaled_convolved_model, oversample_factor,1)

    # return convolved model
    return descaled_convolved_model

def neg_log_Cauchy(model,data,rms_error,skyline_mask=None,spatial_mask=None):
    '''
    A possible L(theta).

    The Cauchy likelihood is (pi*rms_error*(1+((data-model)/rms_error)**2)**-1,
    where we use the rms error of the data as the scale factor of the likelihood.

    So to use as L(theta), we take the negative log of this likelihood.

    :param model: The convolved ROHSA-SNAPD model
    :param data: The observed data
    :param rms_error: The observed RMS error
    :param skyline_mask: A 1D bool array with len(skyline_mask) == n spectral channels, where True is to remove from the L(theta) calculation (or None if no masking).
    :param spatial_mask: A ravelled bool array which when resized is a 2D map of regions to be masked, where True is to remove from the L(theta) calculation (or None if no masking).
    '''

    if (skyline_mask is None) and (spatial_mask is None):
        return jnp.nansum(jnp.log(rms_error)+jnp.log(1 + (((model - data) / rms_error) ** 2)))
    elif (spatial_mask is None):
        return jnp.nansum(jnp.log(rms_error[~skyline_mask,:,:]) + jnp.log(1 + (((model[~skyline_mask,:,:] - data[~skyline_mask,:,:]) / rms_error[~skyline_mask,:,:]) ** 2)))
    elif (skyline_mask is None):
        # must reshape the mask
        reshaped_spatial_mask = jnp.reshape(spatial_mask, (data.shape[1], data.shape[2]))
        return jnp.nansum(jnp.log(rms_error[:, ~reshaped_spatial_mask]) + jnp.log(1 + (((model[:, ~reshaped_spatial_mask] - data[:, ~reshaped_spatial_mask]) / rms_error[:, ~reshaped_spatial_mask]) ** 2)))
    else:
        # must reshape the mask
        reshaped_spatial_mask = jnp.reshape(spatial_mask, (data.shape[1], data.shape[2]))

        # want to spatially nan the cubes
        nan_rms_error = copy.deepcopy(rms_error)
        nan_rms_error[:, reshaped_spatial_mask] = float('nan')
        nan_model = model.at[:, reshaped_spatial_mask].set(float('nan'))
        nan_data = copy.deepcopy(data)
        nan_data[:, reshaped_spatial_mask] = float('nan')

        return jnp.nansum(jnp.log(nan_rms_error[~skyline_mask,:,:]) + jnp.log(1 + (((nan_model[~skyline_mask,:,:] - nan_data[~skyline_mask,:,:]) / nan_rms_error[~skyline_mask,:,:]) ** 2)))

def chi_sqr(model,data,rms_error,skyline_mask=None,spatial_mask=None):
    '''
    A possible L(theta).

    :param model: The convolved ROHSA-SNAPD model
    :param data: The observed data
    :param rms_error: The observed RMS error
    :param skyline_mask: A 1D bool array with len(skyline_mask) == n spectral channels, where True is to remove from the L(theta) calculation (or None if no masking).
    :param spatial_mask: A ravelled bool array which when resized is a 2D map of regions to be masked, where True is to remove from the L(theta) calculation (or None if no masking).
    '''
    if skyline_mask is None:
        return jnp.nansum(((model - data) / rms_error) ** 2)
    elif (spatial_mask is None):
        return jnp.nansum(((model[~skyline_mask,:,:] - data[~skyline_mask,:,:]) / rms_error[~skyline_mask,:,:]) ** 2)
    elif (skyline_mask is None):
        # must reshape the mask
        reshaped_spatial_mask = jnp.reshape(spatial_mask, (data.shape[1], data.shape[2]))
        return jnp.nansum(((model[:,~reshaped_spatial_mask] - data[:,~reshaped_spatial_mask]) / rms_error[:,~reshaped_spatial_mask]) ** 2)
    else:
        # must reshape the mask
        reshaped_spatial_mask = jnp.reshape(spatial_mask, (data.shape[1], data.shape[2]))

        # want to spatially nan the cubes
        nan_rms_error = copy.deepcopy(rms_error)
        nan_rms_error[:, reshaped_spatial_mask] = float('nan')
        nan_model = model.at[:, reshaped_spatial_mask].set(float('nan'))
        nan_data = copy.deepcopy(data)
        nan_data[:, reshaped_spatial_mask] = float('nan')

        return jnp.nansum(((nan_model[~skyline_mask,:,:] - nan_data[~skyline_mask,:,:]) / nan_rms_error[~skyline_mask,:,:]) ** 2)

def cost(params, observed_data, rms, psf_image, lambda_flux, lambda_mu, lambda_sig, kernel,sigma_LSF_channels,oversample_factor,use_Cauchy_likelihood=False,skyline_mask=None,spatial_mask=None):
    """
    This is the cost function that is minimised by the gradient descent fit.
    :param params: A raveled array of Gaussian parameters.
    :param observed_data: The observed galaxy data cube.
    :param rms: The root mean square (RMS) noise of each pixel in the data cube.
    :param psf_image: A 2D image of the PSF of the data (at the resolution of the scaled data.
    :param lambda_flux: The spatial coherence of the flux fields.
    :param lambda_mu: The spatial coherence of the position fields.
    :param lambda_sig: The spatial coherence of the dispersion fields.
    :param kernel: The Laplacian kernel used when calculating the spatial smoothness of the model.
    :param sigma_LSF_channels: The standard deviation of the line spread function (LSF) in units of wavelength channels.
    :param oversample_factor: The factor to oversample the parameter maps to account for sub-pixel velocity gradients.
    :param use_Cauchy_likelihood: Whether to use a negative log Cauchy distribution in the cost function, or if False, chi-sqrd
    :param skyline_mask: A 1D bool array with len(skyline_mask) == n spectral channels, where True is to remove from the L(theta) calculation (or None if no masking).
    :param spatial_mask: A ravelled bool array which when resized is a 2D map of regions to be masked, where True is to remove from the L(theta) calculation (or None if no masking).
    :return:
    """

    # convert pars to a jnp array
    jnp_pars = jnp.asarray(params)

    # need to check which type of params have, either flux, mu and sig or just flux and mu with a final constant sig
    if len(params) == 3 * observed_data.shape[1] * observed_data.shape[2]:
        reshaped_params = jnp.reshape(jnp_pars, (3, observed_data.shape[1], observed_data.shape[2]))
    else:
        # if have just flux and mu with a final constant sig, reshape and make a constant sigma map
        reshaped_flux_mu_params = jnp.reshape(jnp_pars[:-1], (2, observed_data.shape[1], observed_data.shape[2]))

        # create a constant sigma map
        reshaped_sig = jnp.full((observed_data.shape[1], observed_data.shape[2]), jnp_pars[-1])

        # append sigma map to amp and mu maps
        reshaped_params = jnp.append(reshaped_flux_mu_params, jnp.array([reshaped_sig]), axis=0)

    # make copy of original maps to use for enforcing the regularisation
    reshaped_params_for_regularisation = copy.deepcopy(reshaped_params)

    # apply psf convolution to model in order to compare with data - use vmap to convolve each slice individually
    convolved_model = kin_maps_to_convolved_cube(reshaped_params, sigma_LSF_channels, psf_image, observed_data, oversample_factor)

    if use_Cauchy_likelihood:
        J = neg_log_Cauchy(convolved_model,observed_data,rms,skyline_mask, spatial_mask)
    else:
        J = chi_sqr(convolved_model,observed_data,rms,skyline_mask, spatial_mask)

    # this convolves the data with the kernel to enforce smoothness of the solution
    lambda_times_R = 0.

    # make a list of lambda values
    lambdas = jnp.asarray([lambda_flux, lambda_mu, lambda_sig])

    # if use a constant dispersion, only need to regularise the flux and mu maps (so only first two lambdas)
    n_regularised_maps = 3 if len(params) == 3 * observed_data.shape[1] * observed_data.shape[2] else 2
    for i in np.arange(n_regularised_maps):

        # choose respective lambda
        lambda_val = lambdas[i]

        param_map = reshaped_params_for_regularisation[i, :, :]

        # convolve parameter maps with ROHSA Laplacian kernel
        conv = jnp_convolve_kernel(param_map, kernel)

        lambda_times_R += jnp.nansum((conv ** 2)) * lambda_val

    return 0.5 * (J + lambda_times_R)
