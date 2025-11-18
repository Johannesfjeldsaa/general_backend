from __future__ import annotations # allow forward references in type hints
# === === === === === === === === === === === === === === === === === ===
#
# Description: This module provides utility functions for masking data
# used in plotting diagnostics. It will mask data:
# - along a specified dimension in netCDF files.
# - based on a mask array if provided.
# - based on surface type if specified.
# This file is part of the ppe_diag package.
#
# Resources: https://tutorial.xarray.dev/intermediate/indexing/boolean-masking-indexing.html
#
# Author: Johannes Fjeldså
#
# === === === === === === === === === === === === === === === === === ===

import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path
from cartopy import crs as ccrs
from typing import Union
from xarray.plot.utils import label_from_attrs

from reversclim.utils.general.save_utils import save_figure
from reversclim.plotting.visual_profile import get_colormap

def fix_lon_coord(da: xr.DataArray) -> xr.DataArray:
    """fix lon coord from -180-180 format to 0-360

    Parameters
    ----------
    da : xr.DataArray
        a DataArray with a 'lon' coordinate

    Returns
    -------
    xr.DataArray
        a DataArray with the 'lon' coordinate adjusted to 0-360
    """
    lon_coords = da.lon.values
    lon_coords[lon_coords < 0] += 360
    da = da.assign_coords(lon=lon_coords)
    return da

def boolean_mask(da: xr.DataArray) -> xr.DataArray:
    """Convert a binary or float DataArray to boolean values.
    Binary to boolean values (1 -> True, 0 -> False)
    """
    return da.astype(bool)

def threshold_float_mask(da: xr.DataArray, threshold: float) -> xr.DataArray:
    """Convert a float DataArray to boolean values based on a threshold.
    Values above the threshold are True, values below are False.
    """
    return boolean_mask(da > threshold)

def _range_mode(
    min_max_tuple:    tuple[float, float],
    dim_array:  xr.DataArray,
    dim_min_max: tuple[float, float],
    range_mode: str
) -> np.ndarray:
    """Determines the values to include in a mask based on a specified range mode.
    NOTE: The inclusive modes are not yet implemented, so this function
    currently only supports the 'exclusive' mode.

    Parameters
    ----------
    min_max_tuple : tuple[float, float]
        A tuple containing the minimum and maximum values of the range.
        See `range_mode` for details on how this is handled.
    dim_array : xr.DataArray
        The dimension array to use for determining the array of values
        to use for masking. This is typically the dimension array of the
        data to be masked, e.g., `data['lon']` or `data['lat']`.
    dim_min_max : tuple[float, float]
        The minimum and maximum values the dimension can take, only used
        if `range_mode` is 'inclusive' or 'inclusive_min'/'inclusive_max'.
        This is used to determine the bounds for the endpoints of the range
        if inclusive handling is used.
    range_mode : str, optional
        Specifies how to handle the range of longitudes:
        - 'inclusive': includes the longitudes which is "touched" by the range.
            A "touched" longitude is one that min/max is within the bounds of
            assuming bounds is the average of two neighboring longitudes.
        - 'exclusive': includes only the longitudes that are strictly
            within the range, i.e., does not include the endpoints.
        - 'inclusive_min': inclusive handling for the minimum value,
            but exclusive for the maximum value.
        - 'inclusive_max': exclusive handling for the minimum value,
            but inclusive for the maximum value.
        Only applicable if `longitudes` is a tuple.
        Default is 'inclusive'.

    Raises
    ------
    ValueError
        If `min_max_tuple` does not contain exactly two elements.
        If `range_mode` is not one of the valid options.
    TypeError
        If `min_max_tuple` is not a tuple.
    """
    range_modes = ['inclusive', 'exclusive', 'inclusive_min', 'inclusive_max']
    if range_mode not in range_modes:
            err_msg = f"Invalid range_mode '{range_mode}'. " \
                f"Must be one of {range_modes}."
            logging.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
    if isinstance(min_max_tuple, tuple):
        if len(min_max_tuple) != 2:
            err_msg = "'min_max_tuple' must contain exactly two elements (min, max)."
            logging.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

        # Find the values to give to isin based on range_mode
        if 'inclusive' in range_mode:
            # get the bounds by taking the average of the two neighboring midpoints
            midpoints = dim_array.values
            logging.warning("Inclusive modes are not yet implemented. " \
                "Using exclusive mode instead.")
            # calculate the bounds between grid points
            # ...

        # !!!!!!!!!!!!!
        # temporarily use exclusive mode for all cases
        # !!!!!!!!!!!!!

        #elif range_mode == 'exclusive': #

        # take the values of dim_array that are strictly within the range
        # given by min_max_tuple
        included_values: np.ndarray = dim_array.where(
            (dim_array > min_max_tuple[0]) & (dim_array < min_max_tuple[1]),
            drop=True
        ).values

        return included_values
    else:
        err_msg = f"'min_max_tuple' must be a tuple, got {type(min_max_tuple)}"
        logging.error(err_msg, stack_info=True)
        raise TypeError(err_msg)

def _create_coord_mask(
    data: xr.DataArray | xr.Dataset,
    values: np.ndarray | list[float] | tuple[float, float],
    dim: str,
    valid_range: tuple[float, float],
    range_mode: str,
    coord_name: str,
    mask_name: str
) -> xr.DataArray:
    """Base function for creating coordinate masks.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        Input data containing the coordinate dimension
    values : np.ndarray | list[float] | tuple[float, float]
        Coordinate values or range to mask
    dim : str
        Dimension name to operate on
    valid_range : tuple[float, float]
        Minimum and maximum allowed values for validation
    range_mode : str
        How to handle range boundaries (see _range_mode for details)
    coord_name : str
        Name of coordinate for error messages (e.g., "longitude" or "latitude")
    mask_name : str
        Name of the mask to be created, used for attributes

    Raises
    ------
    TypeError
        If `values` is not a numpy array, list of floats, or a tuple.
    ValueError
        If `values` contains values outside of the specified `valid_range`.
    Returns
    -------
    xr.DataArray
        Boolean mask for specified coordinates
    """
    # Type checking
    if not isinstance(values, (np.ndarray, list, tuple)):
        err_msg = f"{coord_name} must be a numpy array, list, or tuple, got {type(values)}"
        logging.error(err_msg, stack_info=True)
        raise TypeError(err_msg)

    # Range validation
    if isinstance(values, (np.ndarray, list)):
        v_min, v_max = valid_range
        bad_values = [val for val in values if val < v_min or val > v_max]
        if bad_values:
            err_msg = (f"All {coord_name} values must be within {valid_range}. "
                      f"Found invalid values: {bad_values}")
            logging.error(err_msg, stack_info=True)
            raise ValueError(err_msg)

    # Handle range specification
    if isinstance(values, tuple):
        values_to_mask = _range_mode(
            min_max_tuple=values,
            dim_array=data[dim],
            dim_min_max=valid_range,
            range_mode=range_mode
        )
    else:
        values_to_mask = np.array(values)
    # create the mask
    mask = data[dim].isin(values_to_mask)

    # add attributes to the mask
    mask.attrs['mask_name'] = mask_name
    mask.attrs['mask_type'] = f"{coord_name}_mask"
    mask.attrs['mask_description'] = (
        f"Mask for {coord_name} values {values} "
        f"along dimension '{dim}' with range mode '{range_mode}'."
    )

    return mask

def _combine_attributes_masks(
    *masks: xr.DataArray,  # Corrected type annotation (DataArrays, not paths)
    combination_method: str
) -> dict:
    """Combine attributes from multiple masks into a single dictionary.

    Parameters
    ----------
    masks : tuple of xarray.DataArray
        Mask data arrays to combine
    combination_method : str
        Method used to combine masks ('intersection'/'union')

    Returns
    -------
    dict
        Combined attributes dictionary
    """
    attributes = {}

    for mask in masks:
        mask_name = mask.attrs.get('mask_name', 'unknown_mask')
        for attr, value in mask.attrs.items():
            if attr == 'mask_name':
                continue  # Skip mask_name attribute

            # Decode bytes to string if needed
            if isinstance(value, bytes):
                value = value.decode('utf-8')

            # Initialize nested dict structure
            if attr not in attributes:
                attributes[attr] = {}

            # Store value keyed by mask name
            attributes[attr][mask_name] = value

    # Post-process attributes
    for attr in list(attributes.keys()):
        # Check if all masks have identical values for this attribute
        unique_values = set(attributes[attr].values())

        if len(unique_values) == 1:
            # Single value: use directly
            attributes[attr] = next(iter(unique_values))
        else:
            # Multiple values: create combined string
            if combination_method == 'intersection':
                values_str = ',\n'.join(
                    f"{name}: {val}"
                    for name, val in attributes[attr].items()
                )
                attributes[attr] = f"intersection of:\n{values_str}"
            elif combination_method == 'union':
                values_str = ',\n'.join(
                    f"{name}: {val}"
                    for name, val in attributes[attr].items()
                )
                attributes[attr] = f"union of:\n{values_str}"
            else:
                err_msg = f"Unknown combination_method '{combination_method}'"
                logging.error(err_msg, stack_info=True)
                raise ValueError(err_msg)

    return attributes


def intersection_of_masks(
    *masks: xr.DataArray
) -> xr.DataArray:
    """Combines multiple boolean masks into a single mask using logical AND operation.

    Parameters
    ----------
    *masks : xr.DataArray
        The boolean masks to combine. All masks must have the same shape.

    Returns
    -------
    xr.DataArray
        A boolean mask where True indicates that all input masks are True.
    """
    if not masks:
        raise ValueError("At least one mask must be provided.")

    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = combined_mask & mask

    # rebuild all the attributes to reflect the combination
    attributes = _combine_attributes_masks(
        *masks,
        combination_method='intersection'
    )
    combined_mask.attrs = attributes

    return combined_mask

def union_of_masks(
    *masks: xr.DataArray
) -> xr.DataArray:
    """Combines multiple boolean masks into a single mask using logical OR operation.

    Parameters
    ----------
    *masks : xr.DataArray
        The boolean masks to combine. All masks must have the same shape.

    Returns
    -------
    xr.DataArray
        A boolean mask where True indicates that any input mask is True.
    """
    if not masks:
        raise ValueError("At least one mask must be provided.")

    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = combined_mask | mask

    # rebuild all the attributes to reflect the combination
    attributes = _combine_attributes_masks(
        *masks,
        combination_method='union'
    )
    combined_mask.attrs = attributes

    return combined_mask

def is_where_compatible(
    data:           Union[xr.DataArray, xr.Dataset],
    cond:           Union[xr.DataArray, np.ndarray],
    exit_on_error:  bool = True
) -> bool:
    """
    Checks if a condition is compatible with the data for masking.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The data to be masked.
    cond : xr.DataArray | np.ndarray
        The condition to check for compatibility with the data. Should
        be a boolean array and broadcastable to the shape of `data`.
    exit_on_error : bool, optional
        If True, will trigger the raise statements if the condition is
        not compatible. Defaults to True. If False, will log an error
        and return False.

    Raises
    ------
    TypeError
        If the condition is not a boolean array.
    ValueError
        If the condition is not broadcastable to the shape of the data.

    Returns
    -------
    bool
        True if the condition is compatible with the data, False otherwise.
    """
    # Check for boolean dtype
    if not hasattr(cond, 'dtype') or cond.dtype != bool:
        err_msg = f"Condition dtype is {cond.dtype} but should be boolean. " \
            "Please ensure the condition is a boolean array."
        logging.error(err_msg, stack_info=True)
        if exit_on_error:
            raise TypeError(err_msg)
        return False
    else:
        logging.debug("Condition is a boolean array.")

    # Check for broadcastability
    try:
        if isinstance(cond, np.ndarray):
            cond = xr.DataArray(cond)
        xr.broadcast(data, cond)
        logging.debug("Condition is broadcastable to the data shape.")
    except Exception:
        err_msg = "Condition is not broadcastable to the data shape. " \
            "Please ensure the condition is compatible with the data shape."
        logging.exception(err_msg, stack_info=True, exc_info=True)
        if exit_on_error:
            raise ValueError(err_msg)
        return False

    return True

def is_float_mask_compatible(
    data:           xr.DataArray,
    mask:           xr.DataArray | np.ndarray,
    exit_on_error:  bool = True,
    check_range:    bool = True,
    min_value:      float = 0.0,
    max_value:      float = 1.0,
) -> bool:
    """Checks if a float mask (weights) is compatible with the data for weighted masking.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The data to be masked.
    mask : xr.DataArray | np.ndarray
        The float mask (weights) to check for compatibility with the data.
    exit_on_error : bool, optional
        If True, will raise exceptions if the mask is not compatible.
        If False, will log an error and return False.
    check_range : bool, optional
        If True, check that all mask values are within [min_value, max_value].
    min_value : float, optional
        Minimum allowed value in the mask.
    max_value : float, optional
        Maximum allowed value in the mask.

    Raises
    ------
    TypeError
        If the mask is not a float array.
    ValueError
        If the mask is not broadcastable to the shape of the data, or values are out of range.

    Returns
    -------
    bool
        True if the mask is compatible with the data, False otherwise.
    """
    # Check for float dtype
    if not hasattr(mask, 'dtype') or not np.issubdtype(mask.dtype, np.floating):
        err_msg = f"Mask dtype is {getattr(mask, 'dtype', None)} but should be a float (weights)."
        logging.error(err_msg, stack_info=True)
        if exit_on_error:
            raise TypeError(err_msg)
        return False
    else:
        logging.debug("Mask is a float array (weights).")

    # Check for broadcastability
    try:
        if isinstance(mask, np.ndarray):
            mask = xr.DataArray(mask)
        xr.broadcast(data, mask)
        logging.debug("Mask is broadcastable to the data shape.")
    except Exception:
        err_msg = "Mask is not broadcastable to the data shape."
        logging.exception(err_msg, stack_info=True, exc_info=True)
        if exit_on_error:
            raise ValueError(err_msg)
        return False

    if check_range:
        valid = ((mask >= min_value) & (mask <= max_value)).all()
        if not bool(valid):
            err_msg = f"Mask values are not all in the range [{min_value}, {max_value}]."
            logging.error(err_msg, stack_info=True)
            if exit_on_error:
                raise ValueError(err_msg)
            return False
        else:
            logging.debug(f"All mask values are within [{min_value}, {max_value}].")

    return True

def _broadcast_mask_to_2d(mask: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Broadcasts a 1D mask to 2D using a reference dataset.

    Parameters
    ----------
    mask : xr.DataArray
        The 1D mask to broadcast.
    reference : xr.DataArray
        The reference dataset to match the mask's dimensions.

    Returns
    -------
    xr.DataArray
        The broadcasted 2D mask.
    """
    # Broadcast to reference grid
    _, mask = xr.broadcast(reference.isel(time=0, drop=True), mask)[1]
    return mask

def visualize_masks(
    created_masks:  dict[str, xr.DataArray],
    output_dir:     str | Path | None = None,
    reference_ds:   xr.DataArray | None = None
) -> None:
    """Visualizes the created masks using xarray's plotting capabilities.
    NOTE: this function will chooise the first entry along all dimensions
    except for 'lat' and 'lon' (or 'latitude' and 'longitude')

    Parameters
    ----------
    created_masks : dict[str, xr.DataArray]
        A dictionary of created masks where keys are mask names and values are DataArrays.
    output_dir : str | Path | None, optional
        The directory where the visualizations will be saved.
        If None, the visualizations will not be saved to disk.
    reference_ds : Optional[xr.Dataset], optional
        A reference dataset to use for broadcasting the masks to 2D.
        If a mask does not have both 'lat' and 'lon' dimensions,
        it will be broadcasted to match the reference dataset's grid.
        If not provided, the masks will only be visualized in one dimension

    Returns
    -------
    None
        Saves visualizations to the specified output directory.
    """
    for mask_name, mask in created_masks.items():

        logging.info(f"Visualizing mask: {mask_name}")

        # choose the first entry along all dimensions except for 'lat' and 'lon'
        # this is to ensure that the mask can be visualized properly
        for dim in mask.dims:
            if dim not in ['lat', 'lon']:
                logging.debug(f"Selecting first entry along dimension '{dim}' for mask '{mask_name}'.")
                mask = mask.isel({dim: 0})
        # Ensure mask is broadcasted to 2D if necessary
        plot_1d = False
        if any(dim not in mask.dims for dim in ['lat', 'lon']):
            if reference_ds is not None:
                mask = _broadcast_mask_to_2d(mask, reference_ds)
                logging.debug(f"Broadcasted mask '{mask_name}' to 2D using reference dataset.")
            else:
                warn_msg = "Mask does not have 'lat' and 'lon' dimensions and no reference dataset provided. " \
                    "Visualizing the mask in 1D only."
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.warning(warn_msg, stack_info=True)
                else:
                    logging.warning(warn_msg)
                plot_1d = True

        # if the mask is a boolean mask we set vmin and vmax to 0 and 1
        if isinstance(mask.dtype, bool):
            vmin, vmax = 0, 1
        else:
            vmin, vmax = mask.min().item(), mask.max().item()

        auto_title = label_from_attrs(mask)
        mask_info: dict  = mask.attrs
        if plot_1d:
            # since the mask is 1D, we can not plot it on a map
            # rather we plot it as a function of the latitude or longitude
            # collapsing the mask along the missing dimension
            # then using the cbar to visualize the value.
            # Determine which dimension is present
            if 'lat' in mask.dims:
                x_dim = 'lat'
            elif 'lon' in mask.dims:
                x_dim = 'lon'
            else:
                raise ValueError(f"Mask '{mask_name}' has neither 'lat' nor 'lon' dimensions, cannot plot.")

            x = mask[x_dim]
            values = mask.values

            # Make a 2D array by repeating the mask values (for "image" effect)
            img = np.tile(values, (10, 1))

            fig, ax = plt.subplots(figsize=(8, 2))
            c = ax.imshow(
                img,
                aspect='auto',
                cmap=get_colormap('mask_visualization'),
                extent=(float(x.min()), float(x.max()), 0.0, 1.0),
                vmin=vmin, vmax=vmax
            )
            ax.set_yticks([])
            ax.set_xlabel(x_dim)
            # get the current title on the axes
            ax.set_title(auto_title + f' - 1D mask: {mask_name}')

            plt.colorbar(c, ax=ax, orientation='vertical', label=mask_name, fraction=0.2)

            mask_info['Note'] = (
                "This mask was visualized in 1D because it does not have both 'lat' and 'lon' dimensions. "
                "The values are repeated vertically for visualization."
            )

        else:
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

            mask.plot(
                ax=ax,
                cmap=get_colormap('mask_visualization'),
                add_colorbar=True,
                cbar_kwargs={'label': mask_name},
                transform=ccrs.PlateCarree(),
                vmin=vmin, vmax=vmax
            ) # type: ignore

            # get the current title on the axes
            ax.set_title(auto_title + f' - Mask: {mask_name}')

        plt.tight_layout()
        if output_dir is not None:
            file_name = f"{mask_name.replace('&', '_')}_mask_visualization.png"
            output_path = Path(output_dir).joinpath(file_name)
            save_figure(fig, output_path, overwrite='prompt')
            logging.info(f'Visualization of mask {mask_name} saved to file {output_path.name} in directory {output_path.parent}.')
        else:
            plt.show()

def apply_mask(da: xr.DataArray, var: str, mask: xr.DataArray) -> xr.Dataset | None:
    try:
        masked_da = da.where(mask)
        ds = masked_da.to_dataset(name=var)
        da_attrs = da.attrs
        mask_attrs = mask.attrs
        ds.attrs['comment'] = ( \
            da_attrs.get('comment', '') +
            '. Masked using: ' + mask_attrs.get('mask_type', '')
        )
        ds.attrs['mask_description'] = mask_attrs.get('mask_description', '')
        return ds
    except Exception as e:
        logging.error(f"Error applying mask: {e}")