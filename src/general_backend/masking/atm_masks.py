"""Atmospheric and geographical masking functions for climate data analysis.

This module provides comprehensive masking capabilities for atmospheric and
oceanic climate data, including geographical, latitudinal, and surface-type
based masks. It supports both individual mask creation and combination of
multiple masks using logical operations.

Key Features:
    - Geographic masking: longitude and latitude based masks
    - Surface type masking: land/ocean masks using fraction data
    - Regional masking: AR6 climate regions and predefined zones
    - Mask combinations: logical AND/OR operations on multiple masks
    - Predefined regions: polar, midlatitude, tropical zones

Supported Mask Types:
    - Longitude/latitude coordinate masks
    - Land and ocean surface masks
    - AR6 climate assessment regions
    - Latitudinal bands (NH/SH polar, midlat, tropics)
    - Global masks
    - Custom combination masks using '&' and '|' operators

Examples:
    # Create individual masks
    lon_mask = create_longitude_mask(data, (-60, -10), "north_atlantic")
    lat_mask = create_latitude_mask(data, (50, 60), "subpolar")

    # Create multiple masks at once
    masks = create_masks(data, ["NH_polar", "land", "CEU"])

    # Combine masks
    combined = create_masks(data, ["NH_polar&land"])
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from regionmask.defined_regions import ar6

from general_backend.logging.setup_logging import get_logger
from general_backend.masking.mask_utils import (
    _create_coord_mask,
    boolean_mask,
    intersection_of_masks,
    threshold_float_mask,
    union_of_masks,
)

# set up logging
logger = get_logger(__name__)


def create_longitude_mask(
    data: xr.DataArray | xr.Dataset,
    longitudes: np.ndarray | list[float] | tuple[float, float],
    mask_name: str,
    dim: str = "lon",
    valid_range: tuple[float, float] = (0, 360),
    range_mode: str = "exclusive",
) -> xr.DataArray:
    """Creates a mask for the specified longitudes along a given dimension.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The data to be masked.
    longitudes : np.ndarray | list[float] | tuple[float, float]
        If np.ndarray or list, the longitudes to mask.
        If tuple, it is interpreted as a range (min, max) of longitudes,
        see 'range_mode' for details on handling.
    mask_name : str
        The name of the mask to be created. This will be used in the mask's
        attributes to identify the mask type and description.
        It should be a descriptive name that indicates the purpose of the mask,
        e.g., 'tropics_mask', 'NH_polar_mask', etc.
    dim : str, optional
        The dimension along which to apply the mask, default is 'lon'.
    valid_range : tuple[float, float], optional
        The valid range of longitudes, default is (0, 360).
        This is used to validate the input longitudes.
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
    TypeError
        If `longitudes` is not a numpy array, list of floats, or a tuple.
    ValueError
        If `longitudes` contains values are outside of `valid_range`.
        If `range_mode` is not one of the valid options.

    Returns
    -------
    xr.DataArray
        A boolean mask where True indicates the specified longitudes.
    """
    return _create_coord_mask(
        data=data,
        values=longitudes,
        dim=dim,
        valid_range=valid_range,
        range_mode=range_mode,
        coord_name="lon",
        mask_name=mask_name,
    )


def create_latitude_mask(
    data: xr.DataArray | xr.Dataset,
    latitudes: np.ndarray | list[float] | tuple[float, float],
    mask_name: str,
    dim: str = "lat",
    valid_range: tuple[float, float] = (-90, 90),
    range_mode: str = "exclusive",
) -> xr.DataArray:
    """Creates a mask for the specified latitudes along a given dimension.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The data to be masked.
    latitudes : np.ndarray | list[float] | tuple[float, float]
        If np.ndarray or list, the latitudes to mask.
        If tuple, it is interpreted as a range (min, max) of latitudes,
        see 'range_mode' for details on handling.
    mask_name : str
        The name of the mask to be created. This will be used in the mask's
        attributes to identify the mask type and description.
        It should be a descriptive name that indicates the purpose of the mask,
        e.g., 'tropics_mask', 'NH_polar_mask', etc.
    dim : str, optional
        The dimension along which to apply the mask, default is 'lon'.
    valid_range : tuple[float, float], optional
        The valid range of latitudes, default is (-90, 90).
        This is used to validate the input latitudes.
    range_mode : str, optional
        Specifies how to handle the range of latitudes:
        - 'inclusive': includes the latitudes which is "touched" by the range.
            A "touched" longitude is one that min/max is within the bounds of
            assuming bounds is the average of two neighboring latitudes.
        - 'exclusive': includes only the latitudes that are strictly
            within the range, i.e., does not include the endpoints.
        - 'inclusive_min': inclusive handling for the minimum value,
            but exclusive for the maximum value.
        - 'inclusive_max': exclusive handling for the minimum value,
            but inclusive for the maximum value.
        Only applicable if `latitudes` is a tuple.
        Default is 'inclusive'.

    Raises
    ------
    TypeError
        If `latitudes` is not a numpy array, list of floats, or a tuple.
    ValueError
        If `latitudes` contains values are outside of `valid_range`.
        If `range_mode` is not one of the valid options.

    Returns
    -------
    xr.DataArray
        A boolean mask where True indicates the specified latitudes.
    """
    return _create_coord_mask(
        data=data,
        values=latitudes,
        dim=dim,
        valid_range=valid_range,
        range_mode=range_mode,
        coord_name="lat",
        mask_name=mask_name,
    )


def create_sea_mask(
    fraction_threshold: float = 0.8,
    sea_frac_data: xr.DataArray | None = None,
    land_frac_data: xr.DataArray | None = None,
) -> xr.DataArray:
    """Creates a mask for the sea surface type.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The data to be masked.
    fraction_threshold : float, optional
        The fraction threshold for the sea surface type.
        If the fraction of the sea surface type is below this threshold,
        the mask will be False. Default is 0.8, meaning that at least 80%
        of the surface must be sea to be considered True.
    sea_frac_data : xr.DataArray, optional
        The sea fraction data to use for masking. If None, the function will
        look for the land_frac_data parameter.
        If both are None, an error is raised.
    land_frac_data : xr.DataArray, optional
        Land fraction data to take inverse of for sea fraction. No
        sea ice fraction considered. Used if sea_frac_data is not provided.
        If both are None, an error is raised.

    Returns
    -------
    xr.DataArray
        A boolean mask where True indicates the sea surface type.
    """
    if sea_frac_data is None:
        logger.debug(
            "No sea fraction data provided, checking for land fraction data."
        )
        if land_frac_data is None:
            raise ValueError("Both sea_frac_data and land_frac_data are None.")
        if "time" in land_frac_data.dims:
            land_frac_data = land_frac_data.squeeze(dim="time")
        sea_frac_data = 1 - land_frac_data
        logger.debug(
            "Sea fraction data computed as inverse of land fraction data."
        )
    else:
        if "time" in sea_frac_data.dims:
            sea_frac_data = sea_frac_data.squeeze(dim="time")

    thresholded = threshold_float_mask(sea_frac_data, fraction_threshold)

    # add attributes to the mask
    thresholded.attrs["mask_name"] = "sea_surface_mask"
    thresholded.attrs["fraction_threshold"] = fraction_threshold
    thresholded.attrs["mask_type"] = "surface_type_mask"
    thresholded.attrs["mask_description"] = (
        "A mask indicating the presence of sea surface type based on a "
        f"fraction threshold of {fraction_threshold * 100:.1f} %."
    )

    return thresholded


def create_land_mask(
    fraction_threshold: float = 0.8,
    land_frac_data: xr.DataArray | None = None,
    sea_frac_data: xr.DataArray | None = None,
) -> xr.DataArray:
    """Creates a mask for the land surface type.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The data to be masked.
    fraction_threshold : float, optional
        The fraction threshold for the land surface type. If the fraction
        of the land surface type is below this threshold, the mask will
        be False. Default is 0.8, meaning that at least 80% of the surface
        must be land to be considered True.
    land_frac_data : xr.DataArray, optional
        The land fraction data to use for masking. If None, the function will
        look for the sea_frac_data parameter.
        If both are None, an error is raised.
    sea_frac_data : xr.DataArray, optional
        Sea fraction data to take inverse of for land fraction. No
        sea ice fraction considered. Used if land_frac_data is not provided.
        If both are None, an error is raised.

    Returns
    -------
    xr.DataArray
        A boolean mask where True indicates the land surface type.
    """
    if land_frac_data is None:
        logger.debug(
            "No land fraction data provided, checking for sea fraction data."
        )
        if sea_frac_data is None:
            raise ValueError("Both land_frac_data and sea_frac_data are None.")
        if "time" in sea_frac_data.dims:
            # remove time dimension if present
            sea_frac_data = sea_frac_data.isel(time=0, drop=True)
        land_frac_data = 1 - sea_frac_data
        logger.debug(
            "Land fraction data computed as inverse of sea fraction data."
        )

    if "time" in land_frac_data.dims:
        land_frac_data = land_frac_data.isel(time=0, drop=True)

    thresholded = threshold_float_mask(land_frac_data, fraction_threshold)

    # add attributes to the mask
    thresholded.attrs["mask_name"] = "land_surface_mask"
    thresholded.attrs["fraction_threshold"] = fraction_threshold
    thresholded.attrs["mask_type"] = "surface_type_mask"
    thresholded.attrs["mask_description"] = (
        "A mask indicating the presence of land surface type based on a "
        f"fraction threshold of {fraction_threshold * 100:.1f} %."
    )

    return thresholded


def create_ar6_region_mask(
    data: xr.DataArray | xr.Dataset, region_abbrev: str
) -> xr.DataArray:
    """Creates a mask for a specified AR6 region.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The data to be masked.
    region_abbrev : str
        The abbreviation of the AR6 region to mask.

    Returns
    -------
    xr.DataArray
        A boolean mask where True indicates the specified AR6 region.
    """
    mask = ar6.all.mask(data)
    index = ar6.all.abbrevs.index(region_abbrev)
    mask = mask.where(mask == index)

    # set the index values to True, else False
    mask = mask.notnull()
    mask = boolean_mask(mask)

    # add attributes to the mask
    mask.attrs["mask_name"] = f"AR6_{region_abbrev}_mask"
    mask.attrs["mask_type"] = "region_mask"
    mask.attrs["mask_description"] = (
        f"A mask indicating the presence of the AR6 region '{region_abbrev}'."
    )

    return mask


ar6_regions = list(ar6.all.abbrevs)
implemented_masks = [
    "global",
    "NH_polar",
    "NH_midlat",
    "NH_tropics",
    "tropics",
    "SH_polar",
    "SH_midlat",
    "SH_tropics",
    "land",
    "ocean",
]
implemented_masks.extend(ar6_regions)
mask_latbnds_mapping = {
    "NH_polar": (60, 90),
    "NH_midlat": (30, 60),
    "NH_tropics": (0, 30),
    "tropics": (-30, 30),
    "SH_polar": (-90, -60),
    "SH_midlat": (-60, -30),
    "SH_tropics": (-30, 0),
}


def _create_an_implemented_mask(
    dataset:        xr.Dataset,
    mask_name:      str,
    landfrac_mask:  xr.DataArray | None = None,
    oceanfrac_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Create a mask based on the provided mask name.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to create the mask for.
    mask_name : str
        The name of the mask to create. Should be one of the implemented masks.

    Returns
    -------
    xr.DataArray
        The created mask as a boolean DataArray.

    Raises
    ------
    ValueError
        If the mask_name is not implemented or if required data for
        land/ocean masks is not provided.
    """
    if mask_name == "global":
        # same lat and lon as dataset
        # then we make a boolean mask of all True values
        return boolean_mask(
            xr.DataArray(
                data=np.ones_like(dataset["lat"]),
                coords={"lat": dataset["lat"]},
                dims=["lat"],
                attrs={
                    "mask_name": "global_mask",
                    "mask_type": "global_mask",
                    "mask_description": "A global mask including all ds lats.",
                },
            )
        )
    elif mask_name in mask_latbnds_mapping:
        latbnds = mask_latbnds_mapping[mask_name]
        return create_latitude_mask(
            dataset,
            latitudes=latbnds,
            mask_name=mask_name,
            dim="lat",
            valid_range=(-90, 90),
            range_mode="exclusive",
        )
    elif mask_name == "land":
        if any(mask is not None for mask in [landfrac_mask, oceanfrac_mask]):
            return create_land_mask(
                fraction_threshold=0.8,
                land_frac_data=landfrac_mask,
                sea_frac_data=oceanfrac_mask,
            )
        else:
            err_msg = (
                "to create the land mask provide either "
                "'landfrac_mask' or 'oceanfrac_mask'"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
    elif mask_name == "ocean":
        if any(mask is not None for mask in [landfrac_mask, oceanfrac_mask]):
            return create_sea_mask(
                fraction_threshold=0.8,
                land_frac_data=landfrac_mask,
                sea_frac_data=oceanfrac_mask,
            )
        else:
            err_msg = (
                "to create the ocean mask provide either "
                "'landfrac_mask' or 'oceanfrac_mask'"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
    elif mask_name in ar6_regions:
        return create_ar6_region_mask(dataset, region_abbrev=mask_name)
    else:
        err_msg = (
            f"Mask '{mask_name}' is not implemented. "
            f"Use one of {',\n'.join(implemented_masks)}."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)


def create_masks(
    dataset:        xr.Dataset,
    masks:          list[str],
    landfrac_mask:  xr.DataArray | None = None,
    oceanfrac_mask: xr.DataArray | None = None,
) -> dict[str, xr.DataArray]:
    """Create multiple masks based on the provided list of mask names.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to create the masks for.
    masks : list[str]
        A list of mask names to create.
    landfrac_mask : xr.DataArray | None, optional
        Land fraction mask data, by default None
    oceanfrac_mask : xr.DataArray | None, optional
        Ocean fraction mask data, by default None

    Returns
    -------
    dict[str, xr.DataArray]
        A dictionary mapping mask names to the created masks.

    Raises
    ------
    ValueError
        If a mask name is not implemented or a combination mask is invalid.
    TypeError
        If a mask name is not a string.
    ValueError
        If there is an error combining masks.
    """
    created_masks = {}
    and_masks = []
    or_masks = []
    for mask in masks:
        if not isinstance(mask, str):
            err_msg = f"Mask '{mask}' is not a string."
            logger.error(err_msg, stack_info=True)
            raise TypeError(err_msg)

        if mask in implemented_masks:
            logger.info("Attempting to create mask %s", mask)

            created_masks[mask] = _create_an_implemented_mask(
                dataset,
                mask,
                landfrac_mask=landfrac_mask,
                oceanfrac_mask=oceanfrac_mask,
            )
            logger.info("Created mask %s", mask)
        else:
            # check if it is a combination mask
            if "&" in mask:
                and_masks.append(mask)
            elif "|" in mask:
                or_masks.append(mask)
            else:
                err_msg = (
                    f"Mask '{mask}' is not implemented and "
                    "is not a combination mask."
                )
                logger.error(err_msg, stack_info=True)
                raise ValueError(err_msg)

    for mask_string, operation in zip(
        [*and_masks, *or_masks], ["&"] * len(and_masks) + ["|"] * len(or_masks)
    ):
        logger.info("Attempting to create mask %s", mask_string)

        mask_tuple = tuple(mask_string.split(operation))
        for mask in mask_tuple:
            if mask in created_masks:
                continue
            elif mask in implemented_masks:
                created_masks[mask] = _create_an_implemented_mask(
                    dataset, mask
                )
            else:
                err_msg = (
                    f"Mask '{mask}' in combination mask"
                    f"'{mask_string}' is not implemented."
                )
                logger.error(err_msg, stack_info=True)
                raise ValueError(err_msg)
        # combine the masks
        try:
            if operation == "|":
                mask_list = [created_masks[m] for m in mask_tuple]
                created_masks[mask_string] = union_of_masks(*mask_list)
                logger.info("Create mask %s", mask_string)
            elif operation == "&":
                mask_list = [created_masks[m] for m in mask_tuple]
                created_masks[mask_string] = intersection_of_masks(*mask_list)
                created_masks[mask_string] = intersection_of_masks(*mask_list)
                logger.info("Create mask %s", mask_string)
        except Exception as err:
            mask_tuple_as_str = ";".join(
                [f"{mask}: \n{created_masks[mask]}" for mask in mask_tuple]
            )
            err_msg = (
                f"Error combining masks for '{mask_string}': {err} \n"
                "Please ensure all masks are compatible for combination "
                f"- mask elements: {mask_tuple_as_str}"
            )
            logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg) from err

    return created_masks
