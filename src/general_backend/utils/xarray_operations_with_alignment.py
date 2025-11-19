"""
Xarray operations with explicit alignment checking.

This module provides wrapper functions for common xarray operations that ensure
proper coordinate alignment before performing calculations. Xarray natively
supports broadcasting and alignment but can silently drop non-aligned
coordinates. To avoid this, these helper functions use
align_kwargs = {"join": "exact"} by default, which will raise an error
if dimensions are not properly aligned.
The module includes functions for:
- Checking alignment between multiple xarray objects
- Performing arithmetic operations with alignment:
add, subtract, multiply, divide, power
- Broadcasting with alignment verification
"""

# ---
# import packages
# ---
import logging
from typing import Any

import xarray as xr

# import source code

# setup logging will be 'reversclim.some_module'
logger = logging.getLogger(__name__)
# align_kwargs default error message
ALIGN_KWARGS_ERR_MSG = """
Input xarray objects are not aligned according to
join='{align_kwargs_values}'.
"""
# ---
# Source code
# ---


def _check_align_kwargs(align_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Check and set default alignment kwargs. Local helper function.

    Parameters
    ----------
    align_kwargs : dict | None
        Dictionary of keyword arguments fed to xr.align() function.
        If None, the returned align_kwargs will be set to a default
        {"join": "exact", 'copy': True}.

    Returns
    -------
    dict
        Dictionary of alignment keyword arguments.

    Raises
    ------
    ValueError
        If align_kwargs is not the expected type.
    """
    if align_kwargs is None:
        align_kwargs = {"join": "exact", "copy": True}
    else:
        if not isinstance(align_kwargs, dict):
            err_msg = "align_kwargs must be a dictionary or None."
            logger.warning(err_msg)
            raise ValueError(err_msg)

    return align_kwargs


def check_alignment(
    *args:          xr.DataArray | xr.Dataset,
    align_kwargs:   dict[str, Any] | None = None
) -> tuple[xr.DataArray | xr.Dataset, ...] | bool:
    """Check if multiple xarray objects are aligned according to the
    specified alignment method. Default kwargs sets xr.align's join to
    'exact' to raise an error if coordinates do not match exactly.
    See xr.align documentation for other options:
    https://docs.xarray.dev/en/latest/generated/xarray.align.html#xarray.align

    Parameters
    ----------
    *args : xr.DataArray | xr.Dataset
        Multiple xarray objects to check for alignment.
    align_kwargs : dict, optional
        Additional keyword arguments to pass to xr.align, default None. If
        None, defaults to {"join": "exact", 'copy': True}.
        If a non-None argument is provided, the align_kwargs will override the
        default. Modify 'join' to change alignment behavior:
        * “outer”: use the union of object indexes
        * “inner”: use the intersection of object indexes
        * “left”: use indexes from the first object with each dimension
        * “right”: use indexes from the last object with each dimension
        * “exact”: instead of aligning, raise ValueError when indexes to be
        aligned are not equal
        * “override”: if indexes are of same size, rewrite indexes to be
        those of the first object with that dimension. Indexes for the same
        dimension must have the same size in all objects.
        See xr.align documentation for more details:
        https://docs.xarray.dev/en/latest/generated/xarray.align.html#xarray.align

    Returns
    -------
    tuple[xr.DataArray | xr.Dataset, ...] | bool
        Tuple of aligned objects if all objects are aligned according to the
        specified method, False otherwise.
    """
    align_kwargs = _check_align_kwargs(
        align_kwargs
    )  # set default align_kwargs if None and check type

    try:
        aligned = xr.align(*args, **align_kwargs)
        logger.debug("Aligned datasets sucessfull")
        return aligned
    except ValueError:
        logger.debug("Align datasets failed")
        return False


def _operation_with_alignment(
    *args:          xr.DataArray | xr.Dataset,
    operation:      str,
    align_kwargs:   dict[str, Any] | None = None,
) -> xr.DataArray | xr.Dataset:
    """Perform an operation on two (or more) xarray objects with alignment.
    Default kwargs sets xr.align's join to 'exact' to raise an error if
    coordinates do not match exactly. See xr.align documentation for other
    options:
    https://docs.xarray.dev/en/latest/generated/xarray.align.html#xarray.align

    Parameters
    ----------
    *args : xr.DataArray | xr.Dataset
        xarray objects to operate on.
    operation : str
        Operation to perform. Supported operations:
        'add', 'subtract', 'multiply', 'divide', 'power'.
    align_kwargs : dict, optional
        Additional keyword arguments to pass to xr.align, by default None.
        If None, defaults to {"join": "exact", 'copy': True}.
        See xr.align documentation for more details:
        https://docs.xarray.dev/en/latest/generated/xarray.align.html#xarray.align

    Returns
    -------
    xr.DataArray | xr.Dataset
        Result of the operation with aligned coordinates.
    """

    align_kwargs = _check_align_kwargs(align_kwargs)

    aligned: tuple[xr.DataArray | xr.Dataset, ...] | bool = check_alignment(
        *args, align_kwargs=align_kwargs
    )
    if aligned is False:
        err_msg = ALIGN_KWARGS_ERR_MSG.format(
            align_kwargs_values=align_kwargs.get("join", "exact")
        )
        logger.error(err_msg)
        raise ValueError(err_msg)
    if not isinstance(aligned, tuple):
        err_msg = (
            "Unexpected error during alignment check. "
            "Returned value is not a tuple nor False."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)
    cls = type(
        aligned[0]
    )  # type of the first argument (DataArray or Dataset) to use its methods
    operations = {
        "add": cls.__add__,
        "subtract": cls.__sub__,
        "multiply": cls.__mul__,
        "divide": cls.__truediv__,
        "power": cls.__pow__,
    }

    if operation not in operations:
        raise ValueError(f"Unsupported operation: {operation}")

    return operations[operation](*aligned)  # type: ignore


def multiply_with_alignment(
    *args:          xr.DataArray | xr.Dataset,
    align_kwargs:   dict[str, Any] | None = None,
) -> xr.DataArray | xr.Dataset:
    """Multiply two or more xarray objects with alignment.

    Parameters
    ----------
    *args : xr.DataArray | xr.Dataset
        xarray objects to multiply.
    align_kwargs : dict | None, optional
        Additional keyword arguments to pass to xr.align, by default None.
        If None, `{"join": "exact", 'copy': True}` is used.
        See xr.align documentation for more details:
        https://docs.xarray.dev/en/latest/generated/xarray.align.html#xarray.align

    Returns
    -------
    xr.DataArray | xr.Dataset
        Result of the multiplication with aligned coordinates.
    """
    return _operation_with_alignment(
        *args, operation="multiply", align_kwargs=align_kwargs
    )


def divide_with_alignment(
    a:              xr.DataArray | xr.Dataset,
    b:              xr.DataArray | xr.Dataset,
    align_kwargs:   dict[str, Any] | None = None,
) -> xr.DataArray | xr.Dataset:
    """Divide one xarray object by another with alignment.

    Parameters
    ----------
    a : xr.DataArray | xr.Dataset
        Numerator xarray object.
    b : xr.DataArray | xr.Dataset
        Denominator xarray object.
    align_kwargs : dict | None, optional
        Additional keyword arguments to pass to xr.align, by default None.
        If None, `{"join": "exact", 'copy': True}` is used.
        See xr.align documentation for more details:
        https://docs.xarray.dev/en/latest/generated/xarray.align.html#xarray.align

    Returns
    -------
    xr.DataArray | xr.Dataset
        Result of the division with aligned coordinates.
    """

    return _operation_with_alignment(
        a, b, operation="divide", align_kwargs=align_kwargs
    )


def add_with_alignment(
    *args:          xr.DataArray | xr.Dataset,
    align_kwargs:   dict[str, Any] | None = None,
) -> xr.DataArray | xr.Dataset:
    """Add two or more xarray objects with alignment.

    Parameters
    ----------
    *args : xr.DataArray | xr.Dataset
        xarray objects to add.
    align_kwargs : dict | None, optional
        Additional keyword arguments to pass to xr.align, by default None.
        If None, `{"join": "exact", 'copy': True}` is used.
        See xr.align documentation for more details:
        https://docs.xarray.dev/en/latest/generated/xarray.align.html#xarray.align

    Returns
    -------
    xr.DataArray | xr.Dataset
        Result of the addition with aligned coordinates.
    """
    return _operation_with_alignment(
        *args, operation="add", align_kwargs=align_kwargs
    )


def subtract_with_alignment(
    a:              xr.DataArray | xr.Dataset,
    b:              xr.DataArray | xr.Dataset,
    align_kwargs:   dict[str, Any] | None = None,
) -> xr.DataArray | xr.Dataset:
    """Subtract one xarray object from another one with alignment.

    Parameters
    ----------
    a : xr.DataArray | xr.Dataset
        Object to subtract from.
    b : xr.DataArray | xr.Dataset
        Object to subtract.
    align_kwargs : dict | None, optional
        Additional keyword arguments to pass to xr.align, by default None.
        If None, `{"join": "exact", 'copy': True}` is used.
        See xr.align documentation for more details:
        https://docs.xarray.dev/en/latest/generated/xarray.align.html#xarray.align

    Returns
    -------
    xr.DataArray | xr.Dataset
        Result of the subtraction with aligned coordinates.
    """
    return _operation_with_alignment(
        a, b, operation="subtract", align_kwargs=align_kwargs
    )


def power_with_alignment(
    a:              xr.DataArray | xr.Dataset,
    b:              xr.DataArray | xr.Dataset,
    align_kwargs:   dict[str, Any] | None = None,
) -> xr.DataArray | xr.Dataset:
    """Raise one xarray object to the power of another with alignment.

    Parameters
    ----------
    a : xr.DataArray | xr.Dataset
        Base xarray object.
    b : xr.DataArray | xr.Dataset
        Exponent xarray object.
    align_kwargs : dict | None, optional
        Additional keyword arguments to pass to xr.align, by default None.
        If None, `{"join": "exact", 'copy': True}` is used.
        See xr.align documentation for more details:
        https://docs.xarray.dev/en/latest/generated/xarray.align.html#xarray.align

    Returns
    -------
    xr.DataArray | xr.Dataset
        Result of a raised to the power of b with aligned coordinates.
    """
    return _operation_with_alignment(
        a, b, operation="power", align_kwargs=align_kwargs
    )


def broadcast_with_alignment(
    *args:              xr.DataArray | xr.Dataset,
    align_kwargs:       dict[str, Any] | None = None,
    broadcast_kwargs:   dict[str, Any] | None = None,
) -> tuple[xr.DataArray | xr.Dataset, ...]:
    """Broadcast *args using xr.broadcast. First check if all input
    xarray objects are aligned.

    Parameters
    ----------
    align_kwargs : dict | None, optional
        Additional keyword arguments to pass to xr.align, by default None.
        If None, `{"join": "exact", 'copy': True}` is used.
        See xr.align documentation for more details:
        https://docs.xarray.dev/en/latest/generated/xarray.align.html#xarray.align
    broadcast_kwargs : dict | None, optional
        Additional keyword arguments to pass to xr.broadcast, by default None.
        If None, no additional kwargs are passed.

    Returns
    -------
    tuple[xr.DataArray | xr.Dataset, ...]
        Broadcasted xarray objects.

    Raises
    ------
    ValueError
        If input xarray objects are not aligned.
    """

    align_kwargs = _check_align_kwargs(align_kwargs)
    if broadcast_kwargs is None:
        broadcast_kwargs = {}
    else:
        if not isinstance(broadcast_kwargs, dict):
            raise ValueError("broadcast_kwargs must be a dictionary or None.")

    aligned: tuple[xr.DataArray | xr.Dataset, ...] | bool = check_alignment(
        *args, align_kwargs=align_kwargs
    )
    if aligned is False:
        err_msg = ALIGN_KWARGS_ERR_MSG.format(
            align_kwargs_values=align_kwargs.get("join", "exact")
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    if isinstance(aligned, tuple):
        return xr.broadcast(*aligned, **broadcast_kwargs)
    else:
        err_msg = (
            "Unexpected error during alignment check. "
            "Returned value is not a tuple nor False."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

def ensure_DataArray(
    data: xr.DataArray | xr.Dataset
) -> xr.DataArray:
    """Ensure the input is an xarray DataArray. If a Dataset is provided,
    convert it to a DataArray if there is only one variable in the Dataset.
    if multiple variables are present, throw an error.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        Input xarray object.

    Returns
    -------
    xr.DataArray
        The input as a DataArray.

    Raises
    ------
    TypeError
        If the input is neither a DataArray nor a Dataset.
    """
    if isinstance(data, xr.DataArray):
        return data
    elif isinstance(data, xr.Dataset):
        if len(data.data_vars) != 1:
            err_msg = (
                "Input Dataset has multiple variables. "
                "Cannot convert to DataArray."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        first_var = list(data.data_vars.keys())[0]
        logger.debug(
            "Input is a Dataset. Converting to DataArray by selecting "
            "the first variable: %s",
            first_var,
        )
        return data[first_var]
    else:
        err_msg = (
            f"Input must be an xarray DataArray or Dataset, "
            f"got {type(data).__name__} instead."
        )
        logger.error(err_msg)
        raise TypeError(err_msg)