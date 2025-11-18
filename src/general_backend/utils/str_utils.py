"""String utility functions for processing and extracting information."""

import re

from general_backend.logging.setup_logging import get_logger

logger = get_logger(__name__)


def extract_realization(member_id: str | None = None) -> str | None:
    """Extract the 'rX' realization part from variant_label,
    e.g. 'r1i1p1f1' -> 'r1'. Tries several patterns in order to
    be robust for common CMIP variant_label formats:

    - r(\\d+)(?=i)   : match 'r1' when it's followed by 'i'(e.g 'r1i1p1f1')
    - r(\\d+)\b      : match 'r1' with a word boundary
    - r(\\d+)        : fallback, match first 'r' followed by digits anywhere

    Parameters
    ----------
    member_id : str | None
        The member_id or variant_label string to extract from.

    Returns
    -------
    str | None
        The extracted realization string (e.g. 'r1').
    """
    if not member_id:
        logger.debug("No member_id provided for realization extraction.")
        return None

    patterns = [r"r(\d+)(?=i)", r"\br(\d+)\b", r"r(\d+)"]
    for pat in patterns:
        m = re.search(pat, member_id)
        if m:
            real = f"r{m.group(1)}"
            logger.debug(
                "Extracted realization '%s' from member_id '%s' with re '%s'.",
                real,
                member_id,
                pat,
            )
            return real

    logger.warning("No realization found in member_id '%s'.", member_id)
    return None
