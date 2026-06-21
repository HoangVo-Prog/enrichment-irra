"""Lightweight result audit warnings."""

from __future__ import annotations

import logging


def audit_outputs(
    logger: logging.Logger,
    row_counts: dict[str, int],
    valid_pair_rate: float,
    mean_cue_shift: float,
    min_pair_cue_shift: float,
) -> None:
    if valid_pair_rate < 0.25:
        logger.warning("Low valid-pair rate: %.4f", valid_pair_rate)
    if mean_cue_shift < max(min_pair_cue_shift, 0.01):
        logger.warning("Weak Cue Shift: %.6f", mean_cue_shift)
    for filename, count in sorted(row_counts.items()):
        if filename.endswith(".csv") and count == 0:
            logger.warning("Output has no data rows: %s", filename)
