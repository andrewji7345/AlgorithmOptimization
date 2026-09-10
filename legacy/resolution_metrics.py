"""Mass-resolution estimators shared by the scan evaluators."""

import numpy as np


def histogram_fwhm(values, minimum_bins=10, maximum_bins=200):
    """Return ``(FWHM, peak_position)`` for the principal histogram peak.

    The histogram starts from the Freedman--Diaconis bin-width rule. Its bin
    count is bounded to avoid an unstable estimate for small samples or a very
    fine histogram when a distribution contains distant tails. Half-maximum
    crossings on either side of the highest bin are linearly interpolated.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan

    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if value_min == value_max:
        return 0.0, value_min

    fd_edges = np.histogram_bin_edges(values, bins="fd")
    bin_count = int(np.clip(len(fd_edges) - 1, minimum_bins, maximum_bins))
    counts, edges = np.histogram(values, bins=bin_count, range=(value_min, value_max))
    centers = 0.5 * (edges[:-1] + edges[1:])

    peak_index = int(np.argmax(counts))
    peak_position = float(centers[peak_index])
    half_maximum = 0.5 * counts[peak_index]

    left_index = peak_index
    while left_index > 0 and counts[left_index - 1] >= half_maximum:
        left_index -= 1
    if left_index == 0:
        left_x, left_y = edges[0], 0.0
    else:
        left_x, left_y = centers[left_index - 1], counts[left_index - 1]
    right_x, right_y = centers[left_index], counts[left_index]
    left_crossing = _linear_crossing(
        left_x, left_y, right_x, right_y, half_maximum
    )

    right_index = peak_index
    while right_index < len(counts) - 1 and counts[right_index + 1] >= half_maximum:
        right_index += 1
    left_x, left_y = centers[right_index], counts[right_index]
    if right_index == len(counts) - 1:
        right_x, right_y = edges[-1], 0.0
    else:
        right_x, right_y = centers[right_index + 1], counts[right_index + 1]
    right_crossing = _linear_crossing(
        left_x, left_y, right_x, right_y, half_maximum
    )

    return float(right_crossing - left_crossing), peak_position


def relative_fwhm_resolution(values):
    """Return ``(FWHM / peak_position, FWHM, peak_position)``."""
    fwhm, peak_position = histogram_fwhm(values)
    resolution = (
        fwhm / peak_position
        if np.isfinite(fwhm) and np.isfinite(peak_position) and peak_position > 0.0
        else np.nan
    )
    return float(resolution), float(fwhm), float(peak_position)


def _linear_crossing(x0, y0, x1, y1, target):
    if y1 == y0:
        return 0.5 * (x0 + x1)
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
