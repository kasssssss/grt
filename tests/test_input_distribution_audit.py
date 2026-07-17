from __future__ import annotations

import numpy as np

from scripts.audit_rads_iq1m_input_distribution import InputStats, js_divergence


def test_js_divergence_is_zero_for_identical_profiles() -> None:
    profile = np.asarray([0.0, 1.0, 3.0, 0.0])

    assert js_divergence(profile, profile) == 0.0


def test_input_stats_preserve_axis_locations_and_phase() -> None:
    sample = np.zeros((2, 64, 8, 1, 256, 2), dtype=np.float32)
    sample[:, 32, 3, 0, 10, 0] = 2.0
    sample[:, 32, 3, 0, 10, 1] = np.pi / 2
    stats = InputStats()

    stats.update(sample)
    summary = stats.summary()

    assert summary["samples"] == 2
    assert summary["doppler_active_bins"] == 1
    assert summary["range_power_q50"] == 10.0
    assert summary["phase_resultant"] == 1.0
    assert np.argmax(summary["profiles"]["azimuth_power"]) == 3
