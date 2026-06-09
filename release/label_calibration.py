"""Etykiety służą tylko do oceny (eval) — live NIE stosuje stałych offsetów pikselowych."""
from __future__ import annotations

# Zachowane dla kompatybilności skryptów; detekcja live nie importuje apply_label_calibration.


def apply_label_calibration(quad):  # noqa: ANN001
    """No-op: kalibracja pikselowa wyłączona (nie skaluje na inne ujęcia)."""
    return quad


def fit_label_calibration(*_args, **_kwargs):
    """Tylko diagnostyka offline — nie używaj wyniku w live."""
    return None
