"""Small Qt/napari helpers shared by the plugin's widgets."""
from __future__ import annotations

from typing import Callable

from qtpy.QtWidgets import QDoubleSpinBox


def refresh_layer_combo(combo, viewer, layer_type, predicate: Callable | None = None) -> list[str]:
    """Repopulate `combo` with the names of `viewer`'s layers of
    `layer_type` (and, if given, passing `predicate`).

    The current selection is kept when that layer is still listed. Signals
    are blocked throughout: a refresh is triggered by layers being added
    or removed anywhere in the viewer, and a host listening for the user
    choosing a layer must not mistake it for a choice.
    """
    current = combo.currentText()
    names = [
        layer.name for layer in viewer.layers
        if isinstance(layer, layer_type) and (predicate is None or predicate(layer))
    ]
    was_blocked = combo.blockSignals(True)
    try:
        combo.clear()
        combo.addItems(names)
        if current in names:
            combo.setCurrentText(current)
    finally:
        combo.blockSignals(was_blocked)
    return names


def calibration_spin(maximum: float, decimals: int) -> QDoubleSpinBox:
    """A calibration field where 0 reads as "unknown" — the plugin-wide
    convention that a blank calibration value is `None`, not zero."""
    spin = QDoubleSpinBox()
    spin.setRange(0.0, maximum)
    spin.setDecimals(decimals)
    spin.setSpecialValueText("unknown")
    return spin
