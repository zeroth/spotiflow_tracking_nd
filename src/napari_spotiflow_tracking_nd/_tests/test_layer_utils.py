from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from napari.layers import Image, Points
from qtpy.QtWidgets import QComboBox


def _viewer(*layers):
    return SimpleNamespace(layers=list(layers))


def test_lists_only_layers_of_the_given_type(qtbot):
    from napari_spotiflow_tracking_nd._layer_utils import refresh_layer_combo

    combo = QComboBox()
    viewer = _viewer(Image(np.zeros((4, 4)), name="img"), Points(name="pts"))

    names = refresh_layer_combo(combo, viewer, Image)

    assert names == ["img"]
    assert [combo.itemText(i) for i in range(combo.count())] == ["img"]


def test_a_predicate_narrows_the_list(qtbot):
    from napari_spotiflow_tracking_nd._layer_utils import refresh_layer_combo

    combo = QComboBox()
    viewer = _viewer(
        Points(name="plain"),
        Points(name="spots", metadata={"csv_path": "a.csv"}),
    )

    refresh_layer_combo(combo, viewer, Points, predicate=lambda l: "csv_path" in l.metadata)

    assert [combo.itemText(i) for i in range(combo.count())] == ["spots"]


def test_keeps_the_current_selection_when_it_survives(qtbot):
    from napari_spotiflow_tracking_nd._layer_utils import refresh_layer_combo

    combo = QComboBox()
    a, b = Image(np.zeros((4, 4)), name="a"), Image(np.zeros((4, 4)), name="b")
    refresh_layer_combo(combo, _viewer(a, b), Image)
    combo.setCurrentText("b")

    refresh_layer_combo(combo, _viewer(Image(np.zeros((4, 4)), name="new"), a, b), Image)

    assert combo.currentText() == "b"


def test_refreshing_emits_no_selection_signals(qtbot):
    # A host reacting to the user picking a layer must not mistake a
    # refresh (a layer added elsewhere) for a pick.
    from napari_spotiflow_tracking_nd._layer_utils import refresh_layer_combo

    combo = QComboBox()
    seen = []
    combo.currentTextChanged.connect(seen.append)

    refresh_layer_combo(combo, _viewer(Image(np.zeros((4, 4)), name="a")), Image)

    assert seen == []
    assert combo.currentText() == "a"
