import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tracing.data import ROOT
from tracing.segments import CHANGE


class SegmentDemoTest(unittest.TestCase):
    def test_display_preserves_whitespace_at_sample_boundaries(self):
        text = " start changes "
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / f"{CHANGE}.joblib").touch()
            with patch.dict(os.environ, {"TRACE_RUN_DIR": directory}), patch("predict.Predictor") as load:
                load.return_value.predict_with_segments.return_value = (
                    {"id": 1, "label": 2, "family": -1}, [(0, 6), (6, len(text))], [0.25],
                )
                app = AppTest.from_file(str(ROOT / "app.py")).run()
                app.text_area[0].input(text)
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertEqual([element.value for element in app.text], ["「 start」", "「 changes 」"])
                self.assertTrue(any("0.250" in item.value for item in app.caption))
