"""Launcher for the Streamlit demo (thin wrapper).

Run this with `streamlit run demo/app.py`.
"""

import runpy
from pathlib import Path

_real_app = Path(__file__).resolve().parent / "streamlit_app.py"
runpy.run_path(str(_real_app), run_name="__main__")

