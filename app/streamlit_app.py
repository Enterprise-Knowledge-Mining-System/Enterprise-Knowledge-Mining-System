"""Compatibility entry point for running the Streamlit app from app/.

The deployed Streamlit app lives at the repository root in streamlit_app.py.
This wrapper keeps `streamlit run app/streamlit_app.py` working.
"""

from pathlib import Path
import runpy


runpy.run_path(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), run_name="__main__")
