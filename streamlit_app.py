"""Root Streamlit entry point for deployments.

The real app lives in app/streamlit_app.py. Some hosts, including Streamlit
Community Cloud, default to looking for streamlit_app.py at the repository root.
Importing the app module here keeps that deployment path working while the
source code remains organized under app/.
"""

from app.streamlit_app import *  # noqa: F401,F403
