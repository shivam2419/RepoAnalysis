"""Start the Streamlit chatbot using the current Python interpreter."""

import subprocess
import sys

if __name__ == "__main__":
    # Use the active interpreter so Streamlit runs in the same environment as the app.
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"], check=True)
