"""Start the FastAPI backend for local development."""

import uvicorn

if __name__ == "__main__":
    # Reload mode restarts the development server when application files change.
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
