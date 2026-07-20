FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# libgomp1 (GNU OpenMP runtime) is required at import time by the
# OpenBLAS-linked numpy/scipy wheels - without it they segfault (exit 139)
# on slim Debian images instead of raising a normal Python ImportError.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Import the heavy native-extension libraries one at a time, at build time.
# If any of them segfaults on import (e.g. a missing system shared library),
# this fails the BUILD with the exact library name in the log, instead of
# shipping a broken image that crash-loops at runtime with no clue why.
RUN python -c "\
import sys; \
libs = ['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'crewai']; \
[ (print(f'importing {lib}...', flush=True), __import__(lib), print(f'{lib} OK', flush=True)) for lib in libs ]; \
print('all native-extension imports OK', flush=True)"

# Copy app files
COPY . .

# Expose port
EXPOSE 8501

# No Docker-level HEALTHCHECK here - Render already does its own external
# health check against /_stcore/health (see render.yaml healthCheckPath /
# the dashboard Settings). An internal HEALTHCHECK would fire curl requests
# against the same process from inside the container every 30s on top of
# that, and its timing lines up with when this service has been segfaulting
# (~60-90s after boot) - removing it as a low-risk diagnostic step.

# Run the Streamlit UI
CMD streamlit run app.py --server.port=8501 --server.address=0.0.0.0
