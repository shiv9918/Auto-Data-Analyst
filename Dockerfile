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

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit UI
CMD streamlit run app.py --server.port=8501 --server.address=0.0.0.0
