# Reproducible execution environment for TradeMind AI.
#
# Pinned to a specific Python patch release, not `3.12` or `latest`. A backtest
# that produces different numbers after a base-image refresh is not reproducible,
# and floating-point behaviour does occasionally shift between builds.

FROM python:3.12.7-slim-bookworm

# PYTHONHASHSEED must be set before the interpreter starts; it cannot be set
# from inside a running process.
ENV PYTHONHASHSEED=42 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so the layer caches across source changes.
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -e ".[dev]" 2>/dev/null || pip install \
        "pandas>=2.2" "numpy>=1.26" "pyarrow>=15.0" "duckdb>=1.0" \
        "pyyaml>=6.0" "scikit-learn>=1.4" "scipy>=1.11" \
        "yfinance>=0.2.40" "pandas-market-calendars>=4.4" "pytest>=8.0" \
        "streamlit>=1.30"

COPY src/ ./src/
COPY tests/ ./tests/
COPY config/ ./config/
COPY dashboard/ ./dashboard/
COPY main.py Makefile README.md PHASE_STATUS.md ./

RUN pip install -e "." --no-deps

# Data and outputs are mounted, never baked in. An image containing market
# data is an image that goes stale silently.
RUN mkdir -p data/raw data/processed data/predictions logs models reports

# Non-root: the container writes only to mounted volumes.
RUN useradd --create-home --uid 1000 trademind && \
    chown -R trademind:trademind /app
USER trademind

# Set PATH to include local user binary directory where pip puts executables
ENV PATH="/home/trademind/.local/bin:${PATH}" \
    PYTHONPATH=/app/src

CMD ["python", "main.py", "--help"]