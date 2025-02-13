# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm AS python-base

# Update curl and install ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download latest uv installer, run it, then remove it
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Set default port and environment variables; ensure installed binary is on `PATH`
ENV PATH="/root/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    WORKERS=1 \
    THREADS=8 \
    JSON_LOGS="true"

# Set the working directory in the container
WORKDIR /app

# Set user to root
USER 0

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src ./src

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-editable --no-dev

# Build and install the package
RUN uv build && uv pip install .

# Copy the application code into the container
COPY . .

# Expose the port on which the app will run
EXPOSE 5000

# Add health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:5000/health || exit 1

# Command to run the FastAPI app with Uvicorn, binding to 0.0.0.0 for external access
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]