# The parser as a container: one image for every workflow step.
#   docker build -f deploy/Dockerfile -t invoice-parser:dev .
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Dependencies first, so a code change does not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home runner && mkdir -p /data /out && chown runner /data /out
USER runner

ENTRYPOINT ["invoice-parser"]
CMD ["--help"]
