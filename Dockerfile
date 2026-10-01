# syntax=docker/dockerfile:1

FROM python:3.13.15-slim-bookworm AS builder

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m venv "${VIRTUAL_ENV}"

COPY requirements-runtime.lock.txt /tmp/requirements-runtime.lock.txt

RUN python -m pip install \
    --no-cache-dir \
    --requirement /tmp/requirements-runtime.lock.txt \
    && python -m pip uninstall --yes pip setuptools wheel


FROM python:3.13.15-slim-bookworm AS runtime

ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="DeployGuard ML" \
      org.opencontainers.image.description="Deployment failure risk prediction API" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app/src"

WORKDIR /app

RUN groupadd \
        --gid 10001 \
        deployguard \
    && useradd \
        --uid 10001 \
        --gid 10001 \
        --no-create-home \
        --home-dir /nonexistent \
        --shell /usr/sbin/nologin \
        deployguard \
    && /usr/local/bin/python -m pip uninstall --yes pip setuptools wheel

COPY --from=builder \
    /opt/venv \
    /opt/venv

COPY --chown=10001:10001 \
    src/deployguard \
    /app/src/deployguard

COPY --chown=10001:10001 \
    artifacts/deployguard_model.joblib \
    /app/artifacts/deployguard_model.joblib

COPY --chown=10001:10001 \
    artifacts/metadata.json \
    /app/artifacts/metadata.json

USER 10001:10001

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=3s \
    --start-period=10s \
    --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"]

ENTRYPOINT ["gunicorn"]

CMD ["--workers", "2", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-", "--worker-tmp-dir", "/tmp", "--timeout", "30", "--graceful-timeout", "30", "--keep-alive", "5", "--preload", "deployguard.app:create_app()"]
