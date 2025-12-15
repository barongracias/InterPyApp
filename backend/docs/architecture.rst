Architecture and Ops
====================

This section describes how the backend executes training jobs, manages asynchronous work and is operated in development and production. It covers the Redis-backed job queue and worker model, fallback synchronous execution for local use, shared artifact storage and the observability and deployment controls used to run and monitor the system reliably.

Job queue (training)
--------------------
- ``/train`` pings Redis and enqueues training jobs to Redis/RQ when ``REDIS_URL`` is set; otherwise it runs synchronously (useful for local dev/tests). If the ping/enqueue fails, the API logs a warning and runs synchronously as a fallback.
- Jobs are consumed by the RQ worker (run ``python worker.py`` or via Docker Compose ``worker`` service). Queue name defaults to ``train`` and is configurable via ``QUEUE_NAME``.
- Job status/metadata is exposed via ``/jobs/{id}``, including backend type and any result payload; job metadata is persisted in Redis job meta.
- Uploads and outputs are volume-backed in Docker so workers and API share artifacts.

Observability
-------------
- Structured logs: ``upload.success``, ``train.enqueued``, ``train.completed``, ``predict.completed``, ``evaluate.completed`` with backend tags and durations/RMSE where applicable.
- Healthcheck: ``/health``; Docker Compose uses curl-based healthcheck for the backend container.
- Set ``ALLOWED_ORIGINS`` for CORS; logs and job metadata are written to stdout/Redis.

Deployment knobs
----------------
- Env: ``REDIS_URL`` (defaults to ``redis://redis:6379/0``; required for async jobs), ``QUEUE_NAME`` (defaults ``train``), ``ALLOWED_ORIGINS`` (CORS).
- Docker Compose includes a Redis service and a dedicated worker service; backend service runs API only.
- TensorFlow CPU is installed by default; images target ``linux/amd64`` for TF wheel compatibility (Apple Silicon users should keep this).