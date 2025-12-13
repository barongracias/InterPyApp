import os

import redis
from rq import Worker, Queue, Connection

from tasks import run_training_job  # noqa: F401  (ensures task importable)


listen = [os.getenv("QUEUE_NAME", "train")]
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def main():
    conn = redis.from_url(redis_url)
    with Connection(conn):
        worker = Worker([Queue(name) for name in listen])
        worker.work()


if __name__ == "__main__":
    main()
