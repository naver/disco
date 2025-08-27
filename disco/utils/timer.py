import time
from contextlib import contextmanager

class Timer:
    """A context manager for timing code execution."""

    def __init__(self, name="Timer"):
        self.name = name
        self.elapsed = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        self.elapsed = end_time - self.start_time

    def __str__(self):
        return f"{self.name}: {self.elapsed:.4f} seconds"


# Alternative using @contextmanager decorator
@contextmanager
def timer(name="Timer"):
    """A simple timer context manager using contextmanager decorator."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed = end_time - start_time