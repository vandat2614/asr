"""Client for the development-only ASR service running on Kaggle.

The final solution must NOT use this backend: inference has to run offline
and locally. Keep the interface (`transcribe(paths) -> list[str]`) identical
to the future local backend so the two are interchangeable.
"""
import os
import time
from pathlib import Path

import requests


class RemoteASR:
    def __init__(self, url=None, token=None, batch_files=16, timeout=300, retries=3):
        self.url = (url or os.environ["ASR_URL"]).rstrip("/")
        self.token = token or os.environ["ASR_TOKEN"]
        self.batch_files = batch_files
        self.timeout = timeout
        self.retries = retries
        self.gpu_seconds = 0.0  # accumulated server-side GPU time

    def health(self):
        r = requests.get(f"{self.url}/health", timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, paths):
        last_error = None
        for attempt in range(self.retries):
            handles = [open(p, "rb") for p in paths]
            try:
                files = [
                    ("files", (Path(p).name, h, "audio/wav"))
                    for p, h in zip(paths, handles)
                ]
                r = requests.post(
                    f"{self.url}/transcribe",
                    files=files,
                    headers={"X-API-Token": self.token},
                    timeout=self.timeout,
                )
                if r.status_code in (400, 401):
                    raise RuntimeError(f"request rejected ({r.status_code}): {r.text}")
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last_error = e
                time.sleep(2**attempt)
            finally:
                for h in handles:
                    h.close()
        raise RuntimeError(f"remote ASR failed after {self.retries} attempts: {last_error}")

    def transcribe(self, paths):
        """Transcribe a list of wav paths. Returns raw texts in the same order."""
        paths = [str(p) for p in paths]
        texts = []
        for i in range(0, len(paths), self.batch_files):
            data = self._post(paths[i : i + self.batch_files])
            self.gpu_seconds += data["gpu_seconds"]
            texts.extend(data["texts"])
        return texts
