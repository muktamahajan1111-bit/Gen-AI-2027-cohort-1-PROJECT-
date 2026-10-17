"""Audio input: a WAV file or the live microphone, behind one ``poll`` API."""
from __future__ import annotations

import queue
import time
import wave

import numpy as np


class WavSource:
    """Replays a 16-bit PCM WAV in sync with the video clock."""

    def __init__(self, path: str, chunk_ms: int = 100):
        with wave.open(path, "rb") as w:
            self.sample_rate, ch = w.getframerate(), w.getnchannels()
            if w.getsampwidth() != 2:
                raise ValueError("Only 16-bit PCM WAV is supported")
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        self._data = data.reshape(-1, ch).mean(axis=1) if ch > 1 else data
        self._n = int(self.sample_rate * chunk_ms / 1000)
        self._pos = 0

    def poll(self, upto_t: float) -> list[tuple[float, np.ndarray]]:
        out = []
        while self._pos < len(self._data) and self._pos / self.sample_rate <= upto_t:
            out.append((self._pos / self.sample_rate, self._data[self._pos:self._pos + self._n]))
            self._pos += self._n
        return out


class MicSource:
    """Live microphone via ``sounddevice``."""

    def __init__(self, sample_rate: int = 16000):
        import sounddevice as sd
        self.sample_rate = sample_rate
        self._q: queue.Queue = queue.Queue()
        self._t0 = time.monotonic()
        self._stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                                      callback=self._cb)
        self._stream.start()

    def _cb(self, indata, frames, time_info, status) -> None:
        self._q.put((time.monotonic() - self._t0, indata[:, 0].copy()))

    def poll(self, upto_t: float) -> list[tuple[float, np.ndarray]]:
        out = []
        while not self._q.empty():
            out.append(self._q.get())
        return out

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
