"""Tiny synthesized sound effects (no audio files needed). Fails silently
when no audio device is available."""
from __future__ import annotations

import numpy as np
import pygame

_SR = 22050


def _tone(freqs, dur, vol=0.3, decay=6.0, noise=0.0):
    t = np.linspace(0, dur, int(_SR * dur), endpoint=False)
    wave = sum(np.sin(2 * np.pi * f * t) for f in freqs) / max(1, len(freqs))
    if noise:
        wave = wave * (1 - noise) + np.random.uniform(-1, 1, t.shape) * noise
    wave *= np.exp(-decay * t) * vol
    pcm = (wave * 32767).astype(np.int16)
    return np.repeat(pcm[:, None], 2, axis=1)


class Audio:
    def __init__(self):
        self.ok = False
        self.sounds = {}
        try:
            pygame.mixer.init(_SR, -16, 2, 512)
            mk = lambda arr: pygame.sndarray.make_sound(np.ascontiguousarray(arr))
            self.sounds = {
                "pickup": mk(_tone([880, 1320], 0.25, 0.25, 8)),
                "terminal": mk(_tone([440, 660, 990], 0.6, 0.25, 3)),
                "alarm": mk(_tone([620, 640], 0.9, 0.3, 1.2)),
                "throw": mk(_tone([200], 0.25, 0.4, 10, noise=0.7)),
                "locker": mk(_tone([120, 180], 0.2, 0.4, 12, noise=0.3)),
                "heartbeat": mk(_tone([55, 60], 0.18, 0.6, 18)),
                "caught": mk(_tone([90, 95, 140], 1.2, 0.5, 2, noise=0.2)),
                "escape": mk(_tone([523, 659, 784, 1046], 1.0, 0.3, 2)),
                "step": mk(_tone([90], 0.06, 0.12, 40, noise=0.5)),
            }
            self.ok = True
        except Exception:
            self.ok = False

    def play(self, name: str):
        if self.ok and name in self.sounds:
            self.sounds[name].play()
