"""Epoch state shared by a dataset and its persistent DataLoader workers."""

import multiprocessing


class SharedEpoch:
    """Share a synchronized integer within one rank's worker processes.

    Create this once before workers start and update it only between iterations.
    A spawn-context lock also works with fork and forkserver workers; a lock
    created in the default fork context cannot be passed to spawn workers.
    This does not change the application's multiprocessing start method.
    """

    def __init__(self, epoch: int = 0):
        self._value = multiprocessing.get_context("spawn").Value("q", int(epoch), lock=True)

    def get(self) -> int:
        return self._value.value

    def set(self, epoch: int) -> None:
        self._value.value = int(epoch)
