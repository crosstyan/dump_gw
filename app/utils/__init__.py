import time
from datetime import timedelta


class Instant:
    """A measurement of a monotonically nondecreasing clock."""

    _time: float

    @staticmethod
    def clock() -> float:
        """Get current clock time in microseconds."""
        return time.monotonic()

    def __init__(self):
        """Initialize with current clock time."""
        self._time = self.clock()

    @classmethod
    def now(cls) -> "Instant":
        """Create new Instant with current time."""
        return cls()

    def elapsed(self) -> timedelta:
        """Get elapsed time as timedelta."""
        now = self.clock()
        diff = now - self._time
        return timedelta(seconds=diff)

    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        return int(self.elapsed().total_seconds() * 1000)

    def has_elapsed_ms(self, ms: int) -> bool:
        """Check if specified milliseconds have elapsed."""
        return self.elapsed_ms() >= ms

    def mut_every_ms(self, ms: int) -> bool:
        """Check if time has elapsed and reset if true."""
        if self.elapsed_ms() >= ms:
            self.mut_reset()
            return True
        return False

    def has_elapsed(self, duration: timedelta) -> bool:
        """Check if specified duration has elapsed."""
        return self.elapsed() >= duration

    def mut_every(self, duration: timedelta) -> bool:
        """Check if duration has elapsed and reset if true."""
        if self.has_elapsed(duration):
            self.mut_reset()
            return True
        return False

    def mut_reset(self) -> None:
        """Reset the timer to current time."""
        self._time = self.clock()

    def mut_elapsed_and_reset(self) -> timedelta:
        """Get elapsed time and reset timer."""
        now = self.clock()
        diff = now - self._time
        duration = timedelta(microseconds=diff)
        self._time = now
        return duration

    def count(self) -> float:
        """Get the internal time counter value."""
        return self._time
