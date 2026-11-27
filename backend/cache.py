
from collections import OrderedDict
from copy import deepcopy
from threading import RLock
from time import monotonic


class TTLCache:

    def __init__(self, ttl_seconds=300, maxsize=16):

        if ttl_seconds <= 0 or maxsize <= 0:
            raise ValueError("Invalid cache configuration")

        self.ttl_seconds = ttl_seconds
        self.maxsize = maxsize

        self._data = OrderedDict()
        self._lock = RLock()

        self.hits = 0
        self.misses = 0

    def get_or_create(self, key, factory):

        with self._lock:

            now = monotonic()

            if key in self._data:

                expires_at, value = self._data[key]

                if now < expires_at:

                    self.hits += 1

                    self._data.move_to_end(key)

                    return deepcopy(value)

                del self._data[key]

            self.misses += 1

            # Holding the lock prevents simultaneous
            # requests from repeating expensive work.

            value = factory()

            self._data[key] = (
                monotonic() + self.ttl_seconds,
                deepcopy(value)
            )

            self._data.move_to_end(key)

            while len(self._data) > self.maxsize:

                self._data.popitem(last=False)

            return deepcopy(value)

    def clear(self):

        with self._lock:

            self._data.clear()

            self.hits = 0
            self.misses = 0

    def stats(self):

        with self._lock:

            now = monotonic()

            expired = [
                key
                for key, (expires_at, _) in self._data.items()
                if now >= expires_at
            ]

            for key in expired:
                del self._data[key]

            total = self.hits + self.misses

            return {
                "entries": len(self._data),
                "hits": self.hits,
                "misses": self.misses,
                "ttl_seconds": self.ttl_seconds,
                "hit_rate_percent": round(
                    self.hits / max(total, 1) * 100,
                    2
                )
            }


# Separate caches for different workloads.

CONJUNCTION_CACHE = TTLCache(
    ttl_seconds=300,
    maxsize=8
)

ANOMALY_CACHE = TTLCache(
    ttl_seconds=300,
    maxsize=8
)

FUSION_CACHE = TTLCache(
    ttl_seconds=300,
    maxsize=8
)

SCHEDULER_CACHE = TTLCache(
    ttl_seconds=300,
    maxsize=8
)

MISSION_CACHE = TTLCache(
    ttl_seconds=60,
    maxsize=8
)

EVALUATION_CACHE = TTLCache(
    ttl_seconds=3600,
    maxsize=4
)


def get_cache_stats():

    return {
        "conjunctions": CONJUNCTION_CACHE.stats(),
        "anomalies": ANOMALY_CACHE.stats(),
        "fusion": FUSION_CACHE.stats(),
        "scheduler": SCHEDULER_CACHE.stats(),
        "mission": MISSION_CACHE.stats(),
        "evaluation": EVALUATION_CACHE.stats()
    }
