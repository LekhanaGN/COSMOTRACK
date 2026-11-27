
from time import perf_counter

from backend.mission import (
    get_mission_overview
)

from backend.cache import (
    MISSION_CACHE,
    CONJUNCTION_CACHE,
    ANOMALY_CACHE,
    FUSION_CACHE,
    SCHEDULER_CACHE,
    EVALUATION_CACHE,
    get_cache_stats
)


for cache in [

    MISSION_CACHE,
    CONJUNCTION_CACHE,
    ANOMALY_CACHE,
    FUSION_CACHE,
    SCHEDULER_CACHE,
    EVALUATION_CACHE

]:

    cache.clear()


def benchmark():

    started = perf_counter()

    result = get_mission_overview(

        hours=6,

        max_objects=5,

        mode="demo",

        include_fusion=True
    )

    elapsed = (
        perf_counter() - started
    )

    return elapsed, result


print("=" * 65)

print("COSMOTRACK PERFORMANCE BENCHMARK")

print("=" * 65)


cold_time, cold_result = benchmark()

warm_time, warm_result = benchmark()


print(
    f"Initial request: "
    f"{cold_time:.4f} seconds"
)

print(
    f"Cached request: "
    f"{warm_time:.4f} seconds"
)

if warm_time > 0:

    print(
        f"Speedup: "
        f"{cold_time / warm_time:.2f}x"
    )

print(
    f"Time reduction: "
    f"{(1 - warm_time / max(cold_time, 1e-9)) * 100:.2f}%"
)

print(
    "Initial status:",
    cold_result["system_status"]
)

print(
    "Cached status:",
    warm_result["system_status"]
)

print(
    "Same generated result:",
    cold_result["generated_at"]
    == warm_result["generated_at"]
)

print("\nCACHE STATISTICS")

for name, stats in get_cache_stats().items():

    print(name, stats)
