import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_benchmark_name(name):
    # Example: BM_DdotOpenBLAS/8192 or BM_GemmAccelerate/1024
    match = re.match(r"BM_([A-Za-z]+)_([A-Za-z]+)[/](\d+)", name)
    if not match:
        return None, None, None
    family = match.group(1).lower()
    library = match.group(2).lower()
    size = int(match.group(3))
    return family, library, size

# Usage (as before):
with open("openblas.json", "r") as f:
    openblas_data = json.load(f)
with open("accelerate.json", "r") as f:
    accelerate_data = json.load(f)

benchmarks = openblas_data["benchmarks"] + accelerate_data["benchmarks"]
grouped = defaultdict(list)

for bm in benchmarks:
    # bm = {'name': 'BM_Ddot_Accelerate/524288', 'family_index': 0, 'per_family_instance_index': 9, 'run_name': 'BM_Ddot_Accelerate/524288', 'run_type': 'iteration', 'repetitions': 1, 'repetition_index': 0, 'threads': 1, 'iterations': 67266, 'real_time': 10425.313912121386, 'cpu_time': 10425.192519251934, 'time_unit': 'ns', 'items_per_second': 50290486149.94984}
    family_idx = bm["family_index"]
    family, library, size = parse_benchmark_name(bm["name"])
    grouped[family].append({
        "family": family,
        "library": library,
        "size": size,
        **bm,
    })


# Plotting per family, with one line per library
for family, data in grouped.items():
    # Organize data by library
    data_by_library = defaultdict(list)
    for entry in data:
        data_by_library[entry["library"]].append((entry["size"], entry["items_per_second"]))

    plt.figure(figsize=(10,6))
    for library, points in data_by_library.items():
        # Sort points by size for a clean plot
        points.sort(key=lambda x: x[0])
        sizes, ips = zip(*points)
        plt.plot(sizes, ips, marker='o', label=library)

    plt.title(f"Benchmark Family: {family}")
    plt.xlabel("Size")
    plt.ylabel("Items per second")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()