

import numpy as np
import matplotlib.pyplot as plt
import m01_Graphs as Gra

graphs = Gra.graphs84
eigenvals = Gra.eigenvals84
spectral_gaps = [ev[1] for ev in eigenvals]
cheegers = Gra.cheegers84

min_gap = min(spectral_gaps)
max_gap = max(spectral_gaps)

sg_counts, sg_bin_edges = np.histogram(spectral_gaps, bins=15)

min_cheeger = min(cheegers)
max_cheeger = max(cheegers)

ch_counts, ch_bin_edges = np.histogram(cheegers, bins=15)

def print_histogram(counts, bin_edges):
    print(f"\n{'Interval':<25}{'Count':>9}{'Percent':>10}")
    print("-" * 44)

    total = counts.sum()
    for i, count in enumerate(counts):
        closing = "]" if i == len(counts) - 1 else ")"
        interval = (
            f"[{bin_edges[i]:.4f}, {bin_edges[i + 1]:.4f}{closing}"
        )
        percentage = 100 * count / total if total else 0
        print(f"{interval:<25}{count:>9,}{percentage:>9.2f}%")

print("spectral gaps:\n")

print_histogram(sg_counts, sg_bin_edges)

print("\n\n")

print("cheeger constants:\n")

print_histogram(ch_counts, ch_bin_edges)