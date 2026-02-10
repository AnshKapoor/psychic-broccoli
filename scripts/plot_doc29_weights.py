#!/usr/bin/env python
"""
Plot Doc29 7-track weight fractions as percentages.

Uses TRACK_DEFS from scripts/doc29_tracks.py.
Outputs a bar chart with a cumulative percentage line.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt

from scripts.doc29_tracks import TRACK_DEFS


def _repo_root() -> Path:
    return ROOT


def main() -> None:
    labels = [name for name, _mult, _w in TRACK_DEFS]
    weights = [w for _name, _mult, w in TRACK_DEFS]
    pct = [w * 100 for w in weights]

    # Cumulative percentage
    cum = []
    total = 0.0
    for w in pct:
        total += w
        cum.append(total)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(labels, pct, color="#4C78A8")
    ax1.set_ylabel("Weight (%)")
    ax1.set_ylim(0, max(pct) * 1.25)
    ax1.set_title("Doc29 7-track weight fractions")
    ax1.tick_params(axis="x", rotation=30, labelsize=8)

    ax2 = ax1.twinx()
    ax2.plot(labels, cum, color="#F58518", marker="o")
    ax2.set_ylabel("Cumulative (%)")
    ax2.set_ylim(0, 100)

    out_path = _repo_root() / "output" / "eda" / "doc29_track_weights.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
