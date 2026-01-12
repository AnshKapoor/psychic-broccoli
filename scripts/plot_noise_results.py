"""Plot noise simulation outputs as a colored scatter map."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot noise simulation results from a CSV.")
    parser.add_argument("csv", help="Path to the noise simulation CSV (e.g., Results_Python/EXP46.csv).")
    parser.add_argument("--column", default="cumulative_res", help="Column to visualize (default: cumulative_res).")
    parser.add_argument("--out", default=None, help="Output PNG path (default: <csv>_<column>.png).")
    parser.add_argument("--sep", default=";", help="CSV separator (default: ';').")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap (default: viridis).")
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI.")
    parser.add_argument("--annotate", action="store_true", help="Annotate points with values.")
    parser.add_argument("--mode", choices=["scatter", "contour"], default="scatter", help="Plot mode (default: scatter).")
    parser.add_argument("--grid-size", type=int, default=150, help="Grid size for contour interpolation.")
    parser.add_argument("--interp", choices=["cubic", "linear", "nearest"], default="cubic", help="Interpolation method.")
    parser.add_argument("--levels", type=int, default=12, help="Number of contour levels.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Contour alpha.")
    parser.add_argument("--fill-nan", action="store_true", help="Fill NaNs in interpolated grid with nearest values.")
    parser.add_argument("--background", default=None, help="Optional background image path.")
    parser.add_argument("--pad-frac", type=float, default=0.2, help="Axis padding fraction of data range.")
    parser.add_argument("--figsize", default=None, help="Figure size as 'width,height' in inches.")
    parser.add_argument("--title", default=None, help="Optional plot title override.")
    parser.add_argument("--xlim", default=None, help="Override x-axis limits as 'min,max'.")
    parser.add_argument("--ylim", default=None, help="Override y-axis limits as 'min,max'.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path, sep=args.sep)

    for col in ("x", "y", args.column):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {csv_path}")

    out_path = Path(args.out) if args.out else csv_path.with_name(f"{csv_path.stem}_{args.column}.png")

    figsize = None
    if args.figsize:
        parts = [float(p.strip()) for p in args.figsize.split(",") if p.strip()]
        if len(parts) == 2:
            figsize = (parts[0], parts[1])
    fig, ax = plt.subplots(figsize=figsize or (8, 6))
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df[args.column].to_numpy()

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    dx = max(x_max - x_min, 1.0)
    dy = max(y_max - y_min, 1.0)
    pad_x = dx * args.pad_frac
    pad_y = dy * args.pad_frac

    if args.mode == "contour":
        xi = np.linspace(x_min, x_max, args.grid_size)
        yi = np.linspace(y_min, y_max, args.grid_size)
        grid_x, grid_y = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (grid_x, grid_y), method=args.interp)
        if args.fill_nan and np.isnan(zi).any():
            zi_near = griddata((x, y), z, (grid_x, grid_y), method="nearest")
            zi = np.where(np.isnan(zi), zi_near, zi)
        zi_masked = np.ma.masked_invalid(zi)
        contour = ax.contourf(grid_x, grid_y, zi_masked, levels=args.levels, cmap=args.cmap, alpha=args.alpha)
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(args.column)
    else:
        sc = ax.scatter(x, y, c=z, cmap=args.cmap, s=80)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(args.column)

    if args.background:
        img = plt.imread(args.background)
        ax.imshow(img, extent=[x_min, x_max, y_min, y_max], aspect="auto")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(args.title or f"{args.column} ({csv_path.name})")
    ax.set_aspect("equal", adjustable="box")
    if args.xlim:
        x_parts = [float(p.strip()) for p in args.xlim.split(",") if p.strip()]
        if len(x_parts) == 2:
            ax.set_xlim(x_parts[0], x_parts[1])
    else:
        ax.set_xlim(x_min - pad_x, x_max + pad_x)

    if args.ylim:
        y_parts = [float(p.strip()) for p in args.ylim.split(",") if p.strip()]
        if len(y_parts) == 2:
            ax.set_ylim(y_parts[0], y_parts[1])
    else:
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    if args.annotate and args.mode == "scatter":
        for _, row in df.iterrows():
            ax.text(row["x"], row["y"], f"{row[args.column]:.2f}", fontsize=7, ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
