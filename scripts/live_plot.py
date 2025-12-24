"""Live plotter that watches `kernel_times.csv` and updates training metrics in real time.

Usage:
    python3 scripts/live_plot.py      # defaults to ../kernel_times.csv and updates every 1.0s
    python3 scripts/live_plot.py --csv ../kernel_times.csv --interval 0.5 --no-show

Features:
- Plots training loss (GPU and CPU MSE values) over epochs
- Shows a bar plot comparing GPU vs CPU times for the latest epoch (if cpu_ columns exist)
"""

import argparse
import time
from pathlib import Path
import signal
import sys

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description='Live plot kernel_times.csv')
    p.add_argument('--csv', default=str(Path(__file__).resolve().parent.parent / 'kernel_times.csv'))
    p.add_argument('--interval', type=float, default=1.0, help='Polling interval in seconds')
    p.add_argument('--no-show', dest='show', action='store_false')
    return p.parse_args()


def safe_read_csv(csv: Path):
    try:
        return pd.read_csv(csv)
    except Exception:
        return None


def run_live(csv_path: Path, interval: float = 1.0, show: bool = True):
    plt.ion()
    fig, (ax_loss, ax_bar) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Live training metrics')

    last_len = 0
    stopped = False

    def _handle_sigint(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, _handle_sigint)

    while not stopped:
        df = safe_read_csv(csv_path)
        if df is None:
            print(f'Waiting for {csv_path} to be created...')
            time.sleep(interval)
            continue

        if 'Epoch' in df.columns:
            df = df.set_index('Epoch')

        # Loss curves
        ax_loss.clear()
        x = df.index.to_numpy()
        if 'mse_value' in df.columns:
            ax_loss.plot(x, df['mse_value'].to_numpy(), label='GPU MSE', marker='o')
        if 'cpu_mse_value' in df.columns:
            ax_loss.plot(x, df['cpu_mse_value'].to_numpy(), label='CPU MSE', marker='x')
        ax_loss.set_ylabel('MSE')
        ax_loss.set_xlabel('Epoch')
        ax_loss.legend()
        ax_loss.grid(True)

        # Latest GPU vs CPU times bar chart (for kernels present)
        ax_bar.clear()
        # find kernels that have cpu_ prefix
        cpu_cols = [c for c in df.columns if c.startswith('cpu_')]
        if cpu_cols:
            # derive base kernel names
            base_names = sorted(set([c[len('cpu_'):] for c in cpu_cols]))
            gpu_vals = []
            cpu_vals = []
            for name in base_names:
                gpu_col = name if name in df.columns else None
                cpu_col = f'cpu_{name}'
                if gpu_col is None and cpu_col not in df.columns:
                    continue
                gpu_vals.append(df[gpu_col].iloc[-1] if gpu_col is not None else float('nan'))
                cpu_vals.append(df[cpu_col].iloc[-1] if cpu_col in df.columns else float('nan'))

            x = range(len(base_names))
            width = 0.35
            ax_bar.bar([i - width/2 for i in x], gpu_vals, width, label='GPU')
            ax_bar.bar([i + width/2 for i in x], cpu_vals, width, label='CPU')
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(base_names, rotation=45, ha='right')
            ax_bar.set_ylabel('Time (ms)')
            ax_bar.legend()
            ax_bar.grid(True, axis='y')
        else:
            ax_bar.text(0.5, 0.5, 'No CPU baseline columns found in CSV', ha='center', va='center')
            ax_bar.set_axis_off()

        fig.tight_layout()
        fig.canvas.draw()
        if show:
            plt.pause(interval)
        else:
            time.sleep(interval)

    plt.close(fig)


def main():
    args = parse_args()
    csv = Path(args.csv)
    if not csv.exists():
        print(f'Warning: {csv} does not exist yet — the script will wait for it and begin plotting once it appears.')
    run_live(csv, interval=args.interval, show=args.show)


if __name__ == '__main__':
    main()
