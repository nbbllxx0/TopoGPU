"""
Generate all figures from experiment CSV results.
Run from the release figures directory:
    python plot_figures.py --results-dir ../rerun_outputs/paper4 --figs-dir ../rerun_outputs/paper4_figs
    python plot_figures.py --results-dir ../rerun_outputs/paper4 --figs-dir ../tmp_figs
Outputs PDFs to the requested figures directory.
"""

import argparse
import os
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Global style (matches paper-3 F3–F9 look)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         12,
    'axes.labelsize':    13,
    'axes.titlesize':    13,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   11,
    'legend.framealpha': 0.9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'savefig.dpi':       300,
    'figure.dpi':        120,
})

BLUE   = '#1f77b4'
ORANGE = '#ff7f0e'
GREEN  = '#2ca02c'
RED    = '#d62728'
PURPLE = '#9467bd'
GRAY   = '#7f7f7f'

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = BASE_DIR.parent / 'rerun_outputs' / 'paper4'
DEFAULT_FIGS = BASE_DIR.parent / 'rerun_outputs' / 'paper4_figs'
RESULTS = DEFAULT_RESULTS
FIGS = DEFAULT_FIGS


def _configure_paths(results_dir=None, figs_dir=None):
    global RESULTS, FIGS
    RESULTS = Path(results_dir) if results_dir else DEFAULT_RESULTS
    FIGS = Path(figs_dir) if figs_dir else DEFAULT_FIGS
    FIGS.mkdir(exist_ok=True, parents=True)

def _savefig(fig, name):
    path = FIGS / name
    fig.savefig(path, bbox_inches='tight')
    print(f'  saved {path}')
    plt.close(fig)


def _read_csv(fname):
    path = RESULTS / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing reference CSV: {path}")
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def _f(row, key, default=float('nan')):
    val = row.get(key, "")
    if val in ("", None):
        return default
    return float(val)


# ===========================================================================
# F1 — h-independent iteration count (E1)
# ===========================================================================
def fig_vcycle_iters():
    rows = _read_csv('e1_vcycle_iters.csv')

    sizes = ['64k', '216k', '512k']
    grouped = {
        size: [int(r['iters']) for r in rows if r['size'] == size]
        for size in sizes
    }
    fail_counts = {
        size: sum(1 for r in rows if r['size'] == size and int(r['converged']) == 0)
        for size in sizes
    }
    totals = {
        size: sum(1 for r in rows if r['size'] == size)
        for size in sizes
    }
    gmg_iters = [float(np.mean(grouped[size])) for size in sizes]
    gmg_err = np.array([
        [mean - min(grouped[size]) for mean, size in zip(gmg_iters, sizes)],
        [max(grouped[size]) - mean for mean, size in zip(gmg_iters, sizes)],
    ])

    x = np.arange(len(sizes))
    width = 0.55

    fig, ax = plt.subplots(figsize=(7.1, 4.7))

    bars = ax.bar(
        x, gmg_iters, width, color=GREEN,
        label='FP64-GMG mean over 9 $V_f/p$ cases',
        yerr=gmg_err, capsize=4,
        error_kw={'elinewidth': 1.0, 'ecolor': GREEN},
    )

    for rect, size in zip(bars, sizes):
        if fail_counts[size] > 0:
            rect.set_hatch('//')
            rect.set_edgecolor(RED)
            rect.set_linewidth(1.6)

    for xi, y, size in zip(x, gmg_iters, sizes):
        ax.text(xi, y + 3, f'{y:.0f}', ha='center', va='bottom',
                fontsize=12, color=GREEN, fontweight='bold')
        if fail_counts[size] > 0:
            ax.text(
                xi, y + 23,
                f"{fail_counts[size]}/{totals[size]} hit 200-cap",
                ha='center', va='bottom', fontsize=9.5,
                color=RED, fontweight='bold'
            )

    cap_line = ax.axhline(
        200, color=RED, ls='--', lw=1.2, alpha=0.7,
        label='200-iteration screening cap'
    )

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=12)
    ax.set_xlabel('Mesh size')
    ax.set_ylabel('Outer PCG iterations')
    ax.set_title('GMG iteration count across heterogeneous mesh-size sweep')
    ax.set_ylim(0, 260)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    fail_proxy = mpl.patches.Patch(
        facecolor=GREEN, edgecolor=RED, hatch='//',
        label='Includes non-converged cases'
    )
    ax.legend(handles=[bars, fail_proxy, cap_line])
    fig.tight_layout()
    _savefig(fig, 'F1_vcycle_iters.pdf')


# ===========================================================================
# F2 — Per-solve wall time scaling (E2)  [log-log line, like paper-3 F3]
# ===========================================================================
def fig_solve_scaling():
    rows = _read_csv('e2_per_solve_wall_time.csv')

    n_elems  = [int(r['n_elem'])            for r in rows]
    t_jac    = [_f(r, 't_jacobi_pcg_s')     for r in rows]
    t_jac_sd = [_f(r, 't_jacobi_pcg_std_s', 0.0) for r in rows]
    t_fp32   = [_f(r, 't_gmg_fp32_s')       for r in rows]
    t_fp32_sd= [_f(r, 't_gmg_fp32_std_s', 0.0) for r in rows]
    t_bf16   = [_f(r, 't_gmg_bf16_s')       for r in rows]
    t_bf16_sd= [_f(r, 't_gmg_bf16_std_s', 0.0) for r in rows]
    labels   = [r['size']                   for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    ax.errorbar(n_elems, t_jac, yerr=t_jac_sd, color=BLUE, marker='s', ms=8, lw=2.0,
            capsize=3,
            label='Jacobi-PCG reference')
    ax.errorbar(n_elems, t_fp32, yerr=t_fp32_sd, color=GREEN, marker='o', ms=8, lw=2.0,
            capsize=3,
            label='FP32-GMG')
    ax.errorbar(n_elems, t_bf16, yerr=t_bf16_sd, color=ORANGE, marker='^', ms=8, lw=2.0,
            capsize=3,
            label='BF16-GMG (FGMRES)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of elements')
    ax.set_ylabel('Per-linear-solve wall time (s)')
    ax.set_title('FEA linear-solve wall-time scaling')
    ax.set_xticks(n_elems)
    ax.set_xticklabels(['64k', '216k', '512k'])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f'{y:.2f}' if y < 1 else f'{y:.1f}'))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    _savefig(fig, 'F2_solve_scaling.pdf')


# ===========================================================================
# F3 - Per-solve capped-baseline wall-time ratio bars (E2)
# ===========================================================================
def fig_solve_speedup():
    rows = _read_csv('e2_per_solve_wall_time.csv')

    labels  = [r['size']                      for r in rows]
    ratio_fp32 = [_f(r, 'speedup_gmg_fp32')  for r in rows]
    ratio_bf16 = [_f(r, 'speedup_gmg_bf16')  for r in rows]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    b1 = ax.bar(x - width/2, ratio_fp32, width, color=GREEN,  label='FP32-GMG / Jacobi-PCG')
    b2 = ax.bar(x + width/2, ratio_bf16, width, color=ORANGE, label='BF16-GMG (FGMRES) / Jacobi-PCG')

    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 0.04,
                f'{h:.2f}x', ha='center', va='bottom', fontsize=10.5)

    ax.axhline(1.0, color='black', lw=1.2, ls='--', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Mesh size')
    ax.set_ylabel('Wall-time ratio vs. capped Jacobi-PCG')
    ax.set_title('Per-linear-solve wall-time ratio vs. capped baseline')
    ax.set_ylim(0, max(ratio_fp32 + ratio_bf16) * 1.2)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    _savefig(fig, 'F3_solve_speedup.pdf')


# ===========================================================================
# F4 — End-to-end SIMP speedup (E3)  [bar chart]
# ===========================================================================
def _legacy_fig_simp_speedup():
    # Backward-compatible alias for the manuscript-tuned version below.
    return fig_simp_speedup()

    bench_labels = {
        'cantilever_216k': 'Cantilever\n(216k)',
        'torsion_small':   'Torsion\n(3k)',
        'mbb_small':       'MBB beam\n(1.5k)',
    }
    presets  = [r['preset']             for r in rows]
    t_paper3 = [float(r['t_paper3_s'])  for r in rows]
    t_gmg    = [float(r['t_gmg_fp32_s'])for r in rows]
    speedups = [float(r['speedup'])     for r in rows]
    xlabels  = [bench_labels.get(p, p)  for p in presets]

    x     = np.arange(len(rows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.6))
    b1 = ax.bar(x - width/2, t_paper3, width, color=ORANGE, label='Jacobi-PCG reference')
    b2 = ax.bar(x + width/2, t_gmg,    width, color=GREEN,  label='FP32-GMG')

    for rect, sp in zip(b2, speedups):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 2.0,
                f'{sp:.1f}x', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=GREEN)

    for rect, preset in zip(b1, presets):
        if preset == 'cantilever_216k':
            ax.text(rect.get_x() + rect.get_width()/2.,
                    rect.get_height() + 1.6,
                    '(stalled)', ha='center', va='bottom',
                    fontsize=9, color=RED, style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel('SIMP wall time (s)')
    ax.set_title('End-to-end SIMP wall time')
    ax.set_ylim(0, max(t_paper3 + t_gmg) * 1.14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    _savefig(fig, 'F4_simp_speedup.pdf')


# ===========================================================================
# F5 — Tensor-core throughput (E4)  [bar chart]
# ===========================================================================
def _legacy_fig_tc_throughput():
    # Backward-compatible alias for the manuscript-tuned version below.
    return fig_tc_throughput()
    dtypes = [r['dtype'].upper() for r in rows]
    gflops = [float(r['gflops']) for r in rows]
    colors = [BLUE, RED]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(dtypes, gflops, color=colors, width=0.45)

    for rect, val in zip(bars, gflops):
        ax.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 30,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11)
    ratio = gflops[1] / max(gflops[0], 1e-12)

    ax.text(0.5, (gflops[0] + gflops[1])*0.5,
            f'{ratio:.0f}x', ha='center', va='center',
            fontsize=13, fontweight='bold', color=GRAY)

    ax.set_ylabel('Throughput (GFLOP/s)')
    ax.set_title('Element matvec throughput')
    ax.set_ylim(0, max(gflops) * 1.12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    _savefig(fig, 'F5_tc_throughput.pdf')


# ===========================================================================
# F6 — ε·κ_eff stability (E5)  [scatter, grouped by mesh size]
# ===========================================================================
def _legacy_fig_kappa_eff():
    rows = _read_csv('e5_kappa_eff.csv')

    sizes_seen = sorted({r['preset'] for r in rows},
                        key=lambda s: int(s.rstrip('k')))
    markers    = {'64k': 'o', '216k': 's', '512k': 'D'}
    colors_s   = {'64k': BLUE, '216k': ORANGE, '512k': PURPLE}

    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    max_kappa = max(float(r['kappa_eff']) for r in rows)
    max_eps   = max(float(r['eps_kappa']) for r in rows)

    for sz in sizes_seen:
        sub = [r for r in rows if r['preset'] == sz]
        x   = [float(r['kappa_eff']) for r in sub]
        y   = [float(r['eps_kappa'])  for r in sub]
        ax.scatter(x, y, marker=markers.get(sz, 'x'),
                   color=colors_s.get(sz, GRAY),
                   s=60, zorder=5, label=f'Mesh size {sz}')

    ax.axhline(1.0, color=RED, lw=1.5, ls='--',
               label=r'Stability threshold ($\varepsilon\!\cdot\!\kappa=1$)')
    ax.fill_between([0, max(200, max_kappa * 1.05)], 0, 1,
                    color=GREEN, alpha=0.07, label='Safe region')

    n_violations = sum(1 for r in rows if float(r['eps_kappa']) >= 1.0)
    total        = len(rows)
    ax.text(0.98, 0.92,
            f'{total - n_violations}/{total} configurations satisfy bound',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GRAY, alpha=0.9))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\kappa_{\mathrm{eff}}$ (Lanczos probe)')
    ax.set_ylabel(r'$\varepsilon_{\mathrm{BF16}}\cdot\kappa_{\mathrm{eff}}$')
    ax.set_title(r'BF16 spectral-proxy diagnostic')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=10)
    fig.tight_layout()
    _savefig(fig, 'F6_kappa_eff.pdf')


# ===========================================================================
# F7 — Ablation study 2x2 (E6)
# ===========================================================================
def fig_ablations():
    e6a = _read_csv('e6a_precision_ablation.csv')
    e6b = _read_csv('e6b_depth_sweep.csv')
    e6c = _read_csv('e6c_vcycle_vs_wcycle.csv')
    e6d = _read_csv('e6d_smoother_type.csv')

    fig = plt.figure(figsize=(11, 8))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── (a) FP64 vs FP32 finest level ──────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    cfgs  = [r['config'].upper() for r in e6a]
    times = [_f(r, 'time_s')     for r in e6a]
    errs  = [_f(r, 'time_std_s', 0.0) for r in e6a]
    iters = [_f(r, 'iters')      for r in e6a]
    bars  = ax_a.bar(cfgs, times, color=[BLUE, GREEN], width=0.4, yerr=errs, capsize=3)
    for b, it, t in zip(bars, iters, times):
        ax_a.text(b.get_x() + b.get_width()/2., t + 0.003,
                  f'{it} iters\n{t:.3f} s', ha='center', va='bottom', fontsize=9.5)
    ax_a.set_ylabel('Solve time (s)')
    ax_a.set_title('(a) Fine-level precision')
    ax_a.set_ylim(0, max(times)*1.35)
    ax_a.grid(True, axis='y', linestyle='--', alpha=0.5)

    # ── (b) FP32 depth sweep ────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    depths = [int(r['n_fp32_levels']) for r in e6b]
    times_b = [_f(r, 'time_s')        for r in e6b]
    errs_b  = [_f(r, 'time_std_s', 0.0) for r in e6b]
    ax_b.bar(depths, times_b, color=PURPLE, width=0.6, yerr=errs_b, capsize=3)
    for d, t in zip(depths, times_b):
        ax_b.text(d, t + 0.003, f'{t:.3f} s', ha='center', va='bottom', fontsize=9.5)
    ax_b.set_xlabel('# FP32 levels (rest FP64)')
    ax_b.set_ylabel('Solve time (s)')
    ax_b.set_title('(b) Precision-depth sweep')
    ax_b.set_xticks(depths)
    ax_b.set_xticklabels([str(d) for d in depths], rotation=30)
    ax_b.set_ylim(0, max(times_b)*1.30)
    ax_b.grid(True, axis='y', linestyle='--', alpha=0.5)

    # ── (c) V-cycle vs W-cycle ──────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    cycles  = [r['cycle'].upper()   for r in e6c]
    times_c = [_f(r, 'time_s')      for r in e6c]
    errs_c  = [_f(r, 'time_std_s', 0.0) for r in e6c]
    iters_c = [_f(r, 'iters')       for r in e6c]
    colors_c = [GREEN, BLUE]
    bars_c   = ax_c.bar(cycles, times_c, color=colors_c, width=0.4, yerr=errs_c, capsize=3)
    for b, it, t in zip(bars_c, iters_c, times_c):
        ax_c.text(b.get_x() + b.get_width()/2., t + 0.004,
                  f'{it} iters\n{t:.3f} s', ha='center', va='bottom', fontsize=9.5)
    ax_c.set_ylabel('Solve time (s)')
    ax_c.set_title('(c) V-cycle vs. W-cycle')
    ax_c.set_ylim(0, max(times_c)*1.35)
    ax_c.grid(True, axis='y', linestyle='--', alpha=0.5)

    # ── (d) Smoother type ───────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    smth_labels = [f"{r['smoother'].capitalize()}, degree {r['degree']}" for r in e6d]
    times_d = [_f(r, 'time_s') for r in e6d]
    errs_d  = [_f(r, 'time_std_s', 0.0) for r in e6d]
    iters_d = [_f(r, 'iters')   for r in e6d]
    colors_d = [BLUE, BLUE, ORANGE, ORANGE]
    alphas_d = [0.9, 0.55, 0.9, 0.55]
    x_d = np.arange(len(smth_labels))
    for xi, (t, lab, col, al, it) in enumerate(
            zip(times_d, smth_labels, colors_d, alphas_d, iters_d)):
        b = ax_d.bar(xi, t, color=col, alpha=al, width=0.6, yerr=errs_d[xi], capsize=3)
        ax_d.text(xi, t + 0.005,
                  f'{it} iters\n{t:.3f} s', ha='center', va='bottom', fontsize=9.5)
    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(smth_labels, fontsize=8.8, rotation=18)
    ax_d.set_ylabel('Solve time (s)')
    ax_d.set_title('(d) Smoother type & degree')
    ax_d.set_ylim(0, max(times_d)*1.35)
    ax_d.grid(True, axis='y', linestyle='--', alpha=0.5)

    fig.suptitle('Ablation study (216k cantilever, ρ=0.5, p=3.0)', fontsize=13)
    _savefig(fig, 'F7_ablations.pdf')


# ===========================================================================
# F8 — Large-scale performance (E7)  [dual-axis: solve time + setup-time VRAM]
# ===========================================================================
def fig_large_scale():
    rows    = _read_csv('e7_large_scale.csv')
    n_elems = [int(r['n_elem'])          for r in rows]
    t_setup = [_f(r, 't_setup_s')        for r in rows]
    t_setup_sd = [_f(r, 't_setup_std_s', 0.0) for r in rows]
    t_solve = [_f(r, 't_solve_s')        for r in rows]
    t_solve_sd = [_f(r, 't_solve_std_s', 0.0) for r in rows]
    vram    = [_f(r, 'vram_delta_mb') / 1024  for r in rows]  # GiB
    vram_sd = [_f(r, 'vram_delta_std_mb', 0.0) / 1024 for r in rows]
    iters   = [_f(r, 'iters')            for r in rows]
    labels  = [r['size']                 for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.9))
    ax2 = ax1.twinx()

    ax1.errorbar(n_elems, t_solve, yerr=t_solve_sd, color=GREEN,  marker='o', ms=9, lw=2.2,
             capsize=3,
             label='Solve time (s)')
    ax1.errorbar(n_elems, t_setup, yerr=t_setup_sd, color=BLUE,   marker='s', ms=9, lw=2.2,
             capsize=3,
             ls='--', label='Setup time (s)')
    ax2.errorbar(n_elems, vram, yerr=vram_sd, color=ORANGE, marker='^', ms=9, lw=2.2,
             capsize=3,
             ls=':', label='Setup delta VRAM (GiB)')

    # Annotate solve times and iter counts
    for x, t, it in zip(n_elems, t_solve, iters):
        ax1.annotate(f'{it} iters\n{t:.2f} s', xy=(x, t),
                     xytext=(0, 14), textcoords='offset points',
                     ha='center', fontsize=9.5, color=GREEN)

    ax1.set_xscale('log')
    ax1.set_xlabel('Number of elements')
    ax1.set_ylabel('Wall time (s)', color='black')
    ax2.set_ylabel('Setup-time delta VRAM (GiB)', color=ORANGE)
    ax2.tick_params(axis='y', labelcolor=ORANGE)
    ax1.set_title('Large-scale FP32-GMG performance (FGMRES)')
    ax1.xaxis.set_major_locator(ticker.FixedLocator(n_elems))
    ax1.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    ax1.xaxis.set_minor_locator(ticker.NullLocator())
    ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax1.set_ylim(0, max(t_setup + t_solve) * 1.80)
    ax2.set_ylim(0, max(vram) * 1.4)
    ax1.grid(True, linestyle='--', alpha=0.5)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc='upper left')

    fig.tight_layout()
    _savefig(fig, 'F8_large_scale.pdf')


# ===========================================================================
# F9 — External baseline: GMG vs PyAMG (E8)  [stacked bar, like paper-3 F8]
# ===========================================================================
def _legacy_fig_external_baseline():
    rows = _read_csv('e8_external_baseline.csv')

    pyamg = next(r for r in rows if r['solver'].startswith('PyAMG'))
    t_build_py = _f(pyamg, 't_build_s')
    t_solve_py = _f(pyamg, 't_solve_s')
    t_build_py_sd = _f(pyamg, 't_build_std_s', 0.0)
    t_solve_py_sd = _f(pyamg, 't_solve_std_s', 0.0)

    gmg_rows = [r for r in rows if 'GMG' in r['solver']]
    if gmg_rows:
        gmg = gmg_rows[0]
        t_gmg_build = _f(gmg, 't_build_s')
        t_gmg_solve = _f(gmg, 't_solve_s')
        t_gmg_build_sd = _f(gmg, 't_build_std_s', 0.0)
        t_gmg_solve_sd = _f(gmg, 't_solve_std_s', 0.0)
    else:
        # Fallback: use E2 solve time; report setup as missing.
        e2    = _read_csv('e2_per_solve_wall_time.csv')
        gmg64 = next(r for r in e2 if r['size'] == '64k')
        t_gmg_solve = _f(gmg64, 't_gmg_fp32_s')
        t_gmg_build = float('nan')
        t_gmg_build_sd = 0.0
        t_gmg_solve_sd = _f(gmg64, 't_gmg_fp32_std_s', 0.0)

    solvers = ['PyAMG (CPU)', 'FP32-GMG (GPU)']
    builds  = [t_build_py, t_gmg_build]
    solves  = [t_solve_py, t_gmg_solve]

    x     = np.arange(len(solvers))
    width = 0.45

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    b1 = ax.bar(x, builds, width, color=BLUE,   label='Build / setup time')
    b2 = ax.bar(x, solves, width, color=GREEN,  label='Solve time',
                bottom=builds)
    totals = [builds[0] + solves[0], builds[1] + solves[1]]
    total_errs = [
        math.sqrt(t_build_py_sd**2 + t_solve_py_sd**2),
        math.sqrt(t_gmg_build_sd**2 + t_gmg_solve_sd**2),
    ]
    ax.errorbar(x, totals, yerr=total_errs, fmt='none', ecolor='black', capsize=4, lw=1.0)

    for xi, (bu, sl) in enumerate(zip(builds, solves)):
        total = bu + sl
        ax.text(xi, total + 0.3, f'{total:.2f} s', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    sp = (t_build_py + t_solve_py) / (t_gmg_build + t_gmg_solve)
    ax.annotate(f'{sp:.0f}x\nfaster',
                xy=(1, (t_gmg_build + t_gmg_solve) * 1.05),
                fontsize=12, fontweight='bold', color=GREEN, ha='center')

    ax.set_xticks(x)
    ax.set_xticklabels(solvers)
    ax.set_ylabel('Wall time (s)')
    ax.set_title('64k cantilever: GMG vs. PyAMG-SA')
    ax.set_ylim(0, (t_build_py + t_solve_py) * 1.18)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    _savefig(fig, 'F9_external_baseline.pdf')


# ===========================================================================
# F10 — Robustness (E10)  [horizontal bar]
# ===========================================================================
def _legacy_fig_robustness():
    rows = _read_csv('e10_robustness.csv')

    cases   = [r['case']         for r in rows]
    iters   = [int(r['iters'])   for r in rows]
    converg = [int(r['converged']) for r in rows]
    kappas  = [float(r['kappa_eff']) for r in rows]
    short_labels = [c.replace('-', '\n') for c in cases]
    bar_colors = [GREEN if c else RED for c in converg]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4.0),
                                      gridspec_kw={'width_ratios': [1.6, 1]})

    y = np.arange(len(rows))
    ax_l.barh(y, iters, color=bar_colors, height=0.55)
    ax_l.axvline(500, color=RED, ls=':', lw=1.2, alpha=0.7, label='maxiter=500')
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(short_labels, fontsize=11)
    ax_l.set_xlabel('FGMRES iterations')
    ax_l.set_title('(a) Iteration count')
    ax_l.grid(True, axis='x', linestyle='--', alpha=0.5)
    for yi, (it, c) in enumerate(zip(iters, converg)):
        ax_l.text(it + 8, yi, str(it), va='center', fontsize=10,
                  color=GREEN if c else RED, fontweight='bold')

    from matplotlib.patches import Patch
    ax_l.legend(handles=[Patch(color=GREEN, label='Converged'),
                          Patch(color=RED,   label='Failed / maxiter'),
                          plt.Line2D([0],[0], color=RED, ls=':', lw=1.2, label='maxiter=500')],
                loc='lower right', fontsize=9.5)

    ax_r.barh(y, kappas, color=[BLUE]*len(rows), height=0.55)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([])
    ax_r.set_xlabel(r'$\kappa_{\mathrm{eff}}$')
    ax_r.set_title(r'(b) $\kappa_{\mathrm{eff}}$')
    ax_r.grid(True, axis='x', linestyle='--', alpha=0.5)
    for yi, k in enumerate(kappas):
        ax_r.text(k + 0.1, yi, f'{k:.1f}', va='center', fontsize=10)

    fig.suptitle('E10: Robustness across density configurations', fontsize=13)
    fig.tight_layout()
    _savefig(fig, 'F10_robustness.pdf')


def fig_residual_histories():
    rows = _read_csv('e2_residual_histories.csv')
    sizes = ['64k', '216k', '512k']
    solvers = ['Jacobi-PCG', 'FP32-GMG', 'BF16-GMG']
    legend_labels = {'Jacobi-PCG': 'Jacobi-PCG', 'FP32-GMG': 'FP32-GMG', 'BF16-GMG': 'BF16-GMG (FGMRES)'}
    colors = {'Jacobi-PCG': ORANGE, 'FP32-GMG': GREEN, 'BF16-GMG': RED}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, size in zip(axes, sizes):
        size_rows = [r for r in rows if r['group'] == size]
        for solver in solvers:
            sr = [r for r in size_rows if r['solver'] == solver]
            if not sr:
                continue
            xs = [int(r['iter']) for r in sr]
            ys = [_f(r, 'rel_residual') for r in sr]
            ax.plot(xs, ys, lw=2.0, color=colors[solver], label=legend_labels[solver])
        ax.set_yscale('log')
        ax.set_xlabel('Outer iteration')
        ax.set_title(size)
        ax.grid(True, linestyle='--', alpha=0.5)
    axes[0].set_ylabel(r'Relative residual $\|r_k\|_2/\|b\|_2$')
    axes[0].legend(fontsize=9.5)
    fig.suptitle('E2: Residual histories by mesh size', fontsize=13)
    fig.tight_layout()
    _savefig(fig, 'F12_residual_histories.pdf')


def fig_simp_trajectory():
    rows = _read_csv('e3_simp_trajectory.csv')
    presets = sorted({r['preset'] for r in rows})
    label_map = {
        'cantilever_216k': 'Cantilever 216k',
        'torsion_small': 'Torsion 3k',
        'mbb_small': 'MBB 1.5k',
    }
    solver_map = {
        'paper3_jacobi': ('Jacobi-PCG', ORANGE),
        'paper4_gmg_fp32': ('FP32-GMG', GREEN),
    }
    fig, axes = plt.subplots(1, len(presets), figsize=(12, 3.8), sharey=False)
    if len(presets) == 1:
        axes = [axes]
    for ax, preset in zip(axes, presets):
        pr = [r for r in rows if r['preset'] == preset]
        for solver, (name, color) in solver_map.items():
            sr = [r for r in pr if r['solver'] == solver]
            xs = [int(r['step']) for r in sr]
            ys = [_f(r, 'compliance') for r in sr]
            ax.plot(xs, ys, color=color, lw=2.0, marker='o', ms=3, label=name)
        ax.set_title(label_map.get(preset, preset))
        ax.set_xlabel('SIMP step')
        ax.grid(True, linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Compliance')
    axes[0].legend(fontsize=9.5)
    fig.suptitle('E3: Compliance trajectory over the auxiliary fixed-penalty 30-step schedule', fontsize=13)
    fig.tight_layout()
    _savefig(fig, 'F13_simp_trajectory.pdf')


def fig_roofline():
    rows = _read_csv('e4_roofline.csv')
    oi = np.logspace(-3, 3, 400)
    peak_bw = max(_f(r, 'peak_bandwidth_gbs') for r in rows)
    peak_fp32 = max(_f(r, 'peak_compute_gflops') for r in rows if r['precision'] == 'fp32')
    peak_bf16 = max(_f(r, 'peak_compute_gflops') for r in rows if r['precision'] == 'bf16')

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(oi, peak_bw * oi, color=GRAY, lw=1.8, ls='--', label='Global-memory roof')
    ax.axhline(peak_fp32, color=BLUE, lw=1.8, label='FP32 compute roof')
    ax.axhline(peak_bf16, color=RED, lw=1.8, label='BF16 compute roof')

    color_map = {
        'fine_bf16_matvec': RED,
        'fine_fp32_matvec': BLUE,
        'level1_spmv': GREEN,
        'coarsest_solve': PURPLE,
    }
    marker_map = {
        'fine_bf16_matvec': '^',
        'fine_fp32_matvec': 'o',
        'level1_spmv': 's',
        'coarsest_solve': 'D',
    }
    anno_map = {
        'fine_bf16_matvec': dict(label='BF16 fine\nmatvec', xytext=(-58, -24), ha='right', va='top'),
        'fine_fp32_matvec': dict(label='FP32 fine\nmatvec', xytext=(18, 18), ha='left', va='bottom'),
        'level1_spmv': dict(label='Level-1\nSpMV', xytext=(-58, 18), ha='right', va='bottom'),
        'coarsest_solve': dict(label='Coarsest\nsolve', xytext=(16, -28), ha='left', va='top'),
    }
    for r in rows:
        x = _f(r, 'operational_intensity')
        y = _f(r, 'achieved_gflops')
        kernel = r['kernel']
        ann = anno_map.get(kernel, dict(label=kernel.replace('_', '\n'), xytext=(8, 12), ha='left', va='bottom'))
        ax.scatter(
            x, y, s=72, color=color_map.get(kernel, 'black'),
            marker=marker_map.get(kernel, 'o'), edgecolors='white', linewidths=0.8,
            zorder=3,
        )
        ax.annotate(
            ann['label'],
            (x, y),
            textcoords='offset points',
            xytext=ann['xytext'],
            ha=ann['ha'],
            va=ann['va'],
            fontsize=9.5,
            arrowprops=dict(arrowstyle='-', color=GRAY, lw=0.8, shrinkA=0, shrinkB=0),
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Operational intensity (FLOP/byte)')
    ax.set_ylabel('Achieved throughput (GFLOP/s)')
    ax.set_title('Roofline placement of representative kernels')
    ax.set_ylim(1e-1, max(peak_bf16, peak_fp32) * 1.2)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(fontsize=9.0, loc='lower right')
    fig.tight_layout()
    _savefig(fig, 'F14_roofline.pdf')


def fig_sensitivity_surface():
    rows = _read_csv('e6_sensitivity_surface.csv')
    available_smoothers = {r['fine_smoother'] for r in rows}
    preferred_order = ['fp32', 'bf16']
    smoothers = [s for s in preferred_order if s in available_smoothers]
    smoothers.extend(sorted(available_smoothers - set(smoothers)))
    restarts = sorted({int(r['restart']) for r in rows})
    fig, axes = plt.subplots(len(smoothers), len(restarts), figsize=(9.0, 5.4), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    degs = sorted({int(r['degree']) for r in rows})
    levels = sorted({int(r['n_levels']) for r in rows})
    finite_times = [_f(r, 'time_s') for r in rows if np.isfinite(_f(r, 'time_s'))]
    norm = mpl.colors.Normalize(vmin=min(finite_times), vmax=max(finite_times))
    fail_proxy = mpl.patches.Patch(facecolor='white', edgecolor=RED, hatch='///',
                                   label='hit iteration limit')
    for i, sm in enumerate(smoothers):
        for j, rst in enumerate(restarts):
            ax = axes[i, j]
            grid = np.full((len(levels), len(degs)), np.nan)
            ann = [["" for _ in degs] for _ in levels]
            failed = np.zeros((len(levels), len(degs)), dtype=bool)
            for r in rows:
                if r['fine_smoother'] != sm or int(r['restart']) != rst:
                    continue
                li = levels.index(int(r['n_levels']))
                di = degs.index(int(r['degree']))
                grid[li, di] = _f(r, 'time_s')
                failed[li, di] = int(r['converged']) == 0
                if failed[li, di]:
                    ann[li][di] = f"{_f(r, 'iters'):.0f}-iter\nlimit"
                else:
                    ann[li][di] = f"{grid[li, di]:.2f}s\n{_f(r, 'iters'):.0f} it"
            im = ax.imshow(grid, cmap='viridis', aspect='auto', norm=norm)
            for li in range(len(levels)):
                for di in range(len(degs)):
                    if np.isfinite(grid[li, di]):
                        if failed[li, di]:
                            ax.add_patch(mpl.patches.Rectangle(
                                (di - 0.5, li - 0.5), 1.0, 1.0, fill=False,
                                hatch='///', edgecolor=RED, linewidth=0.0,
                            ))
                            ax.text(di, li, ann[li][di], ha='center', va='center',
                                    fontsize=8.8, color='black', fontweight='bold',
                                    bbox=dict(facecolor='white', alpha=0.68, edgecolor='none', pad=0.8))
                        else:
                            ax.text(di, li, ann[li][di], ha='center', va='center',
                                    fontsize=8.8, color='white')
            ax.set_xticks(range(len(degs)))
            ax.set_xticklabels([f'degree {d}' for d in degs], fontsize=8.5)
            ax.set_yticks(range(len(levels)))
            ax.set_yticklabels([f'{lv} levels' for lv in levels], fontsize=8.5)
            ax.set_title(f'{sm.upper()} smoother, restart {rst}', fontsize=10.5)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'),
                        ax=axes.ravel().tolist(), fraction=0.024, pad=0.045)
    cbar.set_label('Wall time (s)')
    fig.legend(handles=[fail_proxy], loc='upper center', bbox_to_anchor=(0.5, 0.90),
               fontsize=9.5, frameon=True)
    fig.suptitle('Sensitivity to smoother degree, hierarchy depth, and FGMRES restart', fontsize=12.5, y=0.985)
    fig.subplots_adjust(left=0.09, right=0.86, bottom=0.11, top=0.80,
                        wspace=0.22, hspace=0.45)
    _savefig(fig, 'F15_sensitivity_surface.pdf')


def _legacy_fig_robustness_basin():
    # Backward-compatible alias for the manuscript-tuned version below.
    return fig_robustness_basin()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for penal, marker in [(3.0, 'o'), (4.5, '^')]:
        pr = [r for r in rows if abs(_f(r, 'penal') - penal) < 1e-12]
        xs = [_f(r, 'volfrac') for r in pr]
        ys = [_f(r, 'eps_kappa') for r in pr]
        cols = [GREEN if int(r['converged']) else RED for r in pr]
        ax.scatter(xs, ys, c=cols, marker=marker, s=70, label=f'p={penal}')
    ax.axhline(1.0, color='black', ls='--', lw=1.2, alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Volume fraction')
    ax.set_ylabel(r'$\varepsilon_{\mathrm{BF16}} \kappa_{\mathrm{eff}}$')
    ax.set_title('Robustness basin across contrast and volume fraction')
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker='o', color='w', markerfacecolor=GRAY, label='p=3.0', markersize=8),
        Line2D([0],[0], marker='^', color='w', markerfacecolor=GRAY, label='p=4.5', markersize=8),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=GREEN, label='converged', markersize=8),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=RED, label='failed', markersize=8),
    ], fontsize=9.5, loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    _savefig(fig, 'F16_robustness_basin.pdf')


# ===========================================================================
# Layout-tuned manuscript overrides
# These definitions intentionally supersede the earlier generic versions.
# ===========================================================================
def fig_simp_speedup():
    rows = _read_csv('e3_simp_speedup.csv')

    bench_labels = {
        'cantilever_216k': 'Cantilever\n(216k)',
        'torsion_small': 'Torsion\n(3k)',
        'mbb_small': 'MBB beam\n(1.5k)',
    }
    cap_hit_labels = {
        'cantilever_216k': '(27/30 cap hits)',
        'torsion_small': '(27/30 cap hits)',
        'mbb_small': '(25/30 cap hits)',
    }
    presets = [r['preset'] for r in rows]
    t_paper3 = [float(r['t_paper3_s']) for r in rows]
    t_gmg = [float(r['t_gmg_fp32_s']) for r in rows]
    ratios = [float(r['speedup']) for r in rows]
    xlabels = [bench_labels.get(p, p) for p in presets]

    x = np.arange(len(rows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.6))
    b1 = ax.bar(x - width / 2, t_paper3, width, color=ORANGE, label='Jacobi-PCG reference')
    b2 = ax.bar(x + width / 2, t_gmg, width, color=GREEN, label='FP32-GMG')

    for rect, sp in zip(b2, ratios):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 2.0, f'{sp:.1f}x',
                ha='center', va='bottom', fontsize=11, fontweight='bold', color=GREEN)

    for rect, preset in zip(b1, presets):
        ax.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height() + 1.6,
                cap_hit_labels.get(preset, ''),
                ha='center', va='bottom', fontsize=8.5, color=RED, style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel('SIMP wall time (s)')
    ax.set_title('Auxiliary same-schedule execution time\n(late trajectories diverge)')
    ax.set_ylim(0, max(t_paper3 + t_gmg) * 1.14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()
    fig.text(0.5, 0.01, 'Not a matched-final-design comparison; see trajectory panel for divergence.',
             ha='center', va='bottom', fontsize=9, color='dimgray')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _savefig(fig, 'F4_simp_speedup.pdf')


def fig_tc_throughput():
    rows = _read_csv('e4_tc_throughput.csv')
    dtypes = [r['dtype'].upper() for r in rows]
    gflops = [float(r['gflops']) for r in rows]
    colors = [BLUE, RED]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(dtypes, gflops, color=colors, width=0.45)
    for rect, val in zip(bars, gflops):
        ax.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height() + 28,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Throughput (GFLOP/s)')
    ax.set_title('Element matvec throughput')
    ax.set_ylim(0, max(gflops) * 1.12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    _savefig(fig, 'F5_tc_throughput.pdf')


def fig_kappa_eff():
    rows = _read_csv('e5_kappa_eff.csv')

    sizes_seen = sorted({r['preset'] for r in rows}, key=lambda s: int(s.rstrip('k')))
    markers = {'64k': 'o', '216k': 's', '512k': 'D'}
    colors_s = {'64k': BLUE, '216k': ORANGE, '512k': PURPLE}

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    max_kappa = max(float(r['kappa_eff']) for r in rows)

    for sz in sizes_seen:
        sub = [r for r in rows if r['preset'] == sz]
        x = [float(r['kappa_eff']) for r in sub]
        y = [float(r['eps_kappa']) for r in sub]
        ax.scatter(x, y, marker=markers.get(sz, 'x'), color=colors_s.get(sz, GRAY),
                   s=58, zorder=5, label=sz)

    ax.axhline(1.0, color=RED, lw=1.4, ls='--',
        label=r'Proxy threshold $\varepsilon_{\mathrm{BF16}}\kappa_{\mathrm{eff}}=1$')
    ax.fill_between([0, max(200, max_kappa * 1.05)], 0, 1, color=GREEN, alpha=0.07)

    n_violations = sum(1 for r in rows if float(r['eps_kappa']) >= 1.0)
    total = len(rows)
    ax.text(0.03, 0.10, f'{total - n_violations}/{total} satisfy the bound',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=GRAY, alpha=0.9))

    ax.set_xscale('log')
    ax.set_yscale('log')
    y_vals = [float(r['eps_kappa']) for r in rows]
    lo = int(np.floor(np.log10(min(y_vals))))
    hi = int(np.ceil(np.log10(max(y_vals))))
    ticks = sorted(set([10.0 ** k for k in range(lo, hi + 1)] + [1.0]))
    ax.set_yticks(ticks)
    ax.set_xlabel(r'$\kappa_{\mathrm{eff}}$ (Lanczos probe)')
    ax.set_ylabel(r'$\varepsilon_{\mathrm{BF16}}\kappa_{\mathrm{eff}}$')
    ax.set_title('BF16 spectral-proxy diagnostic')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=9.5, title='Mesh')
    fig.tight_layout()
    _savefig(fig, 'F6_kappa_eff.pdf')


def fig_external_baseline():
    rows = _read_csv('e8_external_baseline.csv')

    pyamg = next(r for r in rows if r['solver'].startswith('PyAMG'))
    t_build_py = _f(pyamg, 't_build_s')
    t_solve_py = _f(pyamg, 't_solve_s')
    t_build_py_sd = _f(pyamg, 't_build_std_s', 0.0)
    t_solve_py_sd = _f(pyamg, 't_solve_std_s', 0.0)

    gmg = next(r for r in rows if 'GMG' in r['solver'])
    t_gmg_build = _f(gmg, 't_build_s')
    t_gmg_solve = _f(gmg, 't_solve_s')
    t_gmg_build_sd = _f(gmg, 't_build_std_s', 0.0)
    t_gmg_solve_sd = _f(gmg, 't_solve_std_s', 0.0)

    solvers = ['PyAMG (CPU)', 'FP32-GMG (GPU)']
    builds = [t_build_py, t_gmg_build]
    solves = [t_solve_py, t_gmg_solve]
    totals = [builds[0] + solves[0], builds[1] + solves[1]]
    total_errs = [
        math.sqrt(t_build_py_sd ** 2 + t_solve_py_sd ** 2),
        math.sqrt(t_gmg_build_sd ** 2 + t_gmg_solve_sd ** 2),
    ]

    x = np.arange(len(solvers))
    width = 0.45

    fig, ax = plt.subplots(figsize=(5.7, 4.6))
    ax.bar(x, builds, width, color=BLUE, label='Build / setup time')
    ax.bar(x, solves, width, color=GREEN, label='Solve time', bottom=builds)
    ax.errorbar(x, totals, yerr=total_errs, fmt='none', ecolor='black', capsize=4, lw=1.0)

    for xi, total in enumerate(totals):
        ax.text(xi, total + 0.35, f'{total:.2f} s', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(solvers)
    ax.set_ylabel('Wall time (s)')
    ax.set_title('64k cantilever: CPU PyAMG post-assembly reference')
    ax.set_ylim(0, totals[0] * 1.18)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    _savefig(fig, 'F9_external_baseline.pdf')


def fig_robustness():
    rows = _read_csv('e10_robustness.csv')

    cases = [r['case'] for r in rows]
    iters = [int(r['iters']) for r in rows]
    converg = [int(r['converged']) for r in rows]
    kappas = [float(r['kappa_eff']) for r in rows]
    label_map = {
        'uniform-vf0.2': r'Uniform $V_f=0.2$',
        'uniform-vf0.5': r'Uniform $V_f=0.5$',
        'uniform-vf0.8': r'Uniform $V_f=0.8$',
        'binary-vf0.2-p1.5': r'Binary $V_f=0.2$, $p=1.5$',
        'binary-vf0.5-p3.0': r'Binary $V_f=0.5$, $p=3.0$',
        'binary-vf0.8-p4.5': r'Binary $V_f=0.8$, $p=4.5$',
        'checkerboard': 'Checkerboard',
        'layered-band': 'Layered band',
        'rho-min-1e-12': r'Random, $\rho_{\mathrm{floor,test}}=10^{-12}$',
        'mixed-very-low': 'Mixed near-void field',
    }
    short_labels = [label_map.get(c, c) for c in cases]
    bar_colors = [GREEN if c else RED for c in converg]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5.2),
                                     gridspec_kw={'width_ratios': [1.9, 1.1]})

    y = np.arange(len(rows))
    ax_l.barh(y, iters, color=bar_colors, height=0.68)
    ax_l.axvline(500, color=RED, ls=':', lw=1.2, alpha=0.7, label='maxiter=500')
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(short_labels, fontsize=10.5)
    ax_l.set_xlabel('FGMRES iterations')
    ax_l.set_title('(a) Iteration count')
    ax_l.grid(True, axis='x', linestyle='--', alpha=0.5)
    for yi, (it, c) in enumerate(zip(iters, converg)):
        ax_l.text(it + 8, yi, str(it), va='center', fontsize=10,
                  color=GREEN if c else RED, fontweight='bold')

    from matplotlib.patches import Patch
    ax_l.legend(handles=[
        Patch(color=GREEN, label='Converged'),
        Patch(color=RED, label='Failed at iteration cap'),
        plt.Line2D([0], [0], color=RED, ls=':', lw=1.2, label='iteration cap = 500'),
    ], loc='lower right', fontsize=9.5)

    ax_r.barh(y, kappas, color=[BLUE] * len(rows), height=0.68)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([])
    ax_r.set_xlabel(r'$\kappa_{\mathrm{eff}}$')
    ax_r.set_title(r'(b) $\kappa_{\mathrm{eff}}$')
    ax_r.grid(True, axis='x', linestyle='--', alpha=0.5)
    for yi, k in enumerate(kappas):
        ax_r.text(k + 2.0, yi, f'{k:.1f}', va='center', fontsize=9.5)

    fig.suptitle('Robustness across density configurations', fontsize=13)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.84, bottom=0.16, wspace=0.10)
    _savefig(fig, 'F10_robustness.pdf')


def fig_robustness_basin():
    rows = _read_csv('e10_basin.csv')
    grouped = {}
    for row in rows:
        key = (_f(row, 'volfrac'), _f(row, 'penal'))
        grouped.setdefault(key, []).append(row)

    collapsed = []
    for (volfrac, penal), grp in sorted(grouped.items()):
        eps_vals = np.array([_f(r, 'eps_kappa') for r in grp], dtype=float)
        iters_vals = np.array([_f(r, 'iters') for r in grp], dtype=float)
        conv_vals = [int(r['converged']) for r in grp]
        collapsed.append({
            'volfrac': volfrac,
            'penal': penal,
            'eps_kappa': float(np.mean(eps_vals)),
            'iters': float(np.mean(iters_vals)),
            'converged': int(all(conv_vals)),
        })

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ymin = min(r['eps_kappa'] for r in collapsed if r['eps_kappa'] > 0)
    ymax = max(r['eps_kappa'] for r in collapsed if np.isfinite(r['eps_kappa']))

    for penal, marker, dx in [(3.0, 'o', -0.012), (4.5, '^', 0.012)]:
        pr = [r for r in collapsed if abs(r['penal'] - penal) < 1e-12]
        xs = [r['volfrac'] + dx for r in pr]
        ys = [r['eps_kappa'] for r in pr]
        cols = [GREEN if r['converged'] else RED for r in pr]
        ax.scatter(xs, ys, c=cols, marker=marker, s=72, alpha=0.9, label=fr'$p={penal:.1f}$')
    ax.axhline(1.0, color='black', ls='--', lw=1.1, alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Volume fraction')
    ax.set_ylabel(r'$\varepsilon_{\mathrm{BF16}}\kappa_{\mathrm{eff}}$')
    ax.set_xlim(0.16, 0.84)
    ax.set_ylim(max(0.75 * ymin, 2e-2), 1.25 * ymax)
    ax.set_title(r'Single-seed robustness screening ($V_f$ vs. $p$)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper center', ncol=2, frameon=True)

    fig.subplots_adjust(top=0.84, bottom=0.18)
    _savefig(fig, 'F16_robustness_basin.pdf')


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default=str(DEFAULT_RESULTS),
                        help='Directory containing locally generated paper4 CSV results to plot')
    parser.add_argument('--figs-dir', default=str(DEFAULT_FIGS),
                        help='Directory where regenerated figures should be written')
    args = parser.parse_args()

    _configure_paths(args.results_dir, args.figs_dir)
    os.makedirs(FIGS, exist_ok=True)
    print('Generating figures...')
    fig_vcycle_iters()
    fig_solve_scaling()
    fig_solve_speedup()
    fig_simp_speedup()
    fig_tc_throughput()
    fig_kappa_eff()
    fig_ablations()
    fig_large_scale()
    fig_external_baseline()
    fig_robustness()
    fig_residual_histories()
    fig_simp_trajectory()
    fig_roofline()
    fig_sensitivity_surface()
    fig_robustness_basin()
    print('Done.')
