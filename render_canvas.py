#!/usr/bin/env python3
"""Render an Obsidian .canvas file to a high-resolution PNG."""

import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Appearance ───────────────────────────────────────────────────────────────
CANVAS_BG = '#161616'
TEXT_COLOR = '#d4d4d4'
EDGE_COLOR = '#888888'
FONT_SIZE  = 7.5    # pt
SCALE      = 0.0085  # canvas-px → figure-inches
DPI        = 200
MARGIN     = 80     # canvas-px margin around the whole graph

# Obsidian color id → (border_hex, fill_hex)
NODE_STYLES = {
    '1': ('#fb464c', '#2d1010'),
    '2': ('#e9973f', '#2d1c0c'),
    '3': ('#e0de71', '#2d2c0c'),
    '4': ('#44cf6e', '#0c2d18'),
    '5': ('#53dfdd', '#0c2c2c'),
    '6': ('#a882ff', '#1c0c30'),
    None: ('#484848', '#1e1e1e'),
}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']

# Emoji → ASCII equivalents (borders already encode status visually)
EMOJI_MAP = {
    '🟩': '[+]', '🟥': '[-]', '🟨': '[~]', '⬜': '[_]', '🟦': '[?]',
    '🔀': '[Q]', '🔬': '[E]', '🌱': '[*]', '🗺': '[M]',
    '✓': '(v)', '·': '-',
}


def clean_text(raw: str, node_width: int) -> str:
    """Strip markdown syntax, replace emoji, wrap lines to fit node width."""
    char_w_px = (FONT_SIZE / 72 * 0.60) / SCALE
    cols = max(8, int((node_width - 12) / char_w_px))

    # Replace emoji before any other processing
    for emoji, sub in EMOJI_MAP.items():
        raw = raw.replace(emoji, sub)

    result = []
    for line in raw.split('\n'):
        line = re.sub(r'^#{1,6}\s*', '', line)
        line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
        line = re.sub(r'\*(.+?)\*', r'\1', line)
        line = re.sub(r'_(.+?)_', r'\1', line)
        line = re.sub(r'`(.+?)`', r'\1', line)
        line = line.strip()
        if line:
            result.append(textwrap.fill(line, width=cols))
        else:
            result.append('')
    return '\n'.join(result).strip()


def edge_point(node, side, tx, ty):
    """Return (fig_x, fig_y) for a connection point on a node side."""
    x, y, w, h = node['x'], node['y'], node['width'], node['height']
    if side == 'top':    return tx(x + w / 2), ty(y)
    if side == 'bottom': return tx(x + w / 2), ty(y + h)
    if side == 'left':   return tx(x),          ty(y + h / 2)
    if side == 'right':  return tx(x + w),       ty(y + h / 2)
    return tx(x + w / 2), ty(y + h / 2)


def render(src: str, dst: str):
    data  = json.loads(Path(src).read_text(encoding='utf-8'))
    nodes = data['nodes']
    edges = data.get('edges', [])
    nmap  = {n['id']: n for n in nodes}

    bx0 = min(n['x']              for n in nodes) - MARGIN
    by0 = min(n['y']              for n in nodes) - MARGIN
    bx1 = max(n['x'] + n['width']  for n in nodes) + MARGIN
    by1 = max(n['y'] + n['height'] for n in nodes) + MARGIN
    W, H = bx1 - bx0, by1 - by0

    fig, ax = plt.subplots(figsize=(W * SCALE, H * SCALE), dpi=DPI)
    fig.patch.set_facecolor(CANVAS_BG)
    ax.set_facecolor(CANVAS_BG)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.axis('off')

    # Canvas → figure data units (y-axis flipped: canvas y↓ → mpl y↑)
    def tx(cx): return cx - bx0
    def ty(cy): return by1 - cy

    # Draw edges behind nodes
    for e in edges:
        s = nmap.get(e['fromNode'])
        d = nmap.get(e['toNode'])
        if not s or not d:
            continue
        ex1, ey1 = edge_point(s, e.get('fromSide', 'bottom'), tx, ty)
        ex2, ey2 = edge_point(d, e.get('toSide',   'top'),    tx, ty)
        ax.annotate('', xy=(ex2, ey2), xytext=(ex1, ey1),
                    arrowprops=dict(arrowstyle='->', color=EDGE_COLOR,
                                   lw=1.2, mutation_scale=12),
                    zorder=1)

    # Draw nodes
    for node in nodes:
        border, fill = NODE_STYLES.get(node.get('color'), NODE_STYLES[None])
        nx = tx(node['x'])
        ny = ty(node['y'] + node['height'])   # bottom-left in mpl coords
        nw, nh = node['width'], node['height']

        ax.add_patch(patches.Rectangle(
            (nx, ny), nw, nh,
            linewidth=1.5, edgecolor=border, facecolor=fill, zorder=2,
        ))

        text = clean_text(node.get('text', ''), nw)
        ax.text(nx + nw / 2, ny + nh / 2, text,
                ha='center', va='center', color=TEXT_COLOR,
                fontsize=FONT_SIZE, zorder=3,
                multialignment='center', linespacing=1.3,
                clip_on=True)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(dst, dpi=DPI, bbox_inches='tight', facecolor=CANVAS_BG)
    plt.close(fig)
    print(f'Saved: {dst}')


if __name__ == '__main__':
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else 'ResearchMap3.canvas'
    dst = sys.argv[2] if len(sys.argv) > 2 else Path(src).with_suffix('.png').name
    render(src, dst)
