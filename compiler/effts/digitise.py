"""Recover the data behind paper Figs. 5 and 6 from the arXiv HTML rendering.

The arXiv HTML of 2601.18756v3 embeds both figures as inline SVG (pgfplots),
so the plotted coordinates are exact up to the SVG's 0.01 px rounding --
about 3e-4 decades on these axes.  The paper's raw data are not published
(upstream README: "available upon request"), so this is the only
quantitative record of the experiments.

    python -m effts.digitise path/to/2601.18756v3.html

writes ``fixtures/paper_fig5_digitised.json`` and
``fixtures/paper_fig6_digitised.json``.  Fetch the HTML with
``curl -sL https://arxiv.org/html/2601.18756v3 -o paper.html``.

Calibration (read from the SVG's own grid lines and tick labels):
* Fig. 5, both panels: vertical grid lines at cost 10^2 and 10^3; horizontal
  grid lines every 2 decades with 10^0 at y = 453.16 and 10^-10 at 109.7
  (the plot frame is y-up: the figure has one matrix(1 0 0 -1) flip).
* Fig. 6: vertical grid lines at L = 2..12; horizontal grid lines every
  decade, 10^-2 at y = 471.53 and 10^-7 at 120.96.

Curve identity.  Fig. 5 colours encode only the order n, so within a colour
the scheme is identified from the data itself: every cost on a curve is a
multiple of that scheme's q (checked, and the result agrees with the legend
order).  Fig. 6 has 11 markers per scheme, drawn in legend order.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIG5_FIXTURE = FIXTURES / "paper_fig5_digitised.json"
FIG6_FIXTURE = FIXTURES / "paper_fig6_digitised.json"

ORDER_COLOURS = ("#DC143C", "#9400D3", "#6495ED", "#32CD32", "#FF8C00")  # n = 2, 4, 6, 8, 10
LEGEND = [  # (label, q, order) in legend order, Figs. 5 and 6
    ("Leapfrog", 1, 2), ("Forest & Ruth", 3, 4), ("Recommended Tab. 2", 6, 4),
    ("Yoshida", 7, 6), ("Blanes & Moan", 10, 6), ("Recommended Tab. 3", 14, 6),
    ("Morales et al.", 17, 8), ("Morales et al.", 33, 10),
]


def _figure(html: str, fig_id: str) -> str:
    i = html.find(f'id="{fig_id}"')
    if i < 0:
        raise ValueError(f"{fig_id} not found")
    return html[i:html.find("</figure>", i)]


def fig5(html: str) -> list[dict]:
    fig = _figure(html, "S3.F5")
    curves: list[tuple[str, list[tuple[float, float]]]] = []
    stack: list[str | None] = []
    for m in re.finditer(r"<(/?)(g|path)([^>]*)>", fig):
        close, tag, attrs = m.groups()
        if tag == "g":
            if close:
                stack.pop()
            else:
                col = re.search(r'stroke="(#[0-9A-Fa-f]+)"', attrs)
                stack.append(col.group(1) if col else None)
        elif not close:
            d = re.search(r'd="([^"]*)"', attrs).group(1)
            pts = [(float(a), float(b)) for a, b in re.findall(r"[ML] ([\d.]+) ([\d.]+)", d)]
            cols = [c for c in stack if c]
            if len(pts) > 8 and cols and cols[-1] in ORDER_COLOURS:
                curves.append((cols[-1], pts))
    grid_x = {"global": (213.09, 423.41), "local": (689.68, 899.99)}  # cost 10^2, 10^3
    y0, ydec = 453.16, (453.16 - 109.7) / 10
    out = []
    for k, (col, pts) in enumerate(curves):
        panel = "global" if pts[0][0] < 580 else "local"
        x2, x3 = grid_x[panel]
        points = [(10 ** (2 + (x - x2) / (x3 - x2)), 10 ** ((y - y0) / ydec)) for x, y in pts]
        label, q, n = LEGEND[k % 8]
        if ORDER_COLOURS.index(col) != n // 2 - 1:
            raise ValueError(f"curve {k}: colour {col} does not match legend order n={n}")
        # interior points only: the first and last are clipped at the axis box
        for cost, _ in points[1:-1]:
            if abs(cost - q * round(cost / q)) > 0.012 * cost:
                raise ValueError(f"curve {k} ({label}): cost {cost:.1f} not a multiple of q={q}")
        out.append(dict(panel=panel, label=label, q=q, order=n, points=points[1:-1]))
    if len(out) != 16:
        raise ValueError(f"expected 16 curves, found {len(out)}")
    return out


def fig6(html: str) -> list[dict]:
    fig = _figure(html, "S3.F6")
    out = []
    k = 0
    for m in re.finditer(r'<g style="--ltx-stroke-color:#[0-9A-F]+;--ltx-fill-color:(#[0-9A-F]+);"[^>]*>', fig):
        if m.group(1) not in ORDER_COLOURS:
            continue
        rest = fig[m.end():]
        nxt = re.search(r'<g style="--ltx-stroke-color', rest)
        body = rest[:nxt.start()] if nxt else rest
        pos = re.findall(r'transform="matrix\(1\.0 0\.0 0\.0 1\.0 ([\d.-]+) ([\d.-]+)\)"', body)
        if len(pos) < 5:
            continue  # legend swatches
        label, q, n = LEGEND[k]
        k += 1
        pts = []
        for x, y in pos:
            L = round((float(x) - 126.54) / 45.758) + 2
            if abs(126.54 + 45.758 * (L - 2) - float(x)) > 0.05:
                raise ValueError(f"marker x {x} not on an L grid line")
            pts.append((L, 10 ** (-2 + (float(y) - 471.53) / ((471.53 - 120.96) / 5))))
        out.append(dict(label=label, q=q, order=n, points=pts))
    if len(out) != 8:
        raise ValueError(f"expected 8 marker series, found {len(out)}")
    return out


def main(path: str) -> None:
    raw = Path(path).read_bytes()
    html = raw.decode()
    meta = dict(source="https://arxiv.org/html/2601.18756v3", sha256=hashlib.sha256(raw).hexdigest(),
                note="SVG coordinates converted with the calibration in effts/digitise.py")
    FIXTURES.mkdir(exist_ok=True)
    FIG5_FIXTURE.write_text(json.dumps(dict(_meta=meta, t=10.0, L=6, model="XXZ", curves=fig5(html)), indent=1))
    FIG6_FIXTURE.write_text(json.dumps(dict(_meta=meta, t=10.0, cost=500, model="XXZ", grouping="global",
                                            series=fig6(html)), indent=1))
    print(f"wrote {FIG5_FIXTURE.name}, {FIG6_FIXTURE.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m effts.digitise paper.html")
    main(sys.argv[1])
