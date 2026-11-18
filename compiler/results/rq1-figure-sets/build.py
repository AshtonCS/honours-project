"""Build rq1-figure-sets.tex: the four RQ1-style full-page figures (CNOT/T x measured r/bound r).

Measured-r pages: figure bodies copied verbatim from results/gallery/chapter4-inline.tex
(fig:rq1-n500, fig:rq1-n500-t, i.e. thesis Figs. 4.3/4.4). Bound-r pages: the generated
results/np-bound-min/*-with-zoom.tex (top) and *-n500-dots.tex (bottom) set the same way;
the dots plot's footnote node is dropped because the thesis figures carry none.
Run from compiler/:  python results/rq1-figure-sets/build.py  then tectonic the .tex.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent
inline = (RES / "gallery/chapter4-inline.tex").read_text()


def thesis_body(label: str) -> tuple[str, str]:
    for m in re.finditer(r"\\begin\{figure\}\[H\](.*?)\\end\{figure\}", inline, re.S):
        if f"\\label{{{label}}}" in m.group(1):
            body, cap = m.group(1).split("\\caption{", 1)
            return body, cap.split("}\n", 1)[0]
    raise ValueError(f"figure {label} not found")


def strip_comments(tex: str) -> str:
    return "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))


def bound_body(metric: str) -> str:
    stem = RES / f"np-bound-min/bound-min-{metric}-all-formulas"
    zoom = strip_comments(Path(f"{stem}-with-zoom.tex").read_text())
    dots = strip_comments(Path(f"{stem}-n500-dots.tex").read_text())
    lines = dots.splitlines()
    note = [l for l in lines if l.startswith("  \\node[anchor=north, text width=")]
    if len(note) != 1:
        raise ValueError(f"expected one note node in {stem}-n500-dots.tex, found {len(note)}")
    dots = "\n".join(l for l in lines if l not in note)
    return ("  \\centering\n  \\begin{fitwidth}\n" + zoom + "\n  \\end{fitwidth}\\par\\bigskip\n"
            "  \\begin{fitwidth}\n" + dots + "\n  \\end{fitwidth}\n")


cnot_m, cnot_m_cap = thesis_body("fig:rq1-n500")
t_m, t_m_cap = thesis_body("fig:rq1-n500-t")
BOUND_NOTE = ("$r$ from the minimised norm bound (Childs et al.\\ 2018 Def.~2, extended to every formula by "
              "counting $2q$ sweeps), computed at every $n$; no fit, no extrapolation.")
pages = [
    ("CNOT count", "Measured $r$", "power-law fit on measured $n = 5$--$11$, extrapolated to $n = 500$",
     "As in the thesis (Figure~4.3)", cnot_m, cnot_m_cap),
    ("CNOT count", "Minimised bound $r$", "norm bound evaluated at every $n = 4$--$500$",
     "New; counterpart of Figure~4.3", bound_body("cnot"),
     "CNOT count against system size, $r$ from the minimised bound: full range and $n = 200$--$500$ "
     "enlarged (top); count at $n = 500$ over seeds (bottom). " + BOUND_NOTE +
     " Same on all 10 seeds, so no spread; M\\&O $q=21$ and Morales Y8m10 coincide (same order and $q$)."),
    ("T count", "Measured $r$", "power-law fit on measured $n = 5$--$11$, extrapolated to $n = 500$",
     "As in the thesis (Figure~4.4)", t_m, t_m_cap),
    ("T count", "Minimised bound $r$", "norm bound evaluated at every $n = 4$--$500$",
     "New; counterpart of Figure~4.4", bound_body("t"),
     "T count against system size, $r$ from the minimised bound: full range and $n = 200$--$500$ "
     "enlarged (top); count at $n = 500$ over seeds (bottom). " + BOUND_NOTE +
     " $\\epsilon = 5\\times10^{-4}$ for the formula plus $5\\times10^{-4}$ for synthesis; the spread "
     "($\\le 0.025\\%$) comes only from synthesising each seed's field angles."),
]

out = [r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{float,graphicx,xcolor}
\usepackage[font=small,labelformat=empty]{caption}
\pagestyle{empty}
\newsavebox{\resultbox}
\newenvironment{fitwidth}%
  {\begin{lrbox}{\resultbox}}%
  {\unskip\end{lrbox}%
   \ifdim\wd\resultbox>\linewidth
     \resizebox{\linewidth}{!}{\usebox{\resultbox}}%
   \else
     \usebox{\resultbox}%
   \fi}
\newcommand{\pagelabel}[5]{%
  \noindent\colorbox{black!8}{\parbox{\dimexpr\linewidth-2\fboxsep}{%
    \textbf{\large Page #1 of 4:\ \ #2 \,|\, #3}\\[2pt]
    \small $r$: #4\hfill #5}}\par\vspace{0.6cm}}
\begin{document}
"""]
for i, (gate, rsrc, how, where, body, cap) in enumerate(pages, 1):
    out.append(f"\\pagelabel{{{i}}}{{{gate}}}{{{rsrc}}}{{{how}}}{{{where}}}\n")
    out.append("\\begin{figure}[H]\n" + body + f"  \\caption{{{cap}}}\n\\end{{figure}}\n")
    out.append("\\clearpage\n" if i < 4 else "")
out.append("\\end{document}\n")
(HERE / "rq1-figure-sets.tex").write_text("".join(out))
print("wrote", HERE / "rq1-figure-sets.tex")
