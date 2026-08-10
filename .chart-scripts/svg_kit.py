"""Minimal SVG building blocks shared by the FastMetal-QAD blog figures.

Every helper returns a string; nothing is mutated in place. Callers collect the
returned fragments into a list and join them once at the end.
"""

from __future__ import annotations

FONT = '-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif'
MONO = 'ui-monospace,SFMono-Regular,Menlo,monospace'

INK = "#1f2a44"
MUTED = "#5d687c"
FAINT = "#93a0b4"
GRID = "#e8ecf2"
RULE = "#c8d0dc"
PAPER = "#ffffff"

# One hue per model, used consistently across every figure in the post.
MODEL_COLORS = {
    "1.3B": "#3b7dd8",
    "5B": "#1f8a4c",
    "14B": "#d1521a",
}
MODEL_TINTS = {
    "1.3B": "#d9e6f9",
    "5B": "#d7efe1",
    "14B": "#fbe3d6",
}


def escape(text: str) -> str:
    """Escape the XML characters that can appear in chart labels."""
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def header(width: int, height: int, label: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)}">\n'
            f'  <rect x="0" y="0" width="{width}" height="{height}" fill="{PAPER}"/>')


def styles(extra: str = "") -> str:
    return f"""  <defs><style>
    .t   {{ font: 700 21px {FONT}; fill: {INK}; }}
    .st  {{ font: 500 14px {FONT}; fill: {MUTED}; }}
    .lbl {{ font: 600 13px {FONT}; fill: {INK}; }}
    .sub {{ font: 500 12px {FONT}; fill: {MUTED}; }}
    .tick{{ font: 500 11px {MONO}; fill: #7a8699; }}
    .val {{ font: 700 13px {MONO}; fill: {INK}; }}
    .vs  {{ font: 500 11px {MONO}; fill: {MUTED}; }}
    .vsw {{ font: 600 11px {MONO}; fill: #ffffff; }}
    .na  {{ font: italic 500 12px {FONT}; fill: {FAINT}; }}
    .cap {{ font: 500 12px {FONT}; fill: #6b7688; }}
    {extra}
  </style></defs>"""


def title(x: float, y: float, text: str, subtitle: str | None = None) -> str:
    out = f'  <text x="{x}" y="{y}" class="t">{escape(text)}</text>'
    if subtitle:
        out += f'\n  <text x="{x}" y="{y + 22}" class="st">{escape(subtitle)}</text>'
    return out


def text(x: float, y: float, body: str, cls: str = "lbl", anchor: str = "start") -> str:
    return (f'  <text x="{x:.1f}" y="{y:.1f}" class="{cls}" '
            f'text-anchor="{anchor}">{escape(body)}</text>')


def bar(x: float, y: float, w: float, h: float, fill: str, radius: float = 5) -> str:
    return (f'  <rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 1.5):.1f}" '
            f'height="{h:.1f}" rx="{radius}" fill="{fill}"/>')


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = GRID,
         width: float = 1, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>')


def swatch(x: float, y: float, fill: str, label: str) -> str:
    return (bar(x, y, 13, 13, fill, radius=3) + "\n" +
            text(x + 20, y + 11, label, cls="sub"))


def render(width: int, height: int, label: str, body: list[str], extra_css: str = "") -> str:
    parts = [header(width, height, label), styles(extra_css), *body, "</svg>", ""]
    return "\n".join(parts)
