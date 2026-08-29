#!/usr/bin/env python3
"""Render the five model-arm architecture diagrams as self-contained SVGs.

The figures are intentionally dependency-free: their source of truth is the
live model contract in ``docs/design.md`` and ``delta_feedback_experiment/model.py``.
Run this file after an architecture change and review every generated SVG.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

WIDTH = 1700
HEIGHT = 950
OUT = Path(__file__).resolve().parent


def text(
    x: float,
    y: float,
    lines: str | list[str],
    *,
    css: str = "label",
    anchor: str = "middle",
    gap: int = 22,
) -> str:
    if isinstance(lines, str):
        lines = [lines]
    tspans = []
    first_y = y - gap * (len(lines) - 1) / 2
    for i, line in enumerate(lines):
        tspans.append(
            f'<tspan x="{x}" y="{first_y + i * gap}">{escape(line)}</tspan>'
        )
    return (
        f'<text class="{css}" x="{x}" y="{first_y}" text-anchor="{anchor}">'
        f'{"".join(tspans)}</text>'
    )


def box(
    x: float,
    y: float,
    w: float,
    h: float,
    lines: str | list[str],
    *,
    kind: str = "neutral",
    css: str = "label",
    radius: int = 16,
) -> str:
    return (
        f'<rect class="node {kind}" x="{x}" y="{y}" width="{w}" height="{h}" '
        f'rx="{radius}"/>'
        + text(x + w / 2, y + h / 2 + 1, lines, css=css)
    )


def circle(x: float, y: float, label: str = "+") -> str:
    return (
        f'<circle class="sum" cx="{x}" cy="{y}" r="19"/>'
        + text(x, y + 6, label, css="sum-label")
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    kind: str = "blue",
    dashed: bool = False,
    arrow: bool = True,
) -> str:
    dash = ' stroke-dasharray="8 7"' if dashed else ""
    marker = f' marker-end="url(#arrow-{kind})"' if arrow else ""
    return (
        f'<line class="edge {kind}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"'
        f"{dash}{marker}/>")


def path(
    points: list[tuple[float, float]],
    *,
    kind: str = "blue",
    dashed: bool = False,
    arrow: bool = True,
) -> str:
    commands = [f"M {points[0][0]} {points[0][1]}"]
    commands.extend(f"L {x} {y}" for x, y in points[1:])
    dash = ' stroke-dasharray="8 7"' if dashed else ""
    marker = f' marker-end="url(#arrow-{kind})"' if arrow else ""
    return f'<path class="edge {kind}" d="{" ".join(commands)}"{dash}{marker}/>'


def start_svg(title: str, subtitle: str, description: str) -> list[str]:
    safe_id = title.lower().replace(" ", "-")
    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'role="img" aria-labelledby="{safe_id}-title {safe_id}-desc">'
        ),
        f'<title id="{safe_id}-title">{escape(title)}</title>',
        f'<desc id="{safe_id}-desc">{escape(description)}</desc>',
        """<defs>
  <marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#2563a8"/></marker>
  <marker id="arrow-purple" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#7c3aed"/></marker>
  <marker id="arrow-orange" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#c86b08"/></marker>
  <marker id="arrow-gray" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#64748b"/></marker>
  <style>
    .canvas { fill: #ffffff; }
    .title { font: 600 34px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #172033; }
    .subtitle { font: 400 17px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #526070; }
    .section { font: 600 18px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #172033; }
    .label { font: 600 16px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #172033; }
    .small { font: 500 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #526070; }
    .formula { font: 500 15px ui-monospace, SFMono-Regular, Menlo, monospace; fill: #172033; }
    .sum-label { font: 600 22px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #172033; }
    .node { stroke-width: 2; }
    .neutral { fill: #f7f9fc; stroke: #9aa7b7; }
    .blue-node { fill: #eaf2ff; stroke: #2563a8; }
    .purple-node { fill: #f2ebff; stroke: #7c3aed; }
    .orange-node { fill: #fff2df; stroke: #c86b08; }
    .green-node { fill: #e5f7f0; stroke: #138a66; }
    .gray-node { fill: #eef2f6; stroke: #64748b; }
    .stack { fill: #fbfcfe; stroke: #9aa7b7; stroke-width: 2; }
    .sum { fill: #ffffff; stroke: #2563a8; stroke-width: 2; }
    .edge { fill: none; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
    .blue { stroke: #2563a8; }
    .purple { stroke: #7c3aed; }
    .orange { stroke: #c86b08; }
    .gray { stroke: #64748b; }
    .divider { stroke: #d5dce6; stroke-width: 1.5; }
  </style>
</defs>""",
        f'<rect class="canvas" width="{WIDTH}" height="{HEIGHT}"/>',
        text(60, 62, title, css="title", anchor="start"),
        text(60, 101, subtitle, css="subtitle", anchor="start"),
        '<line class="divider" x1="60" y1="132" x2="1640" y2="132"/>',
    ]
    return svg


def finish_svg(svg: list[str], footer: str) -> str:
    svg.append('<line class="divider" x1="60" y1="884" x2="1640" y2="884"/>')
    svg.append(text(60, 918, footer, css="small", anchor="start"))
    svg.append("</svg>\n")
    return "\n".join(svg)


def stack(svg: list[str], *, routed: bool, source_label: list[str] | None = None) -> None:
    """Draw the shared trunk on one fixed grid.

    Routed arms reserve the upper half for one source bank and two vertical
    read paths. The residual computation stays on a single horizontal lane;
    no routing edge crosses it.
    """
    x, y, w, h = 500, 205, 680, 500
    stream_y = 515
    svg.append(
        f'<rect class="stack" x="{x}" y="{y}" width="{w}" height="{h}" rx="24"/>'
    )
    svg.append(
        text(
            x + 28,
            y + 38,
            "Transformer block ℓ   × L",
            css="section",
            anchor="start",
        )
    )
    svg.append(
        text(
            x + w - 28,
            y + 38,
            "pre-norm residual trunk",
            css="small",
            anchor="end",
        )
    )

    svg.append(text(x + 36, stream_y - 27, "hℓ", css="formula"))
    svg.append(line(x + 18, stream_y, x + 100, stream_y))
    attn_lines = (
        ["RMSNorm(h + rᵃℓ)", "Attention → Δaℓ"]
        if routed
        else ["RMSNorm", "Attention → Δaℓ"]
    )
    mlp_lines = (
        ["RMSNorm(h + rᵐℓ)", "MLP → Δmℓ"]
        if routed
        else ["RMSNorm", "MLP → Δmℓ"]
    )
    svg.append(box(x + 100, 467, 200, 96, attn_lines, kind="blue-node"))
    svg.append(line(x + 300, stream_y, x + 316, stream_y))
    svg.append(circle(x + 336, stream_y))
    svg.append(line(x + 355, stream_y, x + 370, stream_y))
    svg.append(box(x + 370, 467, 200, 96, mlp_lines, kind="blue-node"))
    svg.append(line(x + 570, stream_y, x + 586, stream_y))
    svg.append(circle(x + 606, stream_y))
    svg.append(line(x + 625, stream_y, x + 680, stream_y))
    svg.append(text(x + 650, stream_y - 27, "hℓ₊₁", css="formula"))

    if routed:
        svg.append(
            box(
                x + 70,
                275,
                540,
                80,
                source_label or ["Sources Sℓ"],
                kind="purple-node",
                css="formula",
            )
        )
        svg.append(
            box(
                x + 120,
                380,
                160,
                66,
                ["rᵃℓ = route", "(Sℓ, qᵃℓ)"],
                kind="purple-node",
                css="formula",
            )
        )
        svg.append(
            box(
                x + 390,
                380,
                160,
                66,
                ["rᵐℓ = route", "(Sℓ, qᵐℓ)"],
                kind="purple-node",
                css="formula",
            )
        )
        svg.append(line(x + 200, 355, x + 200, 378, kind="purple"))
        svg.append(line(x + 470, 355, x + 470, 378, kind="purple"))
        svg.append(line(x + 200, 446, x + 200, 465, kind="purple"))
        svg.append(line(x + 470, 446, x + 470, 465, kind="purple"))
        svg.append(
            text(
                x + w / 2,
                635,
                "Each branch delta joins Sℓ before the next routed read.",
                css="small",
            )
        )
    else:
        svg.append(
            box(
                x + 140,
                295,
                400,
                70,
                ["No source bank  ·  no routing sites"],
                kind="neutral",
            )
        )
        svg.append(
            text(
                x + w / 2,
                635,
                "Each sublayer reads only the current residual stream.",
                css="small",
            )
        )


def common_head(svg: list[str]) -> None:
    svg.append(
        box(
            1215,
            468,
            190,
            94,
            ["Top state", "hₜᵗᵒᵖ"],
            kind="blue-node",
            css="formula",
        )
    )
    svg.append(line(1180, 515, 1213, 515))
    svg.append(
        box(
            1460,
            453,
            195,
            124,
            ["Final RMSNorm", "+ tied LM head", "→ logitsₜ"],
            kind="green-node",
        )
    )
    svg.append(line(1405, 515, 1458, 515, kind="blue"))


def token_input(svg: list[str], *, y: int = 479) -> None:
    svg.append(box(45, y, 120, 72, ["Token", "xₜ"], kind="neutral", css="formula"))
    svg.append(line(165, y + 36, 202, y + 36))
    svg.append(box(205, y - 8, 210, 88, ["Token embedding", "eₜ"], kind="blue-node", css="formula"))


def feedback_input(svg: list[str]) -> None:
    svg.append(
        box(
            45,
            255,
            190,
            88,
            ["Previous payload", "pₜ₋₁"],
            kind="orange-node",
            css="formula",
        )
    )


def gated_entry(svg: list[str]) -> None:
    feedback_input(svg)
    svg.append(box(45, 628, 120, 72, ["Token", "xₜ"], kind="neutral", css="formula"))
    svg.append(line(165, 664, 198, 664))
    svg.append(box(200, 620, 190, 88, ["Embedding", "eₜ"], kind="blue-node", css="formula"))
    svg.append(
        box(
            270,
            440,
            190,
            150,
            [
                "Asymmetric GLU",
                "value: Wᵤpₜ₋₁",
                "gate: σ(Wɢ·norm(eₜ))",
                "RMSNorm → uₜ",
            ],
            kind="orange-node",
            css="small",
        )
    )
    svg.append(path([(235, 299), (365, 299), (365, 438)], kind="orange"))
    svg.append(path([(295, 620), (295, 605), (365, 605), (365, 592)], kind="blue"))
    svg.append(line(460, 515, 498, 515, kind="orange"))
    svg.append(text(479, 488, "uₜ", css="formula"))


def simple_payload(svg: list[str]) -> None:
    svg.append(path([(1310, 562), (1310, 755)], kind="orange"))
    svg.append(
        box(
            1235,
            757,
            150,
            76,
            ["RMSNorm", "pₜ"],
            kind="orange-node",
            css="formula",
        )
    )
    svg.append(line(1385, 795, 1458, 795, kind="orange"))
    svg.append(box(1460, 757, 195, 76, ["Carry to", "step t+1"], kind="orange-node"))


def routed_payload(svg: list[str], *, soft: bool) -> None:
    source = (
        ["From Sℓ", "[null, Δa₀, Δm₀, …]"]
        if soft
        else ["From Sℓ", "[Δa₀, Δm₀, …]"]
    )
    source_kind = "gray-node" if soft else "purple-node"
    source_edge = "gray" if soft else "purple"
    svg.append(box(535, 757, 250, 76, source, kind=source_kind, css="formula"))
    svg.append(line(785, 795, 827, 795, kind=source_edge))
    svg.append(
        box(
            830,
            757,
            180,
            76,
            ["Payload route", "qₚ"],
            kind="purple-node",
            css="formula",
        )
    )
    svg.append(line(1010, 795, 1288, 795, kind="purple"))
    svg.append(text(1145, 773, "routed deltas", css="small"))
    svg.append(circle(1310, 795))
    svg.append(path([(1310, 562), (1310, 774)], kind="blue"))
    svg.append(text(1328, 708, "base hₜᵗᵒᵖ", css="small", anchor="start"))
    svg.append(line(1329, 795, 1360, 795, kind="orange"))
    svg.append(
        box(
            1362,
            757,
            120,
            76,
            ["RMSNorm", "pₜ"],
            kind="orange-node",
            css="formula",
        )
    )
    svg.append(line(1482, 795, 1508, 795, kind="orange"))
    svg.append(box(1510, 757, 145, 76, ["Carry to", "step t+1"], kind="orange-node"))


def vanilla() -> str:
    svg = start_svg(
        "Vanilla transformer",
        "Baseline · ordinary residual depth · no cross-column state",
        "Token embedding enters a standard pre-norm residual transformer stack and the top state feeds the language-model head. There is no depth router or recurrent payload.",
    )
    token_input(svg)
    svg.append(line(415, 515, 498, 515))
    stack(svg, routed=False)
    common_head(svg)
    svg.append(box(650, 760, 380, 70, ["No recurrent payload"], kind="neutral"))
    return finish_svg(svg, "Blue = ordinary residual computation. One column at position t is shown.")


def dar() -> str:
    svg = start_svg(
        "DAR · Delta Attention Residuals",
        "Vertical-axis widening · transient reads over the current column’s decomposition",
        "The embedding seeds a source bank. Before every attention and MLP sublayer, a zero-initialized query routes over the seed and prior sublayer deltas. The routed mixture enriches the read transiently while the residual stream remains clean.",
    )
    token_input(svg)
    svg.append(line(415, 515, 498, 515))
    stack(
        svg,
        routed=True,
        source_label=["Sources Sℓ", "[eₜ  |  Δa₀, Δm₀, …]", "seed + prior deltas"],
    )
    svg.append(path([(310, 469), (310, 315), (568, 315)], kind="purple"))
    svg.append(text(422, 294, "seed eₜ", css="small"))
    common_head(svg)
    svg.append(box(650, 760, 380, 70, ["No cross-column payload"], kind="neutral"))
    return finish_svg(
        svg,
        "Purple = zero-init routing. With only seed eₜ, layer-0 attention routing is a no-op.",
    )


def fbt() -> str:
    svg = start_svg(
        "FBT · Full-Bandwidth Transformer",
        "Horizontal-axis widening · previous top state re-enters layer 0 through a mandatory gate",
        "The previous column payload and current token embedding are fused by an asymmetric GLU. The fused input passes through a standard transformer stack. The normalized top state becomes the payload for the next position.",
    )
    gated_entry(svg)
    stack(svg, routed=False)
    common_head(svg)
    simple_payload(svg)
    return finish_svg(svg, "Orange = cross-column feedback. The trunk has no depth routers; pₜ is the bare normalized top state.")


def df() -> str:
    svg = start_svg(
        "DF · Delta Feedback",
        "Hard combination · gated horizontal feedback + mandatory vertical delta routing",
        "The previous payload and token embedding are fused into u. The fused input seeds DAR-style routing throughout the stack. A dedicated router enriches the top state with a delta-only mixture before forming the next payload.",
    )
    gated_entry(svg)
    stack(
        svg,
        routed=True,
        source_label=["Sources Sℓ", "[uₜ  |  Δa₀, Δm₀, …]", "fused seed + prior deltas"],
    )
    svg.append(path([(480, 515), (480, 315), (568, 315)], kind="purple"))
    svg.append(text(520, 294, "seed uₜ", css="small"))
    common_head(svg)
    routed_payload(svg, soft=False)
    return finish_svg(svg, "Hard-everywhere: neither the routed reads nor the recurrent payload has a null escape.")


def df_soft() -> str:
    svg = start_svg(
        "DF-soft · Optional Delta Feedback",
        "Soft combination · plain token entry + null-enabled routers everywhere",
        "The token embedding remains the residual-stream input. The previous payload is an ungated standing source beside the embedding and prior deltas. Every depth and payload router includes a learnable null source, so the model can regress to vanilla.",
    )
    feedback_input(svg)
    token_input(svg)
    svg.append(line(415, 515, 498, 515))
    stack(
        svg,
        routed=True,
        source_label=[
            "Sources Sℓ",
            "[null  |  pₜ₋₁  |  eₜ  |  Δa₀, Δm₀, …]",
            "pₜ₋₁ masked when absent",
        ],
    )
    svg.append(path([(235, 299), (500, 299), (500, 295), (568, 295)], kind="orange"))
    svg.append(path([(310, 469), (310, 335), (568, 335)], kind="purple"))
    svg.append(text(315, 590, "plain entry  ·  h₀ = eₜ  ·  no GLU", css="small"))
    common_head(svg)
    routed_payload(svg, soft=True)
    return finish_svg(svg, "Gray = null escape. Choosing null at every router recovers the vanilla path; pₜ₋₁ is never fused into h₀.")


def main() -> None:
    diagrams = {
        "vanilla.svg": vanilla(),
        "dar.svg": dar(),
        "fbt.svg": fbt(),
        "df.svg": df(),
        "df-soft.svg": df_soft(),
    }
    for name, contents in diagrams.items():
        svg_path = OUT / name
        svg_path.write_text(contents, encoding="utf-8")


if __name__ == "__main__":
    main()
