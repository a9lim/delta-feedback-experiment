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


def token_input(svg: list[str], *, gated: bool = False) -> None:
    y = 690 if gated else 590
    svg.append(box(45, y, 110, 70, ["Token", "xₜ"], kind="neutral", css="formula"))
    svg.append(line(155, y + 35, 188, y + 35))
    svg.append(box(190, y - 5, 190, 80, ["Embedding", "eₜ"], kind="blue-node", css="formula"))


def feedback_input(svg: list[str]) -> None:
    svg.append(
        box(
            45,
            250,
            170,
            70,
            ["Payload", "pₜ₋₁"],
            kind="orange-node",
            css="formula",
        )
    )


def gated_entry(svg: list[str], *, routed: bool) -> None:
    feedback_input(svg)
    token_input(svg, gated=True)
    svg.append(
        box(
            330,
            565,
            150,
            120,
            ["Asymmetric GLU", "Wᵤpₜ₋₁", "× σ(Wɢ·norm(eₜ))", "norm → uₜ"],
            kind="orange-node",
            css="small",
        )
    )
    svg.append(path([(215, 285), (405, 285), (405, 563)], kind="orange"))
    svg.append(path([(285, 690), (285, 675), (405, 675), (405, 687)], kind="blue"))
    svg.append(line(480, 625, 498, 625, kind="orange"))
    if routed:
        svg.append(path([(490, 625), (490, 290), (498, 290)], kind="purple"))
        svg.append(text(465, 470, "seed uₜ", css="small", anchor="end"))


def plain_entry(svg: list[str], *, routed: bool, soft: bool = False) -> None:
    if soft:
        feedback_input(svg)
    token_input(svg)
    svg.append(line(380, 625, 498, 625))
    if routed:
        if soft:
            svg.append(line(215, 285, 498, 285, kind="orange"))
            svg.append(path([(430, 625), (430, 315), (498, 315)], kind="purple"))
            svg.append(text(285, 705, "plain entry  ·  h₀ = eₜ  ·  no GLU", css="small"))
        else:
            svg.append(path([(430, 625), (430, 290), (498, 290)], kind="purple"))
            svg.append(text(455, 470, "seed eₜ", css="small", anchor="end"))


def wiring_stack(
    svg: list[str], *, routed: bool, source_label: list[str] | None = None
) -> None:
    """Draw one block as an upper source rail and lower residual rail."""
    svg.append(text(500, 195, "Transformer block ℓ   × L", css="section", anchor="start"))
    rail_note = (
        "branch deltas update both rails"
        if routed
        else "branch deltas update the residual rail"
    )
    svg.append(text(1190, 195, rail_note, css="small", anchor="end"))

    if routed:
        svg.append(
            box(
                500,
                245,
                690,
                90,
                source_label or ["Sources Sℓ"],
                kind="purple-node",
                css="formula",
            )
        )
        svg.append(box(550, 365, 140, 65, ["route", "qᵃℓ"], kind="purple-node", css="formula"))
        svg.append(box(910, 365, 140, 65, ["route", "qᵐℓ"], kind="purple-node", css="formula"))
        svg.append(line(620, 335, 620, 363, kind="purple"))
        svg.append(line(980, 335, 980, 363, kind="purple"))
        svg.append(circle(620, 485))
        svg.append(circle(980, 485))
        svg.append(line(620, 430, 620, 464, kind="purple"))
        svg.append(line(980, 430, 980, 464, kind="purple"))
        svg.append(path([(620, 625), (620, 506)], kind="blue"))
        svg.append(path([(980, 625), (980, 506)], kind="blue"))
        svg.append(line(639, 485, 678, 485, kind="blue"))
        svg.append(line(999, 485, 1038, 485, kind="blue"))
        attn_x, mlp_x = 680, 1040
    else:
        svg.append(text(845, 290, "No source rail  ·  no routed reads", css="small"))
        attn_x, mlp_x = 550, 910
        svg.append(path([(620, 625), (620, 512)], kind="blue"))
        svg.append(path([(980, 625), (980, 512)], kind="blue"))

    svg.append(box(attn_x, 445, 140, 80, ["RMSNorm", "Attention"], kind="blue-node"))
    svg.append(box(mlp_x, 445, 140, 80, ["RMSNorm", "MLP"], kind="blue-node"))

    svg.append(line(500, 625, 799, 625))
    svg.append(circle(820, 625))
    svg.append(line(839, 625, 1149, 625))
    svg.append(circle(1170, 625))

    svg.append(path([(attn_x + 140, 485), (820, 485), (820, 604)], kind="blue"))
    svg.append(path([(mlp_x + 140, 485), (1170, 485), (1170, 604)], kind="blue"))
    svg.append(text(842, 460, "Δaℓ", css="formula", anchor="start"))
    svg.append(text(1192, 460, "Δmℓ", css="formula", anchor="start"))

    if routed:
        svg.append(path([(820, 485), (820, 337)], kind="purple"))
        svg.append(path([(1170, 485), (1170, 337)], kind="purple"))
        svg.append(text(735, 352, "deltas return to Sℓ", css="small"))

    svg.append(text(520, 650, "residual stream", css="small", anchor="start"))


def lm_outputs(svg: list[str], *, payload: str) -> None:
    svg.append(line(1189, 625, 1318, 625))
    svg.append(text(1240, 650, "hₜᵗᵒᵖ", css="formula"))
    svg.append(box(1320, 585, 160, 80, ["Final RMSNorm", "+ LM head"], kind="green-node"))
    svg.append(line(1480, 625, 1518, 625, kind="blue"))
    svg.append(box(1520, 590, 135, 70, ["Logits", "logitsₜ"], kind="green-node", css="formula"))

    if payload == "none":
        return

    svg.append(path([(1240, 625), (1240, 360), (1380, 360), (1380, 311)], kind="blue"))
    svg.append(text(1260, 410, "base hₜᵗᵒᵖ", css="small", anchor="start"))
    if payload == "fbt":
        svg.append(box(1210, 250, 140, 80, ["RMSNorm", "hₜᵗᵒᵖ"], kind="orange-node", css="formula"))
        svg.append(line(1350, 290, 1428, 290, kind="orange"))
    else:
        svg.append(line(1190, 290, 1208, 290, kind="purple"))
        svg.append(box(1210, 255, 140, 70, ["Payload route", "qₚ"], kind="purple-node", css="formula"))
        svg.append(line(1350, 290, 1360, 290, kind="purple"))
        svg.append(circle(1380, 290))
        svg.append(line(1399, 290, 1428, 290, kind="orange"))
    svg.append(box(1430, 250, 190, 80, ["Payload", "pₜ → step t+1"], kind="orange-node", css="formula"))


def vanilla() -> str:
    svg = start_svg(
        "Vanilla transformer",
        "Baseline · ordinary residual depth · no cross-column state",
        "Token embedding enters a standard pre-norm residual transformer stack and the top state feeds the language-model head. There is no depth router or recurrent payload.",
    )
    plain_entry(svg, routed=False)
    wiring_stack(svg, routed=False)
    lm_outputs(svg, payload="none")
    return finish_svg(svg, "Blue = ordinary residual computation. One column at position t is shown.")


def dar() -> str:
    svg = start_svg(
        "DAR · Delta Attention Residuals",
        "Vertical-axis widening · transient reads over the current column’s decomposition",
        "The embedding seeds a source bank. Before every attention and MLP sublayer, a zero-initialized query routes over the seed and prior sublayer deltas. The routed mixture enriches the read transiently while the residual stream remains clean.",
    )
    plain_entry(svg, routed=True)
    wiring_stack(
        svg,
        routed=True,
        source_label=["Sources Sℓ", "[eₜ  |  Δa₀, Δm₀, …]", "seed + prior deltas"],
    )
    lm_outputs(svg, payload="none")
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
    gated_entry(svg, routed=False)
    wiring_stack(svg, routed=False)
    lm_outputs(svg, payload="fbt")
    return finish_svg(svg, "Orange = cross-column feedback. The trunk has no depth routers; pₜ is the bare normalized top state.")


def df() -> str:
    svg = start_svg(
        "DF · Delta Feedback",
        "Hard combination · gated horizontal feedback + mandatory vertical delta routing",
        "The previous payload and token embedding are fused into u. The fused input seeds DAR-style routing throughout the stack. A dedicated router enriches the top state with a delta-only mixture before forming the next payload.",
    )
    gated_entry(svg, routed=True)
    wiring_stack(
        svg,
        routed=True,
        source_label=[
            "Sources Sℓ",
            "[uₜ  |  Δa₀, Δm₀, …]",
            "depth: all  ·  payload: drop uₜ",
        ],
    )
    lm_outputs(svg, payload="routed")
    return finish_svg(svg, "Hard-everywhere: neither the routed reads nor the recurrent payload has a null escape.")


def df_soft() -> str:
    svg = start_svg(
        "DF-soft · Optional Delta Feedback",
        "Soft combination · plain token entry + null-enabled routers everywhere",
        "The token embedding remains the residual-stream input. The previous payload is an ungated standing source beside the embedding and prior deltas. Every depth and payload router includes a learnable null source, so the model can regress to vanilla.",
    )
    plain_entry(svg, routed=True, soft=True)
    wiring_stack(
        svg,
        routed=True,
        source_label=[
            "Sources Sℓ",
            "[null  |  pₜ₋₁  |  eₜ  |  Δa₀, Δm₀, …]",
            "depth: all  ·  payload: drop pₜ₋₁ and eₜ",
        ],
    )
    lm_outputs(svg, payload="routed")
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
