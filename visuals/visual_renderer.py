"""
visuals/visual_renderer.py
----------------------------
The Visual Renderer (Phase 4).

Takes an already-validated visual dict from visual_selector.select_visual()
and draws it. This module never receives or executes model-generated code,
HTML, JavaScript, or raw plotting instructions - only plain data (numbers,
short strings, short lists) that was already checked in visual_selector.py.
Every rendering path below is a fixed, hardcoded call WE wrote; the model's
only influence is which plain data values go into it.

render_visual() never raises: if a visual is missing, malformed, or fails
to draw for any reason, it fails silently (or with a small caption) so the
teaching loop is never interrupted by a broken visual.
"""

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; Streamlit's own server has no display
import matplotlib.pyplot as plt
import streamlit as st


def render_visual(visual: dict):
    """Render one visual dict, or do nothing if it's empty/invalid."""
    if not visual:
        return

    visual_type = visual.get("visual_type")
    data = visual.get("visual_data") or {}
    title = visual.get("title")

    renderers = {
        "equation": _render_equation,
        "graph": _render_graph,
        "table": _render_table,
        "code": _render_code,
        "diagram": _render_diagram,
        "image": _render_image_placeholder,
    }
    renderer = renderers.get(visual_type)
    if renderer is None:
        return

    try:
        if title:
            st.markdown(f"**{title}**")
        renderer(data)

        explanation = visual.get("visual_explanation")
        if explanation:
            st.caption(f"Visual explanation: {explanation}")
    except Exception:
        # A broken visual should never take down the lesson.
        return


def _render_equation(data: dict):
    equation = data.get("equation")
    if not equation:
        return
    st.latex(equation)


def _render_graph(data: dict):
    points = data.get("points") or []
    if len(points) < 2:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(data.get("x_label", "x"))
    ax.set_ylabel(data.get("y_label", "y"))
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_table(data: dict):
    columns = data.get("columns") or []
    rows = data.get("rows") or []
    if not columns or not rows:
        return
    table_rows = [dict(zip(columns, row)) for row in rows]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def _render_code(data: dict):
    code = data.get("code")
    if not code:
        return
    st.code(code, language=data.get("language") or "text")


def _render_diagram(data: dict):
    """
    Controlled, dependency-free diagram: nodes as a row of boxes (Streamlit
    columns), connections listed underneath as simple "A -> B" text. No
    external diagramming library or system binary required - deliberately
    simple per the Phase 4 spec ("if a diagram renderer becomes
    unnecessarily complicated, use a simple Streamlit-compatible
    representation"). Every value here is already-validated id/label text,
    never model-generated markup.
    """
    nodes = data.get("nodes") or []
    connections = data.get("connections") or []
    if not nodes:
        return

    cols = st.columns(len(nodes))
    id_to_label = {}
    for col, node in zip(cols, nodes):
        id_to_label[node["id"]] = node["label"]
        with col:
            st.info(node["label"])

    if connections:
        flow_text = "  |  ".join(
            f"{id_to_label.get(a, a)} → {id_to_label.get(b, b)}" for a, b in connections
        )
        st.caption(flow_text)


def _render_image_placeholder(data: dict):
    """
    Image generation/retrieval isn't wired up in this build. Per spec this
    must never block the lesson - show an honest placeholder instead of a
    broken image or an error.
    """
    query = data.get("query")
    st.caption(f"🖼️ An image would help here ({query}), but image generation isn't configured yet.")
