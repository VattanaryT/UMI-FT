#!/usr/bin/env python3
"""δn, clearance, and |F·n| for the best release setting. Port 8812."""

import threading
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import viser

OUT = Path(__file__).resolve().parent / "out_lift_deploy"
blob = np.load(OUT / "best_trace.npz")
t = np.asarray(blob["t_s"])
name = str(blob["name"])
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=("δn mm", "clearance mm", "|F·n| N"),
)
fig.add_trace(go.Scatter(x=t, y=blob["delta_n_mm"], line={"width": 1}), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=blob["clearance_mm"], line={"width": 1}), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=blob["force_n"], line={"width": 1}), row=3, col=1)
fig.update_layout(height=780, showlegend=False, margin={"l": 40, "r": 10, "t": 30, "b": 30})
fig.update_xaxes(title_text="t (s)", row=3, col=1)

server = viser.ViserServer(host="0.0.0.0", port=8812)
server.gui.add_markdown(
    f"**{name}** episode 9. Retract 23.25–23.94 s (T_hold 0.5 s, travel 2 cm), then back in contact."
)
server.gui.add_plotly(fig, aspect=0.7)
print(f"viewer http://0.0.0.0:8812 {name}", flush=True)
threading.Event().wait()
