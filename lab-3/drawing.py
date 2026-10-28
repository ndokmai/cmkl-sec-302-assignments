"""A local canvas widget: draw a reusable alpha mask over a digit."""
from pathlib import Path

import anywidget
import traitlets


class TriggerCanvas(anywidget.AnyWidget):
    _esm = (Path(__file__).with_name("drawing.js")).read_text()
    _css = """
    .trigger-editor { font: 14px system-ui, sans-serif; color: #172033;
        background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 12px;
        padding: 16px; max-width: 620px; }
    .trigger-editor .tools { display: flex; flex-wrap: wrap; align-items: center;
        gap: 10px 16px; margin-bottom: 12px; }
    .trigger-editor label { display: flex; align-items: center; gap: 6px; }
    .trigger-editor button, .trigger-editor select { background: white;
        border: 1px solid #94a3b8; border-radius: 6px; padding: 5px 9px; color: #172033; }
    .trigger-editor button[aria-pressed="true"] { background: #dbeafe; border-color: #2563eb; }
    .trigger-editor input[type="range"] { width: 105px; accent-color: #2563eb; }
    .trigger-editor canvas { display: block; width: min(336px, 100%); height: auto;
        background: white; border: 1px solid #94a3b8; touch-action: none;
        cursor: crosshair; image-rendering: pixelated; }
    .trigger-editor .status { margin: 8px 0 0; font-size: 12px; color: #475569; }
    """
    images = traitlets.List().tag(sync=True)
    mask = traitlets.List(default_value=[[0.0] * 28 for _ in range(28)]).tag(sync=True)
    opacity = traitlets.Float(.45, min=0, max=1).tag(sync=True)
    digit = traitlets.Int(0, min=0, max=9).tag(sync=True)
