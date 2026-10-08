"""A small geometry check for the two 2-by-2 supplemental Matplotlib figures.

The check measures rendered axes and visible panel letters in physical points.
It intentionally does not create an overlay SVG. The JSON report contains the
measured rectangles and checks, so it can be inspected without another render.
This is a layout check, not a text collision or scientific-content audit.
"""
# SPDX-License-Identifier: MIT

from pathlib import Path
import json
import math


class PanelAlignmentError(RuntimeError):
    """The rendered four-panel layout failed its geometry checks."""


def require_matplotlib_panel_alignment(
    fig,
    *,
    json_out=None,
    overlay_svg=None,
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    require_panel_labels=False,
    strict=False,
    **options,
):
    """Return a measured 2-by-2 layout report and raise on failed checks.

    The arguments match the S31/S32 plotting calls. ``strict`` is accepted for
    compatibility; this checker reports only passes and failures, so all failed
    checks block delivery with either value. Extra layout options are rejected.
    ``overlay_svg`` is accepted but no SVG is written, as explained in the report.
    """
    if options:
        raise TypeError(f"Unsupported geometry options: {', '.join(options)}")
    for name, value in (("tolerance_pt", tolerance_pt),
                        ("gutter_tolerance_pt", gutter_tolerance_pt)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = 72.0 / fig.dpi
    axes = [ax for ax in fig.axes if ax.get_visible()]
    checks = []
    panels = []

    def check(name, error, tolerance, detail=None):
        ok = math.isfinite(error) and error <= tolerance
        item = {"name": name, "pass": bool(ok),
                "error_pt": float(error), "tolerance_pt": float(tolerance)}
        if detail is not None:
            item["detail"] = detail
        checks.append(item)

    if len(axes) != 4:
        checks.append({"name": "four_visible_axes", "pass": False,
                       "detail": f"Expected 4 visible axes, received {len(axes)}"})
    else:
        entries = []
        for ax in axes:
            box = ax.get_window_extent(renderer)
            entries.append((ax, [float(v * scale) for v in box.extents]))
        entries.sort(key=lambda entry: -(entry[1][1] + entry[1][3]))
        ordered = sorted(entries[:2], key=lambda entry: entry[1][0])
        ordered += sorted(entries[2:], key=lambda entry: entry[1][0])
        for letter, (ax, box) in zip("abcd", ordered):
            panel = {"id": letter, "bbox_pt": box, "axes_label": ax.get_label()}
            if require_panel_labels:
                candidates = [text for text in ax.texts
                              if text.get_visible() and text.get_text().strip() == letter]
                valid = [text for text in candidates
                         if text.get_window_extent(renderer).width > 0
                         and text.get_window_extent(renderer).height > 0]
                label_ok = len(valid) == 1 and ax.get_label() == letter
                checks.append({"name": f"panel_{letter}_visible_label",
                               "pass": bool(label_ok),
                               "detail": "One rendered matching letter and axes label required"})
                if valid:
                    panel["label_bbox_pt"] = [float(v * scale) for v in
                                               valid[0].get_window_extent(renderer).extents]
            panels.append(panel)

        boxes = [panel["bbox_pt"] for panel in panels]
        # Rendered rectangles are [left, bottom, right, top], in points.
        for col in (0, 1):
            top, bottom = boxes[col], boxes[col + 2]
            for side in (0, 2):
                check(f"column_{col + 1}_edge_{side}",
                      abs(top[side] - bottom[side]), tolerance_pt)
        for row in (0, 1):
            left, right = boxes[row * 2], boxes[row * 2 + 1]
            for side in (1, 3):
                check(f"row_{row + 1}_edge_{side}",
                      abs(left[side] - right[side]), tolerance_pt)
        widths = [box[2] - box[0] for box in boxes]
        heights = [box[3] - box[1] for box in boxes]
        check("equal_panel_widths", max(widths) - min(widths), tolerance_pt)
        check("equal_panel_heights", max(heights) - min(heights), tolerance_pt)

        horizontal = [boxes[1][0] - boxes[0][2], boxes[3][0] - boxes[2][2]]
        vertical = [boxes[0][1] - boxes[2][3], boxes[1][1] - boxes[3][3]]
        for direction, gaps in (("horizontal", horizontal), ("vertical", vertical)):
            checks.append({"name": f"positive_{direction}_gutters",
                           "pass": bool(all(gap > 0 for gap in gaps)),
                           "measured_pt": gaps})
            check(f"equal_{direction}_gutters", abs(gaps[0] - gaps[1]),
                  gutter_tolerance_pt)

    failures = [item for item in checks if not item["pass"]]
    report = {
        "schema_version": 1,
        "checker": "event-local-reliability rendered 2x2 geometry",
        "status": "FAIL" if failures else "PASS",
        "figure_pt": [float(value * 72) for value in fig.get_size_inches()],
        "panels": panels,
        "checks": checks,
        "overlay_svg": {
            "requested": overlay_svg is not None,
            "written": False,
            "reason": "JSON records measured rectangles; no auxiliary overlay rendering is needed.",
        },
    }
    if json_out is not None:
        target = Path(json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise PanelAlignmentError("Failed layout checks: " +
                                  ", ".join(item["name"] for item in failures))
    return report
