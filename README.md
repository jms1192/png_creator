# Pine Poster Renderer

`render_pine_poster` is a unified function for generating Pine-branded PNG charts.  
It supports **bar charts**, **pie charts**, and **dual-axis time-series charts** using a single API.

All outputs render onto a poster template (default: `182.png`) and produce share-ready graphics for dashboards, research, and social content.

---

## Function Overview

```python
render_pine_poster(
    poster_type,            # "bar", "pie", or "dual"
    title,                  # main title text
    subtitle="",            # optional subtitle
    note_value="",          # footer note
    template_path="182.png",
    out_path=None,
    date_str=None,
    colors_hex=None,

    # pie + bar
    labels=None,
    values=None,
    center_image=None,

    # bar-only
    value_axis_label="Volume (USD)",
    label_images=None,
    orientation="horizontal",

    # dual-only
    x_values=None,
    y_series=None,
    ylabel_left="",
    log_left=False,
    include_zero_left=True,
    left_series_type="line",

    right_series=None,
    right_color_hex="#8C3A3A",
    ylabel_right="",
    right_series_type="line",
    log_right=False,
    include_zero_right=True,
    highlight_regions=None,
    highlight_points=None,
)

```

2. Pie Chart (poster_type="pie")

Required

labels

values

Optional

colors_hex

center_image (logo in center)

Example

render_pine_poster(
    poster_type="pie",
    title="DEX Volume Share",
    labels=["Jupiter", "Raydium", "Orca", "Meteora", "Other"],
    values=[45, 25, 15, 10, 5],
)
