Here is a single clean README block you can copy-paste directly into GitHub — no extra formatting needed.

# Pine Poster Renderer

`render_pine_poster` is a unified function for generating Pine-branded PNG charts.  
It supports **bar charts**, **pie charts**, and **dual-axis time-series charts** using a single API.

All outputs render onto a poster template (default: `182.png`) and produce share-ready graphics for dashboards, research, and social posts.

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

Chart Types & Required Inputs
1. Bar Chart (poster_type="bar")

Required

labels: list of category names

values: list of numbers (same length as labels)

Optional

orientation: "horizontal" or "vertical"

value_axis_label: axis name

label_images: per-label avatar/PNG

colors_hex: list of hex colors (one per bar)

Example

render_pine_poster(
    poster_type="bar",
    title="Top Wallets by PnL",
    labels=["W1", "W2", "W3"],
    values=[1_200_000, 950_000, 730_000],
    orientation="vertical",
    colors_hex=["#1C5C3D", "#D97706", "#2563EB"],
)

2. Pie Chart (poster_type="pie")

Required

labels

values

Optional

colors_hex

center_image (logo in the center)

Example

render_pine_poster(
    poster_type="pie",
    title="DEX Volume Share",
    labels=["Jupiter", "Raydium", "Orca", "Meteora", "Other"],
    values=[45, 25, 15, 10, 5],
)

3. Dual / Time-Series Chart (poster_type="dual")

Required

x_values: list of timestamps or categories

y_series: dict of {name: list_of_values} for the left axis

Optional

left axis settings: ylabel_left, log_left, include_zero_left, left_series_type

right axis y-series: right_series, right_series_type, right_color_hex, ylabel_right

optional highlighting: highlight_regions, highlight_points

colors:

left axis series → colors_hex

right axis → right_color_hex

Example

render_pine_poster(
    poster_type="dual",
    title="eUSD Supply vs ENA Price",
    x_values=dates,
    y_series={"eUSD Supply": eusd_supply},
    colors_hex=["#1C5C3D"],
    ylabel_left="eUSD Supply",
    right_series=ena_price,
    right_series_type="bar",
    right_color_hex="#8C3A3A",
    ylabel_right="ENA Price (USD)",
)

Output

If out_path is not provided:

Bar → pine_overlay_bar.png

Pie → pine_overlay_pie.png

Dual → pine_overlay_output_dual.png

Required Files

Your project must include:

pine_poster.py
group_png_creator.py
pie_png_creator.py
metrics_over_time_png_creator.py
182.png   # template background

Dependencies
pip install pillow matplotlib numpy


This README fully documents the function and its inputs so anyone can generate Pine-style charts from your code.
