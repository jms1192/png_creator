this unified Pine poster function.

# 🟢 Pine Poster Renderer  
Unified function for generating Pine-branded PNG charts (Bar, Pie, Dual Over-Time)

This module provides a single entrypoint, `render_pine_poster`, that generates
high-quality, Pine-styled poster PNGs using one of three chart types:

- **Bar charts** – ranked comparisons (e.g., “Top Wallets by PnL”)
- **Pie charts** – share/composition (e.g., “DEX Volume Share”)
- **Dual-Axis Time Series** – over-time charts with optional right Y-axis  
  (e.g., “eUSD Supply vs ENA Price – Last 90 Days”)

The function wraps three specialized renderers:

- `render_pine_poster_bar` (from `group_png_creator.py`)
- `render_pine_poster_pie` (from `pie_png_creator.py`)
- `render_pine_poster_dual` (from `metrics_over_time_png_creator.py`)

All posters are composited onto a Pine template PNG (default: `182.png`).

---

## 📦 Installation

Ensure the following files exist in your project:



group_png_creator.py
pie_png_creator.py
metrics_over_time_png_creator.py
pine_poster.py
182.png # Poster layout background


Install required Python dependencies:

```bash
pip install pillow matplotlib numpy

🧠 render_pine_poster Function Overview
def render_pine_poster(
    poster_type,
    title,
    subtitle="",
    note_value="",
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

📝 Chart Types
1. Bar Charts

poster_type="bar"
or: "horizontal_bar", "vertical_bar"

Required
Argument	Type	Description
labels	list[str]	Bar labels (must match values length)
values	`list[int	float]`
Optional
Argument	Type	Description
orientation	"horizontal" or "vertical"	Direction of bar layout
value_axis_label	str	Label for numeric axis
label_images	list[str|None]	Per-bar avatar/image
colors_hex	list[str]	One hex color per bar
Example
render_pine_poster(
    poster_type="bar",
    title="Top Wallets by PnL",
    subtitle="Last 30 Days",
    note_value="Data: Pine Analytics",
    labels=["W1", "W2", "W3"],
    values=[1_200_000, 950_000, 730_000],
    orientation="vertical",
    value_axis_label="PnL (USD)",
    colors_hex=["#1C5C3D", "#D97706", "#2563EB"],
)

2. Pie Charts

poster_type="pie"

Required
Argument	Type	Description
labels	list[str]	Pie slice labels
values	`list[int	float]`
Optional
Argument	Type	Description
center_image	str	Path/URL for center logo
colors_hex	list[str]	One color per slice
Example
render_pine_poster(
    poster_type="pie",
    title="DEX Volume Share",
    subtitle="Top 5 DEXs – Last 30 Days",
    note_value="Data: Pine Analytics",
    labels=["Jupiter", "Raydium", "Orca", "Meteora", "Other"],
    values=[45, 25, 15, 10, 5],
    colors_hex=["#1C5C3D", "#D97706", "#2563EB", "#6B7280", "#10B981"],
)

3. Dual / Over-Time Charts

poster_type="dual"
or: "time_series", "overtime", "over_time"

Required
Argument	Type	Description
x_values	list[str/date]	X-axis (time or categories)
y_series	dict[str, list]	Left-axis series data
Optional Left-Axis Settings
Argument	Type	Description
ylabel_left	str	Left-axis label
left_series_type	"line", "bar", "area"	Chart style
colors_hex	list[str]	One color per left series
log_left	bool	Log scale
include_zero_left	bool	Force zero into Y-range
Optional Right-Axis Settings
Argument	Type	Description
right_series	list	Second axis series
right_series_type	"line", "bar", "area"	Style
right_color_hex	str	Hex color
ylabel_right	str	Right axis label
log_right	bool	Log scale
include_zero_right	bool	Include zero in range
Highlighting
Argument	Type	Description
highlight_regions	list[dict]	Shaded time windows
highlight_points	list[dict]	Annotated points
Example
render_pine_poster(
    poster_type="dual",
    title="eUSD Supply vs ENA Price",
    subtitle="Last 90 Days",
    note_value="Data: Pine Analytics",
    x_values=dates,
    y_series={"eUSD Supply": eusd_supply},
    ylabel_left="eUSD Supply",
    left_series_type="line",
    colors_hex=["#1C5C3D"],
    right_series=ena_price,
    right_series_type="bar",
    right_color_hex="#8C3A3A",
    ylabel_right="ENA Price (USD)",
    include_zero_left=False,
    include_zero_right=False,
)

🎨 Colors

Bar/Pie:
colors_hex=["#HEX", "#HEX", ...] must match the number of bars or slices.

Dual Chart:

Left axis → colors_hex

Right axis → right_color_hex

📁 Output

If out_path is not provided:

Bar → pine_overlay_bar.png

Pie → pine_overlay_pie.png

Dual → pine_overlay_output_dual.png

All charts are saved in PNG format.

⚠️ Error Handling

The function raises errors for:

Invalid poster_type

Missing required arguments (labels, values, x_values, y_series)

Length mismatches:

len(labels) != len(values)

Series length != len(x_values)

Incorrect color list lengths

🧩 Recommended Folder Structure
/your-project
  ├── pine_poster.py
  ├── group_png_creator.py
  ├── pie_png_creator.py
  ├── metrics_over_time_png_creator.py
  ├── 182.png                      # template background
  ├── README.md

🤝 Contributing

Pull requests welcome — especially for:

Additional chart types

New templates

More flexible layout controls

Automated data → chart pipelines
