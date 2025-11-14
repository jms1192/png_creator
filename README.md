# `render_pine_poster` – Unified Pine Chart Renderer

This module provides a single entrypoint, `render_pine_poster`, that generates
Pine-branded PNG posters using one of three chart types:

- **Bar** – ranked comparisons (e.g. “Top Wallets by PnL”)  
- **Pie** – share / composition (e.g. “DEX Volume Share”)  
- **Dual** – over-time charts with optional second Y-axis  
  (e.g. “eUSD Supply vs ENA Price – Last 90 Days”)

Under the hood it calls:

- `render_pine_poster_bar`   (from `group_png_creator.py`)
- `render_pine_poster_pie`   (from `pie_png_creator.py`)
- `render_pine_poster_dual`  (from `metrics_over_time_png_creator.py`)

All outputs are composited onto the Pine template image (default: `182.png`) and
saved as PNG files.

---

## Function Signature

```python
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
Shared Arguments (All Poster Types)
These apply regardless of poster_type.

poster_type (required)
Type: str

Allowed values:

"bar", "bar_chart", "horizontal_bar", "vertical_bar"

"pie", "pie_chart"

"dual", "overtime", "over_time", "line_chart", "time_series"

Behavior:

"pie" → calls render_pine_poster_pie

"bar" / "horizontal_bar" / "vertical_bar" → calls render_pine_poster_bar

"dual" / "time_series" etc. → calls render_pine_poster_dual

title
Type: str

Usage: Main chart title at the top of the poster.

subtitle (optional)
Type: str

Usage: Smaller text under the title (e.g. “Last 30 Days”).

note_value (optional)
Type: str

Usage: Footer note at the bottom (e.g. “Data: Pine Analytics”).

template_path (optional)
Type: str

Default: "182.png"

Usage: Path to the Pine poster background PNG template.

out_path (optional)
Type: str | None

Default behavior (if None):

"pie" → "pine_overlay_pie.png"

"bar" → "pine_overlay_bar.png"

"dual" → "pine_overlay_output_dual.png"

Usage: Output PNG file path.

date_str (optional)
Type: str | None

Default: current date formatted as "Month DD, YYYY".

Usage: Date shown in the footer. Override if you want a specific date.

colors_hex (optional; meaning depends on poster type)
Type: list[str] | None

Format: list of hex color strings, e.g. ["#1C5C3D", "#D97706", "#2563EB"]

Behavior:

For bar/pie: one color per bar/slice (length must equal len(values)).

For dual: one color per left-axis series (length must equal
the number of series in y_series).

If None, each underlying renderer falls back to automatic palettes.

Arguments for Bar Posters
Bar posters are selected with:

python
Copy code
poster_type in ("bar", "bar_chart", "horizontal_bar", "vertical_bar")
Required for Bar
labels
Type: list[str]

Usage: Category names for each bar (e.g. wallet IDs, DEX names).

Length must equal len(values).

values
Type: list[int | float]

Usage: Numeric value for each bar.

Length must equal len(labels).

Optional / Bar-specific
orientation
Type: str

Default: "horizontal"

Allowed:

"horizontal" → horizontal bars (categories on Y-axis)

"vertical" → vertical bars (categories on X-axis)

Note: If poster_type == "vertical_bar" or "horizontal_bar",
that overrides this argument.

value_axis_label
Type: str

Default: "Volume (USD)"

Usage: Label for numeric axis (e.g. "PnL (USD)", "Volume (USD)").

label_images
Type: list[str | None] | None

Usage:

Per-category image (URL or local path) to draw near each bar.

Length must equal len(labels) if provided.

Use None for entries with no image.

center_image
Type: str | None

Usage: Optional watermark / logo in the center of the chart region.
(URL or local path, depends on underlying bar renderer.)

Arguments for Pie Posters
Pie posters are selected with:

python
Copy code
poster_type in ("pie", "pie_chart")
Required for Pie
labels
Type: list[str]

Usage: Labels for each slice (e.g. DEX names).

Length must equal len(values).

values
Type: list[int | float]

Usage: Numeric values per slice (shares, volumes, percentages, etc.).

Length must equal len(labels).

Optional / Pie-specific
center_image
Type: str | None

Usage: Optional center image (e.g. a logo) cropped into a circle
and placed in the pie center. Can be:

Local path: "./logo.png"

URL: "https://.../logo.png"

When provided, some slice percentage labels may be disabled to avoid clutter.

Arguments for Dual / Over-Time Posters
Dual posters are selected with:

python
Copy code
poster_type in ("dual", "overtime", "over_time", "line_chart", "time_series")
These are over-time charts with:

Left Y-axis: one or more series (y_series)

Right Y-axis: optional second series (right_series)

X-axis: time or index (x_values)

Required for Dual
x_values
Type: list[str | datetime | any]

Usage:

If parsable as dates (datetime objects or ISO-like strings), the chart
treats them as dates and formats the X-axis as a time axis.

Otherwise, treated as numeric / categorical positions.

y_series
Type:

dict[str, Sequence[int | float]] (recommended)

or Sequence[Sequence[int | float]]

or Sequence[int | float] (single series)

Usage:

Left-axis data series.

All series must be the same length as x_values.

Examples:

python
Copy code
# Single series (dict form)
y_series = {"Volume (USD)": [100, 120, 90, 150]}

# Multi-series (dict form)
y_series = {
    "Volume (USD)": [100, 120, 90, 150],
    "Fees (USD)":   [5,   7,   4,  10],
}
Optional / Left-axis Options
ylabel_left
Type: str

Usage: Label for the left Y-axis (e.g. "Volume (USD)",
"eUSD Supply").

left_series_type
Type: str

Default: "line"

Allowed: "line", "bar", "area"

Usage: How to plot left series:

"line" – standard lines

"bar" – stacked bars

"area" – stacked area chart

log_left
Type: bool

Default: False

Usage: Use log scale on left Y-axis. All left series must be > 0.

include_zero_left
Type: bool

Default: True

Usage:

True → Y-axis will attempt to include 0.

False → scales more tightly around data range.

Optional / Right-axis Options
right_series
Type: Sequence[int | float] | None

Usage: Single series to plot on the right Y-axis (e.g. token price).

Length must equal len(x_values).

ylabel_right
Type: str

Usage: Label for the right Y-axis (e.g. "Price (USD)").

right_series_type
Type: str

Default: "line"

Allowed: "line", "bar", "area"

right_color_hex
Type: str

Default: "#8C3A3A" (deep red)

Usage: Color for the right-axis series.

log_right
Type: bool

Default: False

Usage: Log scale on right Y-axis (values must be > 0).

include_zero_right
Type: bool

Default: True

Usage: Same behavior as include_zero_left, but for right axis.

Optional / Highlighting
These are passed verbatim into render_pine_poster_dual and used for
visual annotations.

highlight_regions
Type: list[dict] | None

Each dict may contain:

"start": x-value (same type as x_values – date string, datetime, or numeric)

"end": x-value (same type as x_values)

"label": str (label placed near the band, optional)

Example:

python
Copy code
highlight_regions = [
    {
        "start": "2025-10-01",
        "end":   "2025-10-07",
        "label": "Launch Week",
    }
]
highlight_points
Type: list[dict] | None

Each dict may contain:

"x": x-value to highlight (same domain as x_values)

"series": series name or index (for left axis) or 0 for right axis

"axis": "left" or "right" (default "left")

"label": text to display near the point

Example:

python
Copy code
highlight_points = [
    {"x": "2025-11-01", "series": "eUSD Supply", "axis": "left", "label": "Major Mint"},
    {"x": "2025-11-10", "axis": "right", "label": "Price Spike"},
]
Example Usage
1. Bar – Top Wallets by PnL
python
Copy code
render_pine_poster(
    poster_type="bar",
    title="Top Wallets by PnL",
    subtitle="Last 30 Days",
    note_value="Data: Pine Analytics",
    labels=["W1", "W2", "W3"],
    values=[1_200_000, 950_000, 730_000],
    label_images=[None, None, None],
    orientation="vertical",
    value_axis_label="PnL (USD)",
    colors_hex=[
        "#1C5C3D",  # Pine green
        "#D97706",  # amber
        "#2563EB",  # blue
    ],
)
2. Pie – DEX Volume Share
python
Copy code
render_pine_poster(
    poster_type="pie",
    title="DEX Volume Share",
    subtitle="Top 5 DEXs – Last 30 Days",
    note_value="Data: Pine Analytics",
    labels=["Jupiter", "Raydium", "Orca", "Meteora", "Other"],
    values=[45, 25, 15, 10, 5],
    colors_hex=[
        "#1C5C3D",  # Jupiter
        "#D97706",  # Raydium
        "#2563EB",  # Orca
        "#6B7280",  # Meteora
        "#10B981",  # Other
    ],
)
3. Dual – eUSD Supply vs ENA Price (90 Days)
python
Copy code
render_pine_poster(
    poster_type="dual",
    title="eUSD Supply vs ENA Price",
    subtitle="Last 90 Days (Synthetic Example)",
    note_value="Data: Example synthetic series",

    x_values=dates,  # list of 90 date strings

    # Left axis: eUSD supply
    y_series={"eUSD Supply": eusd_supply},
    ylabel_left="eUSD Supply",
    left_series_type="line",
    include_zero_left=False,
    colors_hex=[
        "#1C5C3D",  # left axis series color (eUSD)
    ],

    # Right axis: ENA price
    right_series=ena_price,
    ylabel_right="ENA Price (USD)",
    right_series_type="bar",
    right_color_hex="#8C3A3A",
    include_zero_right=False,
)
Error Conditions
The function will raise ValueError in these common cases:

poster_type is not one of "pie", "bar", or "dual" (or synonyms).

labels and values not provided for "pie" or "bar".

x_values or y_series not provided for "dual".

Length mismatches:

len(labels) != len(values)

For dual: series lengths not equal to len(x_values).

colors_hex length not matching required number of series/slices/bars.
