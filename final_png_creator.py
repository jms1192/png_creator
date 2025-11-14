# pine_poster.py (unified entry point)

from group_png_creator import render_pine_poster_bar
from metrics_over_time_png_creator import render_pine_poster_dual
from pie_png_creator import render_pine_poster_pie
from datetime import datetime, timedelta
import math


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
):
    """
    Unified Pine poster entrypoint.

    poster_type:
        - "pie"  -> render_pine_poster_pie(...)
        - "bar"  -> render_pine_poster_bar(...)
        - "dual" -> render_pine_poster_dual(...)
    """
    pt = str(poster_type).lower().strip()

    # ---------- PIE ----------
    if pt in ("pie", "pie_chart"):
        if labels is None or values is None:
            raise ValueError("For poster_type='pie', provide labels and values.")
        final_out = out_path or "pine_overlay_pie.png"
        return render_pine_poster_pie(
            title=title,
            subtitle=subtitle,
            note_value=note_value,
            labels=labels,
            values=values,
            colors_hex=colors_hex,
            template_path=template_path,
            out_path=final_out,
            date_str=date_str,
            center_image=center_image,
        )

    # ---------- BAR ----------
    elif pt in ("bar", "bar_chart", "horizontal_bar", "vertical_bar"):
        if labels is None or values is None:
            raise ValueError("For poster_type='bar', provide labels and values.")

        bar_orientation = orientation
        if pt == "vertical_bar":
            bar_orientation = "vertical"
        elif pt == "horizontal_bar":
            bar_orientation = "horizontal"

        final_out = out_path or "pine_overlay_bar.png"
        return render_pine_poster_bar(
            title=title,
            subtitle=subtitle,
            note_value=note_value,
            labels=labels,
            values=values,
            colors_hex=colors_hex,
            template_path=template_path,
            out_path=final_out,
            date_str=date_str,
            center_image=center_image,
            value_axis_label=value_axis_label,
            label_images=label_images,
            orientation=bar_orientation,
        )

    # ---------- DUAL / OVER-TIME ----------
    elif pt in ("dual", "overtime", "over_time", "line_chart", "time_series"):
        if x_values is None or y_series is None:
            raise ValueError("For poster_type='dual', provide x_values and y_series.")
        final_out = out_path or "pine_overlay_output_dual.png"
        return render_pine_poster_dual(
            title=title,
            subtitle=subtitle,
            note_value=note_value,
            x_values=x_values,
            y_series=y_series,
            colors_hex=colors_hex,
            ylabel_left=ylabel_left,
            log_left=log_left,
            include_zero_left=include_zero_left,
            chart_type=left_series_type,
            right_series=right_series,
            right_color_hex=right_color_hex,
            ylabel_right=ylabel_right,
            right_chart_type=right_series_type,
            log_right=log_right,
            include_zero_right=include_zero_right,
            template_path=template_path,
            out_path=final_out,
            highlight_regions=highlight_regions,
            highlight_points=highlight_points,
            date_str=date_str,
        )

    else:
        raise ValueError("poster_type must be one of: 'pie', 'bar', 'dual'")


# =========================
# Example usages with colors
# =========================

# 1) Bar chart – Top Wallets by PnL
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

# 2) Pie chart – DEX Volume Share
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

# 3) Dual over-time – eUSD Supply vs ENA Price (90 days)

# ---- 90 Days of Dates ----
today = datetime.today()
dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(90)][::-1]

# ---- More realistic eUSD supply (left axis) ----
eusd_supply = []
supply = 25_000_000

for i in range(90):
    if i < 30:            # ramp-up phase
        daily_delta = 180_000
    elif i < 60:          # consolidation / slower growth
        daily_delta = 70_000
    else:                 # renewed growth after catalysts
        daily_delta = 220_000

    cyclical = 40_000 * math.sin(i / 7)  # weekly-ish noise

    if i in (17, 38, 59, 73):            # occasional net-outflow days
        shock = -350_000
    else:
        shock = 0

    supply = max(0, supply + daily_delta + cyclical + shock)
    eusd_supply.append(supply)

# ---- More realistic ENA price (right axis) ----
ena_price = []
price = 0.55

for i in range(90):
    if i < 25:
        drift = 0.0025      # grind up
    elif i < 50:
        drift = -0.0015     # mid-cycle dump
    elif i < 75:
        drift = 0.0030      # renewed strength
    else:
        drift = 0.0005      # flatten out

    vol_wave = 0.01 * math.sin(i / 3.5) + 0.005 * math.cos(i / 9.0)

    if i in (26, 27):
        shock = -0.04       # selloff
    elif i in (55, 56):
        shock = 0.05        # pump
    else:
        shock = 0.0

    price = max(0.12, price + drift + vol_wave + shock)
    ena_price.append(round(price, 4))

# ---- Render dual chart with explicit colors ----
render_pine_poster(
    poster_type="dual",
    title="eUSD Supply vs ENA Price",
    subtitle="Last 90 Days (Synthetic Example)",
    note_value="Data: Example synthetic series",
    x_values=dates,
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
    right_series_type="line",
    right_color_hex="#8C3A3A",  # explicit right-side color
    include_zero_right=False,
)
