import json

import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.graph_objects as go

# Thresholds and constants for filtering and analysis
threshold_ert_changed = 0.5
threshold_ert_steady = 0.5
min_change_ert = 0.1
start_year = 2000

# Predefined country groupings for plotting
if os.path.exists("0_chosen_countries.json"):
    with open("0_chosen_countries.json", "r") as f:
        choices = json.load(f)
else:
    choices = {}

# Folder where plots will be saved
plot_folder = f"./1_plots/"

# Mapping of continents to their ISO3 country codes
with open("0_iso3_by_continent.json", "r") as f:
    iso3_by_continent = json.load(f)


def load_data(ert_path: str, polity_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads ERT and Polity datasets from CSV files.

    Args:
        ert_path (str): Path to the ERT dataset CSV file.
        polity_path (str): Path to the Polity dataset CSV file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Loaded ERT and Polity DataFrames.
    """
    ert = pd.read_csv(ert_path, encoding="utf-8")
    polity = pd.read_csv(polity_path, encoding="utf-8")
    return ert, polity


def prefilter_ert(ert: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters ERT data to identify countries with democratic backsliding or stable democracy.

    Args:
        ert (pd.DataFrame): ERT dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames of backsliding and steady countries.
    """
    global threshold_ert_changed, threshold_ert_steady, min_change_ert, start_year

    # Filter to include only data from the specified start year
    ert = ert.loc[ert["year"] >= start_year]

    changed_countries = []
    steady_countries = []

    for country in ert["country_text_id"].unique():
        country_df = ert.loc[ert["country_text_id"] == country]
        max_v2x = country_df["v2x_polyarchy"].max()
        min_v2x = country_df["v2x_polyarchy"].min()

        # Skip countries that never surpassed the "changed" threshold
        if max_v2x <= threshold_ert_changed:
            continue

        # Classify as backsliding if sufficient change and negative trend
        if abs(max_v2x - min_v2x) >= min_change_ert:
            if calc_trend(country_df) < 0:
                changed_countries.append(country)
            continue

        # Classify as steady if values consistently exceed "steady" threshold
        elif min_v2x >= threshold_ert_steady:
            steady_countries.append(country)

    changed = ert.loc[ert["country_text_id"].isin(changed_countries)]
    steady = ert.loc[ert["country_text_id"].isin(steady_countries)]
    return changed, steady


def calc_trend(country: pd.DataFrame) -> float:
    """
    Calculates the linear trend (slope) of the v2x_polyarchy index over time.

    Args:
        country (pd.DataFrame): Subset of ERT data for a single country.

    Returns:
        float: Slope of the linear regression fit.
    """
    increase, _ = np.polyfit(country['year'], country['v2x_polyarchy'], 1)
    return increase


def plot_scatter_by_continent(backsliding: pd.DataFrame, steady: pd.DataFrame):
    """
    Plots interactive scatter plots for countries by continent and democracy trend.

    Args:
        backsliding (pd.DataFrame): Countries with negative democracy trends.
        steady (pd.DataFrame): Countries with stable democracy indices.
    """
    global choices, iso3_by_continent

    backsliding['type'] = 'backsliding'
    steady['type'] = 'steady'
    df_all = pd.concat([backsliding, steady], ignore_index=True)

    # Map countries to their continent
    country_to_continent = {}
    for continent, countries in iso3_by_continent.items():
        for country in countries:
            country_to_continent[country] = continent

    df_all['continent'] = df_all['country_text_id'].map(country_to_continent)
    df_all = df_all.dropna(subset=['continent'])

    traces = []
    visibility_lookup = []

    # Create traces per country and type
    for _, group in df_all.groupby(['country_text_id', 'type']):
        trace = go.Scatter(
            x=group['year'],
            y=group['v2x_polyarchy'],
            mode='lines+markers',
            name=f"{group['country_text_id'].iloc[0]} ({group['type'].iloc[0]})"
        )
        traces.append(trace)
        visibility_lookup.append((group['country_text_id'].iloc[0], group['type'].iloc[0]))

    # Setup filter buttons
    buttons = []
    buttons.append(dict(label="All Countries",
                        method="update",
                        args=[{"visible": [True] * len(traces)},
                              {"title": {"text": "Democracy Index by Country"}}]))

    for label, countries in choices.items():
        visible = [(country in countries) for country, _ in visibility_lookup]
        buttons.append(dict(label=label,
                            method="update",
                            args=[{"visible": visible},
                                  {"title": {"text": f"Democracy Index by Country: {label}"}}]))

    for continent, countries in iso3_by_continent.items():
        visible = [(country in countries) for country, _ in visibility_lookup]
        buttons.append(dict(label=continent,
                            method="update",
                            args=[{"visible": visible},
                                  {"title": {"text": f"Democracy Index by Country: {continent}"}}]))

    fig = go.Figure(data=traces)

    # Add threshold line
    fig.add_shape(
        type="line",
        x0=backsliding['year'].min(), x1=backsliding['year'].max(),
        y0=threshold_ert_changed, y1=threshold_ert_changed,
        line=dict(color="Red", width=2, dash="dash"),
    )

    fig.update_layout(
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.5,
            "xanchor": "center",
            "y": 1.15,
            "yanchor": "top",
            "pad": {"r": 10, "t": 10},
            "bgcolor": "lightgray",
            "bordercolor": "black",
            "borderwidth": 1,
            "font": {"size": 14, "color": "black"}
        }],
        title="Democracy Index by Country",
        xaxis_title="Year",
        yaxis_title="v2x_polyarchy",
        yaxis={"range": [0, 1]},
        showlegend=True
    )

    fig.write_html(f"{plot_folder}/plot_with_filters.html")


def main() -> None:
    """
    Main function to run the data loading, filtering, and plotting pipeline.
    """
    ert_df, polity_df = load_data("0_raw_data/ert.csv", "0_raw_data/polity_v.csv")
    changed_df, steady_df = prefilter_ert(ert_df)
    plot_scatter_by_continent(changed_df, steady_df)


if __name__ == "__main__":
    main()
