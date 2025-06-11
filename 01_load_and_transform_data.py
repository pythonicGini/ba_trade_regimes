import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Threshold values are defined for categorizing countries based on democratic trends
THRESHOLD_BACKSLIDING = 0.5
THRESHOLD_STEADY = 0.5
MIN_CHANGE = 0.1
START_YEAR = 2000

# List of countries to be excluded from analysis
IGNORE_COUNTRIES = [
    "TWN", "ARG", "ARM", "BGD", "CIV", "FJI", "GEO", "GNB", "GUY", "KEN", "LBY", "LKA", "LSO", "MDA", "MDG", "MEX", "MKD", "MLI", "MNG", "NER", "NPL", "PER", "PSE", "ROU", "SEN", "SLE", "SRB", "SVK", "THA", "TUN", "UKR", "VEN", "ZMB"
]

# Main procedure for processing data and generating visual output
def main() -> None:
    # Dataset is loaded and filtered for years starting from the specified threshold
    df_ert_raw = pd.read_csv("0_raw_data/ert.csv", encoding="UTF-8")
    df_ert_raw = df_ert_raw[df_ert_raw["year"] >= START_YEAR]

    # Relevant countries are identified and separated based on democracy trends
    relevant_countries = get_relevant_countries(df_ert_raw)

    # Visualization is created based on continent and country status
    plot_by_continents(df_ert_raw, relevant_countries)
    return

# Function to determine and return relevant countries categorized as "steady" or "backsliding"
def get_relevant_countries(df_ert: pd.DataFrame) -> dict:
    valid_countries = {
        "backsliding": [],
        "steady": []
    }

    # Each country in the dataset is evaluated
    for country in df_ert["country_text_id"].unique():
        if country in IGNORE_COUNTRIES:
            continue

        # Dataset is filtered for the current country
        df_country = df_ert[df_ert["country_text_id"] == country].reset_index()
        max_regime = df_country["v2x_polyarchy"].max()
        min_regime = df_country["v2x_polyarchy"].min()

        # Countries with minimal regime change and meeting a democratic threshold are classified as steady
        if (max_regime - min_regime <= MIN_CHANGE) and (min_regime >= THRESHOLD_STEADY):
            valid_countries["steady"].append(country)
            continue

        # Countries showing significant democratic decline are classified as backsliding
        if is_backsliding(df_country):
            valid_countries["backsliding"].append(country)

    # Alphabetical sorting of countries in both categories
    valid_countries["steady"].sort()
    valid_countries["backsliding"].sort()

    # Result is saved to a JSON file
    with open("00_steady_and_backsliding_countries.json", "w") as outfile:
        json.dump(valid_countries, outfile, indent=4)

    return valid_countries

# Function to assess whether a country exhibits democratic backsliding
def is_backsliding(df: pd.DataFrame) -> bool:
    for i, val in enumerate(df['v2x_polyarchy']):
        if val > THRESHOLD_BACKSLIDING:
            # Any subsequent year with a drop exceeding the minimum change is checked
            threshold = val - MIN_CHANGE
            if (df['v2x_polyarchy'].iloc[i + 1:] <= threshold).any():
                return True
    return False

# Function to create an interactive plot grouped by continents and democracy trends
def plot_by_continents(df_ert: pd.DataFrame, relevant_countries: dict) -> None:
    # Mapping of countries to their classification
    i_relevant_countries = invert_list_dict(relevant_countries)

    # Dictionary linking country codes to continents is retrieved
    cont_dict = get_continent_dict()
    unique_conts = set(cont_dict.values())
    traces = []

    # Plot trace is created for each country
    for country in i_relevant_countries.keys():
        country_df = df_ert[df_ert["country_text_id"] == country]
        trace = go.Scatter(
            x=country_df['year'],
            y=country_df['v2x_polyarchy'],
            mode='lines+markers',
            name=f"{country} ({i_relevant_countries[country][:1]})",
            text=f"{country_df['country_name'].iloc[0]} ({country})",
            hoverinfo="x+y+text",
            meta={"Cont": cont_dict[country], "Status": i_relevant_countries[country]},
        )
        traces.append(trace)

    buttons = []

    # Button to display all countries
    buttons.append(dict(label="All Countries",
                        method="update",
                        args=[{"visible": [True] * len(traces)},
                              {"title": {"text": "Democracy Index for all relevant Country"}}]))

    # Buttons to filter data by continent
    for cont in unique_conts:
        visible = []
        for trace in traces:
            if trace.meta["Cont"] == cont:
                visible.append(True)
            else:
                visible.append(False)

        buttons.append(dict(label=f"{cont}",
                            method="update",
                            args=[{"visible": visible},
                                  {"title": {"text": f"Democracy Index for Continent {cont}"}}]))

    # Buttons to filter data by democracy status
    for status in ["backsliding", "steady"]:
        visible = []
        for trace in traces:
            if trace.meta["Status"] == status:
                visible.append(True)
            else:
                visible.append(False)

        buttons.append(dict(label=f"{status} countries",
                            method="update",
                            args=[{"visible": visible},
                                  {"title": {"text": f"Democracy Index for {status} countries"}}]))

    # Figure object is created with all traces
    fig = go.Figure(data=traces)

    # A reference line for the backsliding threshold is added
    fig.add_shape(
        type="line",
        x0=df_ert['year'].min(), x1=df_ert['year'].max(),
        y0=THRESHOLD_BACKSLIDING, y1=THRESHOLD_BACKSLIDING,
        line=dict(color="Red", width=2, dash="dash"),
    )

    # Layout of the visualization is updated with interactive menu
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

    # Display of the final figure
    fig.write_html("1_plots/plot_with_filters.html")

    return

# Function to invert dictionary structure from key->list to item->key
def invert_list_dict(relevant_countries: dict) -> dict:
    inverted_countries = {}
    for key in relevant_countries.keys():
        for value in relevant_countries[key]:
            inverted_countries[value] = key
    return inverted_countries

# Function to load continent mapping and return an inverted dictionary
def get_continent_dict() -> dict:
    with open("0_iso3_by_continent.json", "r") as infile:
        json_data = json.load(infile)
    return invert_list_dict(json_data)

# Execution of the main function when the script is run directly
if __name__ == "__main__":
    main()
