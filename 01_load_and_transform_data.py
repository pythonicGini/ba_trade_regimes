# These lines bring in tools to help work with data, numbers, and graphs
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# These are values used later to make decisions about what kind of countries we're dealing with
THRESHOLD_BACKSLIDING = 0.5  # If democracy falls below this after being higher, we say it's "backsliding"
THRESHOLD_STEADY = 0.5  # A steady country should be at or above this level
MIN_CHANGE = 0.1  # We care about changes bigger than this number
START_YEAR = 2000  # We only want to look at data from this year onward

# This is a list of countries we don’t want to include in our analysis
IGNORE_COUNTRIES = [
    "TWN"
]


# This is where the program starts running
def main() -> None:
    # Read in a file containing raw data about democracy scores
    df_ert_raw = pd.read_csv("0_raw_data/ert.csv", encoding="UTF-8")

    # Keep only the data from the year 2000 and later
    df_ert_raw = df_ert_raw[df_ert_raw["year"] >= START_YEAR]

    # Find out which countries are steady or backsliding
    relevant_countries = get_relevant_countries(df_ert_raw)

    # Make a graph to show how democracy is changing, split by continent
    plot_by_continents(df_ert_raw, relevant_countries)
    return


# This function figures out which countries are worth focusing on
def get_relevant_countries(df_ert: pd.DataFrame) -> dict:
    # Prepare two empty lists to hold countries that are steady or backsliding
    valid_countries = {
        "backsliding": [],
        "steady": []
    }

    # Go through each country in the data
    for country in df_ert["country_text_id"].unique():
        if country in IGNORE_COUNTRIES:
            continue  # Skip countries we want to ignore

        # Get all the data for one country
        df_country = df_ert[df_ert["country_text_id"] == country].reset_index()

        # Find the highest and lowest democracy score for this country
        max_regime = df_country["v2x_polyarchy"].max()
        min_regime = df_country["v2x_polyarchy"].min()

        # If the country's democracy level didn’t change much and was generally high, mark it as "steady"
        if (max_regime - min_regime <= MIN_CHANGE) and (min_regime >= THRESHOLD_STEADY):
            valid_countries["steady"].append(country)
            continue

        # If the country's democracy score went down significantly, mark it as "backsliding"
        if is_backsliding(df_country):
            valid_countries["backsliding"].append(country)

    # Sort the countries alphabetically
    valid_countries["steady"].sort()
    valid_countries["backsliding"].sort()

    # Save the results into a file so they can be used again later
    with open("00_steady_and_backsliding_countries.json", "w") as outfile:
        json.dump(valid_countries, outfile, indent=4)
    return valid_countries


# This checks if a country's democracy score got worse over time
def is_backsliding(df: pd.DataFrame) -> bool:
    for i, val in enumerate(df['v2x_polyarchy']):
        if val > THRESHOLD_BACKSLIDING:
            # If any later score is at least MIN_CHANGE lower, then it's backsliding
            threshold = val - MIN_CHANGE
            if (df['v2x_polyarchy'].iloc[i + 1:] <= threshold).any():
                return True
    return False


# This makes a visual graph of the countries’ democracy scores, grouped by continent
def plot_by_continents(df_ert: pd.DataFrame, relevant_countries: dict) -> None:
    # Rearrange the country data to make it easier to work with
    i_relevant_countries = invert_list_dict(relevant_countries)

    # Get info about which country belongs to which continent
    cont_dict = get_continent_dict()
    unique_conts = set(cont_dict.values())
    traces = []

    # Make a line for each country on the graph
    for country in i_relevant_countries.keys():
        country_df = df_ert[df_ert["country_text_id"] == country]
        trace = go.Scatter(
            x=country_df['year'],  # years go on the x-axis
            y=country_df['v2x_polyarchy'],  # democracy scores go on the y-axis
            mode='lines+markers',
            name=f"{country} ({i_relevant_countries[country][:1]})",
            text=f"{country_df['country_name'].iloc[0]} ({country})",
            hoverinfo="x+y+text",
            meta={"Cont": cont_dict[country], "Status": i_relevant_countries[country]},
        )
        traces.append(trace)

    buttons = []

    # Create button to show all countries
    buttons.append(dict(label="All Countries",
                        method="update",
                        args=[{"visible": [True] * len(traces)},
                              {"title": {"text": "Democracy Index for all relevant Country"}}]))

    # Create buttons to filter by continent
    for cont in unique_conts:
        visible = []
        for trace in traces:
            visible.append(trace.meta["Cont"] == cont)

        buttons.append(dict(label=f"{cont}",
                            method="update",
                            args=[{"visible": visible},
                                  {"title": {"text": f"Democracy Index for Continent {cont}"}}]))

    # Create buttons to filter by whether countries are steady or backsliding
    for status in ["backsliding", "steady"]:
        visible = []
        for trace in traces:
            visible.append(trace.meta["Status"] == status)

        buttons.append(dict(label=f"{status} countries",
                            method="update",
                            args=[{"visible": visible},
                                  {"title": {"text": f"Democracy Index for {status} countries"}}]))

    # Put all the country lines onto the graph
    fig = go.Figure(data=traces)

    # Add a dashed red line to show the threshold where backsliding is considered
    fig.add_shape(
        type="line",
        x0=df_ert['year'].min(), x1=df_ert['year'].max(),
        y0=THRESHOLD_BACKSLIDING, y1=THRESHOLD_BACKSLIDING,
        line=dict(color="Red", width=2, dash="dash"),
    )

    # Set how the graph looks, including the title, axis labels, and filter buttons
    fig.update_layout(
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.5,
            "xanchor": "auto",
            "y": 1.1,
            "yanchor": "auto",
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

    # Show the finished graph
    fig.show()
    return


# This flips a list-of-lists structure, making it easier to look up info by country
def invert_list_dict(relevant_countries: dict) -> dict:
    inverted_countries = {}
    for key in relevant_countries.keys():
        for value in relevant_countries[key]:
            inverted_countries[value] = key
    return inverted_countries


# This reads a file that connects countries to continents and flips the data for easy use
def get_continent_dict() -> dict:
    with open("0_iso3_by_continent.json", "r") as infile:
        json_data = json.load(infile)
    return invert_list_dict(json_data)


# This tells the computer to start everything by running the main() function
if __name__ == "__main__":
    main()
