# Essential libraries are imported for data manipulation, machine learning, and interactive visualization
import os.path
import pandas as pd
import json
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Main routine that orchestrates the creation of treatment periods and visualization of trade data
def main() -> None:
    create_country_treatments()  # Initializes or updates country treatment periods

    with open("00_treatment_countries.json", "r") as f:
        relevant_countries = json.load(f)

    # List of countries classified as "steady" democracies
    control_countries = [c for c, v in relevant_countries.items() if v["dem"] == "steady"]

    # Preprocessed trade and weight data are loaded
    df_trade_data = pd.read_csv("02_combined_trade_data.csv")
    df_weights = pd.read_csv("03_combined_weight_data.csv")

    # Regime types and trade flow categories are defined
    regime_indices = [0, 1, 2, 3]
    flow_codes = ["M", "X"]

    # Visualization structure is set up with a 2x2 subplot layout
    fig = make_subplots(rows=2,
                        cols=2,
                        subplot_titles=[f"Trade with regime of index {int(r)}" for r in regime_indices])

    # Each regime index is assigned a position in the subplot grid
    regime_positions = {0: [1, 1],
                        1: [1, 2],
                        2: [2, 1],
                        3: [2, 2],
                        }

    # Processing is done for each country that has defined treatment periods
    for country, values in relevant_countries.items():
        treat_start = values["treat_start"]
        treat_end = values["treat_end"]
        df_local_trade_data = df_trade_data.copy()
        df_local_weights = df_weights.copy()

        if not treat_start or not treat_end:
            continue

        # Historical weights for both treated and control countries before treatment are extracted
        control_weights = df_local_weights[(df_local_weights["Country Code"] == country) | 
                                           (df_local_weights["Country Code"].isin(control_countries))]
        control_weights = control_weights[control_weights["Time"] < treat_start]
        relevant_cols = [col for col in control_weights.columns.tolist() if col not in ["Country Code", "Time"]]

        # Average values per country are calculated to build feature matrix
        X_all = control_weights.groupby("Country Code")[relevant_cols].mean()
        treated_index = X_all.index.get_loc(country)

        # Standardization is applied to normalize data
        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(X_all)

        control_indicies = [i for i in range(len(X_all_scaled)) if i != treated_index]
        local_control_countries = [X_all.iloc[x].name for x in control_indicies]

        X_control_scaled = X_all_scaled[control_indicies]
        X_treated_scaled = X_all_scaled[treated_index]

        # Ridge regression is used to determine weights for synthetic control construction
        try:
            ridge = Ridge(alpha=1e-5, fit_intercept=False, positive=True)
            ridge.fit(X_control_scaled.T, X_treated_scaled.T)
        except ValueError:
            continue

        weights = ridge.coef_
        weights /= weights.sum()

        # Dictionary of countries and their corresponding weights
        weights_countries = {x: y for x, y in zip(local_control_countries, weights)}

        # Trade data for treated country and synthetic control group is retrieved
        df_country = df_local_trade_data.loc[df_local_trade_data["reporterISO"] == country]
        df_trade_control = df_local_trade_data[df_local_trade_data["reporterISO"].isin(local_control_countries)]
        df_trade_control = df_trade_control[df_trade_control["refYear"] < 2024]

        # Weighted trade shares for synthetic control are calculated
        df_trade_control["weight"] = df_trade_control["reporterISO"].map(weights_countries)
        df_trade_control["weighted_share"] = df_trade_control["weight"] * df_trade_control["share"]
        synthetic_control = df_trade_control.groupby(["refYear", "flowCode", "v2x_regime"])["weighted_share"].sum()
        synthetic_control = synthetic_control.reset_index()

        # Treatment period markers are added as vertical lines
        treat_start_line = go.Scatter(
            x=[treat_start] * 100,
            y=list(range(-1, 101, 1)),
            mode='lines',
            text=f"treatment_start",
            line=dict(dash='dash', width=1, color='green'),
            hoverinfo="text",
            showlegend=False,
            meta=country
        )
        treat_end_line = go.Scatter(
            x=[treat_end] * 100,
            y=list(range(-1, 101, 1)),
            mode='lines',
            line=dict(dash='dash', width=1, color='green'),
            text=f"treatment_end",
            hoverinfo="text",
            showlegend=False,
            meta=country
        )

        # Both actual and synthetic trade data are plotted for each combination of flow and regime type
        for flow_code in flow_codes:
            for regime_index in regime_indices:
                row, col = regime_positions[regime_index]

                treated = df_country.loc[
                    (df_country["flowCode"] == flow_code) & (df_country["v2x_regime"] == regime_index)]
                synthetic = synthetic_control.loc[
                    (synthetic_control["flowCode"] == flow_code) & (synthetic_control["v2x_regime"] == regime_index)]

                plot_treated = go.Scatter(
                    x=treated["refYear"],
                    y=treated["share"],
                    mode='lines+markers',
                    marker={"color": "blue" if flow_code == "M" else "red"},
                    text=f"{'Import' if flow_code == 'M' else 'Export'} of {country}",
                    hoverinfo="x+y+text",
                    showlegend=False,
                    meta=country,
                )

                plot_synthetic = go.Scatter(
                    x=synthetic["refYear"],
                    y=synthetic["weighted_share"],
                    mode='lines+markers',
                    marker={"color": "MediumSlateBlue" if flow_code == "X" else "PaleVioletRed"},
                    text=f"{'Import' if flow_code == 'X' else 'Export'} of synthetic_{country}",
                    hoverinfo="x+y+text",
                    showlegend=False,
                    meta=country
                )

                # Plots are added to the relevant subplot
                fig.add_trace(plot_treated, row=row, col=col)
                fig.add_trace(plot_synthetic, row=row, col=col)
                fig.add_trace(treat_start_line, row=row, col=col)
                fig.add_trace(treat_end_line, row=row, col=col)

    # Interactive button setup to control country selection in the plot
    buttons = []
    visibility = []

    for trace in fig.data:
        trace.visible = False
        visibility.append(False)

    buttons.append(dict(
        label="None",
        method="update",
        args=[{"visible": visibility}],
    ))

    # A button is created for each country to toggle its visibility in the plot
    defined_countries = []
    for trace in fig.data:
        if trace.meta not in defined_countries:
            defined_countries.append(trace.meta)

    for country in defined_countries:
        visibility = []

        for trace in fig.data:
            visibility.append(trace.meta == country)

        # Uncomment the block below to save country-specific plots as images
        fig_small = fig
        for trace, visible in zip(fig_small.data, visibility):
            trace.visible = visible
        fig_small.update_layout(title_text=f"Summarized tradevolume (Import and Export) in % with countries of different regime indices for {country}")
        fig_small.write_image(f"images/{country}.png", width=1280, height=720)

        buttons.append(dict(
            label=country,
            method="update",
            args=[{"visible": visibility}]
        ))

    # Plot layout is finalized and saved
    fig.update_yaxes(range=[0, 100])
    fig.update_yaxes(title="Percentage of trade")
    fig.update_xaxes(range=[1997, 2027])
    fig.update_xaxes(title="year")
    fig.update_layout(
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": False,
            "x": 1.05,
            "y": 1.1
        }],
        title_text="Summarized tradevolume (Import and Export) in % with countries of different regime indices ",
    )
    fig.write_html("1_plots/SCM_Plot.html")
    fig.show()
    return

# Loads previously saved weight data for reuse
def load_weight_data() -> pd.DataFrame:
    return pd.read_csv("03_combined_weight_data.csv")

# Initializes treatment periods for each country based on its democracy classification
def create_country_treatments():
    with open("00_steady_and_backsliding_countries.json", "r") as f:
        json_data = json.load(f)
        relevant_countries = {}
        for x in json_data.keys():
            for y in json_data[x]:
                if x == "backsliding":
                    relevant_countries[y] = {
                        "dem": x,
                        "treat_start": None,
                        "treat_end": None,
                    }
                else:
                    treat_start = random.randint(2005, 2015)
                    treat_end = treat_start + random.randint(3, 5)
                    relevant_countries[y] = {
                        "dem": x,
                        "treat_start": treat_start,
                        "treat_end": treat_end,
                    }

    # If treatment data already exists, it is preserved and updated
    if os.path.exists("00_treatment_countries.json"):
        with open("00_treatment_countries.json", "r") as f:
            defined_countries = json.load(f)
            relevant_countries.update(defined_countries)

    with open("00_treatment_countries.json", "w") as f:
        json.dump(relevant_countries, f, indent=4)

# Ensures the main function is executed when the script is run directly
if __name__ == "__main__":
    main()
