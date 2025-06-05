import pandas as pd
import json
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go


with open("0_chosen_countries.json", "r") as f:
    json_data = json.load(f)
    backsliding_countries = json_data["Chosen backsliding Countries"]
    steady_countries = json_data["Chosen steady Countries"]

with open("0_treatment_periods.json", "r") as f:
    treatment_periods = json.load(f)

PREDICTORS = [
        "GDP per capita, PPP (current international $) [NY.GDP.PCAP.PP.CD]",
        #"Merchandise trade (% of GDP) [TG.VAL.TOTL.GD.ZS]",
        "Unemployment, total (% of total labor force) (national estimate) [SL.UEM.TOTL.NE.ZS]",
        "GDP growth (annual %) [NY.GDP.MKTP.KD.ZG]",
        "Control of Corruption: Estimate [CC.EST]",
        # "External debt stocks (% of GNI) [DT.DOD.DECT.GN.ZS]",
        "Inflation, consumer prices (annual %) [FP.CPI.TOTL.ZG]",
        "General government final consumption expenditure (% of GDP) [NE.CON.GOVT.ZS]",
        #"Trade (% of GDP) [NE.TRD.GNFS.ZS]",
        "Tax revenue (% of GDP) [GC.TAX.TOTL.GD.ZS]",
        # "Literacy rate, adult total (% of people ages 15 and above) [SE.ADT.LITR.ZS]",
        "Life expectancy at birth, total (years) [SP.DYN.LE00.IN]",
        "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population) [SI.POV.DDAY]",
        #"Taxes on international trade (% of revenue) [GC.TAX.INTT.RV.ZS]",
        "Tariff rate, applied, simple mean, all products (%) [TM.TAX.MRCH.SM.AR.ZS]",
        "Rule of Law: Percentile Rank [RL.PER.RNK]",
        "Regulatory Quality: Percentile Rank [RQ.PER.RNK]",
        "Political Stability and Absence of Violence/Terrorism: Percentile Rank [PV.PER.RNK]"
    ]




def load_and_pre_filter_weight_data() -> (pd.DataFrame, pd.DataFrame):
    name_mapping = {
        "Country Name":                                                         "country",
        "Country Code":                                                         "iso3_country_code",
        "Time":                                                                 "year",
    }

    relevant_cols = PREDICTORS + ["country", "iso3_country_code", "year"]

    df = pd.read_csv("./0_raw_data/weight_data/world_bank_weight_data.csv", skipfooter=5, engine='python')
    df = df.rename(columns=name_mapping)

    control = df.loc[df["iso3_country_code"].isin(steady_countries)][relevant_cols]
    treated = df.loc[df["iso3_country_code"].isin(backsliding_countries)][relevant_cols]
    control = interpolate_df(control)
    treated = interpolate_df(treated)
    for index, value in control.isnull().sum().items():
        if value > len(control) * 0.1:
            print(control.isnull().sum())
            raise Exception(f"To many NA Values in column {index}")
        elif value > 0:
            mean = control[index].mean()
            control[index] = control[index].fillna(mean)

    for index, value in treated.isnull().sum().items():
        if value > len(treated) * 0.1:
            print(treated.isnull().sum())
            raise Exception(f"To many NA Values in column {index}")

    return control, treated


def interpolate_df(df: pd.DataFrame) -> pd.DataFrame:
    df_interpolated = df.copy()

    # Ensure the data is sorted for meaningful interpolation
    df_interpolated = df_interpolated.sort_values(by=['country', 'year'])

    # Interpolate numeric columns grouped by 'country'
    for col in df.columns:
        if col in PREDICTORS:
            df_interpolated[col] = (
                df_interpolated
                .groupby('country')[col]
                .transform(lambda group: group.interpolate(method='linear', limit_direction='both'))
            )

    return df_interpolated


def do_scm(control: pd.DataFrame, treated: pd.DataFrame) -> dict[str, pd.DataFrame]:

    weight_dict = {}

    for treated_country in treatment_periods.keys():
        df_treated_country = treated.loc[treated["iso3_country_code"] == treated_country]
        pre_period = treatment_periods[treated_country]["pre"]
        treat_period = treatment_periods[treated_country]["treat"]

        control_pre = control.loc[control["iso3_country_code"] != treated_country]
        control_pre = control_pre.loc[(pre_period[0] <= control_pre["year"]) & (control_pre["year"] <= pre_period[1])]
        treated_pre = df_treated_country.loc[(pre_period[0] <= df_treated_country["year"]) & (df_treated_country["year"] <= pre_period[1])]

        X_control = control_pre.groupby("iso3_country_code")[PREDICTORS].mean()
        X_treated = treated_pre.groupby("iso3_country_code")[PREDICTORS].mean()

        x_all = pd.concat([X_control, X_treated], ignore_index=True)
        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(x_all)
        X_control_scaled = X_all_scaled[:-1]
        X_treated_scaled = X_all_scaled[-1]

        ridge = Ridge(alpha=1e-5, fit_intercept=False, positive=True)
        ridge.fit(X_control_scaled.T, X_treated_scaled.T)

        weights = ridge.coef_
        weights /= weights.sum()

        weight_df = pd.DataFrame({"country": X_control.index,
                                  "weight": weights}).sort_values("weight", ascending=False)

        weight_dict[treated_country] = weight_df

    return weight_dict


def make_synthetic_controls(weights: dict) -> dict:
    df_steady_countries = pd.read_csv("3_trade_data_with_regime_indices/steady_countries_trade_and_regime.csv")

    synthetic_controls = {}

    for treated_country in weights.keys():
        treated_weights = weights[treated_country]
        treated_weights: pd.DataFrame

        controls_df = df_steady_countries.loc[df_steady_countries["reporterISO"].isin(treated_weights["country"].unique())]
        controls_df = controls_df.merge(treated_weights, left_on="reporterISO", right_on="country", how="left").drop("country", axis=1)

        controls_df["weighted_share"] = controls_df["share"] * controls_df["weight"]


        controls_df = controls_df.groupby(["refYear", "flowCode", "regime_index"], as_index=False)["weighted_share"].sum()
        controls_df.rename(columns={"weighted_share": "share"}, inplace=True)
        synthetic_controls[treated_country] = controls_df

    return synthetic_controls

def make_plot(synthetic_controls: dict) -> None:
    df_backsliding = pd.read_csv("3_trade_data_with_regime_indices/backsliding_countries_trade_and_regime.csv")

    regime_indices = [0, 1, 2, 3]
    flow_codes = ["M", "X"]

    fig = make_subplots(rows=2,
                        cols=2,
                        subplot_titles=[f"Trade with regime of index {int(r)}" for r in regime_indices])

    regime_positions = {0: [1, 1],
                        1: [1, 2],
                        2: [2, 1],
                        3: [2, 2],
                        }

    for country in synthetic_controls.keys():
        df_synthetic = synthetic_controls[country]
        df_country = df_backsliding.loc[df_backsliding["reporterISO"] == country]
        if country == "GRC":
            print(df_country)
        treat_start = treatment_periods[country]["treat"][0]
        treat_end = treatment_periods[country]["treat"][1]

        treat_start_line = go.Scatter(
            x=[treat_start] * 100,
            y=list(range(-1,101,1)),
            mode='lines',
            text=f"treatment_start",
            line=dict(dash='dash', width=1, color='green'),
            hoverinfo="text",
            showlegend=False,
            meta = country
        )
        treat_end_line = go.Scatter(
            x=[treat_end] * 100,
            y=list(range(-1,101,1)),
            mode='lines',
            line=dict(dash='dash', width=1, color='green'),
            text=f"treatment_end",
            hoverinfo="text",
            showlegend=False,
            meta = country
        )

        for flow_code in flow_codes:
            for regime_index in regime_indices:
                row, col = regime_positions[regime_index]

                treated = df_country.loc[(df_country["flowCode"] == flow_code) & (df_country["regime_index"] == regime_index)]
                synthetic = df_synthetic.loc[(df_synthetic["flowCode"] == flow_code) & (df_synthetic["regime_index"] == regime_index)]

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
                    y=synthetic["share"],
                    mode='lines+markers',
                    marker={"color": "MediumSlateBlue" if flow_code == "X" else "PaleVioletRed"},
                    text=f"{'Import' if flow_code == 'X' else 'Export'} of synthetic_{country}",
                    hoverinfo="x+y+text",
                    showlegend=False,
                    meta=country
                )

                fig.add_trace(plot_treated, row = row, col = col)
                fig.add_trace(plot_synthetic, row = row, col = col)
                fig.add_trace(treat_start_line, row = row, col = col)
                fig.add_trace(treat_end_line, row = row, col = col)

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

    for country in synthetic_controls.keys():
        visibility = []
        for trace in fig.data:
            if trace.meta == country:
                visibility.append(True)
            else:
                visibility.append(False)

        buttons.append(dict(
            label=country,
            method="update",
            args=[{"visible": visibility}]
        ))

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



def main():
    control, treated = load_and_pre_filter_weight_data()
    weights = do_scm(control, treated)
    synthetic_controls = make_synthetic_controls(weights)
    make_plot(synthetic_controls)
    return


if __name__ == "__main__":
    main()