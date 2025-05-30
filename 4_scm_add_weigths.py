import pandas as pd
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

with open("0_chosen_countries.json", "r") as f:
    json_data = json.load(f)
    backsliding_countries = json_data["Chosen backsliding Countries"]
    steady_countries = json_data["Chosen steady Countries"]

with open("0_treatment_periods.json", "r") as f:
    treatment_periods = json.load(f)

def load_and_pre_filter_weight_data() -> (pd.DataFrame, pd.DataFrame):
    name_mapping = {
        "Country Name":                                                         "country",
        "Country Code":                                                         "iso3_country_code",
        "Time":                                                                 "year",
        "GDP (current US$) [NY.GDP.MKTP.CD]":                                   "gdp",
        "Population, total [SP.POP.TOTL]":                                      "population",
        "GDP per capita, PPP (current international $) [NY.GDP.PCAP.PP.CD]":    "gdp_pc",
        "Net trade in goods (BoP, current US$) [BN.GSR.MRCH.CD]":               "net_trade_goods",
    }

    df = pd.read_csv("./0_raw_data/weight_data/world_bank_weight_data.csv", skipfooter=5, engine='python')
    df = df.rename(columns=name_mapping)

    control = df.loc[df["iso3_country_code"].isin(steady_countries)]
    treated = df.loc[df["iso3_country_code"].isin(backsliding_countries)]

    return control, treated

def define_synthetic_controls(control: pd.DataFrame, treated: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        "gdp",
        "population",
        "gdp_pc",
        #"net_trade_goods",
    ]

    for treated_country in treatment_periods.keys():
        print(treated_country)
        df_treated_country = treated.loc[treated["iso3_country_code"] == treated_country]
        pre_period = treatment_periods[treated_country]["pre"]
        treat_period = treatment_periods[treated_country]["treat"]
        post_period = treatment_periods[treated_country]["post"]

        control_pre = control.loc[(pre_period[0] <= control["year"]) & (control["year"] <= pre_period[1])]
        treated_pre = df_treated_country.loc[(pre_period[0] <= df_treated_country["year"]) & (df_treated_country["year"] <= pre_period[1])]


        X_control = control_pre.groupby("iso3_country_code")[predictors].mean()
        X_treated = treated_pre.groupby("iso3_country_code")[predictors].mean()


        scaler = StandardScaler()
        X_control_scaled = scaler.fit_transform(X_control)
        X_treated_scaled = scaler.transform(X_treated)

        ridge = Ridge(alpha=1e-5, fit_intercept=False, positive=True)
        ridge.fit(X_control_scaled.T, X_treated_scaled.T)

        weights = ridge.coef_.flatten()
        weights /= weights.sum()

        weight_df = pd.DataFrame({"country": X_control.index,
                                  "weight": weights}).sort_values("weight", ascending=False)

        print(weight_df)
        break

    return


def main():
    control, treated = load_and_pre_filter_weight_data()
    define_synthetic_controls(control, treated)
    return


if __name__ == "__main__":
    main()