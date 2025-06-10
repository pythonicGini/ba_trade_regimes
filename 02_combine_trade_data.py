import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import os
import json

REQUIRED_COLUMNS = ['reporterCode', 'reporterISO', 'reporterDesc', 'partnerCode', 'partnerISO', 'partnerDesc', 'flowCode', 'flowDesc', 'refYear', 'primaryValue']

NOT_USED_COUNTRY_CODES = [0, 899, 490, 837, 568]

def invert_list_dict(relevant_countries: dict) -> dict:
    inverted_countries = {}
    for key in relevant_countries.keys():
        for value in relevant_countries[key]:
            inverted_countries[value] = key
    return inverted_countries

with open("00_steady_and_backsliding_countries.json", "r") as f:
    json_data = json.load(f)
    COUNTRIES = invert_list_dict(json_data)


def get_csv_paths(path: str) -> list[str]:
    files = os.listdir(path)

    return [f"{path}/{file}" for file in files if ".csv" in file]

def main() -> None:
    df = concat_csv_files()
    df = add_dem_rating(df)
    group_and_create_relative_trade(df)


def group_and_create_relative_trade(df) -> pd.DataFrame:
    df = df.groupby(["reporterISO","refYear", "flowCode", "v2x_regime"], as_index=False)["primaryValue"].sum()
    df = df.reset_index()
    print(df)
    return df

def add_dem_rating(df) -> pd.DataFrame:
    df_ert = pd.read_csv("0_raw_data/ert.csv", encoding="utf-8")
    columns = REQUIRED_COLUMNS + ["v2x_regime"]
    df = pd.merge(df, df_ert, how="left", left_on=["partnerISO", "refYear"], right_on=["country_text_id", "year"])[columns]
    df.to_csv("0_test.csv", index=False)
    return df


def concat_csv_files() -> pd.DataFrame:
    base = "./0_raw_data/trade_data"

    files = get_csv_paths(base)

    df_list_csv = []

    for file in files:
        sub_df = pd.read_csv(file, encoding="WINDOWS-1252", index_col=False)[REQUIRED_COLUMNS]
        sub_df = sub_df.loc[~sub_df["partnerCode"].isin(NOT_USED_COUNTRY_CODES)]
        df_list_csv.append(sub_df)
    df = pd.concat(df_list_csv, ignore_index=True)
    df = df.reset_index()

    return df

if __name__ == "__main__":
    main()