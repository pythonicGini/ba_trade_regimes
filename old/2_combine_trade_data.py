import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import os
import json

REQUIRED_COLUMNS = ['reporterCode', 'reporterISO', 'reporterDesc', 'partnerCode', 'partnerISO', 'partnerDesc', 'flowCode', 'flowDesc', 'refYear', 'primaryValue']

with open("0_chosen_countries.json", "r") as f:
    json_data = json.load(f)
    backsliding_countries = json_data["Chosen backsliding Countries"]
    steady_countries = json_data["Chosen steady Countries"]


def get_csv_paths(path: str) -> list[str]:
    files = os.listdir(path)

    return [f"{path}/{file}" for file in files if ".csv" in file]



def main() -> None:
    base = "./0_raw_data/trade_data"

    files = get_csv_paths(base)

    df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    for file in files:
        sub_df = pd.read_csv(file, encoding="WINDOWS-1252", index_col=False)[REQUIRED_COLUMNS]
        sub_df = sub_df.loc[~sub_df["partnerCode"].isin([0, 899])]
        df = pd.concat([df, sub_df], ignore_index=True)

    backsliding_df = df.loc[df["reporterISO"].isin(backsliding_countries)]
    steady_df = df.loc[df["reporterISO"].isin(steady_countries)]

    backsliding_df.to_csv("2_combined_trade_data/backsliding_countries.csv", index=False)
    steady_df.to_csv("2_combined_trade_data/steady_countries.csv", index=False)
    return


if __name__ == "__main__":
    main()

