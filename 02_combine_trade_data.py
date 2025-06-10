# Future warnings are disabled to avoid cluttering the output
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Required libraries are imported for data handling, file operations, and reading JSON data
import pandas as pd
import os
import json

# Definition of the essential columns expected in the trade datasets
REQUIRED_COLUMNS = ['reporterCode', 'reporterISO', 'reporterDesc', 'partnerCode', 'partnerISO', 'partnerDesc', 'flowCode', 'flowDesc', 'refYear', 'primaryValue']

# A list of country codes that are to be excluded from the dataset
NOT_USED_COUNTRY_CODES = [0, 899, 490, 837, 568]

# Converts a dictionary where each key maps to a list into one where each item maps back to its original key
def invert_list_dict(relevant_countries: dict) -> dict:
    inverted_countries = {}
    for key in relevant_countries.keys():
        for value in relevant_countries[key]:
            inverted_countries[value] = key
    return inverted_countries

# A JSON file containing categorized countries is loaded and inverted to map individual countries to their status
with open("00_steady_and_backsliding_countries.json", "r") as f:
    json_data = json.load(f)
    COUNTRIES = invert_list_dict(json_data)

# Returns a list of paths to all CSV files located in the specified directory
def get_csv_paths(path: str) -> list[str]:
    files = os.listdir(path)
    return [f"{path}/{file}" for file in files if ".csv" in file]

# Main function to execute all processing steps and return the final dataset
def main() -> None:
    df = concat_csv_files()        # Combines all relevant trade CSV files into one dataset
    df = add_dem_rating(df)        # Merges democratic ratings into the dataset
    df = group_and_create_relative_trade(df)  # Aggregates and normalizes trade data
    df = df.drop(columns=["index"])
    df.to_csv("02_combined_trade_data.csv", index=False)
    return

# Groups trade data by country, year, and trade flow; calculates relative trade share in percentage
def group_and_create_relative_trade(df) -> pd.DataFrame:
    df = df.groupby(["reporterISO", "refYear", "flowCode", "v2x_regime"], as_index=False)["primaryValue"].sum()
    df = df.reset_index()
    df["share"] = df.groupby(["reporterISO", "flowCode", "refYear"])["primaryValue"].transform(lambda x: x / x.sum() * 100).round(2)
    return df

# Adds democratic regime data to each trade record based on country and year
def add_dem_rating(df) -> pd.DataFrame:
    df_ert = pd.read_csv("0_raw_data/ert.csv", encoding="utf-8")
    columns = REQUIRED_COLUMNS + ["v2x_regime"]
    df = pd.merge(df, df_ert, how="left", left_on=["partnerISO", "refYear"], right_on=["country_text_id", "year"])[columns]
    return df

# Combines multiple trade data files into a single structured dataset, excluding irrelevant country codes
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

# Ensures the main function is only run when this script is executed directly
if __name__ == "__main__":
    main()