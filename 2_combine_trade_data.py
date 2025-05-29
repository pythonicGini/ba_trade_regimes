import pandas as pd
import os


REQUIRED_COLUMNS = ['reporterCode', 'reporterISO', 'reporterDesc', 'partnerCode', 'partnerISO', 'partnerDesc', 'flowCode', 'flowDesc', 'refYear', 'primaryValue']


def get_csv_paths(path: str) -> list[str]:
    files = os.listdir(path)

    return [f"{path}/{file}" for file in files if ".csv" in file]



def main() -> None:
    base = "./raw_data/trade_data/"
    folders = ["backsliding", "steady"]

    for folder in folders:
        files = get_csv_paths(base + folder)

        df = pd.DataFrame(columns=REQUIRED_COLUMNS)

        for file in files:
            sub_df = pd.read_csv(file, encoding="WINDOWS-1252", index_col=False)[REQUIRED_COLUMNS]
            sub_df = sub_df.loc[~sub_df["partnerCode"].isin([0, 899])]
            df = pd.concat([df, sub_df], ignore_index=True)

        df.to_csv(f"2_combined_trade_data/{folder}_countries.csv", index=False)
        print(f"Wrote {len(df)} datapoints for {folder}")
    return


if __name__ == "__main__":
    main()

