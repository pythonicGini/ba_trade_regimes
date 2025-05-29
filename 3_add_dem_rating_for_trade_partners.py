import pandas as pd


# für jeden datensatz in backsliding.csv und steady.csv -> jahr und handelspartner auslesen -> ert v2x_regime auslesen
# -> als neue spalte an backsliding.csv und steady.csv anhängen



ERT_DF = pd.read_csv("0_raw_data/ert.csv", encoding="utf-8", index_col=False)

def get_regime_index(row: pd.Series):
    regime_index = ERT_DF.loc[ERT_DF["year"] == row.get("refYear")].loc[ERT_DF["country_text_id"] == row.get("partnerISO"), "v2x_regime"].values
    if len(regime_index) == 0:
        return None
    return regime_index[0]


def main() -> None:
    backsliding_df = pd.read_csv("./2_combined_trade_data/backsliding_countries.csv", encoding="utf-8", index_col=False)
    steady_df = pd.read_csv("./2_combined_trade_data/steady_countries.csv", encoding="utf-8", index_col=False)

    backsliding_df["regime_index"] = backsliding_df.apply(get_regime_index, axis=1)
    steady_df["regime_index"] = steady_df.apply(get_regime_index, axis=1)

    backsliding_df.to_csv("./3_trade_data_with_regime_indices/backsliding_countries_trade_and_regime.csv", encoding="utf-8", index=False)
    steady_df.to_csv("./3_trade_data_with_regime_indices/steady_countries_trade_and_regime.csv", encoding="utf-8", index=False)
    return

if __name__ == "__main__":
    main()