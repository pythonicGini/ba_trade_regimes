import pandas as pd


# für jeden datensatz in backsliding.csv und steady.csv -> jahr und handelspartner auslesen -> ert v2x_regime auslesen
# -> als neue spalte an backsliding.csv und steady.csv anhängen



def main() -> None:
    ert_df = pd.read_csv("0_raw_data/ert.csv", encoding="utf-8", index_col=False)[["year", "country_text_id", "v2x_regime"]]
    ert_df = ert_df.loc[(ert_df["year"] >= 2000)]


    backsliding_df = pd.read_csv("./2_combined_trade_data/backsliding_countries.csv", encoding="utf-8", index_col=False)
    steady_df = pd.read_csv("./2_combined_trade_data/steady_countries.csv", encoding="utf-8", index_col=False)


    backsliding_df = backsliding_df.merge(ert_df[["country_text_id", "year", "v2x_regime"]], how="left", left_on=["partnerISO", "refYear"], right_on=["country_text_id", "year"])
    backsliding_df = backsliding_df.drop(columns=["country_text_id", "year"], axis=1)
    backsliding_df = backsliding_df.rename({"v2x_regime": "regime_index"}, axis=1)

    steady_df = steady_df.merge(ert_df[["country_text_id", "year", "v2x_regime"]], how="left", left_on=["partnerISO", "refYear"], right_on=["country_text_id", "year"])
    steady_df = steady_df.drop(columns=["country_text_id", "year"], axis=1)
    steady_df = steady_df.rename({"v2x_regime": "regime_index"}, axis=1)


    backsliding_df = backsliding_df.groupby(["reporterISO", "flowCode", "refYear", "regime_index"], as_index=False)["primaryValue"].sum()
    steady_df = steady_df.groupby(["reporterISO", "flowCode", "refYear", "regime_index"], as_index=False)["primaryValue"].sum()

    backsliding_df.to_csv("./3_trade_data_with_regime_indices/backsliding_countries_trade_and_regime.csv", encoding="utf-8", index=False)
    steady_df.to_csv("./3_trade_data_with_regime_indices/steady_countries_trade_and_regime.csv", encoding="utf-8", index=False)
    return

if __name__ == "__main__":
    main()