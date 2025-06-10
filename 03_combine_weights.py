import pandas as pd
from sklearn.preprocessing import StandardScaler


def main() -> None:
    df = load_and_combine_data()
    df = interpolate_weights(df)
    df.to_csv("03_combined_weight_data.csv", index=False)
    return

def interpolate_weights(df) -> pd.DataFrame:
    df_interpolated = df.copy()

    # Ensure the data is sorted for meaningful interpolation
    df_interpolated = df_interpolated.sort_values(by=["Country Code", "Time"])

    # Interpolate numeric columns grouped by 'country'
    for col in df_interpolated.columns:
        if col not in ["Country Code", "Time"]:
            df_interpolated[col] = (
                df_interpolated
                .groupby("Country Code")[col]
                .transform(lambda group: group.interpolate(method='linear', limit_direction='both'))
            )

    for index, value in df_interpolated.isnull().sum().items():
        if value > len(df_interpolated) * 0.15:
            print(f"Column {index} has too many empty values and will be dropped")
            df_interpolated.drop(columns=[index], inplace=True)

        elif value > 0:
            mean = df_interpolated[index].mean()
            df_interpolated[index] = df_interpolated[index].fillna(mean)

    return df_interpolated

def load_and_combine_data() -> pd.DataFrame:
    world_bank_data = pd.read_csv("0_raw_data/weight_data/world_bank_weight_data.csv")
    trade_data = pd.read_csv("02_combined_trade_data.csv")

    trade_data = trade_data.pivot_table(index=["reporterISO", "refYear"],
                                                    columns=["flowCode", "v2x_regime"], values="share")
    trade_data.columns = [f'regime_{int(col[1])}_{col[0]}_share' for col in trade_data.columns]
    trade_data = trade_data.reset_index()

    df_weight_data = pd.merge(world_bank_data, trade_data, how="left", left_on=["Country Code", "Time"], right_on=["reporterISO", "refYear"])
    df_weight_data = df_weight_data.drop(columns=["reporterISO", "refYear", "Time Code", "Country Name"])

    distance_data = pd.read_csv("0_raw_data/weight_data/dist_cepii.CSV", delimiter=";")
    distance_data = distance_data.groupby(["iso_o"])["dist"].mean().round(0).reset_index()

    df_weight_data = df_weight_data.merge(distance_data, left_on="Country Code", right_on="iso_o", how="left").drop(columns=["iso_o"])

    df_member = normalize_membership_data()
    df_weight_data = df_weight_data.merge(df_member, left_on=["Country Code", "Time"], right_on=["iso", "year"], how="left").drop(columns=["iso", "year"])
    df_weight_data["memberships"] = df_weight_data["memberships"].fillna(0)

    df_liberty = get_transformed_liberty_scores()
    return df_weight_data

def normalize_membership_data() -> pd.DataFrame:
    df_member = pd.read_csv("0_raw_data/weight_data/Membership Data.txt", delimiter="\t")
    df_member = df_member.loc[df_member["year"] >= 2000]
    df_member = df_member.groupby(["iso", "year"], as_index=False)["O_Parent"].count()
    df_member.rename(columns={"O_Parent": "memberships"}, inplace=True)
    return df_member


def get_transformed_liberty_scores() -> pd.DataFrame:
    df_liberty = pd.read_csv("0_raw_data/weight_data/Country_and_Territory_Ratings_and_Statuses.CSV", delimiter=";",skiprows=[0,2])
    columns = df_liberty.columns
    name_map = {}

    year = None
    skip = False

    for col in columns:
        if col == "Year(s) Under Review":
            continue
        if col.isnumeric():
            year = col
            skip = True
            continue
        if skip:
            skip = False
            continue
        else:
            if int(year) < 2000:
                continue
            name_map[col] = year + "_Status"

    needed_columns = list(name_map.keys()) + ["Year(s) Under Review"]
    df_liberty = df_liberty[needed_columns]
    df_liberty = df_liberty.rename(columns=name_map)

if __name__ == "__main__":
    main()
