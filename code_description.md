# Structured Description of Data Analysis Pipeline for Democratic Backsliding and Trade Patterns

## Overview
This pipeline implements a synthetic control method analysis to examine the relationship between democratic backsliding and international trade patterns. The system processes multiple data sources to identify countries experiencing democratic decline and compares their trade behaviors with synthetically constructed control groups.

## Script 1: Democratic Classification and Visualization

### Core Classification Logic
```python
def get_relevant_countries(df_ert: pd.DataFrame) -> dict:
    valid_countries = {
        "backsliding": [],
        "steady": []
    }
    
    for country in df_ert["country_text_id"].unique():
        if country in IGNORE_COUNTRIES:
            continue
            
        df_country = df_ert[df_ert["country_text_id"] == country].reset_index()
        max_regime = df_country["v2x_polyarchy"].max()
        min_regime = df_country["v2x_polyarchy"].min()
        
        if (max_regime - min_regime <= MIN_CHANGE) and (min_regime >= THRESHOLD_STEADY):
            valid_countries["steady"].append(country)
            continue
            
        if is_backsliding(df_country):
            valid_countries["backsliding"].append(country)
```

**Data Preparation Process:**
- Loads electoral regime type (ERT) data from CSV files containing V-Dem democracy indicators
- Filters data to focus on years from 2000 onwards (START_YEAR = 2000)
- Excludes predefined problematic countries from the IGNORE_COUNTRIES list
- Applies dual classification criteria:
  - **Steady democracies**: Countries with minimal regime change (≤0.1) and consistently high democratic scores (≥0.5)
  - **Backsliding countries**: Countries showing significant democratic decline identified through temporal analysis

### Backsliding Detection Algorithm
```python
def is_backsliding(df: pd.DataFrame) -> bool:
    for i, val in enumerate(df['v2x_polyarchy']):
        if val > THRESHOLD_BACKSLIDING:
            threshold = val - MIN_CHANGE
            if (df['v2x_polyarchy'].iloc[i + 1:] <= threshold).any():
                return True
    return False
```

**Evaluation Methodology:**
- Implements a sequential threshold-based detection system
- Identifies periods where democratic scores exceed 0.5 (THRESHOLD_BACKSLIDING)
- Detects subsequent drops of at least 0.1 points (MIN_CHANGE) as evidence of backsliding
- Uses the V-Dem polyarchy index (v2x_polyarchy) as the primary democracy measure
- Outputs classification results to "00_steady_and_backsliding_countries.json"

## Script 2: Trade Data Integration and Democratic Regime Mapping

### Data Consolidation Framework
```python
def concat_csv_files() -> pd.DataFrame:
    base = "./0_raw_data/trade_data"
    files = get_csv_paths(base)
    
    df_list_csv = []
    
    for file in files:
        sub_df = pd.read_csv(file, encoding="WINDOWS-1252", index_col=False)[REQUIRED_COLUMNS]
        sub_df = sub_df.loc[~sub_df["partnerCode"].isin(NOT_USED_COUNTRY_CODES)]
        df_list_csv.append(sub_df)
    
    df = pd.concat(df_list_csv, ignore_index=True)
    return df
```

**Data Preparation Process:**
- Processes multiple trade data CSV files from the raw data directory
- Standardizes data structure using REQUIRED_COLUMNS specification
- Filters out irrelevant entities using NOT_USED_COUNTRY_CODES (excludes codes: 0, 899, 490, 837, 568)
- Handles encoding issues with WINDOWS-1252 character set

### Democratic Regime Integration
```python
def add_dem_rating(df) -> pd.DataFrame:
    df_ert = pd.read_csv("0_raw_data/ert.csv", encoding="utf-8")
    columns = REQUIRED_COLUMNS + ["v2x_regime"]
    df = pd.merge(df, df_ert, how="left", left_on=["partnerISO", "refYear"], 
                  right_on=["country_text_id", "year"])[columns]
    return df
```

**Evaluation Framework:**
- Merges trade flow data with democratic regime classifications
- Links trading partner countries to their democratic status using ISO codes and years
- Preserves temporal alignment between trade transactions and democratic ratings
- Adds v2x_regime scores to enable regime-based trade analysis

### Trade Share Calculation
```python
def group_and_create_relative_trade(df) -> pd.DataFrame:
    df = df.groupby(["reporterISO", "refYear", "flowCode", "v2x_regime"], as_index=False)["primaryValue"].sum()
    df = df.reset_index()
    df["share"] = df.groupby(["reporterISO", "flowCode", "refYear"])["primaryValue"].transform(lambda x: x / x.sum() * 100).round(2)
    return df
```

**Data Transformation Process:**
- Aggregates trade values by reporter country, year, flow direction, and partner regime type
- Calculates relative trade shares as percentages of total trade volume
- Normalizes data to enable cross-country and cross-temporal comparisons
- Outputs processed data to "02_combined_trade_data.csv"

## Script 3: Weight Data Processing and Feature Engineering

### Data Integration Architecture
```python
def load_and_combine_data() -> pd.DataFrame:
    world_bank_data = pd.read_csv("0_raw_data/weight_data/world_bank_weight_data.csv")
    trade_data = pd.read_csv("02_combined_trade_data.csv")
    
    trade_data = trade_data.pivot_table(index=["reporterISO", "refYear"],
                                        columns=["flowCode", "v2x_regime"], values="share")
    trade_data.columns = [f'regime_{int(col[1])}_{col[0]}_share' for col in trade_data.columns]
    trade_data = trade_data.reset_index()
```

**Data Preparation Process:**
- Integrates World Bank economic indicators with processed trade data
- Pivots trade data to create regime-specific trade share variables
- Generates column names following the pattern: "regime_{regime_index}_{flow_direction}_share"
- Merges datasets on country codes and time periods

### Missing Data Handling Strategy
```python
def interpolate_weights(df) -> pd.DataFrame:
    df_interpolated = df.copy()
    df_interpolated = df_interpolated.sort_values(by=["Country Code", "Time"])
    
    for col in df_interpolated.columns:
        if col not in ["Country Code", "Time"]:
            df_interpolated[col] = (
                df_interpolated
                .groupby("Country Code")[col]
                .transform(lambda group: group.interpolate(method='linear', limit_direction='both'))
            )
```

**Evaluation and Quality Control:**
- Implements linear interpolation for missing values within country time series
- Applies data quality thresholds: drops columns with >15% missing values
- Uses mean imputation for remaining missing values after interpolation
- Ensures temporal consistency through country-grouped sorting

### Additional Feature Integration
```python
distance_data = pd.read_csv("0_raw_data/weight_data/dist_cepii.CSV", delimiter=";")
distance_data = distance_data.groupby(["iso_o"])["dist"].mean().round(0).reset_index()

df_member = normalize_membership_data()
df_weight_data = df_weight_data.merge(df_member, left_on=["Country Code", "Time"], right_on=["iso", "year"], how="left")
```

**Feature Engineering Process:**
- Incorporates geographic distance data from CEPII database
- Calculates average distances for each country to create geographic controls
- Adds international organization membership counts as institutional variables
- Processes membership data from 2000 onwards with annual aggregation

## Script 4: Synthetic Control Method Implementation

### Treatment Period Assignment
```python
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
```

**Evaluation Framework Setup:**
- Assigns random treatment periods to steady democracies (placebo treatments)
- Treatment starts randomly between 2005-2015, lasting 3-5 years
- Backsliding countries receive actual treatment periods (to be defined manually)
- Creates persistent treatment assignment through JSON file storage

### Synthetic Control Construction
```python
control_weights = df_local_weights[(df_local_weights["Country Code"] == country) | 
                                   (df_local_weights["Country Code"].isin(control_countries))]
control_weights = control_weights[control_weights["Time"] < treat_start]
relevant_cols = [col for col in control_weights.columns.tolist() if col not in ["Country Code", "Time"]]

X_all = control_weights.groupby("Country Code")[relevant_cols].mean()
treated_index = X_all.index.get_loc(country)

scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)
```

**Data Preparation for Synthetic Control:**
- Extracts pre-treatment characteristics for treated and control countries
- Calculates country-level averages across all feature dimensions
- Applies standardization to ensure equal weighting of different variable scales
- Separates treated country characteristics from potential control pool

### Weight Optimization Process
```python
ridge = Ridge(alpha=1e-5, fit_intercept=False, positive=True)
ridge.fit(X_control_scaled.T, X_treated_scaled.T)

weights = ridge.coef_
weights /= weights.sum()

weights_countries = {x: y for x, y in zip(local_control_countries, weights)}
```

**Evaluation Methodology:**
- Implements Ridge regression with positive constraint for weight determination
- Uses minimal regularization (alpha=1e-5) to prioritize fit quality
- Normalizes weights to sum to unity for proper synthetic control construction
- Maps optimized weights to corresponding control countries

### Synthetic Control Outcome Construction
```python
df_trade_control["weight"] = df_trade_control["reporterISO"].map(weights_countries)
df_trade_control["weighted_share"] = df_trade_control["weight"] * df_trade_control["share"]
synthetic_control = df_trade_control.groupby(["refYear", "flowCode", "v2x_regime"])["weighted_share"].sum()
```

**Final Evaluation Process:**
- Applies optimized weights to control countries' trade outcomes
- Constructs synthetic control time series through weighted aggregation
- Maintains disaggregation by trade flow direction and partner regime type
- Enables comparison between actual treated outcomes and synthetic counterfactuals

## Overall Pipeline Assessment

**Strengths:**
- Comprehensive integration of multiple data sources (V-Dem, trade, World Bank, geographic)
- Robust missing data handling with quality control thresholds
- Sophisticated synthetic control methodology with proper weight optimization
- Temporal alignment preservation across all data integration steps

**Key Limitations:**
- Random treatment assignment for placebo tests may not reflect realistic policy timing
- Limited validation of democracy classification thresholds
- Potential selection bias in IGNORE_COUNTRIES list
- Ridge regression approach may not capture complex non-linear relationships in synthetic control construction