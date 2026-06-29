"""
Load parquet files and join fold data to car_with_dep_factor on vin_date.
"""

import json
import pandas as pd


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    # Load configuration
    config = load_config()
    paths = config["data_paths"]
    
    print("=" * 60)
    print("Loading parquet files from config...")
    print("=" * 60)
    
    # Load fold parquet
    print(f"\nLoading fold parquet from:\n  {paths['fold_parquet']}")
    df_fold = pd.read_parquet(paths["fold_parquet"])
    print(f"Fold data shape: {df_fold.shape}")
    
    # Load car_with_dep_factor parquet
    print(f"\nLoading car_with_dep_factor from:\n  {paths['car_with_dep_factor']}")
    df_car = pd.read_parquet(paths["car_with_dep_factor"])
    print(f"Car with dep factor shape: {df_car.shape}")
    
    # Display head of each dataframe
    print("\n" + "=" * 60)
    print("HEAD OF FOLD DATA")
    print("=" * 60)
    print(df_fold.head())
    print(f"\nColumns: {list(df_fold.columns)}")
    
    print("\n" + "=" * 60)
    print("HEAD OF CAR WITH DEP FACTOR DATA")
    print("=" * 60)
    print(df_car.head())
    print(f"\nColumns: {list(df_car.columns)}")
    
    # Join fold data to car_with_dep_factor on vin_date
    print("\n" + "=" * 60)
    print("JOINING FOLD DATA TO CAR_WITH_DEP_FACTOR ON vin_date")
    print("=" * 60)
    
    df_joined = df_car.merge(df_fold, on="vin_date", how="left")
    
    print(f"\nJoined data shape: {df_joined.shape}")
    print(f"Original car data rows: {len(df_car)}")
    print(f"Fold data rows: {len(df_fold)}")
    print(f"Rows with fold match: {df_joined['fold'].notna().sum()}")
    print(f"Rows without fold match: {df_joined['fold'].isna().sum()}")
    
    print("\n" + "=" * 60)
    print("HEAD OF JOINED DATA")
    print("=" * 60)
    print(df_joined.head())
    print(f"\nColumns: {list(df_joined.columns)}")
    
    # Save joined data to parquet
    output_path = paths["output_joined"]
    print("\n" + "=" * 60)
    print(f"SAVING JOINED DATA TO:\n  {output_path}")
    print("=" * 60)
    df_joined.to_parquet(output_path, index=False)
    print("Saved successfully!")
    
    return df_fold, df_car, df_joined


if __name__ == "__main__":
    df_fold, df_car, df_joined = main()
