import polars as pl
import sys
from datetime import datetime

def consolidate_policies(input_csv, output_csv, max_iterations=14):
    df = pl.read_csv(input_csv)
    
    print(f"Loaded {df.shape[0]:,} rows")
    
    df = df.with_columns(
        pl.col('ID').min().over('pol').alias('superpolicy_id')
    )
    
    print("Initial assignment complete")
    print()
    
    for iteration in range(max_iterations):
        iter_num = iteration + 1
        
        crossing_vins = (
            df.group_by('VIN')
            .agg(pl.col('superpolicy_id').n_unique().alias('policy_count'))
            .filter(pl.col('policy_count') > 1)
        )
        
        crossing_count = crossing_vins.shape[0]
        print(f"Iteration {iter_num:2d}: {crossing_count:,} VINs cross policies", end="")
        
        if crossing_count == 0:
            print(" → CONVERGED")
            break
        
        print()
        
        df = df.with_columns(
            pl.col('superpolicy_id').min().over('pol').alias('superpolicy_id')
        )
    
    df = df.sort('ID')
    df.write_csv(output_csv)
    
    print(f"\nOutput saved: {output_csv}")
    
    unique_policies = df['pol'].n_unique()
    unique_vins = df['VIN'].n_unique()
    unique_superpolicies = df['superpolicy_id'].n_unique()
    
    print(f"Original policies:      {unique_policies:,}")
    print(f"Consolidated policies:  {unique_superpolicies:,}")
    print(f"Unique VINs:           {unique_vins:,}")
    print(f"Total records:         {df.shape[0]:,}")
    
    return df

if __name__ == "__main__":
    input_file = "/Users/o/dev/aps/data/2025_CFX_superpolicy/raw_in/glm_data_step2_final_specificolumns.csv"
    output_file = "/Users/o/dev/aps/data/2025_CFX_superpolicy/raw_out/superpolicy_final.csv"
    
    try:
        consolidate_policies(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
