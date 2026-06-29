import pandas as pd

data_path = '/Users/Mach/dev/aps/data/2026_Dmodel_data'

# Read car parquet
print("Reading car parquet file...")
car_df = pd.read_parquet(f'{data_path}/master_dataset_car.parquet')
print(f'Car dataframe shape: {car_df.shape}')
print(f'Car dataframe columns: {car_df.columns.tolist()[:10]}...')

# Read VIN_BSST parquet
print("\nReading VIN_BSST parquet file...")
vin_bsst_df = pd.read_parquet(f'{data_path}/VIN_BSST.parquet')
print(f'VIN_BSST dataframe shape: {vin_bsst_df.shape}')
print(f'VIN_BSST dataframe columns: {vin_bsst_df.columns.tolist()}')

# Extract VIN from vin_date
print("\nExtracting VIN from vin_date...")
car_df['vin'] = car_df['vin_date'].str.split('_').str[0]
print(f'Sample vin_date: {car_df["vin_date"].head(3).tolist()}')
print(f'Extracted VIN: {car_df["vin"].head(3).tolist()}')

# Get unique VINs
car_vins = set(car_df['vin'].unique())
print(f'\nTotal unique VINs in car: {len(car_vins)}')

# Get BSST VINs
vin_col = 'VIN' if 'VIN' in vin_bsst_df.columns else 'vin'
bsst_vins = set(vin_bsst_df[vin_col].unique())
print(f'Total unique VINs in BSST: {len(bsst_vins)}')

# VINs without BSST
vins_without_bsst = car_vins - bsst_vins
print(f'\n=== RESULT ===')
print(f'Number of VINs without BSST: {len(vins_without_bsst)}')
print(f'Percentage: {len(vins_without_bsst)/len(car_vins)*100:.2f}%')
