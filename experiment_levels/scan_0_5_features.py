"""
Scan training data for all features with values only in 0-5 range.
"""
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, 'code')
from encoding_strategies import _fix_numeric_object_columns, _is_0_5_col

# Load sample
print('Loading 10K training sample...')
train_path = '/Users/Mach/dev/aps/data/2026_Dmodel_data/train_combined.parquet'
train_sample = pd.read_parquet(train_path).head(10000)

# Fix dtypes (critical!)
print('Fixing dtypes...')
train_sample = _fix_numeric_object_columns(train_sample)

# Scan for 0-5 features
print('\nScanning for 0-5 features...')
features_0_5 = []
for col in train_sample.select_dtypes(include=[np.number]).columns:
    if _is_0_5_col(train_sample[col]):
        features_0_5.append(col)

print(f'\nFound {len(features_0_5)} features with values only in 0-5 range:')
print('='*70)
for feat in sorted(features_0_5):
    unique_vals = sorted(train_sample[feat].dropna().unique())
    print(f'  {feat:<50} values: {unique_vals}')
print('='*70)

# Save to file for processing
with open('config/0_5_features_found.txt', 'w') as f:
    for feat in sorted(features_0_5):
        f.write(feat + '\n')
print(f'\nSaved feature list to: config/0_5_features_found.txt')
