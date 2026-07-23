"""
Quick test to verify Type 4 encoding works correctly.

Type 4 should:
1. Remap level 2 -> level 4 for all 0-5 features
2. Apply binary grouping: {0,1,4}->0 | {3,5}->1
3. Result: only levels {0,1,3,4,5} remain (no level 2)
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'code')

from encoding_strategies import encode_type4_custom

# Create synthetic test data
print("=" * 70)
print("Testing Type 4 Encoding: Map 2->4, then Binary {0,1,4}->0 | {3,5}->1")
print("=" * 70)

# Create test dataframe with 0-5 features
test_data = pd.DataFrame({
    'feature_0_5_A': [0, 1, 2, 3, 4, 5, 0, 1, 2],  # Has level 2
    'feature_0_5_B': [0, 0, 2, 2, 3, 3, 5, 5, 4],  # Has level 2
    'feature_other': [10, 20, 30, 40, 50, 60, 70, 80, 90],  # Non 0-5
    'pp_bi': [100, 200, 300, 400, 500, 600, 700, 800, 900],  # Target
})

print("\n1. Original Data (before encoding):")
print(test_data[['feature_0_5_A', 'feature_0_5_B', 'feature_other']].head())

# Apply Type 4 encoding
X_encoded, feature_names, encoders = encode_type4_custom(test_data)

print("\n2. After Type 4 Encoding:")
print(X_encoded[['feature_0_5_A', 'feature_0_5_B', 'feature_other']].head(9))

print("\n3. Encoding Logic Verification:")
print("   Original -> After 2->4 remap -> After binary grouping")
print("   --------------------------------------------------------")
original_levels = [0, 1, 2, 3, 4, 5]
for level in original_levels:
    remapped = 4 if level == 2 else level
    if remapped in {0, 1, 4}:
        binary = 0
    elif remapped in {3, 5}:
        binary = 1
    else:
        binary = "?"
    print(f"   {level} -> {remapped} -> {binary}")

print("\n4. Unique values in encoded features:")
print(f"   feature_0_5_A: {sorted(X_encoded['feature_0_5_A'].unique())}")
print(f"   feature_0_5_B: {sorted(X_encoded['feature_0_5_B'].unique())}")
print(f"   feature_other: {sorted(X_encoded['feature_other'].unique())}")

print("\n5. Expected behavior:")
print("   ✓ 0-5 features should only have values {0, 1} (binary)")
print("   ✓ Non-0-5 features should pass through unchanged")
print("   ✓ Original level 2 -> becomes 4 -> then maps to 0 (low group)")

# Validation checks
print("\n6. Validation:")
all_0_5_vals = set(X_encoded['feature_0_5_A'].unique()) | set(X_encoded['feature_0_5_B'].unique())
if all_0_5_vals == {0.0, 1.0}:
    print("   ✅ PASS: 0-5 features contain only {0, 1}")
else:
    print(f"   ❌ FAIL: 0-5 features contain {all_0_5_vals}, expected {{0.0, 1.0}}")

if set(X_encoded['feature_other'].unique()) == {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0}:
    print("   ✅ PASS: Non-0-5 feature passed through unchanged")
else:
    print("   ❌ FAIL: Non-0-5 feature was modified")

print("\n7. Feature count:")
print(f"   Total features: {len(feature_names)}")
print(f"   (3 numeric features from test data)")

print("\n" + "=" * 70)
print("Type 4 Encoding Test Complete!")
print("=" * 70)
