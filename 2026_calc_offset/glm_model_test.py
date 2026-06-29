"""
GLM Model Test Script
======================
This script loads a GLM model configuration from JSON, creates a dummy dataset,
and tests the model input/output functionality.

Model Type: Generalized Linear Model (Gamma family with log link)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_glm_model(json_path: str) -> Dict[str, Any]:
    """
    Load GLM model configuration from a JSON file.
    
    Parameters:
    -----------
    json_path : str
        Path to the JSON file containing GLM model configuration
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing model configuration
    """
    with open(json_path, 'r') as f:
        model_config = json.load(f)
    
    print(f"✓ Loaded model: {model_config['model_name']}")
    print(f"  Family: {model_config['family']}")
    print(f"  Link: {model_config['link']}")
    print(f"  Intercept: {model_config['intercept']:.6f}")
    print(f"  Number of coefficients: {len(model_config['coefficients'])}")
    print(f"  Original training observations: {model_config['n_obs']:,}")
    
    return model_config


def parse_formula(formula: str) -> Tuple[str, List[str]]:
    """
    Parse the GLM formula to extract target and predictor variables.
    
    Parameters:
    -----------
    formula : str
        Formula string in format "target ~ var1 + var2 + ..."
        
    Returns:
    --------
    Tuple[str, List[str]]
        Target variable name and list of predictor names
    """
    parts = formula.split('~')
    target = parts[0].strip()
    predictors = [p.strip() for p in parts[1].split('+')]
    return target, predictors


def extract_variable_info(predictors: List[str]) -> Dict[str, List[str]]:
    """
    Categorize predictors into continuous and categorical variables.
    
    Parameters:
    -----------
    predictors : List[str]
        List of predictor variable names
        
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary with 'continuous', 'states', and 'makes' keys
    """
    continuous = []
    states = []
    makes = []
    
    for p in predictors:
        if p.startswith('STATE_'):
            states.append(p.replace('STATE_', ''))
        elif p.startswith('MAKE_'):
            makes.append(p.replace('MAKE_', ''))
        else:
            continuous.append(p)
    
    return {
        'continuous': continuous,
        'states': states,
        'makes': makes
    }


def create_dummy_dataset(
    n_samples: int = 100,
    variable_info: Dict[str, List[str]] = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Create a dummy dataset for testing the GLM model.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    variable_info : Dict[str, List[str]]
        Dictionary containing variable categorization
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with dummy data for all predictor variables
    """
    np.random.seed(random_seed)
    
    # Default variable info if not provided
    if variable_info is None:
        variable_info = {
            'continuous': ['ODOMETER', 'geo_pop_density_ntile', 'CALC_VEH_AGE'],
            'states': ['AR', 'AZ', 'CA', 'CT', 'FL', 'IL', 'LA', 'MA', 'MI', 'MN',
                      'MO', 'MS', 'MT', 'NC', 'ND', 'NH', 'NJ', 'NM', 'NV', 'NY',
                      'OH', 'OR', 'PA', 'SC', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA'],
            'makes': ['BUICK', 'CHEVROLET', 'CHRYSLER', 'DAEWOO', 'DODGE', 'FORD',
                     'HONDA', 'HYUNDAI', 'KIA', 'LAMBORGHINI', 'MAZDA', 'MERCURY',
                     'MITSUBISHI', 'NISSAN', 'PLYMOUTH', 'PONTIAC', 'SCION',
                     'SUBARU', 'TOYOTA', 'VOLKSWAGEN']
        }
    
    data = {}
    
    # Generate continuous variables with realistic ranges
    # ODOMETER: miles driven (typically 0 - 200,000)
    data['ODOMETER'] = np.random.uniform(5000, 150000, n_samples)
    
    # geo_pop_density_ntile: population density percentile (1-100)
    data['geo_pop_density_ntile'] = np.random.randint(1, 101, n_samples)
    
    # CALC_VEH_AGE: vehicle age in years (typically 0-20)
    data['CALC_VEH_AGE'] = np.random.uniform(0, 15, n_samples)
    
    # Generate state indicators (one state per vehicle)
    states = variable_info['states']
    # Randomly assign one state to each vehicle
    assigned_states = np.random.choice(states + ['OTHER'], n_samples)
    for state in states:
        data[f'STATE_{state}'] = (assigned_states == state).astype(bool)
    
    # Generate make indicators (one make per vehicle)
    makes = variable_info['makes']
    # Randomly assign one make to each vehicle
    assigned_makes = np.random.choice(makes + ['OTHER'], n_samples)
    for make in makes:
        data[f'MAKE_{make}'] = (assigned_makes == make).astype(bool)
    
    df = pd.DataFrame(data)
    
    print(f"\n✓ Created dummy dataset with {n_samples} samples")
    print(f"  Continuous variables: {len(variable_info['continuous'])}")
    print(f"  State indicators: {len(states)}")
    print(f"  Make indicators: {len(makes)}")
    
    return df


def predict_glm(
    data: pd.DataFrame,
    model_config: Dict[str, Any]
) -> pd.Series:
    """
    Apply GLM model to make predictions.
    
    For Gamma family with log link:
    - Linear predictor: η = β₀ + Σ(βᵢ * Xᵢ)
    - Link function: log(μ) = η
    - Prediction: μ = exp(η)
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data with predictor variables
    model_config : Dict[str, Any]
        Model configuration with intercept and coefficients
        
    Returns:
    --------
    pd.Series
        Predicted values (dep_factor)
    """
    intercept = model_config['intercept']
    coefficients = model_config['coefficients']
    rebase_factor = model_config.get('rebase_factor', 1.0)
    link_function = model_config.get('link', 'log')
    
    # Calculate linear predictor (η)
    linear_predictor = np.full(len(data), intercept)
    
    # Add contribution from each coefficient
    for coef_name, coef_value in coefficients.items():
        # Parse coefficient name to get variable name
        # Format: "STATE_AR[T.True]" -> "STATE_AR"
        # or "ODOMETER" -> "ODOMETER"
        
        if '[T.True]' in coef_name:
            # Boolean indicator variable
            var_name = coef_name.replace('[T.True]', '')
            if var_name in data.columns:
                linear_predictor += coef_value * data[var_name].astype(float)
        else:
            # Continuous variable
            if coef_name in data.columns:
                linear_predictor += coef_value * data[coef_name]
    
    # Apply inverse link function to get predictions
    if link_function == 'log':
        predictions = np.exp(linear_predictor)
    elif link_function == 'identity':
        predictions = linear_predictor
    elif link_function == 'inverse':
        predictions = 1.0 / linear_predictor
    else:
        raise ValueError(f"Unknown link function: {link_function}")
    
    # Apply rebase factor
    predictions = predictions * rebase_factor
    
    return pd.Series(predictions, name='predicted_dep_factor')


def validate_predictions(
    predictions: pd.Series,
    model_config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Validate predictions and compute summary statistics.
    
    Parameters:
    -----------
    predictions : pd.Series
        Model predictions
    model_config : Dict[str, Any]
        Model configuration
        
    Returns:
    --------
    Dict[str, float]
        Summary statistics of predictions
    """
    stats = {
        'mean': predictions.mean(),
        'std': predictions.std(),
        'min': predictions.min(),
        'max': predictions.max(),
        'median': predictions.median()
    }
    
    print("\n✓ Prediction Statistics:")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std:  {stats['std']:.6f}")
    print(f"  Min:  {stats['min']:.6f}")
    print(f"  Max:  {stats['max']:.6f}")
    print(f"  Median: {stats['median']:.6f}")
    
    return stats


def save_results(
    data: pd.DataFrame,
    predictions: pd.Series,
    output_path: str = 'output/test_results.csv'
) -> None:
    """
    Save test results to a CSV file.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    predictions : pd.Series
        Model predictions
    output_path : str
        Path to save results
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine data and predictions
    results = data.copy()
    results['predicted_dep_factor'] = predictions
    
    # Save to CSV
    results.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")


def main():
    """Main function to run the GLM model test."""
    print("=" * 60)
    print("GLM Model Test - Basic Economy Car")
    print("=" * 60)
    
    # 1. Load the GLM model configuration
    json_path = 'JSON/Basic_Economy_Car_GLM.json'
    print(f"\n[1] Loading model from: {json_path}")
    model_config = load_glm_model(json_path)
    
    # 2. Parse the formula
    print(f"\n[2] Parsing formula...")
    target, predictors = parse_formula(model_config['formula'])
    print(f"  Target variable: {target}")
    print(f"  Number of predictors: {len(predictors)}")
    
    # 3. Extract variable information
    variable_info = extract_variable_info(predictors)
    
    # 4. Create dummy dataset
    print(f"\n[3] Creating dummy dataset...")
    n_samples = 100
    dummy_data = create_dummy_dataset(
        n_samples=n_samples,
        random_seed=42
    )
    
    # 5. Display sample data
    print("\n[4] Sample data (first 5 rows):")
    print("-" * 60)
    display_cols = ['ODOMETER', 'geo_pop_density_ntile', 'CALC_VEH_AGE']
    # Find one True state and make for display
    state_cols = [c for c in dummy_data.columns if c.startswith('STATE_')]
    make_cols = [c for c in dummy_data.columns if c.startswith('MAKE_')]
    
    sample_display = dummy_data[display_cols].head()
    print(sample_display.to_string())
    
    # Show which state/make is True for first 5 rows
    print("\nState assignments (first 5 rows):")
    for i in range(min(5, len(dummy_data))):
        state_true = [c for c in state_cols if dummy_data.loc[i, c]]
        make_true = [c for c in make_cols if dummy_data.loc[i, c]]
        state_str = state_true[0] if state_true else "OTHER"
        make_str = make_true[0] if make_true else "OTHER"
        print(f"  Row {i}: {state_str}, {make_str}")
    
    # 6. Make predictions
    print(f"\n[5] Making predictions with GLM model...")
    predictions = predict_glm(dummy_data, model_config)
    
    # 7. Validate predictions
    print(f"\n[6] Validating predictions...")
    stats = validate_predictions(predictions, model_config)
    
    # 8. Display sample predictions
    print("\n[7] Sample predictions (first 10 rows):")
    print("-" * 60)
    results_display = pd.DataFrame({
        'ODOMETER': dummy_data['ODOMETER'].head(10),
        'VEH_AGE': dummy_data['CALC_VEH_AGE'].head(10),
        'PREDICTED': predictions.head(10)
    })
    print(results_display.to_string(index=False, float_format='%.4f'))
    
    # 9. Save results
    print(f"\n[8] Saving results...")
    save_results(dummy_data, predictions, 'output/test_results.csv')
    
    # 10. Summary
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  - Model: {model_config['model_name']}")
    print(f"  - Family: {model_config['family']} with {model_config['link']} link")
    print(f"  - Test samples: {n_samples}")
    print(f"  - Prediction range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  - Output file: output/test_results.csv")
    
    return dummy_data, predictions


if __name__ == '__main__':
    data, predictions = main()
