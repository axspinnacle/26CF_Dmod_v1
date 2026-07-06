import json
import numpy as np
import pandas as pd


def load_glm_model(json_path):
    with open(json_path, "r") as f:
        model_config = json.load(f)
    return model_config


def parse_formula(formula):
    parts = formula.split("~")
    target = parts[0].strip()
    predictors = [p.strip() for p in parts[1].split("+")]
    return target, predictors


def predict_glm(data, model_config):
    intercept = model_config["intercept"]
    coefficients = model_config["coefficients"]
    rebase_factor = model_config.get("rebase_factor", 1.0)
    link_function = model_config.get("link", "log")

    linear_predictor = np.full(len(data), intercept)

    for coef_name, coef_value in coefficients.items():
        if "[T.True]" in coef_name:
            var_name = coef_name.replace("[T.True]", "")
            if var_name in data.columns:
                linear_predictor += coef_value * data[var_name].astype(float)
        else:
            if coef_name in data.columns:
                linear_predictor += coef_value * data[coef_name]

    if link_function == "log":
        predictions = np.exp(linear_predictor)
    elif link_function == "identity":
        predictions = linear_predictor
    elif link_function == "inverse":
        predictions = 1.0 / linear_predictor
    else:
        raise ValueError(f"Unknown link function: {link_function}")

    predictions = predictions * rebase_factor

    return pd.Series(predictions, name="predicted_dep_factor")
