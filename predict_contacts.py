#!/usr/bin/env python3
"""
predict_contacts.py

Predict residue-residue interaction types for a protein structure.

Usage:
    python3 predict_contacts.py path/to/protein.cif --model logistic --output predictions.tsv
    python3 predict_contacts.py path/to/protein.cif --model random_forest --output predictions.tsv
    python3 predict_contacts.py path/to/protein.cif --model xgboost --output predictions.tsv
"""

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── PyTorch 2.6 Bypass Monkeypatch ──────────────────────────────────────────
# This automatically patches torch globally to ensure older weights load safely 
try:
    import torch
    _original_load = torch.load
    def _safe_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)
    torch.load = _safe_load
except ImportError:
    pass

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL_COLS = ['HBOND', 'VDW', 'IONIC', 'PIPISTACK', 'PICATION', 'SSBOND', 'PIHBOND']

PAIR_COLS = ['pdb_id', 's_ch', 's_resi', 's_ins', 's_resn',
             't_ch', 't_resi', 't_ins', 't_resn']

# FIX: Added 't_3di_state' which was missing from the original script's list
NUM_FEATURES = [
    's_rsa', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5',
    's_3di_state',
    't_rsa', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5',
    't_3di_state',
]

CAT_FEATURES = ['s_ss8', 's_3di_letter', 't_ss8', 't_3di_letter']

MODEL_PATHS = {
    'logistic':      'classification_ring/models/logistic_classifier/logistic_model.pkl',
    'random_forest': 'classification_ring/models/random_forest/random_forest_model.pkl',
    'xgboost':       'classification_ring/models/xgb_classifier/xgb_model.pkl',
}

# Best thresholds per label (from validation set tuning in the notebooks)
DEFAULT_THRESHOLDS = {
    'HBOND': 0.63, 'VDW': 0.56, 'IONIC': 0.17,
    'PIPISTACK': 0.33, 'PICATION': 0.08, 'SSBOND': 0.37, 'PIHBOND': 0.33
}

# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict residue interaction network contacts from a protein structure."
    )
    parser.add_argument("input_structure", type=str,
                        help="Path to a valid .pdb or .cif coordinate file.")
    parser.add_argument("--model", type=str, required=True,
                        choices=['logistic', 'random_forest', 'xgboost'],
                        help="Which trained model to use.")
    parser.add_argument("--config", type=str,
                        default="classification_ring/configuration.json",
                        help="Path to configuration JSON (default: classification_ring/configuration.json).")
    parser.add_argument("--output", type=str, default="prediction_output.tsv",
                        help="Output TSV file path (default: prediction_output.tsv).")
    return parser.parse_args()


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(input_file: str, config_path: str):
    """Run calc_features.py and calc_3di.py, return paths to output dirs."""
    feat_dir = "tmp_features"
    di3_dir  = "tmp_3di"
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(di3_dir,  exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(config_path))

    logger.info("Extracting structural features (DSSP, RSA, Atchley, Ramachandran)...")
    subprocess.run(
        [sys.executable,
         os.path.join(script_dir, "calc_features.py"),
         input_file, "-out_dir", feat_dir],
        check=True
    )

    logger.info("Extracting 3Di alphabet features...")
    subprocess.run(
        [sys.executable,
         os.path.join(script_dir, "calc_3di.py"),
         input_file, "-out_dir", di3_dir],
        check=True
    )

    return feat_dir, di3_dir


# ── Feature merging ───────────────────────────────────────────────────────────
def merge_features(feat_dir: str, di3_dir: str, pdb_id: str) -> pd.DataFrame:
    """
    Merge structural features (calc_features.py output) with
    3Di features (calc_3di.py output) into one DataFrame.
    """
    feat_path = os.path.join(feat_dir, f"{pdb_id}.tsv")
    di3_path  = os.path.join(di3_dir,  f"{pdb_id}.tsv")

    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Structural features file not found: {feat_path}")
    if not os.path.exists(di3_path):
        raise FileNotFoundError(f"3Di features file not found: {di3_path}")

    feat_df = pd.read_csv(feat_path, sep="\t")
    di3_df  = pd.read_csv(di3_path,  sep="\t")

    logger.info(f"Structural features: {feat_df.shape[0]} contacts")
    logger.info(f"3Di features: {di3_df.shape[0]} residues")

    # Rename 3Di columns for source residue merge
    di3_src = di3_df[['ch', 'resi', 'ins', 'resn', '3di_state', '3di_letter']].copy()
    di3_src.columns = ['s_ch', 's_resi', 's_ins', 's_resn', 's_3di_state', 's_3di_letter']

    # Rename 3Di columns for target residue merge
    di3_tgt = di3_df[['ch', 'resi', 'ins', 'resn', '3di_state', '3di_letter']].copy()
    di3_tgt.columns = ['t_ch', 't_resi', 't_ins', 't_resn', 't_3di_state', 't_3di_letter']

    # Merge on source residue
    merged = feat_df.merge(
        di3_src,
        on=['s_ch', 's_resi', 's_ins', 's_resn'],
        how='left'
    )

    # Merge on target residue
    merged = merged.merge(
        di3_tgt,
        on=['t_ch', 't_resi', 't_ins', 't_resn'],
        how='left'
    )

    logger.info(f"Merged feature matrix: {merged.shape[0]} contacts, {merged.shape[1]} columns")
    return merged


# ── Biological constraints ────────────────────────────────────────────────────
def apply_biological_constraints(proba_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    """Zero out probabilities that are biologically impossible."""
    s_resn = feat_df['s_resn'].str.upper()
    t_resn = feat_df['t_resn'].str.upper()

    acidic   = {"D", "E"}
    basic    = {"K", "R", "H"}
    aromatic = {"F", "Y", "W", "H"}
    cysteine = {"C"}

    ionic_ok    = (s_resn.isin(acidic) & t_resn.isin(basic)) | (t_resn.isin(acidic) & s_resn.isin(basic))
    pipi_ok     = s_resn.isin(aromatic) & t_resn.isin(aromatic)
    pication_ok = (s_resn.isin(aromatic) & t_resn.isin(basic)) | (t_resn.isin(aromatic) & s_resn.isin(basic))
    ssbond_ok   = s_resn.isin(cysteine) & t_resn.isin(cysteine)
    pihbond_ok  = s_resn.isin(aromatic) | t_resn.isin(aromatic)

    constrained = proba_df.copy()
    constrained.loc[~ionic_ok.values,    'IONIC']    = 0.0
    constrained.loc[~pipi_ok.values,     'PIPISTACK'] = 0.0
    constrained.loc[~pication_ok.values, 'PICATION'] = 0.0
    constrained.loc[~ssbond_ok.values,   'SSBOND']   = 0.0
    constrained.loc[~pihbond_ok.values,  'PIHBOND']  = 0.0

    return constrained


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    input_file = args.input_structure
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    pdb_id = Path(input_file).stem

    # 1. Extract features
    feat_dir, di3_dir = extract_features(input_file, args.config)

    # 2. Merge structural + 3Di features
    logger.info("Merging feature tables...")
    merged_df = merge_features(feat_dir, di3_dir, pdb_id)

    # 3. Build feature matrix (same columns as training)
    feature_cols = NUM_FEATURES + CAT_FEATURES
    missing = [c for c in feature_cols if c not in merged_df.columns]
    if missing:
        logger.warning(f"Missing feature columns (will be NaN): {missing}")
        for c in missing:
            merged_df[c] = np.nan

    X = merged_df[feature_cols].copy()

    # 4. Load model
    model_path = MODEL_PATHS[args.model]
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}\n"
                     f"Train the model first by running the corresponding notebook.")
        sys.exit(1)

    logger.info(f"Loading {args.model} model from {model_path}...")
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)

    # 5. Predict probabilities
    logger.info("Running predictions...")
    
    # FIX: Check if using xgboost to handle categorical mismatch/one-hot alignments
    if args.model == 'xgboost':
        # Safely convert categories and set pipeline parameters
        for est in pipeline.estimators_:
            if hasattr(est, 'set_params'):
                est.set_params(enable_categorical=True)
        
        # Manually one-hot encode text data for raw XGBoost compatibility
        X = pd.get_dummies(X)
        
        # Match features against original model booster structure
        try:
            expected_features = pipeline.estimators_[0].get_booster().feature_names
            if expected_features is not None:
                for col in expected_features:
                    if col not in X.columns:
                        X[col] = 0.0
                X = X[expected_features].astype(float)
        except Exception as e:
            logger.warning(f"Could not automatically match booster feature array shape: {e}")

    y_proba = pipeline.predict_proba(X)
    if isinstance(y_proba, list):
        y_scores = np.column_stack([p[:, 1] for p in y_proba])
    else:
        y_scores = y_proba[:, 1:]

    proba_df = pd.DataFrame(y_scores, columns=LABEL_COLS, index=merged_df.index)

    # 6. Apply biological constraints
    logger.info("Applying biological constraints...")
    proba_df = apply_biological_constraints(proba_df, merged_df)

    # 7. Apply thresholds → binary predictions
    thresholds = DEFAULT_THRESHOLDS
    pred_df = (proba_df >= pd.Series(thresholds)).astype(int)

    # 8. Build output table
    output_rows = []
    for idx in merged_df.index:
        row = merged_df.loc[idx]
        scores = proba_df.loc[idx]
        preds  = pred_df.loc[idx]

        predicted_types = [label for label in LABEL_COLS if preds[label] == 1]
        interaction_str = ','.join(predicted_types) if predicted_types else 'Unclassified'

        out = {
            's_ch':   row['s_ch'],
            's_resi': row['s_resi'],
            's_ins':  row['s_ins'],
            's_resn': row['s_resn'],
            't_ch':   row['t_ch'],
            't_resi': row['t_resi'],
            't_ins':  row['t_ins'],
            't_resn': row['t_resn'],
            'Interaction': interaction_str,
        }
        for label in LABEL_COLS:
            out[f'score_{label}'] = round(float(scores[label]), 4)

        output_rows.append(out)

    output_df = pd.DataFrame(output_rows)

    # 9. Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output_df.to_csv(args.output, sep='\t', index=False)
    logger.info(f"Predictions saved to: {args.output}")
    logger.info(f"Total contacts predicted: {len(output_df)}")

    # Summary
    for label in LABEL_COLS:
        n = pred_df[label].sum()
        logger.info(f"  {label}: {n} predicted contacts")

    # Cleanup temp dirs
    import shutil
    shutil.rmtree("tmp_features", ignore_errors=True)
    shutil.rmtree("tmp_3di",      ignore_errors=True)


if __name__ == "__main__":
    main()