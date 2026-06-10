"""
AmEx Customer Churn Prediction — Main Training Pipeline
Run this script to train all models end-to-end.

Usage:
    python train.py
    python train.py --data data/amex_churn_dataset.csv --smote --cv 5
"""

import argparse
import os
import sys
import pickle
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.preprocessor import AmExPreprocessor, FeatureEngineer
from models.trainer import train_all_models, evaluate_model, cross_validate_model
from models.explainability import ShapExplainer

MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description='AmEx Churn Training Pipeline')
    p.add_argument('--data', default='data/amex_churn_dataset.csv')
    p.add_argument('--test-size', type=float, default=0.20)
    p.add_argument('--smote', action='store_true', default=True)
    p.add_argument('--cv', type=int, default=5)
    p.add_argument('--shap', action='store_true', default=True)
    p.add_argument('--no-smote', dest='smote', action='store_false')
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*60}")
    print(f"  🏦 AmEx Customer Churn Prediction — Training Pipeline")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ── Load Data ──────────────────────────────────────────────────
    print(f"📂 Loading dataset from: {args.data}")
    df = pd.read_csv(args.data)
    print(f"   Shape: {df.shape} | Churn Rate: {df['Churn'].mean()*100:.2f}%")

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # ── Train/Test Split ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
    print(f"\n📊 Split — Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Feature Engineering ────────────────────────────────────────
    print("\n⚙️  Feature Engineering...")
    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    X_test_fe = fe.transform(X_test)
    print(f"   Features after engineering: {X_train_fe.shape[1]}")

    # ── Preprocessing ──────────────────────────────────────────────
    print("\n🔧 Preprocessing...")
    prep = AmExPreprocessor(scaler_type='robust')
    X_train_proc = prep.fit_transform(X_train_fe)
    X_test_proc = prep.transform(X_test_fe)
    print(f"   Final feature matrix: {X_train_proc.shape}")

    # Save preprocessors
    with open(os.path.join(MODELS_DIR, 'feature_engineer.pkl'), 'wb') as f:
        pickle.dump(fe, f)
    with open(os.path.join(MODELS_DIR, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(prep, f)
    print("   💾 Preprocessors saved.")

    # ── Train Models ───────────────────────────────────────────────
    print(f"\n🤖 Training Models (SMOTE={'ON' if args.smote else 'OFF'})...")
    trained_models, results = train_all_models(
        X_train_proc, X_test_proc,
        y_train.values, y_test.values,
        use_smote=args.smote
    )

    # ── Cross-Validation ───────────────────────────────────────────
    if args.cv > 1:
        print(f"\n🔄 Running {args.cv}-fold Cross-Validation on best model...")
        best_model_name = max(
            results, key=lambda k: results[k]['metrics']['roc_auc']
        )
        best_model = trained_models[best_model_name]
        cv_scores = cross_validate_model(
            best_model, X_train_proc, y_train.values,
            cv=args.cv, model_name=best_model_name
        )

    # ── SHAP Explainability ────────────────────────────────────────
    if args.shap:
        print("\n🔍 Computing SHAP Explanations...")
        try:
            from models.explainability import ShapExplainer
            best_name = max(results, key=lambda k: results[k]['metrics']['roc_auc'])
            # Use XGBoost for SHAP (best support)
            xgb_model = trained_models.get('XGBoost', trained_models[best_name])

            explainer = ShapExplainer(
                xgb_model,
                model_name='XGBoost',
                feature_names=list(X_test_proc.columns)
            )
            explainer.fit(X_train_proc, model_type='tree')
            # Compute on a sample for speed
            sample_size = min(2000, len(X_test_proc))
            X_sample = X_test_proc.sample(sample_size, random_state=42)
            explainer.compute_shap_values(X_sample)
            top_features = explainer.global_feature_importance(save=True)
            explainer.summary_plot(save=True)
            explainer.local_explanation(0, X_sample, save=True)

            # Save top feature list
            top_feat_path = os.path.join(MODELS_DIR, 'top_features.json')
            with open(top_feat_path, 'w') as f:
                json.dump(top_features.index.tolist(), f, indent=2)
            print(f"   💾 Top features saved → {top_feat_path}")

        except Exception as e:
            print(f"   ⚠️ SHAP failed: {e}")

    # ── Final Summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  📈 FINAL MODEL COMPARISON")
    print(f"{'='*60}")
    header = f"{'Model':<25} {'AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}"
    print(header)
    print('-' * 62)
    for name, res in results.items():
        m = res['metrics']
        print(f"{name:<25} {m['roc_auc']:>8.4f} {m['f1']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}")

    print(f"\n✅ Training complete. Models saved to: {MODELS_DIR}/")
    print(f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == '__main__':
    main()
