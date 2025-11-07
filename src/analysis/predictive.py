"""
Predictive Modeling Module

Builds regression models to predict resolution time and identify important features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline


class PredictiveModeling:
    """Predictive models for resolution time"""

    def __init__(self, data_path: str = "data/processed/forum_data.csv"):
        """
        Initialize with processed data

        Args:
            data_path: Path to processed CSV file
        """
        self.df = pd.read_csv(data_path)
        self.output_dir = Path("data/models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = Path("data/processed/figures")
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for modeling"""
        print("\n=== Preparing Data ===")

        # Filter to resolved topics only
        self.df_resolved = self.df[self.df["time_to_resolution_hours"].notna()].copy()
        print(f"Total resolved topics: {len(self.df_resolved)}")

        # Define features
        self.feature_cols = [
            "is_cuda_related",
            "has_code_block",
            "code_block_count",
            "question_length",
            "has_error_trace",
            "views",
            "reply_count",
            "hour_of_day",
        ]

        # Add category dummies
        self.df_resolved = pd.get_dummies(
            self.df_resolved, columns=["category_id"], prefix="cat", drop_first=True
        )
        category_cols = [col for col in self.df_resolved.columns if col.startswith("cat_")]
        self.feature_cols.extend(category_cols)

        # Convert boolean to int
        bool_cols = ["is_cuda_related", "has_code_block", "has_error_trace"]
        for col in bool_cols:
            self.df_resolved[col] = self.df_resolved[col].astype(int)

        # Target variable
        self.target_col = "time_to_resolution_hours"

        # Remove outliers (resolution time > 99th percentile)
        threshold = self.df_resolved[self.target_col].quantile(0.99)
        original_len = len(self.df_resolved)
        self.df_resolved = self.df_resolved[self.df_resolved[self.target_col] <= threshold]
        print(f"Removed {original_len - len(self.df_resolved)} outliers (>99th percentile)")

        # Train/test split
        X = self.df_resolved[self.feature_cols]
        y = self.df_resolved[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")

    def simple_linear_regression(self) -> Dict:
        """
        Simple linear regression: time ~ is_cuda_related

        Returns:
            Model metrics dictionary
        """
        print("\n=== Simple Linear Regression ===")

        X_train = self.X_train[["is_cuda_related"]]
        X_test = self.X_test[["is_cuda_related"]]

        model = LinearRegression()
        model.fit(X_train, self.y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))

        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test MAE: {test_mae:.2f} hours")
        print(f"Test RMSE: {test_rmse:.2f} hours")
        print(f"\nCoefficient (CUDA effect): {model.coef_[0]:.2f} hours")
        print(f"Intercept: {model.intercept_:.2f} hours")

        return {
            "name": "Simple Linear Regression",
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
        }

    def multiple_linear_regression(self) -> Dict:
        """
        Multiple linear regression with all features

        Returns:
            Model metrics dictionary
        """
        print("\n=== Multiple Linear Regression ===")

        # Create pipeline with scaling
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("regressor", LinearRegression())]
        )

        pipeline.fit(self.X_train, self.y_train)

        # Predictions
        y_pred_train = pipeline.predict(self.X_train)
        y_pred_test = pipeline.predict(self.X_test)

        # Metrics
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))

        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test MAE: {test_mae:.2f} hours")
        print(f"Test RMSE: {test_rmse:.2f} hours")

        # Cross-validation
        cv_scores = cross_val_score(
            pipeline, self.X_train, self.y_train, cv=5, scoring="r2"
        )
        print(f"\n5-Fold CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Feature importance (coefficients)
        regressor = pipeline.named_steps["regressor"]
        coef_df = pd.DataFrame(
            {"feature": self.feature_cols, "coefficient": regressor.coef_}
        ).sort_values("coefficient", key=abs, ascending=False)

        print("\nTop 10 Most Important Features:")
        print(coef_df.head(10))

        # Save model
        with open(self.output_dir / "multiple_regression_model.pkl", "wb") as f:
            pickle.dump(pipeline, f)

        return {
            "name": "Multiple Linear Regression",
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

    def polynomial_regression(self) -> Dict:
        """
        Non-linear regression with interaction terms

        Returns:
            Model metrics dictionary
        """
        print("\n=== Polynomial Regression (with Interactions) ===")

        # Add interaction term: is_cuda * has_code_block
        X_train_poly = self.X_train.copy()
        X_test_poly = self.X_test.copy()

        X_train_poly["cuda_x_code"] = (
            X_train_poly["is_cuda_related"] * X_train_poly["has_code_block"]
        )
        X_test_poly["cuda_x_code"] = (
            X_test_poly["is_cuda_related"] * X_test_poly["has_code_block"]
        )

        # Add squared term for question length
        X_train_poly["question_length_sq"] = X_train_poly["question_length"] ** 2
        X_test_poly["question_length_sq"] = X_test_poly["question_length"] ** 2

        # Create pipeline
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("regressor", LinearRegression())]
        )

        pipeline.fit(X_train_poly, self.y_train)

        # Predictions
        y_pred_train = pipeline.predict(X_train_poly)
        y_pred_test = pipeline.predict(X_test_poly)

        # Metrics
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))

        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test MAE: {test_mae:.2f} hours")
        print(f"Test RMSE: {test_rmse:.2f} hours")

        return {
            "name": "Polynomial Regression",
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
        }

    def lasso_regression(self) -> Dict:
        """
        Lasso regression for feature selection

        Returns:
            Model metrics dictionary
        """
        print("\n=== Lasso Regression (L1 Regularization) ===")

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", LassoCV(cv=5, random_state=42, max_iter=10000)),
            ]
        )

        pipeline.fit(self.X_train, self.y_train)

        # Predictions
        y_pred_test = pipeline.predict(self.X_test)

        # Metrics
        test_r2 = r2_score(self.y_test, y_pred_test)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))

        print(f"Test R²: {test_r2:.4f}")
        print(f"Test MAE: {test_mae:.2f} hours")
        print(f"Test RMSE: {test_rmse:.2f} hours")

        # Selected features (non-zero coefficients)
        regressor = pipeline.named_steps["regressor"]
        selected_features = [
            feat for feat, coef in zip(self.feature_cols, regressor.coef_) if abs(coef) > 0.01
        ]
        print(f"\nSelected {len(selected_features)} features (out of {len(self.feature_cols)})")
        print("Selected features:", selected_features[:10])

        return {
            "name": "Lasso Regression",
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "n_features": len(selected_features),
        }

    def plot_predictions(self, model_name: str, y_pred: np.ndarray):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(8, 8))
        plt.scatter(self.y_test, y_pred, alpha=0.5, edgecolors="black", linewidth=0.5)
        plt.plot([0, self.y_test.max()], [0, self.y_test.max()], "r--", lw=2)
        plt.xlabel("Actual Resolution Time (hours)")
        plt.ylabel("Predicted Resolution Time (hours)")
        plt.title(f"{model_name}: Actual vs Predicted")
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"{model_name.lower().replace(' ', '_')}_predictions.png", dpi=300)
        plt.close()

    def compare_models(self, results: list):
        """Compare all models"""
        print("\n=== Model Comparison ===")

        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))

        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # R² comparison
        axes[0].bar(
            range(len(df_results)),
            df_results["test_r2"],
            color="skyblue",
            edgecolor="black",
        )
        axes[0].set_xticks(range(len(df_results)))
        axes[0].set_xticklabels(df_results["name"], rotation=45, ha="right")
        axes[0].set_ylabel("Test R²")
        axes[0].set_title("Model Comparison: R²")
        axes[0].set_ylim(0, 1)

        # MAE comparison
        axes[1].bar(
            range(len(df_results)),
            df_results["test_mae"],
            color="coral",
            edgecolor="black",
        )
        axes[1].set_xticks(range(len(df_results)))
        axes[1].set_xticklabels(df_results["name"], rotation=45, ha="right")
        axes[1].set_ylabel("Test MAE (hours)")
        axes[1].set_title("Model Comparison: MAE")

        plt.tight_layout()
        plt.savefig(self.figures_dir / "model_comparison.png", dpi=300)
        plt.close()

    def run_all_models(self):
        """Run all predictive models"""
        print("\n" + "=" * 60)
        print("PREDICTIVE MODELING")
        print("=" * 60)

        results = []

        # Run models
        results.append(self.simple_linear_regression())
        results.append(self.multiple_linear_regression())
        results.append(self.polynomial_regression())
        results.append(self.lasso_regression())

        # Compare models
        self.compare_models(results)

        print("\n" + "=" * 60)
        print("PREDICTIVE MODELING COMPLETE")
        print("=" * 60)


def main():
    """Main execution function"""
    modeling = PredictiveModeling()
    modeling.run_all_models()


if __name__ == "__main__":
    main()
