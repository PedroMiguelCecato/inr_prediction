import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class INRModelPlot:
    """
    Classe para visualização, diagnóstico e interpretação
    de modelos preditivos de INR.
    """
    # =========================
    # Predição vs Real
    # =========================
    @staticmethod
    def plot_inr_prediction(
        dates,
        y_true,
        y_pred,
        low_vals,
        high_vals,
        title="INR Real x Previsto"
    ):
        dates = np.asarray(dates)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if np.isscalar(low_vals):
            low_vals = np.full_like(y_pred, low_vals, dtype=float)
        if np.isscalar(high_vals):
            high_vals = np.full_like(y_pred, high_vals, dtype=float)

        plt.figure(figsize=(14, 6))
        plt.plot(dates, y_true, label="INR real", marker="o")
        plt.plot(dates, y_pred, label="INR previsto", marker="x")

        plt.fill_between(
            dates,
            low_vals,
            high_vals,
            color="green",
            alpha=0.15,
            label="Faixa alvo"
        )

        plt.xlabel("Data")
        plt.ylabel("INR")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # =========================
    # Resíduos
    # =========================
    @staticmethod
    def plot_residuals(
        dates,
        y_true,
        y_pred,
        title="Modelo"
    ):
        dates = np.asarray(dates)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        resid = y_true - y_pred

        # Série temporal dos resíduos
        plt.figure(figsize=(14, 4))
        plt.plot(dates, resid, marker="o")
        plt.axhline(0, color="k", linestyle="--")
        plt.title(f"Resíduos (real − previsto) — {title}")
        plt.xlabel("Data")
        plt.ylabel("Resíduo")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Distribuição dos resíduos
        plt.figure(figsize=(8, 4))
        sns.histplot(resid, bins=25, kde=True)
        plt.title(f"Distribuição dos Resíduos — {title}")
        plt.xlabel("Resíduo")
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.show()

    # =========================
    # XGBoost
    # =========================
    @staticmethod
    def plot_xgb_feature_importance(
        model,
        top_n=15,
        importance_type="gain",
        title=None
    ):
        booster = model.get_booster()
        importance = booster.get_score(importance_type=importance_type)

        if not importance:
            raise ValueError("O modelo XGBoost não retornou importâncias.")

        imp_df = (
            pd.DataFrame({
                "feature": list(importance.keys()),
                "value": list(importance.values())
            })
            .sort_values("value", ascending=True)
            .tail(top_n)
        )

        plt.figure(figsize=(20, 8))
        bars = plt.barh(imp_df["feature"], imp_df["value"])

        max_val = imp_df["value"].max()
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                va="center"
            )

        plt.xlabel(importance_type.capitalize())
        plt.title(title or f"Importância das Features (XGBoost — {importance_type.upper()})")
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    # =========================
    # LightGBM
    # =========================
    @staticmethod
    def plot_lgb_feature_importance(
        model,
        feature_names=None,
        top_n=20,
        importance_type="gain",
        title=None,
        figsize=(12, 8),
        show_values=True
    ):
        if importance_type not in ("gain", "split"):
            raise ValueError("importance_type deve ser 'gain' ou 'split'.")

        if hasattr(model, "booster_") and model.booster_ is not None:
            booster = model.booster_
        else:
            booster = model

        try:
            names = booster.feature_name()
        except Exception:
            names = feature_names

        try:
            vals = booster.feature_importance(importance_type=importance_type)
        except Exception:
            vals = booster.feature_importance()

        if names is None:
            names = [f"f{i}" for i in range(len(vals))]

        min_len = min(len(names), len(vals))
        names = names[:min_len]
        vals = vals[:min_len]

        imp_df = (
            pd.DataFrame({"feature": names, "importance": vals})
            .query("importance != 0")
            .sort_values("importance", ascending=True)
            .tail(top_n)
        )

        plt.figure(figsize=figsize)
        bars = plt.barh(imp_df["feature"], imp_df["importance"], alpha=0.9)

        if show_values and not imp_df.empty:
            max_val = imp_df["importance"].max()
            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width + max_val * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.4f}",
                    va="center",
                    fontsize=9
                )

        plt.xlabel(importance_type.capitalize())
        plt.title(title or f"Importância das Features (LightGBM — top {len(imp_df)})")
        plt.grid(axis="x", linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.show()

    # =========================
    # Random Forest
    # =========================
    @staticmethod
    def plot_rf_feature_importance(
        model,
        feature_names,
        top_n=20,
        title="Importância das Features — RandomForest",
        figsize=(12, 8),
        show_values=True
    ):
        importances = model.feature_importances_

        if len(importances) != len(feature_names):
            raise ValueError("feature_names não corresponde às importâncias do modelo.")

        imp_df = (
            pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            })
            .sort_values("importance", ascending=True)
            .tail(top_n)
        )

        plt.figure(figsize=figsize)
        bars = plt.barh(imp_df["feature"], imp_df["importance"], color="steelblue")

        if show_values:
            max_val = imp_df["importance"].max()
            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width + max_val * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.4f}",
                    va="center",
                    fontsize=9
                )

        plt.title(title)
        plt.xlabel("Importância")
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()