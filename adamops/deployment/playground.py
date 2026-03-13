"""
AdamOps Model Playground

Launch an interactive Streamlit UI to test, visualize, and explore ML models.
One-liner:
    >>> from adamops.deployment.playground import launch_playground
    >>> launch_playground(model, X_test, y_test)
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import joblib

from adamops.utils.logging import get_logger
from adamops.utils.helpers import infer_task_type, ensure_dir

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App Template  (written to /tmp and run via `streamlit run`)
# ─────────────────────────────────────────────────────────────────────────────

_STREAMLIT_APP = r'''
"""AdamOps Model Playground — auto-generated Streamlit app."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Load artifacts ────────────────────────────────────────────────────────
ARTIFACT_DIR = os.environ.get("ADAMOPS_PLAYGROUND_DIR", "")
model      = joblib.load(os.path.join(ARTIFACT_DIR, "model.joblib"))
X_test     = joblib.load(os.path.join(ARTIFACT_DIR, "X_test.joblib"))
y_test     = joblib.load(os.path.join(ARTIFACT_DIR, "y_test.joblib"))
meta       = json.load(open(os.path.join(ARTIFACT_DIR, "meta.json")))

task_type      = meta["task_type"]
feature_names  = meta["feature_names"]
model_name     = meta["model_name"]

# Convert to DataFrames / arrays
if isinstance(X_test, pd.DataFrame):
    _X_arr = X_test.values
    _X_df  = X_test
else:
    _X_arr = np.array(X_test)
    _X_df  = pd.DataFrame(_X_arr, columns=feature_names)

_y_arr = np.array(y_test)

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"AdamOps Playground — {model_name}",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #111827; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ AdamOps Playground")
    st.caption(f"Model: **{model_name}**")
    st.caption(f"Task: **{task_type}**")
    st.caption(f"Features: **{len(feature_names)}**")
    st.caption(f"Test samples: **{len(_y_arr)}**")
    st.divider()
    st.caption("Built with [AdamOps](https://github.com/adamops)")

# ── Tabs ─────────────────────────────────────────────────────────────────
tab_metrics, tab_predict, tab_viz, tab_explore, tab_batch = st.tabs([
    "📊 Metrics", "🔮 Live Prediction", "📈 Visualizations",
    "📋 Data Explorer", "📥 Batch Prediction"
])

# ═════════════════════════════════════════════════════════════════════════
# TAB 1 — Metrics
# ═════════════════════════════════════════════════════════════════════════
with tab_metrics:
    y_pred = model.predict(_X_arr)

    if task_type in ("classification", "multiclass"):
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                     f1_score, matthews_corrcoef, roc_auc_score)
        acc  = accuracy_score(_y_arr, y_pred)
        prec = precision_score(_y_arr, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(_y_arr, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(_y_arr, y_pred, average="weighted", zero_division=0)

        cols = st.columns(4)
        for col, (name, val) in zip(cols, [
            ("Accuracy", acc), ("Precision", prec),
            ("Recall", rec), ("F1 Score", f1)
        ]):
            col.markdown(f"""
            <div class="metric-card">
                <div class="value">{val:.4f}</div>
                <div class="label">{name}</div>
            </div>
            """, unsafe_allow_html=True)

        # Extra row
        extras = {}
        if len(np.unique(_y_arr)) == 2:
            extras["MCC"] = matthews_corrcoef(_y_arr, y_pred)
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(_X_arr)
                    if y_prob.ndim == 2:
                        y_prob_1 = y_prob[:, 1]
                    else:
                        y_prob_1 = y_prob
                    extras["ROC-AUC"] = roc_auc_score(_y_arr, y_prob_1)
                except Exception:
                    pass
        if extras:
            st.markdown("")
            cols2 = st.columns(len(extras))
            for col, (name, val) in zip(cols2, extras.items()):
                col.markdown(f"""
                <div class="metric-card">
                    <div class="value">{val:.4f}</div>
                    <div class="label">{name}</div>
                </div>
                """, unsafe_allow_html=True)

    else:  # regression
        from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                                     r2_score, explained_variance_score)
        mse  = mean_squared_error(_y_arr, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(_y_arr, y_pred)
        r2   = r2_score(_y_arr, y_pred)

        cols = st.columns(4)
        for col, (name, val) in zip(cols, [
            ("RMSE", rmse), ("MAE", mae),
            ("R²", r2), ("Explained Var", explained_variance_score(_y_arr, y_pred))
        ]):
            col.markdown(f"""
            <div class="metric-card">
                <div class="value">{val:.4f}</div>
                <div class="label">{name}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    st.caption("Predictions computed on the full test set.")

# ═════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Prediction
# ═════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("🔮 Live Prediction")
    st.caption("Adjust feature values and see the model's prediction in real time.")

    input_cols = st.columns(min(3, len(feature_names)))
    input_values = {}

    for i, fname in enumerate(feature_names):
        col = input_cols[i % len(input_cols)]
        series = _X_df[fname] if fname in _X_df.columns else pd.Series(_X_arr[:, i])

        if pd.api.types.is_numeric_dtype(series):
            vmin = float(series.min())
            vmax = float(series.max())
            vmean = float(series.mean())
            step = (vmax - vmin) / 100.0 if vmax != vmin else 0.01
            input_values[fname] = col.slider(
                fname, min_value=vmin, max_value=vmax, value=vmean, step=step
            )
        else:
            options = sorted(series.dropna().unique().tolist())
            input_values[fname] = col.selectbox(fname, options)

    if st.button("🔮 Predict", type="primary", use_container_width=True):
        input_arr = np.array([[input_values[f] for f in feature_names]])
        pred = model.predict(input_arr)

        st.success(f"**Prediction: {pred[0]}**")

        if hasattr(model, "predict_proba") and task_type in ("classification", "multiclass"):
            try:
                prob = model.predict_proba(input_arr)[0]
                prob_df = pd.DataFrame({
                    "Class": [str(c) for c in (model.classes_ if hasattr(model, 'classes_') else range(len(prob)))],
                    "Probability": prob
                })
                st.bar_chart(prob_df.set_index("Class"))
            except Exception:
                pass

# ═════════════════════════════════════════════════════════════════════════
# TAB 3 — Visualizations
# ═════════════════════════════════════════════════════════════════════════
with tab_viz:
    st.subheader("📈 Visualizations")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_pred_viz = model.predict(_X_arr)

    if task_type in ("classification", "multiclass"):
        viz_option = st.selectbox("Select visualization", [
            "Confusion Matrix", "Feature Importance", "Class Distribution"
        ])

        if viz_option == "Confusion Matrix":
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            cm = confusion_matrix(_y_arr, y_pred_viz)
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(ax=ax, cmap="Blues")
            ax.set_title("Confusion Matrix", color="white", fontsize=14)
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            st.pyplot(fig)

        elif viz_option == "Feature Importance":
            importance = None
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance = np.abs(model.coef_).flatten()

            if importance is not None:
                imp_df = pd.DataFrame({
                    "Feature": feature_names[:len(importance)],
                    "Importance": importance
                }).sort_values("Importance", ascending=True).tail(20)
                st.bar_chart(imp_df.set_index("Feature"))
            else:
                st.info("This model type does not expose feature importances.")

        elif viz_option == "Class Distribution":
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.patch.set_facecolor('#0f172a')
            for ax in axes:
                ax.set_facecolor('#0f172a')
                ax.tick_params(colors="white")

            unique_true, counts_true = np.unique(_y_arr, return_counts=True)
            unique_pred, counts_pred = np.unique(y_pred_viz, return_counts=True)

            axes[0].bar(unique_true.astype(str), counts_true, color="#6366f1")
            axes[0].set_title("Actual", color="white")
            axes[1].bar(unique_pred.astype(str), counts_pred, color="#22c55e")
            axes[1].set_title("Predicted", color="white")
            plt.tight_layout()
            st.pyplot(fig)

    else:  # regression
        viz_option = st.selectbox("Select visualization", [
            "Actual vs Predicted", "Residuals", "Feature Importance"
        ])

        if viz_option == "Actual vs Predicted":
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            ax.scatter(_y_arr, y_pred_viz, alpha=0.5, c="#6366f1", s=20)
            lims = [min(_y_arr.min(), y_pred_viz.min()),
                    max(_y_arr.max(), y_pred_viz.max())]
            ax.plot(lims, lims, 'r--', alpha=0.7, label="Perfect")
            ax.set_xlabel("Actual", color="white")
            ax.set_ylabel("Predicted", color="white")
            ax.set_title("Actual vs Predicted", color="white")
            ax.tick_params(colors="white")
            ax.legend()
            st.pyplot(fig)

        elif viz_option == "Residuals":
            residuals = _y_arr - y_pred_viz
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor('#0f172a')
            for ax in axes:
                ax.set_facecolor('#0f172a')
                ax.tick_params(colors="white")

            axes[0].scatter(y_pred_viz, residuals, alpha=0.5, c="#6366f1", s=20)
            axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
            axes[0].set_xlabel("Predicted", color="white")
            axes[0].set_ylabel("Residuals", color="white")
            axes[0].set_title("Residuals vs Predicted", color="white")

            axes[1].hist(residuals, bins=30, color="#6366f1", edgecolor="#1e293b", alpha=0.8)
            axes[1].set_xlabel("Residuals", color="white")
            axes[1].set_ylabel("Frequency", color="white")
            axes[1].set_title("Residual Distribution", color="white")
            plt.tight_layout()
            st.pyplot(fig)

        elif viz_option == "Feature Importance":
            importance = None
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance = np.abs(model.coef_).flatten()

            if importance is not None:
                imp_df = pd.DataFrame({
                    "Feature": feature_names[:len(importance)],
                    "Importance": importance
                }).sort_values("Importance", ascending=True).tail(20)
                st.bar_chart(imp_df.set_index("Feature"))
            else:
                st.info("This model type does not expose feature importances.")

# ═════════════════════════════════════════════════════════════════════════
# TAB 4 — Data Explorer
# ═════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.subheader("📋 Data Explorer")

    sub_tab = st.radio("View", ["Data Preview", "Statistics", "Distributions"],
                       horizontal=True)

    if sub_tab == "Data Preview":
        n_rows = st.slider("Rows to display", 5, min(100, len(_X_df)), 20)
        st.dataframe(_X_df.head(n_rows), use_container_width=True)
        st.caption(f"Shape: {_X_df.shape[0]} rows × {_X_df.shape[1]} columns")

    elif sub_tab == "Statistics":
        st.dataframe(_X_df.describe().T, use_container_width=True)
        missing = _X_df.isnull().sum()
        if missing.any():
            st.warning("Missing values detected:")
            st.dataframe(missing[missing > 0], use_container_width=True)
        else:
            st.success("No missing values in the dataset.")

    elif sub_tab == "Distributions":
        col_to_plot = st.selectbox("Select column",
                                   [c for c in _X_df.columns if pd.api.types.is_numeric_dtype(_X_df[c])])
        if col_to_plot:
            st.bar_chart(_X_df[col_to_plot].dropna().value_counts().sort_index().head(50))

# ═════════════════════════════════════════════════════════════════════════
# TAB 5 — Batch Prediction
# ═════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("📥 Batch Prediction")
    st.caption("Upload a CSV file to get predictions for all rows.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.dataframe(batch_df.head(10), use_container_width=True)
            st.caption(f"Uploaded: {batch_df.shape[0]} rows × {batch_df.shape[1]} columns")

            if st.button("🚀 Predict All", type="primary", use_container_width=True):
                # Try to align columns with training features
                missing_cols = [c for c in feature_names if c not in batch_df.columns]
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                else:
                    X_batch = batch_df[feature_names].values
                    preds = model.predict(X_batch)
                    batch_df["prediction"] = preds

                    if hasattr(model, "predict_proba") and task_type in ("classification", "multiclass"):
                        try:
                            probs = model.predict_proba(X_batch)
                            for i, cls in enumerate(model.classes_ if hasattr(model, 'classes_') else range(probs.shape[1])):
                                batch_df[f"prob_{cls}"] = probs[:, i]
                        except Exception:
                            pass

                    st.dataframe(batch_df, use_container_width=True)

                    csv_data = batch_df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download Results CSV",
                        csv_data,
                        "predictions.csv",
                        "text/csv",
                        use_container_width=True,
                    )
        except Exception as e:
            st.error(f"Error reading file: {e}")
'''


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def launch_playground(
    model: Any,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    model_name: str = "Model",
    feature_names: Optional[List[str]] = None,
    task_type: Optional[str] = None,
    port: int = 8501,
    open_browser: bool = True,
) -> None:
    """
    Launch an interactive Streamlit playground for a trained model.

    Args:
        model: A trained scikit-learn compatible model.
        X_test: Test features (DataFrame or ndarray).
        y_test: Test labels (Series or ndarray).
        model_name: Display name for the model.
        feature_names: Feature names (auto-detected from DataFrame columns).
        task_type: 'classification', 'regression', or None (auto-detect).
        port: Streamlit server port.
        open_browser: Whether to open the browser automatically.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = load_iris(return_X_y=True, as_frame=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
        >>> model = RandomForestClassifier().fit(X_train, y_train)
        >>> launch_playground(model, X_test, y_test, model_name="Iris Classifier")
    """
    # Auto-detect feature names
    if feature_names is None:
        if isinstance(X_test, pd.DataFrame):
            feature_names = X_test.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

    # Auto-detect task type
    if task_type is None:
        task_type = infer_task_type(y_test)

    # Create temp directory for artifacts
    artifact_dir = os.path.join(tempfile.gettempdir(), "adamops_playground")
    ensure_dir(artifact_dir)

    # Save artifacts
    joblib.dump(model, os.path.join(artifact_dir, "model.joblib"))
    joblib.dump(X_test, os.path.join(artifact_dir, "X_test.joblib"))
    joblib.dump(y_test, os.path.join(artifact_dir, "y_test.joblib"))

    meta = {
        "task_type": task_type,
        "feature_names": feature_names,
        "model_name": model_name,
    }
    with open(os.path.join(artifact_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    # Write Streamlit app script
    app_path = os.path.join(artifact_dir, "playground_app.py")
    with open(app_path, "w", encoding="utf-8") as f:
        f.write(_STREAMLIT_APP)

    logger.info(f"Playground artifacts saved to {artifact_dir}")

    # Launch Streamlit
    print(f"\n{'='*60}")
    print(f"  ⚡ AdamOps Model Playground")
    print(f"  Model: {model_name}  |  Task: {task_type}")
    print(f"  Features: {len(feature_names)}  |  Test samples: {len(y_test)}")
    print(f"  Open in browser: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env["ADAMOPS_PLAYGROUND_DIR"] = artifact_dir

    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", str(port),
        "--server.headless", str(not open_browser).lower(),
        "--browser.gatherUsageStats", "false",
        "--theme.base", "dark",
        "--theme.primaryColor", "#6366f1",
        "--theme.backgroundColor", "#0f172a",
        "--theme.secondaryBackgroundColor", "#1e293b",
        "--theme.textColor", "#f1f5f9",
    ]

    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n\nShutting down AdamOps Playground...")
    except FileNotFoundError:
        logger.error(
            "Streamlit not found. Install with: pip install streamlit"
        )
        raise ImportError(
            "Streamlit is required for the Model Playground. "
            "Install with: pip install streamlit"
        )
