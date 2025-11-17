"""
Streamlit Dashboard for CUDA Error Resolution Analysis

Interactive dashboard for predicting resolution times and visualizing causal effects.
"""

import streamlit as st
import pandas as pd
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="CUDA Error Resolution Analysis",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 CUDA Error Resolution Analysis Dashboard")
st.markdown("Interactive analysis of PyTorch forum questions with ML-based predictions.")

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Data Overview", "Predict Resolution Time", "Causal Analysis Results", "Fun Insights"],
)


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("data/models")


@st.cache_data
def load_data():
    path = DATA_DIR / "forum_data.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_causal_results():
    path = DATA_DIR / "causal_results.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


@st.cache_resource
def load_model():
    path = MODEL_DIR / "multiple_regression_model.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


df = load_data()

if df is None or df.empty:
    st.error("Data file not found. Please run data collection and processing first.")
    st.stop()

resolved_df = df[df["time_to_resolution_hours"].notna()].copy()
if not resolved_df.empty:
    resolved_df["code_density"] = resolved_df["code_block_count"] / (resolved_df["question_length"] + 1)
    resolved_df["engagement_score"] = (
        resolved_df["views"] + resolved_df["reply_count"]
    ) / (resolved_df["question_length"] + 1)


def _category_options(frame):
    if "category_id" not in frame:
        return []
    return sorted(frame["category_id"].dropna().astype(str).unique().tolist())


def _numeric_bounds(series, fallback_min: int, fallback_max: int) -> tuple[int, int]:
    if series is None:
        return fallback_min, fallback_max
    clean = series.dropna()
    if clean.empty:
        return fallback_min, fallback_max
    min_val = int(clean.min())
    max_val = int(clean.max())
    if min_val == max_val:
        max_val = min_val + 1
    return min_val, max_val


def _numeric_default(series, fallback: int) -> int:
    if series is None:
        return fallback
    clean = series.dropna()
    if clean.empty:
        return fallback
    return int(clean.median())


def _bool_default(series, fallback: bool) -> bool:
    if series is None:
        return fallback
    clean = series.dropna()
    if clean.empty:
        return fallback
    return bool(round(clean.mean()))


def _compute_derived_features(values: dict) -> dict:
    question_length = max(values.get("question_length", 0), 0)
    code_blocks = max(values.get("code_block_count", 0), 0)
    views = max(values.get("views", 0), 0)
    replies = max(values.get("reply_count", 0), 0)

    derived = {
        "code_density": code_blocks / (question_length + 1),
        "engagement_score": (views + replies) / (question_length + 1),
    }
    return derived


def _expected_feature_names(model):
    if model is None:
        return None
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    scaler = getattr(getattr(model, "named_steps", {}), "get", lambda _key: None)("scaler") if hasattr(model, "named_steps") else None
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    return None


def _format_float(value, decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"

if page == "Data Overview":
    st.header("Data Overview")

    category_options = _category_options(df)
    filter_cols = st.columns(3)
    selected_categories = filter_cols[0].multiselect(
        "Categories",
        options=category_options,
        default=category_options[: min(8, len(category_options))] if category_options else [],
    )
    cuda_filter = filter_cols[1].selectbox("CUDA Filter", ["All", "CUDA", "Non-CUDA"], index=0)
    view_mode = filter_cols[2].selectbox("Focus", ["All posts", "Resolved only"], index=1)

    working_df = resolved_df if view_mode == "Resolved only" else df.copy()

    if selected_categories:
        working_df = working_df[working_df["category_id"].astype(str).isin(selected_categories)]

    if cuda_filter != "All":
        is_cuda_flag = 1 if cuda_filter == "CUDA" else 0
        working_df = working_df[working_df["is_cuda_related"] == is_cuda_flag]

    filtered_resolved = working_df[working_df["time_to_resolution_hours"].notna()].copy()

    if filtered_resolved.empty:
        st.info("No rows match the current filters. Try broadening your selection.")
    else:
        comparison = filtered_resolved.groupby("is_cuda_related").agg({
            "time_to_resolution_hours": ["mean", "median"],
            "is_resolved": "mean",
            "views": "mean",
            "reply_count": "mean",
            "has_code_block": "mean",
        }).round(2)
        comparison.index = comparison.index.map({0: "Non-CUDA", 1: "CUDA"}).rename("Question Type")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filtered Topics", f"{len(working_df):,}")
    col2.metric("CUDA Share", f"{working_df['is_cuda_related'].mean()*100:.1f}%")
    col3.metric("Resolved Rate", f"{working_df['is_resolved'].mean()*100:.1f}%")
    col4.metric(
        "Avg Resolution Time",
        f"{working_df['time_to_resolution_hours'].mean():.1f}h" if working_df["time_to_resolution_hours"].notna().any() else "N/A",
    )

    if not filtered_resolved.empty:
        st.subheader("CUDA vs Non-CUDA Comparison")
        st.dataframe(comparison, width="stretch")
        csv_bytes = filtered_resolved.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered resolved data",
            data=csv_bytes,
            file_name="filtered_forum_resolved.csv",
            mime="text/csv",
        )

    st.subheader("Resolution Time Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        hist_series = working_df["time_to_resolution_hours"].dropna()
        if hist_series.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xlim(0, 1)
        else:
            ax.hist(hist_series, bins=40, edgecolor="black")
        ax.set_xlabel("Time to Resolution (hours)")
        ax.set_ylabel("Frequency")
        ax.set_title("Overall Distribution")
        st.pyplot(fig)

    with col2:
        if filtered_resolved.empty:
            st.info("More resolved topics needed for comparison plot.")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            filtered_resolved.boxplot(column="time_to_resolution_hours", by="is_cuda_related", ax=ax)
            group_values = sorted(filtered_resolved["is_cuda_related"].unique())
            tick_labels = ["CUDA" if val == 1 else "Non-CUDA" for val in group_values]
            ax.set_xticks(range(1, len(group_values) + 1))
            ax.set_xticklabels(tick_labels)
            ax.set_ylabel("Time to Resolution (hours)")
            ax.set_title("CUDA vs Non-CUDA")
            plt.suptitle("")
            st.pyplot(fig)

    if not filtered_resolved.empty:
        st.subheader("Top Threads Explorer")
        display_cols = [
            "topic_id",
            "title",
            "time_to_resolution_hours",
            "views",
            "reply_count",
            "is_cuda_related",
        ]
        tabs = st.tabs(["Longest Wait", "Fastest Resolved", "Most Viewed"])

        with tabs[0]:
            slowest = filtered_resolved.sort_values("time_to_resolution_hours", ascending=False).head(10)
            st.dataframe(slowest[display_cols], width="stretch")

        with tabs[1]:
            fastest = filtered_resolved.sort_values("time_to_resolution_hours").head(10)
            st.dataframe(fastest[display_cols], width="stretch")

        with tabs[2]:
            most_viewed = filtered_resolved.sort_values("views", ascending=False).head(10)
            st.dataframe(most_viewed[display_cols], width="stretch")

elif page == "Predict Resolution Time":
    st.header("Predict Resolution Time")
    st.markdown("Enter question characteristics to predict resolution time.")

    model = load_model()
    expected_features = _expected_feature_names(model)

    # Derive dynamic bounds from data where possible
    code_min, code_max = _numeric_bounds(df.get("code_block_count"), 0, 10)
    code_default = min(max(_numeric_default(df.get("code_block_count"), 1), code_min), code_max)

    qlen_min, qlen_max = _numeric_bounds(df.get("question_length"), 50, 20000)
    qlen_default = min(max(_numeric_default(df.get("question_length"), 500), qlen_min), qlen_max)

    views_min, views_max = _numeric_bounds(df.get("views"), 0, 10000)
    views_default = min(max(_numeric_default(df.get("views"), 100), views_min), views_max)

    replies_min, replies_max = _numeric_bounds(df.get("reply_count"), 0, 50)
    replies_default = min(max(_numeric_default(df.get("reply_count"), 5), replies_min), replies_max)

    hour_min, hour_max = _numeric_bounds(df.get("hour_of_day"), 0, 23)
    hour_default = min(max(_numeric_default(df.get("hour_of_day"), 12), hour_min), hour_max)

    # Input widgets
    col1, col2 = st.columns(2)

    with col1:
        is_cuda = st.checkbox("CUDA-related question", value=False)
        has_code_series = df["has_code_block"] if "has_code_block" in df else None
        has_error_series = df["has_error_trace"] if "has_error_trace" in df else None
        has_code_default = _bool_default(has_code_series, True)
        has_error_default = _bool_default(has_error_series, False)
        has_code = st.checkbox("Includes code snippet", value=has_code_default)
        code_count = st.slider("Number of code blocks", code_min, code_max, code_default)
        has_error = st.checkbox("Includes error trace", value=has_error_default)

    with col2:
        question_length = st.slider("Question length (characters)", qlen_min, qlen_max, qlen_default)
        views = st.number_input("Expected views", min_value=views_min, max_value=views_max, value=views_default)
        replies = st.number_input("Expected replies", min_value=replies_min, max_value=replies_max, value=replies_default)
        hour_of_day = st.slider("Hour of day posted", hour_min, hour_max, hour_default)

    # Dynamic category list from data
    if df is not None and "category_id" in df.columns:
        categories = sorted(df["category_id"].dropna().unique().tolist())
    else:
        categories = ["General", "Vision", "NLP", "Distributed", "Mobile", "Other"]

    category = st.selectbox("Category", categories)

    if st.button("Predict Resolution Time", type="primary"):
        if model is None:
            st.warning("Prediction model not available. Please train the model first.")
        else:
            # Build input row compatible with training preprocessing
            # Base numeric features
            row = {
                "is_cuda_related": int(is_cuda),
                "has_code_block": int(has_code),
                "code_block_count": int(code_count),
                "question_length": int(question_length),
                "has_error_trace": int(has_error),
                "views": int(views),
                "reply_count": int(replies),
                "hour_of_day": int(hour_of_day),
                "category_id": category,
            }

            row.update(_compute_derived_features(row))

            row_df = pd.DataFrame([row])

            # Create category dummy columns consistent with training (drop_first=True)
            # Use the full dataset categories to ensure same dummy columns
            try:
                full_cats = pd.get_dummies(df["category_id"], prefix="cat", drop_first=True).columns.tolist()
            except Exception:
                full_cats = []

            row_dummies = pd.get_dummies(row_df, columns=["category_id"], prefix="cat", drop_first=True)

            # Ensure all category dummy columns exist
            for c in full_cats:
                if c not in row_dummies.columns:
                    row_dummies[c] = 0

            # Build final feature vector in expected order
            if expected_features:
                # Ensure all expected features exist
                for feature in expected_features:
                    if feature not in row_dummies.columns:
                        if feature in {"code_density", "engagement_score"}:
                            row_dummies[feature] = row.get(feature, 0)
                        elif feature.startswith("cat_"):
                            row_dummies[feature] = 0
                        else:
                            row_dummies[feature] = row.get(feature, 0)
                X_pred = row_dummies[expected_features]
            else:
                base_features = [
                    "is_cuda_related",
                    "has_code_block",
                    "code_block_count",
                    "question_length",
                    "has_error_trace",
                    "views",
                    "reply_count",
                    "hour_of_day",
                    "code_density",
                    "engagement_score",
                ]
                feature_cols = base_features + full_cats
                X_pred = row_dummies.reindex(columns=feature_cols, fill_value=0)

            # Predict
            try:
                pred = model.predict(X_pred)[0]
                st.metric("Predicted Resolution Time", f"{pred:.1f} hours")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            median_resolution = df["time_to_resolution_hours"].median()
            if pd.notna(median_resolution):
                if pred <= median_resolution:
                    st.balloons()
                    st.success("Lightning-fast! This question is projected to resolve faster than the typical thread ⚡")
                else:
                    st.warning("Heads up: expect a longer discussion. Consider adding logs, repro steps, or more context.")

            st.subheader("Key Factors")
            factors = []
            if is_cuda:
                factors.append("CUDA-related (expected increase)")
            if has_code:
                factors.append("Includes code (expected decrease)")
            if has_error:
                factors.append("Has error trace (expected increase)")

            for factor in factors:
                st.markdown(factor)

elif page == "Causal Analysis Results":
    st.header("Causal Analysis Results")

    st.markdown("""
    ### Research Question
    **Do CUDA-related questions causally increase resolution time?**

    We use multiple causal inference methods to answer this question.
    """)

    st.subheader("Causal Structure")

    dag = """
    digraph {
        rankdir=LR
        node [shape=rectangle, style="rounded,filled", fontname="Helvetica", fontsize=12]

        subgraph cluster_confounders {
            label="Observed Confounders"
            color="#cfd8dc"
            style="rounded,dashed"
            Category [label="Category", fillcolor="#e3f2fd", color="#1e88e5"]
            Code [label="Code Present", fillcolor="#e8f5e9", color="#43a047"]
            Complexity [label="Question Length", fillcolor="#fff3e0", color="#fb8c00"]
        }

        Treatment [label="CUDA-related", fillcolor="#f3e5f5", color="#8e24aa"]
        Outcome [label="Resolution Time", fillcolor="#ffebee", color="#e53935"]

        Category -> Treatment
        Category -> Outcome
        Code -> Treatment
        Code -> Outcome
        Complexity -> Treatment
        Complexity -> Outcome
        Treatment -> Outcome
    }
    """

    st.graphviz_chart(dag)
    st.caption("Directed acyclic graph showing treatment (CUDA-related) and key confounders feeding into resolution time.")

    st.subheader("Treatment Effect Estimates")

    causal_results = load_causal_results()
    if causal_results is None:
        st.warning("Causal results not found. Run `03_causal_inference.ipynb` to generate `data/processed/causal_results.json`.")
        comparison = []
        results_df = pd.DataFrame()
    else:
        comparison = causal_results.get("comparison", [])
        results_df = pd.DataFrame(comparison)
        if not results_df.empty:
            results_df = results_df.rename(columns={
                "method": "Method",
                "ate_hours": "ATE (hours)",
                "ci_lower": "95% CI Lower",
                "ci_upper": "95% CI Upper",
            })

    if results_df.empty:
        st.info("No causal estimates available yet.")
    else:
        st.dataframe(results_df, width="stretch")

    st.subheader("Key Findings")

    def _select_method(preferred: str | None = None):
        if not comparison:
            return None
        if preferred:
            for item in comparison:
                if item.get("method") == preferred:
                    return item
        return comparison[0]

    highlight = _select_method("Double ML")

    col1, col2 = st.columns(2)

    with col1:
        if highlight:
            ate = highlight.get("ate_hours")
            ci_low = highlight.get("ci_lower")
            ci_high = highlight.get("ci_upper")
            method_name = highlight.get("method", "Selected method")

            ci_text = "N/A"
            if ci_low is not None and ci_high is not None:
                ci_text = f"[{ci_low:.1f}, {ci_high:.1f}]"

            st.info(f"""
            **{method_name} Effect**

            CUDA-related questions take **{_format_float(ate)} hours** longer to resolve
            95% CI: {ci_text}

            Estimates reflect causal effects after adjusting for confounders.
            """)
        else:
            st.info("Causal effect summary unavailable.")

    with col2:
        subgroup_effects = (causal_results or {}).get("subgroup_effects", {})
        subgroup = None
        preferred_key = highlight.get("method") if highlight else None

        if preferred_key and preferred_key in subgroup_effects:
            subgroup = subgroup_effects[preferred_key]
        elif subgroup_effects:
            first_key = next(iter(subgroup_effects))
            subgroup = subgroup_effects[first_key]
            preferred_key = first_key

        if subgroup:
            with_code = subgroup.get("with_code")
            without_code = subgroup.get("without_code")
            diff = subgroup.get("difference")
            st.success(f"""
            **Heterogeneous Effects ({preferred_key})**

            - With code: **{_format_float(with_code)} hours**
            - Without code: **{_format_float(without_code)} hours**
            - Delta: **{_format_float(diff)} hours**

            Including code substantially changes the effect.
            """)
        else:
            st.success("No subgroup estimates available.")

    st.subheader("Recommendations")
    st.markdown("""
    1. **Prompt for code snippets** to reduce resolution time for CUDA questions
    2. **Specialized moderators** for CUDA topics when causal effects are large
    3. **Template responses** to address recurring CUDA issues
    4. **Priority tagging** so longer-running CUDA threads get attention
    """)

elif page == "Fun Insights":
    st.header("Fun Insights 🎉")

    if resolved_df.empty:
        st.info("Fun insights unlock after at least one resolved topic is available.")
    else:
        spotlight_cols = st.columns(3)

        fastest_hour = None
        hour_stats = resolved_df.groupby("hour_of_day")["time_to_resolution_hours"].mean()
        if not hour_stats.empty:
            fastest_hour = int(hour_stats.idxmin())
            spotlight_cols[0].metric(
                "Fastest Posting Hour ⏱️",
                f"{fastest_hour:02d}:00",
                f"avg {hour_stats.min():.1f}h",
            )

        code_effect = (
            resolved_df.groupby("has_code_block")["time_to_resolution_hours"].mean().to_dict()
        )
        if code_effect:
            with_code = code_effect.get(1, float("nan"))
            without_code = code_effect.get(0, float("nan"))
            if pd.notna(with_code) and pd.notna(without_code):
                delta = without_code - with_code
                spotlight_cols[1].metric(
                    "Code Snippet Boost 💻",
                    f"{delta:+.1f}h",
                    "faster with code" if delta > 0 else "longer with code",
                )

        cuda_share = resolved_df["is_cuda_related"].mean() * 100
        spotlight_cols[2].metric("CUDA Share 🔥", f"{cuda_share:.1f}%", "of resolved posts")

        st.subheader("Category Speed League")
        category_summary = (
            resolved_df.groupby("category_id").agg(
                avg_time=("time_to_resolution_hours", "mean"),
                volume=("topic_id", "count"),
                cuda_ratio=("is_cuda_related", "mean"),
            )
            .sort_values("avg_time")
            .round({"avg_time": 2, "cuda_ratio": 2})
            .head(10)
        )
        category_summary.rename(columns={
            "avg_time": "Avg Hours",
            "volume": "Topics",
            "cuda_ratio": "CUDA %",
        }, inplace=True)
        category_summary["CUDA %"] = (category_summary["CUDA %"] * 100).round(1)
        category_summary.index = category_summary.index.astype(str)
        st.dataframe(category_summary, width="stretch")

        st.subheader("Resolution Playground")
        playground_df = resolved_df.copy()
        playground_df["category_id"] = playground_df["category_id"].astype(str)
        numeric_options = [
            "views",
            "reply_count",
            "question_length",
            "code_density",
            "engagement_score",
        ]
        x_axis = st.selectbox("X-axis", numeric_options, index=0)
        hue_options = ["is_cuda_related", "has_code_block", "category_id"]
        hue_choice = st.selectbox("Color by", hue_options, index=0)
        max_points = len(playground_df)
        if max_points <= 500:
            sample_size = max_points
            st.caption("Using all available resolved topics (<=500) for the scatter plot.")
        else:
            slider_max = min(max_points, 5000)
            slider_min = min(500, slider_max)
            default_val = min(2000, slider_max)
            sample_size = st.slider(
                "Sample size for scatter (to keep things speedy)",
                min_value=slider_min,
                max_value=slider_max,
                value=default_val,
                step=100,
            )
        sample_df = (
            playground_df.sample(sample_size, random_state=42)
            if sample_size < max_points
            else playground_df
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.scatterplot(
            data=sample_df,
            x=x_axis,
            y="time_to_resolution_hours",
            hue=hue_choice,
            palette="coolwarm",
            alpha=0.7,
            ax=ax,
        )
        ax.set_ylabel("Resolution Time (hours)")
        ax.set_xlabel(x_axis.replace("_", " ").title())
        ax.set_title("Explore how features relate to resolution speed")
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
        st.pyplot(fig)

        with st.expander("Quick Fun Facts"):
            fastest_cat = category_summary.index[0] if not category_summary.empty else "N/A"
            slowest_cat = category_summary.index[-1] if len(category_summary) > 1 else "N/A"
            most_viewed = (
                resolved_df.sort_values("views", ascending=False)
                .head(1)[["views", "time_to_resolution_hours", "is_cuda_related"]]
            )
            if not most_viewed.empty:
                top_views = int(most_viewed.iloc[0]["views"])
                top_time = most_viewed.iloc[0]["time_to_resolution_hours"]
                cuda_flag = "CUDA" if most_viewed.iloc[0]["is_cuda_related"] else "non-CUDA"
                st.markdown(
                    f"- 🏆 **Most viewed resolved thread:** {top_views:,} views ({cuda_flag}, {top_time:.1f}h to resolve)"
                )
            st.markdown(f"- 🥇 **Fastest category ID:** {fastest_cat}")
            st.markdown(f"- 🐢 **Slowest category ID:** {slowest_cat}")
            if fastest_hour is not None:
                st.markdown(f"- 📅 **Best hour to post:** {fastest_hour:02d}:00 (based on historical averages)")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project**: CUDA Error Resolution Analysis

**Data**: PyTorch Discussion Forum

**Methods**: Causal Inference, Machine Learning
""")