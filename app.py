import streamlit as st
import pandas as pd
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

# Classification models
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

# Regression models
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    HuberRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.svm import SVR

if "results_df" not in st.session_state:
    st.session_state.results_df = None

if "best_model" not in st.session_state:
    st.session_state.best_model = None

if "models_ran" not in st.session_state:
    st.session_state.models_ran = False


st.markdown(
    """
    <style>
    .toggle-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .toggle-label {
        font-weight: 600;
        font-size: 14px;
    }

    .pulse {
        animation: pulse 1.5s infinite;
        color: #4CAF50;
        font-weight: bold;
    }

    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Helper functions ----------

def detect_problem_type(y: pd.Series) -> str:
    """
    Auto-detect if the target column is better treated as
    Classification or Regression.
    """
    # If it's clearly non-numeric -> classification
    if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):
        return "Classification"

    if pd.api.types.is_bool_dtype(y):
        return "Classification"

    # Drop NaNs for analysis
    y_non_null = y.dropna()
    n_unique = y_non_null.nunique()

    # If few unique integer values -> likely classification
    if pd.api.types.is_integer_dtype(y_non_null) and n_unique <= 15:
        return "Classification"

    # Fallback
    return "Regression"


def map_cafv_eligibility(value):
    """
    Map CAFV eligibility text to 0/1.
    1 = eligible, 0 = not eligible / others.
    """
    v = str(value).lower()
    if "not eligible" in v:
        return 0
    elif "eligible" in v:
        return 1
    else:
        return 0

def estimate_runtime(num_rows, num_models, problem_type):
    """
    Rough runtime estimation (in seconds)
    """
    # Base time per model
    if problem_type == "Classification":
        base_time = 1.5
    else:
        base_time = 1.2

    # Scale with data size (3000 rows = baseline)
    size_factor = num_rows / 3000

    estimated_seconds = base_time * num_models * size_factor

    # Correction factor (accounts for CV, encoding, overhead)
    estimated_seconds *= 2.5

    # Keep estimate reasonable
    estimated_seconds = max(10, min(estimated_seconds, 900))

    return int(estimated_seconds)

# ---------- Model Explanation Knowledge Base ----------


MODEL_EXPLANATIONS = {

    # =====================
    # CLASSIFICATION MODELS
    # =====================

    "Logistic Regression": {
        "why": "Uses a linear decision boundary to separate classes efficiently.",
        "best_for": "Large datasets, binary or multi-class problems with linear separability.",
        "limits": "Cannot capture complex non-linear relationships without feature engineering."
    },

    "Ridge Classifier": {
        "why": "Adds regularization to reduce overfitting in linear classification.",
        "best_for": "High-dimensional data with correlated features.",
        "limits": "Still assumes linear decision boundaries."
    },

    "Naive Bayes": {
        "why": "Applies probabilistic reasoning assuming feature independence.",
        "best_for": "Text data, categorical features, and fast baseline models.",
        "limits": "Independence assumption is often unrealistic."
    },

    "Decision Tree Classifier": {
        "why": "Splits data based on feature thresholds to learn decision rules.",
        "best_for": "Interpretable models and capturing non-linear relationships.",
        "limits": "Prone to overfitting without depth control."
    },

    "Random Forest Classifier": {
        "why": "Combines multiple decision trees to improve robustness and accuracy.",
        "best_for": "Non-linear data, mixed feature types, noisy datasets.",
        "limits": "Less interpretable and more computationally expensive."
    },

    "Extra Trees Classifier": {
        "why": "Introduces extra randomness to reduce variance and improve generalization.",
        "best_for": "Large datasets with complex feature interactions.",
        "limits": "May underperform on small datasets."
    },

    "Gradient Boosting Classifier": {
        "why": "Sequentially corrects errors made by previous models.",
        "best_for": "Structured data with complex patterns.",
        "limits": "Sensitive to noise and slower to train."
    },

    "AdaBoost Classifier": {
        "why": "Focuses more on misclassified samples to improve performance.",
        "best_for": "Clean datasets with moderate size.",
        "limits": "Sensitive to noisy data and outliers."
    },

    "Bagging Classifier": {
        "why": "Reduces variance by training multiple models on bootstrapped samples.",
        "best_for": "High-variance models like decision trees.",
        "limits": "Does not reduce bias."
    },

    "KNN Classifier": {
        "why": "Classifies based on similarity to nearest neighbors.",
        "best_for": "Small datasets with well-separated classes.",
        "limits": "Very slow and memory-intensive for large datasets."
    },

    "Linear SVM": {
        "why": "Finds a maximum-margin linear hyperplane between classes.",
        "best_for": "High-dimensional data with linear separation.",
        "limits": "Cannot handle non-linear boundaries."
    },

    "Kernel SVM": {
        "why": "Uses kernel tricks to model non-linear decision boundaries.",
        "best_for": "Complex but small-to-medium datasets.",
        "limits": "Extremely slow and memory-heavy on large datasets."
    },

    # =====================
    # REGRESSION MODELS
    # =====================

    "Linear Regression": {
        "why": "Models a straight-line relationship between features and target.",
        "best_for": "Continuous numeric targets with linear relationships.",
        "limits": "Fails on non-linear patterns."
    },

    "Ridge Regression": {
        "why": "Adds L2 regularization to stabilize linear regression.",
        "best_for": "Multicollinearity and noisy numeric features.",
        "limits": "Still limited to linear relationships."
    },

    "Lasso Regression": {
        "why": "Performs feature selection by shrinking coefficients to zero.",
        "best_for": "Sparse datasets and feature selection.",
        "limits": "Unstable when features are highly correlated."
    },

    "ElasticNet": {
        "why": "Combines Ridge and Lasso regularization.",
        "best_for": "High-dimensional data with correlated features.",
        "limits": "Requires careful tuning of regularization parameters."
    },

    "Huber Regressor": {
        "why": "Balances squared and absolute loss to reduce outlier impact.",
        "best_for": "Regression with outliers.",
        "limits": "Less efficient when data is clean."
    },

    "Decision Tree Regressor": {
        "why": "Learns piecewise constant predictions via splits.",
        "best_for": "Non-linear relationships and interpretability.",
        "limits": "High variance without pruning."
    },

    "Random Forest Regressor": {
        "why": "Ensembles multiple trees to reduce variance.",
        "best_for": "Complex non-linear regression problems.",
        "limits": "Less interpretable and slower."
    },

    "Extra Trees Regressor": {
        "why": "Uses randomized splits to improve generalization.",
        "best_for": "Large datasets with complex patterns.",
        "limits": "May lose accuracy on small datasets."
    },

    "Gradient Boosting Regressor": {
        "why": "Sequentially improves predictions by correcting residuals.",
        "best_for": "Highly structured non-linear data.",
        "limits": "Sensitive to noise and slower training."
    },

    "AdaBoost Regressor": {
        "why": "Boosts weak learners focusing on difficult samples.",
        "best_for": "Moderate-size clean datasets.",
        "limits": "Sensitive to outliers."
    },

    "KNN Regressor": {
        "why": "Predicts values based on neighboring samples.",
        "best_for": "Small datasets with smooth trends.",
        "limits": "Very slow and memory-heavy for large datasets."
    },

    "SVM Regressor": {
        "why": "Uses margin-based optimization for regression.",
        "best_for": "Small datasets with complex non-linear relationships.",
        "limits": "Not scalable to large datasets."
    }
}


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Automotive ML Advisor", layout="wide")
col1, col2 = st.columns([1, 6])

with col1:
    st.image("logo.png", width=200)

with col2:
    st.markdown(
        "<h1 style='margin-bottom: 0;'> Automotive ML Algorithm Advisor</h1>",
        unsafe_allow_html=True
    )
st.write(
    """
Upload an automotive dataset (e.g., EV population / car specs),  
select a target column, and this app will **suggest the best ML model**  
by testing several algorithms automatically.
"""
)

st.sidebar.header("⚙️ Controls")
st.sidebar.subheader("🎛️ Display Options")

advanced_mode = st.sidebar.toggle(
    "Advanced Mode",
    value=False
)

if advanced_mode:
    st.sidebar.markdown(
        "<div class='pulse'>⚡ Advanced features enabled</div>",
        unsafe_allow_html=True
    )


uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# ================= DATASET CHANGE DETECTION =================
if uploaded_file is not None:
    file_id = uploaded_file.name

    if st.session_state.get("last_uploaded_file") != file_id:
        # New dataset uploaded → reset results
        st.session_state.last_uploaded_file = file_id
        st.session_state.results_df = None
        st.session_state.best_model = None
        st.session_state.models_ran = False
# ============================================================


if uploaded_file is not None:
    # Read data
    df_full = pd.read_csv(uploaded_file)

    st.subheader("Preview of Data")
    preview_rows = st.slider("Preview rows", 5, 100, 10)
    st.dataframe(df_full.head(preview_rows))
    st.write(f"Original Shape: {df_full.shape[0]} rows × {df_full.shape[1]} columns")

    df = df_full.copy()
    
    # ---------- Advanced Options (controlled by toggle) ----------
    if advanced_mode:
        with st.expander("⚙️ Advanced Options"):
            max_rows = st.slider(
                "Max rows for training",
                min_value=1000,
                max_value=50000,
                value=3000,
                step=1000
            )
    else:
        max_rows = 3000

        # ---------- Dataset sampling for training ----------
    if df.shape[0] > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    st.subheader("📊 Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isna().sum().sum()))

    # Feature engineering / cleanup
    if "VIN (1-10)" in df.columns:
        df.drop(columns=["VIN (1-10)"], inplace=True)
    
    if "Model Year" in df.columns:
        df["Vehicle Age"] = 2025 - df["Model Year"]

    
    st.write(f"Training Shape (after sampling): {df.shape[0]} rows × {df.shape[1]} columns")

    # Optionally drop clearly unhelpful columns
    cols_to_drop = [
        "DOL Vehicle ID",
        "Vehicle Location",
        "VIN",
        "2020 Census Tract",
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Target selection
    target_col = st.sidebar.selectbox("Select the target column (what you want to predict)", df.columns)

    # ---------- Time-Series Detection ----------
    is_time_series = False

    datetime_cols = df.select_dtypes(include=["datetime64", "datetime64[ns]"]).columns

    if len(datetime_cols) > 0:
        is_time_series = True
        time_col = datetime_cols[0]  # use first datetime column
        df = df.sort_values(by=time_col)
    # --------------------------------------------

    if target_col:
        y_raw = df[target_col]

        # Drop rows with missing target
        mask = y_raw.notna()
        df = df[mask]
        y_raw = y_raw[mask]

        # Auto-detect problem type
        suggested_type = detect_problem_type(y_raw)

        st.write(f" Auto-detected problem type: `{suggested_type}` based on target column.")

        # Allow manual override but default to suggested type
        problem_type = st.sidebar.radio(
            "Select / confirm problem type",
            ("Classification", "Regression"),
            index=0 if suggested_type == "Classification" else 1,
            help="Default is auto-detected. You can override if needed.",
        )

        st.sidebar.subheader("🤖 Model Selection Mode")

        model_mode = st.sidebar.radio(
            "Choose how models are selected",
            ["Automatic (Recommended)", "Manual (Select Models)"]
        )

        # ---------- Auto Time-Series Feature Engineering ----------
        if is_time_series and problem_type == "Regression":

            st.subheader("🕒 Auto Time-Series Feature Engineering")

            df[f"{target_col}_lag_1"] = df[target_col].shift(1)
            df[f"{target_col}_lag_7"] = df[target_col].shift(7)

            df[f"{target_col}_rolling_mean_7"] = (
                df[target_col].rolling(window=7).mean()
            )
            df[f"{target_col}_rolling_std_7"] = (
                df[target_col].rolling(window=7).std()
            )

            df.dropna(inplace=True)

            st.success(
                "Time-series lag & rolling features generated automatically"
            )

        # Prepare target (y)
        if problem_type == "Classification":
            # Special handling if this looks like CAFV eligibility
            if "cafv" in target_col.lower() or "elig" in target_col.lower():
                y = y_raw.apply(map_cafv_eligibility)
                st.info("CAFV-like column detected. Converted to binary classes: 1 = eligible, 0 = not eligible/other.")
            else:
                # For other classification tasks, keep labels as-is
                y = y_raw
        else:
            # Regression: make sure y is numeric
            y = pd.to_numeric(y_raw, errors="coerce")
            mask = y.notna()
            df = df[mask]
            y = y[mask]

        # ---------- TIME SERIES DETECTION ----------
        time_cols = [
            c for c in df.columns
            if any(k in c.lower() for k in ["date", "time", "year", "month", "day"])
        ]

        is_time_series = len(time_cols) > 0

        if is_time_series:
            time_col = time_cols[0]
            df = df.sort_values(by=time_col)
            st.info(f"⏳ Time-series dataset detected using column: `{time_col}`")
        
        # Features (X) = all other columns
        X = df.drop(columns=[target_col])

        # Identify categorical and numeric columns
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns

        st.write(f"Using {len(numeric_cols)} numeric and {len(categorical_cols)} categorical feature columns.")

        # One-hot encode categorical features
        if len(categorical_cols) > 0:
            X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        else:
            X_encoded = X.copy()

        # Replace infinities and NaNs
        X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)
        X_encoded = X_encoded.fillna(0)

        from sklearn.preprocessing import StandardScaler

        # Scale numeric features (important for Logistic Regression & SVM)
        scaler = StandardScaler()
        X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])


        if X_encoded.shape[1] == 0:
            st.error("No usable features after encoding. Please check your dataset.")
        else:
            # ------------ SAFE SPLITTING + SMOTE LOGIC ------------
            can_stratify = False
            apply_smote = False

            if is_time_series:
                st.warning("⚠ Time-series detected — disabling SMOTE & stratified split")
                can_stratify = False
                apply_smote = False

            if problem_type == "Classification":
                class_counts = y.value_counts()
                st.write("Class distribution:", class_counts)

                # If at least 2 samples in every class, we can stratify
                if class_counts.min() >= 2:
                    can_stratify = True
                    # SMOTE needs at least 2–3 samples in minority class for k_neighbors
                    apply_smote = class_counts.min() >= 3
                else:
                    st.warning(
                        "One of the classes has fewer than 2 samples. "
                        "Skipping stratified split and SMOTE for stability."
                    )

            # Train/test split
            if is_time_series:
                tscv = TimeSeriesSplit(n_splits=5)
                train_idx, test_idx = list(tscv.split(X_encoded))[-1]

                X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            else:
                if can_stratify:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_encoded, y, test_size=0.2, random_state=42, stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_encoded, y, test_size=0.2, random_state=42
                    )
            # ================= CROSS-VALIDATION STRATEGY =================
            if is_time_series:
                cv = TimeSeriesSplit(n_splits=5)
            elif problem_type == "Classification":
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
            # =============================================================

            # Apply SMOTE only when it is safe
            if problem_type == "Classification" and apply_smote and not is_time_series:
                sm = SMOTE(random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                st.info("Applied SMOTE to balance classes.")
            elif problem_type == "Classification" and is_time_series:
                st.info("SMOTE not applied (too few samples in a class). Using original class distribution."
                        "🕒 SMOTE disabled for time-series data to preserve temporal order.")

            # ------------------------------------------------------
            # ================= DATASET VALIDATION =================
            st.subheader("🧪 Dataset Validation")

            issues = []
            warnings = []

            # Row count check
            if df.shape[0] < 30:
                issues.append("Dataset has fewer than 30 rows. ML results may be unreliable.")

            # Feature count check
            if X_encoded.shape[1] < 2:
                issues.append("Dataset has fewer than 2 usable features.")

            # Target uniqueness check
            if problem_type == "Classification":
                unique_classes = y.nunique()
                if unique_classes < 2:
                    issues.append("Target column has only one class. Classification is not possible.")
                elif unique_classes > 10:
                    warnings.append("Target has many classes. Some classifiers may struggle.")

            # Missing values check
            missing_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
            if missing_ratio > 0.3:
                warnings.append("High percentage of missing values detected.")

            # Display validation results
            if issues:
                st.error("❌ Dataset is NOT suitable for reliable ML:")
                for i in issues:
                    st.write("•", i)
            elif warnings:
                st.warning("⚠️ Dataset may produce unstable results:")
                for w in warnings:
                    st.write("•", w)
            else:
                st.success("✅ Dataset looks suitable for ML modeling.")
            # =====================================================

            # Auto-detect large dataset to disable slow models
            is_large_dataset = max_rows > 10000

            if problem_type == "Classification":
                models = {}

                # ⚡ Fast & stable classifiers (always enabled)
                models["Logistic Regression"] = LogisticRegression(max_iter=2000,solver="lbfgs",n_jobs=-1)
                models["Ridge Classifier"] = RidgeClassifier()
                models["Naive Bayes"] = GaussianNB()
                models["Decision Tree Classifier"] = DecisionTreeClassifier(max_depth=10)
                models["Random Forest Classifier"] = RandomForestClassifier(
                    n_estimators=30, max_depth=12, n_jobs=-1
                )
                models["Gradient Boosting Classifier"] = GradientBoostingClassifier()
                models["AdaBoost Classifier"] = AdaBoostClassifier()
                models["Bagging Classifier"] = BaggingClassifier()

                # 🐢 Slow models → only if safe
                if not is_large_dataset and not is_time_series:
                    models["KNN Classifier"] = KNeighborsClassifier(n_neighbors=5)
                    models["Kernel SVM"] = SVC()
                    models["Extra Trees Classifier"] = ExtraTreesClassifier(
                        n_estimators=30, max_depth=12, n_jobs=-1
                    )
            
            if problem_type == "Classification" and model_mode == "Manual (Select Models)":
                st.sidebar.subheader("🎯 Select Classification Models")

                selected_models = st.sidebar.multiselect(
                    "Choose models to run",
                    list(models.keys()),
                    default=list(models.keys())
                )

                models = {k: v for k, v in models.items() if k in selected_models}

                if not models:
                    st.warning("⚠ Please select at least one classification model.")
            
            if problem_type == "Regression":
                models = {}

                # ⚡ Fast & stable regressors
                models["Linear Regression"] = LinearRegression()
                models["Ridge Regression"] = Ridge()
                models["ElasticNet"] = ElasticNet()
                models["Huber Regressor"] = HuberRegressor()
                models["Decision Tree Regressor"] = DecisionTreeRegressor(max_depth=10)
                models["Random Forest Regressor"] = RandomForestRegressor(
                    n_estimators=30, max_depth=12, n_jobs=-1
                )
                models["Gradient Boosting Regressor"] = GradientBoostingRegressor()
                models["AdaBoost Regressor"] = AdaBoostRegressor()

                # 🐢 Slow regressors → only if safe
                if not is_large_dataset and not is_time_series:
                    models["KNN Regressor"] = KNeighborsRegressor(n_neighbors=5)
                    models["SVR"] = SVR()
                    models["Extra Trees Regressor"] = ExtraTreesRegressor(
                        n_estimators=30, max_depth=12, n_jobs=-1
                    )
            
            if problem_type == "Regression" and model_mode == "Manual (Select Models)":
                st.sidebar.subheader("🎯 Select Regression Models")

                selected_models = st.sidebar.multiselect(
                    "Choose models to run",
                    list(models.keys()),
                    default=list(models.keys())
                )

                models = {k: v for k, v in models.items() if k in selected_models}

                if not models:
                    st.warning("⚠ Please select at least one regression model.")


            if st.button("Run Model Suggestion"):

                if not models:
                    st.error("❌ No models selected.")
                    st.stop()

                # 🔹 Animated status badge (only in Advanced Mode)
                if advanced_mode:
                    st.markdown(
                        "<div class='pulse'>🧠 Running extended model benchmarking...</div>",
                        unsafe_allow_html=True
                    )
                results = []

               
                # ⏱️ Runtime Estimation BEFORE training
                estimated_time = estimate_runtime(
                    num_rows=X_train.shape[0],
                    num_models=len(models),
                    problem_type=problem_type
                )

                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(90deg, #1f4037, #99f2c8);
                        padding: 12px;
                        border-radius: 10px;
                        color: black;
                        font-weight: 700;
                        font-size: 16px;
                        margin-bottom: 10px;
                    ">
                        ⏱️ Estimated Training Time: ~{estimated_time} seconds
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if problem_type == "Classification":
                    with st.spinner("Training and evaluating models..."):
                        import time
                        start_time = time.time()
                    
                        progress = st.progress(0)
                        status = st.empty()
                        
                        for i, (name, model) in enumerate(models.items()):
                            elapsed = int(time.time() - start_time)
                            status.text(f"Training {name}... ⏱ {elapsed}s elapsed")
                            progress.progress((i + 1) / len(models))
                            try:
                                # Cross-validation on TRAIN data
                                cv_scores = cross_val_score(
                                    model,
                                    X_train,
                                    y_train,
                                    cv=5,
                                    scoring="accuracy",
                                    n_jobs=-1
                                )

                                cv_accuracy = cv_scores.mean()

                                # Fit once on full TRAIN set
                                model.fit(X_train, y_train)

                                # Predict on TEST set
                                y_pred = model.predict(X_test)
                                prec = precision_score(
                                    y_test,
                                    y_pred,
                                    average="weighted",
                                    zero_division=0,
                                )
                                rec = recall_score(
                                    y_test,
                                    y_pred,
                                    average="weighted",
                                    zero_division=0
                                )
                                f1 = f1_score(
                                    y_test,
                                    y_pred,
                                    average="weighted",
                                    zero_division=0
                                )
                                results.append(
                                    {
                                        "Model": name,
                                        "Primary Metric": "Accuracy (CV)",
                                        "Score": cv_accuracy,
                                        "CV Std": cv_scores.std(),
                                        "Precision": prec,
                                        "Recall": rec,
                                        "F1 Score": f1,
                                    }
                                )
                            except Exception:
                                results.append(
                                    {
                                        "Model": name,
                                        "Primary Metric": "Error",
                                        "Score": 0,
                                        "CV Std": 0,
                                        "Precision": 0,
                                        "Recall": 0,
                                        "F1 Score": 0,
                                    }
                                )
                        status.text("Training completed ✅")

                        total_time = int(time.time() - start_time)

                        st.success(
                            f"✅ Training completed in {total_time} seconds "
                            f"(Estimated ~{estimated_time}s)"
                        )

                        # ✅ SAVE RESULTS TO SESSION STATE
                        results_df = pd.DataFrame(results)
                        results_df = results_df.sort_values(by="Score", ascending=False).reset_index(drop=True)

                        st.session_state.results_df = results_df
                        st.session_state.best_model = results_df.iloc[0]
                        st.session_state.models_ran = True


                if problem_type == "Regression":   
                    with st.spinner("Training and evaluating models..."):
                        import time
                        start_time = time.time()

                        progress = st.progress(0)
                        status = st.empty()
                        
                        for i, (name, model) in enumerate(models.items()):
                            elapsed = int(time.time() - start_time)
                            status.text(f"Training {name}... ⏱ {elapsed}s elapsed")
                            progress.progress((i + 1) / len(models))
                            try:
                                
                                # Cross-validated R²
                                cv_scores = cross_val_score(
                                    model,
                                    X_train,
                                    y_train,
                                    cv=5,
                                    scoring="r2",
                                    n_jobs=-1
                                )

                                cv_r2 = cv_scores.mean()

                                # 2️⃣ Fit model on full TRAIN set
                                model.fit(X_train, y_train)

                                # 3️⃣ Predict on TEST set
                                y_pred = model.predict(X_test)

                                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                mae = mean_absolute_error(y_test, y_pred)

                                # Avoid division by zero in MAPE
                                mape = np.mean(
                                    np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))
                                ) * 100
                                results.append(
                                    {
                                        "Model": name,
                                        "Primary Metric": "R² (CV)",
                                        "Score": cv_r2,
                                        "CV Std": cv_scores.std(),
                                        "RMSE": rmse,
                                        "MAE": mae,
                                        "MAPE (%)": mape
                                    }
                                )
                            except Exception:
                                results.append(
                                    {
                                        "Model": name,
                                        "Primary Metric": "Error",
                                        "Score": 0,
                                        "CV Std": 0,
                                        "RMSE": None,
                                        "MAE": None,
                                        "MAPE (%)": None
                                    }
                                )
                        status.text("Training completed ✅")

                        total_time = int(time.time() - start_time)

                        st.success(
                            f"✅ Training completed in {total_time} seconds "
                            f"(Estimated ~{estimated_time}s)"
                        )

                        # ✅ SAVE RESULTS TO SESSION STATE
                        results_df = pd.DataFrame(results)
                        results_df = results_df.sort_values(by="Score", ascending=False).reset_index(drop=True)

                        st.session_state.results_df = results_df
                        st.session_state.best_model = results_df.iloc[0]
                        st.session_state.models_ran = True


            # ---------- STEP 5: Post-Run Model Analysis ----------
            if st.session_state.models_ran:

                results_df = st.session_state.results_df
                best_row = st.session_state.best_model

                # 🆚 Compare Two Models
                st.subheader("🆚 Compare Two Models")

                compare_models = st.multiselect(
                    "Select exactly two models to compare",
                    options=results_df["Model"].tolist(),
                    max_selections=2,
                    key="compare_models"
                )

                if len(compare_models) == 2:
                    model_a = results_df[results_df["Model"] == compare_models[0]].iloc[0]
                    model_b = results_df[results_df["Model"] == compare_models[1]].iloc[0]

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"### 🔵 {model_a['Model']}")
                        st.write(model_a.drop("Model"))

                    with col_b:
                        st.markdown(f"### 🟣 {model_b['Model']}")
                        st.write(model_b.drop("Model"))

                # 🔍 Why this model fits your data
                st.subheader("🔍 Why this model fits your data")

                selected_model = st.selectbox(
                    "Select a model for explanation",
                    options=results_df["Model"].tolist(),
                    key="model_explain"
                )

                info = MODEL_EXPLANATIONS.get(selected_model)
                if info:
                    st.info(
                        f"""
                        **Model:** {selected_model}

                        **Why it works well:**  
                        {info['why']}

                        **Best suited for:**  
                        {info['best_for']}

                        **Limitations:**  
                        {info['limits']}
                        """
                    )
                else:
                    st.info("ℹ No detailed explanation available for this model.")
                
                st.session_state.results_df = results_df
                st.session_state.best_model = results_df.iloc[0]
                st.session_state.models_ran = True



            # Show results
            if st.session_state.models_ran and st.session_state.results_df is not None:
                
                results_df = st.session_state.results_df
                best_row = st.session_state.best_model
                
                # ---------- Prepare download data ----------
                results_df_display = results_df.copy()            

                results_df_display.insert(0, "Rank", ["🥇", "🥈", "🥉"] + [""] * (len(results_df) - 3))
                
                def highlight_top_models(row):
                    if row.name == 0:
                        return ["background-color: #668F07; color: #ecfdf5; font-weight: bold"] * len(row) #green
                    elif row.name == 1:
                        return ["background-color: #FAC907; color: #fffbeb; font-weight: bold"] * len(row) #yellow
                    elif row.name == 2:
                        return ["background-color: #F78E25; color: #fff7ed; font-weight: bold"] * len(row) #orange
                    else:
                        return [""] * len(row)

                tab1, tab2 = st.tabs(["📋 Model Ranking", "📊 Score Chart"])

                with tab1:
                    st.dataframe(results_df_display.style.apply(highlight_top_models, axis=1),
                    width='stretch')

                    # Best model (based on "Score" column)
                    best_row = results_df.iloc[0]


                    st.markdown("### 📊 Dataset-based reasoning")

                    reasons = []

                    if X_train.shape[0] > 5000:
                        reasons.append("Large dataset favors ensemble models.")

                    if X_train.shape[1] > 20:
                        reasons.append("High feature count suggests non-linear interactions.")

                    if is_time_series:
                        reasons.append("Temporal dependency detected in the dataset.")

                    for r in reasons:
                        st.write("•", r)

                    st.subheader("🧠 Auto Insights")

                    best_model_name = best_row["Model"]
                    num_samples = X_train.shape[0]
                    num_features = X_train.shape[1]

                    if problem_type == "Classification":
                        if "Forest" in best_model_name or "Tree" in best_model_name:
                            insight = "Tree-based models performed best, indicating non-linear relationships in the data."
                        elif "Logistic" in best_model_name:
                            insight = "Linear models performed well, suggesting mostly linear decision boundaries."
                        else:
                            insight = "This model best matched the feature distribution and class structure."
                    else:
                        if "Forest" in best_model_name:
                            insight = "Ensemble regressors captured complex, non-linear patterns effectively."
                        else:
                            insight = "The model achieved the best trade-off between bias and variance."

                    st.info(
                        f"""
                    **Why this model?**  
                    • Samples used: {num_samples}  
                    • Feature count: {num_features}  
                    • Selected model: **{best_model_name}**

                    {insight}
                    """
                    )

                    #Confidence badge
                    st.subheader("🔍 Result Confidence")

                    best_score = best_row["Score"]

                    if problem_type == "Classification":
                        if best_score >= 0.95 and num_samples > 1000:
                            confidence = "🟢 High Confidence"
                            reason = "Large dataset with consistently high classification accuracy."
                        elif best_score >= 0.85:
                            confidence = "🟡 Medium Confidence"
                            reason = "Good performance with some uncertainty."
                        else:
                            confidence = "🔴 Low Confidence"
                            reason = "Performance may be affected by data limitations."
                    else:
                        if best_score >= 0.8 and num_samples > 1000:
                            confidence = "🟢 High Confidence"
                            reason = "Strong explanatory power with sufficient data."
                        elif best_score >= 0.6:
                            confidence = "🟡 Medium Confidence"
                            reason = "Moderate regression fit."
                        else:
                            confidence = "🔴 Low Confidence"
                            reason = "Weak relationship between features and target."

                    st.success(f"{confidence}\n\n{reason}")

                    st.subheader("📊 Model Agreement Score")

                    score_range = results_df["Score"].max() - results_df["Score"].min()

                    if score_range < 0.05:
                        agreement = "🟢 High agreement between models (stable predictions)."
                    elif score_range < 0.15:
                        agreement = "🟡 Moderate agreement between models."
                    else:
                        agreement = "🔴 Low agreement — model choice significantly affects results."

                    st.info(
                        f"""
                    **Score Range:** {score_range:.4f}  
                    {agreement}
                    """
                    )

                    if problem_type == "Classification":
                        st.success(
                            f"Best model based on Accuracy: **{best_row['Model']}** "
                            f"with Accuracy = **{best_row['Score']:.3f}**"
                        )
                    else:
                        st.success(
                            f"Best model based on R²: **{best_row['Model']}** "
                            f"with R² = **{best_row['Score']:.3f}**"
                        )

                    with tab2:
                        st.bar_chart(results_df.set_index("Model")["Score"])
                    
                    with st.expander("📥 Download Results"):
                        csv_results = results_df.to_csv(index=False).encode("utf-8")
                        
                        st.download_button(
                            label="⬇️ Download Model Comparison (CSV)",
                            data=csv_results,
                            file_name="model_comparison_results.csv",
                            mime="text/csv"

                        )


                        # ---------- Best model summary download ----------
                        summary_text = f"""
                        Best Model Summary
                        ------------------
                        Model: {best_row['Model']}
                        Primary Metric: {best_row['Primary Metric']}
                        Score: {best_row['Score']:.4f}
                        """

                        st.download_button(
                            label="🏆 Download Best Model Summary",
                            data=summary_text,
                            file_name="best_model_summary.txt",
                            mime="text/plain"
                        )

else:
    st.info("Please upload a CSV file to get started.")
