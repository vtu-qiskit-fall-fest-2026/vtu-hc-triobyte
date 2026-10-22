
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

st.set_page_config(
    page_title="Quantum ML Software Bug Prediction",
    page_icon="🔬",
    layout="wide",
)

st.title("Quantum ML-Based Software Bug Prediction")
st.write(
    "Predict defective software modules using NASA CM1 metrics "
    "and compare a classical SVM with a quantum-kernel classifier."
)

st.info(
    "This is an experimental prototype. Quantum QSVC uses a "
    "simulated quantum kernel by default, not physical quantum hardware."
)

uploaded_file = st.file_uploader(
    "Upload NASA CM1 dataset (CSV)",
    type=["csv"],
)

if uploaded_file is None:
    st.warning("Upload cm1.csv to start the experiment.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if df.empty:
    st.error("The CSV contains no rows.")
    st.stop()

df.columns = df.columns.astype(str).str.strip()

st.subheader("1. Dataset overview")
st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
st.dataframe(df.head(10), use_container_width=True)

target = st.selectbox(
    "Select the defect-label column",
    options=df.columns,
    index=(
        list(df.columns).index("defects")
        if "defects" in df.columns else 0
    ),
)

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
feature_candidates = [c for c in numeric_columns if c != target]

if not feature_candidates:
    st.error("No numeric feature columns were found.")
    st.stop()

default_features = feature_candidates[:min(4, len(feature_candidates))]

features = st.multiselect(
    "Select numeric software metrics (maximum 4)",
    options=feature_candidates,
    default=default_features,
)

sample_size = st.slider(
    "Maximum rows to use",
    min_value=min(20, len(df)),
    max_value=len(df),
    value=len(df),
    step=1,
)

test_percent = st.select_slider(
    "Test-set percentage",
    options=[20, 25, 30],
    value=25,
)

if st.button("Train and compare models", type="primary"):
    if not features:
        st.error("Select at least one numeric feature.")
        st.stop()

    if len(features) > 4:
        st.error("Select no more than four features for this experiment.")
        st.stop()

    data = df[features + [target]].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[target])
    data = data.sample(
        n=min(sample_size, len(data)),
        random_state=42,
    )

    # Keep original class names as strings for clear label handling.
    y_raw = data[target].astype(str).str.strip()
    classes = sorted(y_raw.unique().tolist())

    if len(classes) != 2:
        st.error(
            f"Expected two target classes, but found: {classes}"
        )
        st.stop()

    # NASA CM1 uses False for non-defective and True for defective.
    # Explicitly map the defective class to 1.
    normalized = {c.lower(): c for c in classes}
    if "true" in normalized and "false" in normalized:
        negative_name = normalized["false"]
        positive_name = normalized["true"]
    else:
        st.error(
            "Could not identify True/False defect labels. "
            f"Detected classes: {classes}. Check the target column."
        )
        st.stop()

    y = (y_raw == positive_name).astype(int).to_numpy()

    X = data[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().all().any():
        bad_columns = X.columns[X.isna().all()].tolist()
        st.error(f"These features contain no numeric data: {bad_columns}")
        st.stop()

    counts = pd.Series(y).value_counts()
    if len(counts) != 2 or counts.min() < 2:
        st.error("Each class must have at least two examples.")
        st.stop()

    st.subheader("2. Class distribution")
    st.write(
        f"Non-defective: {int((y == 0).sum())} | "
        f"Defective: {int((y == 1).sum())}"
    )
    st.caption(
        "Defective modules are treated as the positive class "
        "for precision, recall, and F1-score."
    )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_percent / 100,
            random_state=42,
            stratify=y,
        )
    except ValueError as exc:
        st.error(f"Could not split the data: {exc}")
        st.stop()

    # Fit imputation and scaling on training data only.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Quantum feature maps work better when feature values are bounded.
    # Map scaled values into a limited angular range.
    X_train_quantum = np.clip(X_train, -2, 2) * (np.pi / 4)
    X_test_quantum = np.clip(X_test, -2, 2) * (np.pi / 4)

    results = {}
    predictions = {}
    model_errors = {}

    st.subheader("3. Model training")

    with st.spinner("Training classical SVM..."):
        try:
            classical_model = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", class_weight="balanced"),
            )

            # X_train and X_test are already scaled above.
            # Use a direct SVC to avoid scaling a second time.
            classical_model = SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=42,
            )
            classical_model.fit(X_train, y_train)
            classical_pred = classical_model.predict(X_test)

            predictions["Classical SVM"] = classical_pred

        except Exception as exc:
            model_errors["Classical SVM"] = str(exc)

    # Quantum kernel classifier with Qiskit.
    with st.spinner("Training quantum-kernel QSVC..."):
        try:
            from qiskit.circuit.library import ZZFeatureMap
            from qiskit_machine_learning.kernels import (
                FidelityQuantumKernel,
            )
            from qiskit_machine_learning.algorithms import QSVC

            feature_map = ZZFeatureMap(
                feature_dimension=len(features),
                reps=2,
                entanglement="linear",
            )

            quantum_kernel = FidelityQuantumKernel(
                feature_map=feature_map
            )

            quantum_model = QSVC(
                quantum_kernel=quantum_kernel,
                class_weight="balanced",
            )

            quantum_model.fit(X_train_quantum, y_train)
            quantum_pred = quantum_model.predict(X_test_quantum)

            predictions["Quantum QSVC"] = quantum_pred

        except Exception as exc:
            model_errors["Quantum QSVC"] = str(exc)

    # Calculate metrics with defective (label 1) as positive.
    for model_name, pred in predictions.items():
        results[model_name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, pos_label=1, zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, pos_label=1, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, pred, pos_label=1, zero_division=0
            ),
        }

    st.subheader("4. Evaluation results")

    if results:
        results_df = pd.DataFrame(results).T
        st.dataframe(
            results_df.style.format("{:.3f}"),
            use_container_width=True,
        )
        st.bar_chart(results_df)

    if model_errors:
        st.subheader("Model errors")
        for model_name, message in model_errors.items():
            st.warning(f"{model_name} could not run.")
            st.code(message)

    st.subheader("5. Confusion matrices")

    for model_name, pred in predictions.items():
        st.markdown(f"**{model_name}**")
        cm = confusion_matrix(y_test, pred, labels=[0, 1])

        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_title(model_name)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks([0, 1], labels=["Non-defective", "Defective"])
        ax.set_yticks([0, 1], labels=["Non-defective", "Defective"])

        for row in range(2):
            for col in range(2):
                ax.text(
                    col,
                    row,
                    str(cm[row, col]),
                    ha="center",
                    va="center",
                )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander(f"{model_name} classification report"):
            st.text(
                classification_report(
                    y_test,
                    pred,
                    labels=[0, 1],
                    target_names=["Non-defective", "Defective"],
                    zero_division=0,
                )
            )

    st.subheader("6. Test-set predictions")

    prediction_df = X_test.copy()
    prediction_df = pd.DataFrame(
        scaler.inverse_transform(X_test),
        columns=features,
    ).reset_index(drop=True)

    prediction_df["Actual label"] = np.where(
        y_test == 1, positive_name, negative_name
    )

    for model_name, pred in predictions.items():
        prediction_df[f"{model_name} prediction"] = np.where(
            pred == 1, positive_name, negative_name
        )

    st.dataframe(prediction_df, use_container_width=True)

    st.download_button(
        "Download predictions as CSV",
        data=prediction_df.to_csv(index=False).encode("utf-8"),
        file_name="software_defect_predictions.csv",
        mime="text/csv",
    )

    st.caption(
        "This experiment uses a simulated quantum kernel on your laptop. "
        "Metrics are experimental and do not establish quantum advantage. "
        "For reliable conclusions, repeat the evaluation with multiple "
        "splits or cross-validation."
    )
