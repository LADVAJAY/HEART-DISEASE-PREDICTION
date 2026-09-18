
# ============================================================
# HEART DISEASE PREDICTION - STREAMLIT WEB APP
# Python for Data Science - BE05000231
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("❤️ Heart Disease Prediction")

st.write(
    "Python for Data Science - BE05000231"
)

st.markdown("---")


# ============================================================
# 1. LOAD DATASET
# ============================================================

try:

    df = pd.read_csv("heart.csv")

except FileNotFoundError:

    st.error("heart.csv file not found!")

    st.info(
        "Please keep heart.csv in the same folder as app.py."
    )

    st.stop()


# ============================================================
# 2. DATA CLEANING
# ============================================================

df = df.drop_duplicates()

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


# ============================================================
# 3. CHECK TARGET COLUMN
# ============================================================

if "target" not in df.columns:

    st.error(
        "The dataset must contain a 'target' column."
    )

    st.stop()


# ============================================================
# 4. SEPARATE INPUT AND OUTPUT
# ============================================================

X = df.drop("target", axis=1)

y = df["target"]


# ============================================================
# 5. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. NORMALIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# 8. SIDEBAR
# ============================================================

st.sidebar.header("📋 Patient Information")

st.sidebar.write(
    "Enter the patient's information below."
)


# ============================================================
# 9. PATIENT INPUT
# ============================================================

# AGE

age = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50,
    step=1
)


# SEX

sex = st.sidebar.selectbox(
    "Sex",
    [0, 1],
    format_func=lambda x:
        "Female (0)" if x == 0 else "Male (1)"
)


# CHEST PAIN

cp = st.sidebar.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)


# BLOOD PRESSURE

trestbps = st.sidebar.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120,
    step=1
)


# CHOLESTEROL

chol = st.sidebar.number_input(
    "Cholesterol",
    min_value=50,
    max_value=600,
    value=200,
    step=1
)


# FASTING BLOOD SUGAR

fbs = st.sidebar.selectbox(
    "Fasting Blood Sugar",
    [0, 1],
    format_func=lambda x:
        "No (0)" if x == 0 else "Yes (1)"
)


# RESTING ECG

restecg = st.sidebar.selectbox(
    "Resting ECG",
    [0, 1, 2]
)


# MAXIMUM HEART RATE

thalach = st.sidebar.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150,
    step=1
)


# EXERCISE ANGINA

exang = st.sidebar.selectbox(
    "Exercise Induced Angina",
    [0, 1],
    format_func=lambda x:
        "No (0)" if x == 0 else "Yes (1)"
)


# OLDPEAK

oldpeak = st.sidebar.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)


# SLOPE

slope = st.sidebar.selectbox(
    "Slope",
    [0, 1, 2]
)


# CA

ca = st.sidebar.selectbox(
    "Number of Major Vessels (CA)",
    [0, 1, 2, 3]
)


# THAL

thal = st.sidebar.selectbox(
    "Thal",
    [0, 1, 2, 3]
)


# ============================================================
# 10. CREATE PATIENT DATA
# ============================================================

patient_data = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}


# ============================================================
# 11. PREDICTION BUTTON
# ============================================================

predict_button = st.sidebar.button(
    "🔮 Predict Heart Disease",
    use_container_width=True
)


# ============================================================
# 12. MAIN PAGE INFORMATION
# ============================================================

st.subheader("📊 Heart Disease Prediction System")

st.write(
    "Enter patient information from the sidebar "
    "and click the Predict button."
)


# ============================================================
# 13. PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create patient dataframe
    # --------------------------------------------------------

    patient_df = pd.DataFrame(
        [patient_data],
        columns=X.columns
    )


    # --------------------------------------------------------
    # Scale patient data
    # --------------------------------------------------------

    patient_scaled = scaler.transform(
        patient_df
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        patient_scaled
    )


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability = model.predict_proba(
        patient_scaled
    )


    no_disease_probability = (
        probability[0][0] * 100
    )

    disease_probability = (
        probability[0][1] * 100
    )


    # ========================================================
    # 14. RESULT
    # ========================================================

    st.markdown("---")

    st.subheader("🔍 Prediction Result")


    if prediction[0] == 1:

        st.error(
            "❤️ HEART DISEASE DETECTED"
        )

    else:

        st.success(
            "✅ NO HEART DISEASE DETECTED"
        )


    # ========================================================
    # 15. PROBABILITY
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "No Disease Probability",
            f"{no_disease_probability:.2f}%"
        )


    with col2:

        st.metric(
            "Disease Probability",
            f"{disease_probability:.2f}%"
        )


    # ========================================================
    # 16. PATIENT INPUT TABLE
    # ========================================================

    st.markdown("---")

    st.subheader("👤 Patient Information")

    display_patient = pd.DataFrame(
        {
            "Feature": patient_df.columns,
            "Patient Value": patient_df.iloc[0].values
        }
    )

    st.dataframe(
        display_patient,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # 17. PREDICTION PROBABILITY GRAPH
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📊 Patient Prediction Probability"
    )

    fig1, ax1 = plt.subplots(
        figsize=(8, 5)
    )

    probabilities = [
        no_disease_probability,
        disease_probability
    ]

    bars = ax1.bar(
        ["No Disease", "Disease"],
        probabilities
    )

    ax1.set_ylabel(
        "Probability (%)"
    )

    ax1.set_title(
        "Patient Prediction Probability"
    )

    ax1.set_ylim(
        0,
        100
    )

    for bar, value in zip(
        bars,
        probabilities
    ):

        ax1.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 2,
            f"{value:.2f}%",
            ha="center"
        )

    st.pyplot(fig1)

    plt.close(fig1)


    # ========================================================
    # 18. PATIENT INPUT VALUES
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📊 Patient Input Values"
    )

    selected_columns = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]

    available_columns = [
        column
        for column in selected_columns
        if column in X.columns
    ]

    patient_values = [
        patient_df[column].iloc[0]
        for column in available_columns
    ]


    fig2, ax2 = plt.subplots(
        figsize=(10, 5)
    )

    bars = ax2.bar(
        available_columns,
        patient_values
    )

    ax2.set_title(
        "Patient Input Values"
    )

    ax2.set_xlabel(
        "Health Parameters"
    )

    ax2.set_ylabel(
        "Patient Value"
    )

    for bar, value in zip(
        bars,
        patient_values
    ):

        ax2.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom"
        )

    st.pyplot(fig2)

    plt.close(fig2)


    # ========================================================
    # 19. PATIENT VS DATASET AVERAGE
    # ========================================================

    st.subheader(
        "📊 Patient vs Dataset Average"
    )

    average_values = [
        df[column].mean()
        for column in available_columns
    ]

    x = np.arange(
        len(available_columns)
    )

    width = 0.35

    fig3, ax3 = plt.subplots(
        figsize=(10, 5)
    )

    patient_bars = ax3.bar(
        x - width / 2,
        patient_values,
        width,
        label="Patient"
    )

    average_bars = ax3.bar(
        x + width / 2,
        average_values,
        width,
        label="Dataset Average"
    )

    ax3.set_xticks(x)

    ax3.set_xticklabels(
        available_columns
    )

    ax3.set_ylabel(
        "Value"
    )

    ax3.set_title(
        "Patient Values vs Dataset Average"
    )

    ax3.legend()

    st.pyplot(fig3)

    plt.close(fig3)


    # ========================================================
    # 20. AGE DISTRIBUTION
    # ========================================================

    if "age" in X.columns:

        st.subheader(
            "📈 Patient Age Compared With Dataset"
        )

        patient_age = (
            patient_df["age"].iloc[0]
        )

        fig4, ax4 = plt.subplots(
            figsize=(8, 5)
        )

        sns.histplot(
            df["age"],
            bins=10,
            kde=True,
            ax=ax4
        )

        ax4.axvline(
            patient_age,
            linestyle="--",
            linewidth=3,
            label=f"Patient Age = {patient_age}"
        )

        ax4.set_title(
            "Age Distribution"
        )

        ax4.set_xlabel(
            "Age"
        )

        ax4.set_ylabel(
            "Number of Patients"
        )

        ax4.legend()

        st.pyplot(fig4)

        plt.close(fig4)


    # ========================================================
    # 21. CHOLESTEROL DISTRIBUTION
    # ========================================================

    if "chol" in X.columns:

        st.subheader(
            "📈 Patient Cholesterol Compared With Dataset"
        )

        patient_chol = (
            patient_df["chol"].iloc[0]
        )

        fig5, ax5 = plt.subplots(
            figsize=(8, 5)
        )

        sns.histplot(
            df["chol"],
            bins=10,
            kde=True,
            ax=ax5
        )

        ax5.axvline(
            patient_chol,
            linestyle="--",
            linewidth=3,
            label=f"Patient Cholesterol = {patient_chol}"
        )

        ax5.set_title(
            "Cholesterol Distribution"
        )

        ax5.set_xlabel(
            "Cholesterol"
        )

        ax5.set_ylabel(
            "Number of Patients"
        )

        ax5.legend()

        st.pyplot(fig5)

        plt.close(fig5)


    # ========================================================
    # 22. BLOOD PRESSURE DISTRIBUTION
    # ========================================================

    if "trestbps" in X.columns:

        st.subheader(
            "📈 Patient Blood Pressure Compared With Dataset"
        )

        patient_bp = (
            patient_df["trestbps"].iloc[0]
        )

        fig6, ax6 = plt.subplots(
            figsize=(8, 5)
        )

        sns.histplot(
            df["trestbps"],
            bins=10,
            kde=True,
            ax=ax6
        )

        ax6.axvline(
            patient_bp,
            linestyle="--",
            linewidth=3,
            label=f"Patient BP = {patient_bp}"
        )

        ax6.set_title(
            "Resting Blood Pressure Distribution"
        )

        ax6.set_xlabel(
            "Resting Blood Pressure"
        )

        ax6.set_ylabel(
            "Number of Patients"
        )

        ax6.legend()

        st.pyplot(fig6)

        plt.close(fig6)


    # ========================================================
    # 23. CHEST PAIN
    # ========================================================

    if "cp" in X.columns:

        st.subheader(
            "📊 Chest Pain Type"
        )

        patient_cp = (
            patient_df["cp"].iloc[0]
        )

        cp_counts = (
            df["cp"]
            .value_counts()
            .sort_index()
        )

        fig7, ax7 = plt.subplots(
            figsize=(8, 5)
        )

        bars = ax7.bar(
            cp_counts.index.astype(str),
            cp_counts.values
        )

        ax7.set_title(
            "Chest Pain Type Distribution"
        )

        ax7.set_xlabel(
            "Chest Pain Type"
        )

        ax7.set_ylabel(
            "Number of Patients"
        )

        for bar, value in zip(
            bars,
            cp_counts.values
        ):

            ax7.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
                str(value),
                ha="center",
                va="bottom"
            )

        ax7.text(
            0.5,
            0.95,
            f"Patient Chest Pain Type = {patient_cp}",
            transform=ax7.transAxes,
            ha="center"
        )

        st.pyplot(fig7)

        plt.close(fig7)


    # ========================================================
    # 24. AGE VS MAXIMUM HEART RATE
    # ========================================================

    if (
        "age" in X.columns
        and "thalach" in X.columns
    ):

        st.subheader(
            "📊 Patient Age vs Maximum Heart Rate"
        )

        patient_age = (
            patient_df["age"].iloc[0]
        )

        patient_thalach = (
            patient_df["thalach"].iloc[0]
        )

        fig8, ax8 = plt.subplots(
            figsize=(9, 5)
        )

        sns.scatterplot(
            data=df,
            x="age",
            y="thalach",
            hue="target",
            ax=ax8
        )

        ax8.scatter(
            patient_age,
            patient_thalach,
            s=200,
            marker="X",
            label="Patient"
        )

        ax8.set_title(
            "Age vs Maximum Heart Rate"
        )

        ax8.set_xlabel(
            "Age"
        )

        ax8.set_ylabel(
            "Maximum Heart Rate"
        )

        ax8.legend()

        st.pyplot(fig8)

        plt.close(fig8)


    # ========================================================
    # 25. CATEGORICAL FEATURES
    # ========================================================

    categorical_columns = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal"
    ]

    available_categorical = [
        column
        for column in categorical_columns
        if column in X.columns
    ]

    if len(available_categorical) > 0:

        st.subheader(
            "📊 Patient Categorical Feature Values"
        )

        categorical_values = [
            patient_df[column].iloc[0]
            for column in available_categorical
        ]

        fig9, ax9 = plt.subplots(
            figsize=(10, 5)
        )

        bars = ax9.bar(
            available_categorical,
            categorical_values
        )

        ax9.set_title(
            "Patient Categorical Feature Values"
        )

        ax9.set_xlabel(
            "Features"
        )

        ax9.set_ylabel(
            "Patient Input"
        )

        for bar, value in zip(
            bars,
            categorical_values
        ):

            ax9.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.0f}",
                ha="center",
                va="bottom"
            )

        st.pyplot(fig9)

        plt.close(fig9)


    # ========================================================
    # 26. CONFUSION MATRIX
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📊 Confusion Matrix"
    )

    y_pred = model.predict(
        X_test_scaled
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig10, ax10 = plt.subplots(
        figsize=(7, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "No Disease",
            "Disease"
        ],
        yticklabels=[
            "No Disease",
            "Disease"
        ],
        ax=ax10
    )

    ax10.set_title(
        "Confusion Matrix"
    )

    ax10.set_xlabel(
        "Predicted"
    )

    ax10.set_ylabel(
        "Actual"
    )

    st.pyplot(fig10)

    plt.close(fig10)


    # ========================================================
    # 27. MODEL ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.subheader(
        "📈 Model Accuracy"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Test Dataset Records",
            len(y_test)
        )


    fig11, ax11 = plt.subplots(
        figsize=(6, 5)
    )

    ax11.bar(
        ["Accuracy"],
        [accuracy * 100]
    )

    ax11.set_ylim(
        0,
        100
    )

    ax11.set_ylabel(
        "Accuracy (%)"
    )

    ax11.set_title(
        "Logistic Regression Model Accuracy"
    )

    ax11.text(
        0,
        accuracy * 100 + 2,
        f"{accuracy * 100:.2f}%",
        ha="center"
    )

    st.pyplot(fig11)

    plt.close(fig11)


    # ========================================================
    # 28. PROJECT COMPLETED
    # ========================================================

    st.markdown("---")

    st.success(
        "✅ Patient prediction completed successfully!"
    )

    st.info(
        "This project is for educational purposes only "
        "and should not be used as a medical diagnosis."
    )


# ============================================================
# 29. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Heart Disease Prediction | "
    "Python for Data Science - BE05000231"
)
