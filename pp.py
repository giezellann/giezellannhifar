import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

st.set_page_config(page_title="Campus Facility Reservation Monitoring", page_icon="🏫", layout="wide")

st.title("🏫 Campus Facility Reservation Monitoring")
st.write(
    "Explore reservation data, clean records, view trends, and train a simple "
    "Decision Tree model to predict whether a reservation is completed."
)

REQUIRED_COLUMNS = [
    "Facility_Type", "User_Category", "Lead_Time_Days", "Participants",
    "Usage_Status", "Actual_Attendance"
]

def make_demo_data(rows=300, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Reservation_ID": [f"RES{i:03d}" for i in range(1, rows + 1)],
        "Facility_Type": rng.choice(
            ["Auditorium", "Conference Room", "Laboratory", "Sports Complex"], rows
        ),
        "User_Category": rng.choice(["Student", "Faculty", "Staff", "External"], rows),
        "Lead_Time_Days": rng.integers(-2, 30, rows),
        "Participants": rng.integers(5, 100, rows),
        "Usage_Status": rng.choice(["Used", "Cancelled", "No-Show"], rows, p=[0.6, 0.25, 0.15]),
        "Actual_Attendance": rng.integers(0, 100, rows),
        "Cancellation_Days_Before": rng.integers(0, 10, rows),
    })

def clean_data(data):
    cleaned = data.drop_duplicates().copy()
    categorical_cols = cleaned.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        cleaned[col] = cleaned[col].astype("string").str.strip().str.title()

    if "Lead_Time_Days" in cleaned.columns:
        cleaned["Lead_Time_Days"] = pd.to_numeric(cleaned["Lead_Time_Days"], errors="coerce")
        cleaned.loc[cleaned["Lead_Time_Days"] < 0, "Lead_Time_Days"] = np.nan
        median = cleaned["Lead_Time_Days"].median()
        cleaned["Lead_Time_Days"] = cleaned["Lead_Time_Days"].fillna(0 if pd.isna(median) else median)

    if "Usage_Status" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["Usage_Status"])
        cleaned["Reservation_Outcome"] = cleaned["Usage_Status"].apply(
            lambda value: "Completed" if str(value).strip().title() == "Used" else "Not Completed"
        )

    if "Participants" in cleaned.columns:
        cleaned["Participants"] = pd.to_numeric(cleaned["Participants"], errors="coerce")
    if "Actual_Attendance" in cleaned.columns:
        cleaned["Actual_Attendance"] = pd.to_numeric(cleaned["Actual_Attendance"], errors="coerce")

    if "Lead_Time_Days" in cleaned.columns:
        def lead_category(days):
            if days <= 2:
                return "Last Minute"
            if days <= 7:
                return "Short Lead"
            if days <= 14:
                return "Medium Lead"
            return "Early Booking"
        cleaned["Lead_Time_Category"] = cleaned["Lead_Time_Days"].apply(lead_category)

    if {"Participants", "Actual_Attendance"}.issubset(cleaned.columns):
        participants = cleaned["Participants"]
        attendance = cleaned["Actual_Attendance"]
        ratio = attendance / participants.replace(0, np.nan)
        cleaned["Attendance_Ratio"] = ratio.clip(lower=0, upper=1).fillna(0)

    return cleaned

with st.sidebar:
    st.header("Data source")
    uploaded_file = st.file_uploader("Upload facility_reservations.csv", type=["csv"])
    use_demo = st.checkbox("Use sample demonstration data if no CSV is uploaded", value=True)
    st.caption("Expected columns include Facility_Type, User_Category, Lead_Time_Days, Participants, Usage_Status, and Actual_Attendance.")

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success("CSV uploaded successfully.")
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
elif use_demo:
    raw_df = make_demo_data()
    st.info("Showing generated sample data. Upload your own CSV in the sidebar to analyze your actual reservations.")
else:
    st.warning("Upload a CSV or enable the sample data option in the sidebar.")
    st.stop()

missing = [col for col in REQUIRED_COLUMNS if col not in raw_df.columns]
if missing:
    st.error("The dataset is missing required column(s): " + ", ".join(missing))
    st.write("Columns found:", list(raw_df.columns))
    st.stop()

cleaned_df = clean_data(raw_df)

tab_overview, tab_cleaning, tab_eda, tab_model, tab_data = st.tabs(
    ["Overview", "Data Cleaning", "Exploratory Analysis", "Prediction Model", "Data Table"]
)

with tab_overview:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Original records", f"{len(raw_df):,}")
    c2.metric("Cleaned records", f"{len(cleaned_df):,}")
    c3.metric("Columns after processing", f"{len(cleaned_df.columns):,}")
    st.write("Preview of the original dataset")
    st.dataframe(raw_df.head(10), use_container_width=True)
    with st.expander("Column data types"):
        st.dataframe(pd.DataFrame({"Column": raw_df.columns, "Data type": raw_df.dtypes.astype(str).values}), use_container_width=True)
    with st.expander("Summary statistics"):
        st.dataframe(raw_df.describe(include="all").transpose(), use_container_width=True)

with tab_cleaning:
    st.subheader("Data Cleaning and Feature Engineering")
    st.write("Cleaning steps applied: remove duplicate rows, standardize text categories, fix negative lead times, fill missing lead times, and create derived columns.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Duplicate rows removed", int(raw_df.duplicated().sum()))
    c2.metric("Missing values remaining", int(cleaned_df.isna().sum().sum()))
    c3.metric("Rows removed", max(0, len(raw_df) - len(cleaned_df)))
    st.dataframe(cleaned_df.head(20), use_container_width=True)
    csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download cleaned CSV", data=csv_bytes, file_name="facility_reservations_cleaned.csv", mime="text/csv")

with tab_eda:
    st.subheader("Exploratory Data Analysis")
    facility_counts = cleaned_df["Facility_Type"].value_counts()
    user_outcome = pd.crosstab(cleaned_df["User_Category"], cleaned_df["Reservation_Outcome"], normalize="index") * 100
    lead_outcome = pd.crosstab(cleaned_df["Lead_Time_Category"], cleaned_df["Reservation_Outcome"], normalize="index") * 100

    if not facility_counts.empty:
        st.markdown("**1. Reservation count by facility type**")
        st.bar_chart(facility_counts)
        st.success(f"Most reserved facility: {facility_counts.idxmax()} ({facility_counts.max()} reservations)")
    st.markdown("**2. Reservation outcome percentage by user category**")
    st.dataframe(user_outcome.round(2), use_container_width=True)
    if "Not Completed" in user_outcome.columns and not user_outcome.empty:
        top_user = user_outcome["Not Completed"].idxmax()
        st.info(f"Highest non-completion rate: {top_user} ({user_outcome.loc[top_user, 'Not Completed']:.2f}%)")
    st.markdown("**3. Reservation outcome percentage by lead-time category**")
    st.dataframe(lead_outcome.round(2), use_container_width=True)
    if not lead_outcome.empty:
        st.bar_chart(lead_outcome)

with tab_model:
    st.subheader("Decision Tree Prediction Model")
    st.write("The model uses facility type, user category, lead time, participant count, and lead-time category to predict Reservation_Outcome.")
    predictors = ["Facility_Type", "User_Category", "Lead_Time_Days", "Participants", "Lead_Time_Category"]
    model_df = cleaned_df.dropna(subset=predictors + ["Reservation_Outcome"]).copy()
    if len(model_df) < 5 or model_df["Reservation_Outcome"].nunique() < 2:
        st.warning("There is not enough varied data to train the model. Upload a larger dataset with both completed and not-completed reservations.")
    else:
        X = pd.get_dummies(model_df[predictors], drop_first=True)
        y = model_df["Reservation_Outcome"]
        test_size = st.slider("Test data percentage", min_value=10, max_value=40, value=20, step=5)
        class_counts = y.value_counts()
        stratify_y = y if class_counts.min() >= 2 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42, stratify=stratify_y
            )
            model = DecisionTreeClassifier(max_depth=4, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            a, b, c, d = st.columns(4)
            a.metric("Accuracy", f"{accuracy_score(y_test, predictions):.2%}")
            b.metric("Precision (weighted)", f"{precision_score(y_test, predictions, average='weighted', zero_division=0):.2%}")
            c.metric("Recall (weighted)", f"{recall_score(y_test, predictions, average='weighted', zero_division=0):.2%}")
            d.metric("F1 score (weighted)", f"{f1_score(y_test, predictions, average='weighted', zero_division=0):.2%}")
            st.markdown("**Confusion matrix**")
            labels = sorted(y.unique())
            cm = confusion_matrix(y_test, predictions, labels=labels)
            st.dataframe(pd.DataFrame(cm, index=[f"Actual {x}" for x in labels], columns=[f"Predicted {x}" for x in labels]), use_container_width=True)
            st.markdown("**Classification report**")
            report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
            st.markdown("**Top 3 influential predictors**")
            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.bar_chart(importances.head(3))
            with st.expander("View decision tree"):
                fig, ax = plt.subplots(figsize=(18, 8))
                plot_tree(model, feature_names=list(X.columns), class_names=list(model.classes_), filled=True, rounded=True, ax=ax, fontsize=8)
                st.pyplot(fig)
                plt.close(fig)
        except ValueError as exc:
            st.warning(f"Could not train the model with this dataset split: {exc}")

with tab_data:
    st.subheader("Processed Dataset")
    st.dataframe(cleaned_df, use_container_width=True, height=450)
    st.download_button(
        "⬇️ Download processed dataset",
        data=cleaned_df.to_csv(index=False).encode("utf-8"),
        file_name="facility_reservations_cleaned.csv",
        mime="text/csv",
    )

st.caption("Note: If sample data is enabled, the results are demonstrations and not real campus reservation records.")
