import json
import os

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

st.set_page_config(page_title="SnapAddy Dashboard", layout="wide")

# Config / API Setup

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("GEMINI_API_KEY not set")
    st.stop()

genai.configure(api_key=API_KEY)
MODEL = genai.GenerativeModel("gemini-2.0-flash")

DEPARTMENT_LABELS = [
    "Information Technology",
    "Consulting",
    "Sales",
    "Project Management",
    "Business Development",
    "Marketing",
    "Administrative",
    "Human Resources",
    "Purchasing",
    "Customer Support",
]

SYSTEM_DEPT = (
    "You are a classifier. "
    f"Return exactly ONE label from: {', '.join(DEPARTMENT_LABELS)}. "
    "Output ONLY the label."
)


def predict_department(position, organization):
    prompt = f"""{SYSTEM_DEPT}

Job title: {position}
Company: {organization}
"""
    response = MODEL.generate_content(prompt)
    return response.text.strip()


@st.cache_data(show_spinner=False)
def predict_department_cached(position: str, organization: str) -> str:
    return predict_department(position, organization)



# UI Header
col1, col2 = st.columns([1, 4])

with col1:
    st.image("files/snapAddy_Logo.png", width=400)

with col2:
    st.title("SnapAddy Dashboard")


# File Upload

uploaded_file = st.file_uploader("Upload LinkedIn JSON", type=["json"])

if uploaded_file is None:
    st.stop()

data = json.load(uploaded_file)

st.success("Datei erfolgreich geladen")
st.write("Anzahl Profiles:", len(data))


# Extract Jobs
jobs_all = []
jobs_active = []

for person_id, cv in enumerate(data):
    for job in cv:
        row = {**job, "person_id": person_id}
        jobs_all.append(row)
        if job.get("status") == "ACTIVE":
            jobs_active.append(row)

st.write("Anzahl Jobs gesamt:", len(jobs_all))
st.write("Anzahl ACTIVE Jobs:", len(jobs_active))


# Filter ACTIVE Jobs (Text)
st.subheader("Filter ACTIVE Jobs")

search_text = st.text_input("Suche nach Position oder Unternehmen", value="")

df_active = pd.DataFrame(jobs_active)

if search_text:
    mask = (
        df_active["position"].str.contains(search_text, case=False, na=False)
        | df_active["organization"].str.contains(search_text, case=False, na=False)
    )
    df_filtered = df_active[mask]
else:
    df_filtered = df_active

cols = [
    c
    for c in ["person_id", "position", "organization", "status", "startDate", "endDate"]
    if c in df_filtered.columns
]

st.write(f"Gefilterte ACTIVE Jobs: {len(df_filtered)}")
st.dataframe(df_filtered[cols], use_container_width=True)



# Seniority Predictions (TF-IDF model)

st.subheader("Predictions")

@st.cache_resource
def load_seniority_model():
    return joblib.load("src/notebooks/seniority_tfidf_model.joblib")

seniority_model = load_seniority_model()

df_active["text_seniority"] = (
    df_active["position"].fillna("") + " " + df_active["organization"].fillna("")
).str.lower()

if st.button("Run Seniority Predictions"):
    df_active["seniority_pred"] = seniority_model.predict(df_active["text_seniority"])
    st.session_state["df_active_pred"] = df_active


# Use Case Search + Department Predictions
if "df_active_pred" in st.session_state:
    df_pred = st.session_state["df_active_pred"]

    st.subheader("Search (Use Case)")

    desired_seniority = st.multiselect(
        "Gewünschte Seniority",
        options=sorted(df_pred["seniority_pred"].unique().tolist()),
        default=[],
    )

    desired_departments = st.multiselect(
        "Gewünschtes Department (Prediction via API)",
        options=DEPARTMENT_LABELS,
        default=[],
    )

    # Apply text filter + seniority filter (before API)
    df_filtered = df_pred.copy()

    if search_text:
        mask = (
            df_filtered["position"].str.contains(search_text, case=False, na=False)
            | df_filtered["organization"].str.contains(search_text, case=False, na=False)
        )
        df_filtered = df_filtered[mask]

    if desired_seniority:
        df_filtered = df_filtered[df_filtered["seniority_pred"].isin(desired_seniority)]

    st.write(f"Aktuelle Treffer (ohne Department-Filter): {len(df_filtered)}")

    max_api = st.slider(
        "Max API Calls (für Department)", min_value=1, max_value=200, value=30, step=1
    )

    if st.button("Run Department Predictions (API)"):
        with st.spinner("Department wird predicted..."):
            df_tmp = df_filtered.head(max_api).copy()

            dept_preds = []
            for _, r in df_tmp.iterrows():
                dept_preds.append(
                    predict_department_cached(r.get("position", ""), r.get("organization", ""))
                )

            df_tmp["department_pred"] = dept_preds

        # Ensure column exists
        if "department_pred" not in df_pred.columns:
            df_pred["department_pred"] = pd.NA

        # Merge predictions back into df_pred
        merge_keys = ["person_id", "position", "organization"]
        df_pred = df_pred.merge(
            df_tmp[merge_keys + ["department_pred"]],
            on=merge_keys,
            how="left",
            suffixes=("", "_new"),
        )
        df_pred["department_pred"] = df_pred["department_pred_new"].combine_first(
            df_pred["department_pred"]
        )
        df_pred = df_pred.drop(columns=["department_pred_new"])

        st.session_state["df_active_pred"] = df_pred
        st.success("Department Predictions ergänzt.")

    # Rebuild view from session_state (so it includes new dept preds)
    df_show = st.session_state["df_active_pred"].copy()

    # Re-apply filters consistently
    if search_text:
        mask = (
            df_show["position"].str.contains(search_text, case=False, na=False)
            | df_show["organization"].str.contains(search_text, case=False, na=False)
        )
        df_show = df_show[mask]

    if desired_seniority:
        df_show = df_show[df_show["seniority_pred"].isin(desired_seniority)]

    if desired_departments:
        df_show = df_show[df_show["department_pred"].isin(desired_departments)]

    cols_show = [
        c
        for c in [
            "person_id",
            "position",
            "organization",
            "seniority_pred",
            "department_pred",
            "status",
            "startDate",
            "endDate",
        ]
        if c in df_show.columns
    ]

    st.write(f"Treffer (mit Department-Filter falls gesetzt): {len(df_show)}")
    st.dataframe(df_show[cols_show], use_container_width=True)