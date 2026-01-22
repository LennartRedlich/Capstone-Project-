import pandas as pd
from sklearn.metrics import accuracy_score
import json
import streamlit as st
from pathlib import Path


LOGO_PATH = Path("files/snapAddy_Logo.png")

st.set_page_config(page_title="CV Explorer", layout="wide")

st.image(str(LOGO_PATH), width=400)

st.title("LinkedIn CV Explorer (minimal)")

DATA_PATH = "files/linkedin-cvs-annotated.json"  
LOGO_PATH = Path("files/snapAddy_Logo.png")

@st.cache_data
def load_jobs(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        cvs = json.load(f)

    rows = []
    for person_id, cv in enumerate(cvs):
        for job in cv:
            rows.append({
                "person_id": person_id,
                "status": job.get("status"),
                "position": job.get("position"),
                "organization": job.get("organization"),
                "startDate": job.get("startDate"),
                "endDate": job.get("endDate"),
                "department": job.get("department"),
                "seniority": job.get("seniority"),
                "linkedin": job.get("linkedin"),
            })
    return pd.DataFrame(rows)

df = load_jobs(DATA_PATH)

# --- Sidebar: Filter ---
st.sidebar.header("Filter")

only_active = st.sidebar.checkbox("Only ACTIVE jobs", value=True)
if only_active:
    df_view = df[df["status"] == "ACTIVE"].copy()
else:
    df_view = df.copy()

# simple text search
query = st.sidebar.text_input("Search (position / organization)", "")
if query.strip():
    q = query.strip().lower()
    pos = df_view["position"].fillna("").str.lower()
    org = df_view["organization"].fillna("").str.lower()
    df_view = df_view[pos.str.contains(q) | org.str.contains(q)]

# optional categorical filter if columns exist
if "department" in df_view.columns:
    dep_vals = sorted([x for x in df_view["department"].dropna().unique()])
    dep_sel = st.sidebar.multiselect("Department", dep_vals)
    if dep_sel:
        df_view = df_view[df_view["department"].isin(dep_sel)]

if "seniority" in df_view.columns:
    sen_vals = sorted([x for x in df_view["seniority"].dropna().unique()])
    sen_sel = st.sidebar.multiselect("Seniority", sen_vals)
    if sen_sel:
        df_view = df_view[df_view["seniority"].isin(sen_sel)]

# --- KPIs ---
c1, c2 = st.columns(2)
c1.metric("Total jobs", len(df))
c2.metric("Shown jobs", len(df_view))

# --- Table ---
st.subheader("Jobs")
cols = ["person_id", "status", "position", "organization", "department", "seniority", "startDate", "endDate"]
cols = [c for c in cols if c in df_view.columns]
st.dataframe(df_view[cols], use_container_width=True)

# --- Optional: show raw row ---
st.subheader("Inspect one job")
idx = st.number_input("Row index (in current view)", min_value=0, max_value=max(len(df_view)-1, 0), value=0, step=1)
if len(df_view) > 0:
    st.json(df_view.iloc[int(idx)].to_dict())

