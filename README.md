# LinkedIn Career Domain & Seniority Prediction

Capstone project for predicting career domain and seniority level from LinkedIn profiles.

## Project Overview

This project uses machine learning to automatically predict two target variables from LinkedIn CVs:
- **Domain** (e.g. Sales, IT, Marketing, HR)
- **Seniority** (e.g. Management, Professional, Junior)


## Project Structure

```
Capstone-Project-/
├── files/                          # Data files
│   	├── department-v2.csv 
│         ├── linkedin-cvs-annotated.json # Main evaluation dataset
│         ├── linkedin-cvs-not-annotated.json
│   	├── seniority-v2.csv
│         └── snapAddy_Logo.png                      # Logo
├── src/
│   └── notebooks/                 # Analysis notebooks
│       ├── 01_EDA.ipynb
│       ├── 02_Baseline Department.ipynb
│       ├── 02_Baseline Seniority.ipynb
│       ├── 03_Embedding_Department.ipynb
│       ├── 03_Embedding_Seniority.ipynb
│       ├── 04_TF-IDF_logreg_department.ipynb
│       ├── 04_TF-IDF_logreg_seniority.ipynb
│       ├── 05_API_Seniority.ipynb
│       ├── 05_API_Department.ipynb
│       ├──06_Feature Engineering Domain.ipynb
│       ├──07_Finetune_Distilbert_CV_Classification.ipynb
│       ├──08_Pseudo-Labeling_CV_Domain_Seniority.ipynb
│       └── *.joblib               # Saved models
├── app.py			#Streamlit dashboard
├── requirements.txt
└── README.md



## Setup

```bash
# Clone repository
git clone https://github.com/LennartRedlich/Capstone-Project-.git
cd Capstone-Project-

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run notebooks in `src/notebooks/` sequentially:

```bash
jupyter notebook src/notebooks/01_EDA.ipynb
```

## Streamlit Dashboard

To run the interactive dashboard:

```bash
streamlit run app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`
