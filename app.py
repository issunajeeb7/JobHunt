import streamlit as st
import pandas as pd
import os
from model import cluster_jobs, assign_cluster_to_new_jobs, CLUSTERED_JOBS_PATH
from scrape import scrape_karkidi_jobs, save_jobs_to_csv


st.set_page_config(page_title="JobHunt: Smart Job Discovery", layout="wide")
st.title("🔎 JobHunt: Smart Job Discovery & Alerts")

# --- Load or cluster jobs ---
def get_clustered_jobs():
    if not os.path.exists(CLUSTERED_JOBS_PATH):
        st.info("Clustering jobs for the first time...")
        df = cluster_jobs('data/jobs.csv')
    else:
        df = pd.read_csv(CLUSTERED_JOBS_PATH)
    return df

df = get_clustered_jobs()

# --- User Skill Interests ---
st.header("Find Jobs by Your Skills")
all_skills = set()
for s in df['Skills'].dropna():
    all_skills.update([x.strip().lower() for x in s.split(',') if x.strip()])

user_skills = st.multiselect(
    "Select your skills of interest:",
    sorted(all_skills),
    help="You'll see jobs that require at least one of your selected skills."
)



if user_skills:
    mask = df['Skills'].str.lower().apply(lambda s: any(skill in s for skill in user_skills) if isinstance(s, str) else False)
    matched_jobs = df[mask]
    st.success(f"Found {len(matched_jobs)} jobs matching your skills!")
    st.dataframe(matched_jobs[['Title', 'Company', 'Location', 'Experience', 'Skills', 'Summary']])
else:
    st.info("Select skills to see matching jobs.")


if st.button("🔄 Refresh Job Listings Now"):
    df_jobs = scrape_karkidi_jobs(keyword="data science", pages=2)
    save_jobs_to_csv(df_jobs)
    cluster_jobs('data/jobs.csv')
    st.success("Job listings updated!")
    df = get_clustered_jobs()  # Reload the updated data





    