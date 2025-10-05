# file: prepare_target_and_features.py
import pandas as pd
import numpy as np

CSV_IN = "outputs/arxiv_cs_university.csv"   # ผลจาก Day1
CSV_OUT = "outputs/arxiv_for_model.csv"      # ผลลัพธ์ที่พร้อมเรียนรู้

df = pd.read_csv(CSV_IN, low_memory=False)

# ensure basic columns exist
for c in ["id","title","abstract","authors","categories","update_date","university_match"]:
    if c not in df.columns:
        df[c] = ""

# extract main category and year
df["category_main"] = df["categories"].fillna("").str.split().str[0].fillna("unknown")
df["update_date"] = pd.to_datetime(df["update_date"], errors="coerce")
df["year"] = df["update_date"].dt.year.fillna(0).astype(int)

# --- Ensure we have citation_count ---
if "citation_count" not in df.columns:
    # fallback: create a proxy (REMOVE this in real run)
    np.random.seed(0)
    df["citation_count"] = np.random.poisson(lam=5, size=len(df))

# compute Top10% threshold per (category_main, year)
# for categories/years with too few papers, compute per-category only
group_cols = ["category_main", "year"]
# if not enough entries per (category, year), groupby category only threshold
df["citation_threshold"] = np.nan

# first try (category, year)
th_cv = df.groupby(group_cols)["citation_count"].quantile(0.90).reset_index()
th_cv = th_cv.rename(columns={"citation_count":"th90"})
df = df.merge(th_cv, on=group_cols, how="left")
df["citation_threshold"] = df["th90"]

# for rows where threshold is NaN (too few data), fallback to category-level 90th
th_cat = df.groupby("category_main")["citation_count"].quantile(0.90).rename("th90_cat")
df = df.merge(th_cat, on="category_main", how="left")
df["citation_threshold"] = df["citation_threshold"].fillna(df["th90_cat"])

# final fallback: global 90th
global_th = df["citation_count"].quantile(0.90)
df["citation_threshold"] = df["citation_threshold"].fillna(global_th)

# define target
df["is_high_impact"] = (df["citation_count"] >= df["citation_threshold"]).astype(int)

# save for modeling
KEEP = ["id","title","abstract","authors","categories","category_main","year","university_match","citation_count","citation_threshold","is_high_impact"]
df[KEEP].to_csv(CSV_OUT, index=False)
print("Saved:", CSV_OUT)
