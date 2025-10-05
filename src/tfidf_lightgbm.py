import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
from lightgbm import early_stopping
import warnings
warnings.filterwarnings("ignore")

# ==========================
# CONFIG
# ==========================
INPUT_FILE = "outputs/arxiv_for_model.csv"
OUTPUT_METRICS_FILE = "outputs/baseline_metrics.csv"

# ==========================
# LOAD DATA
# ==========================
print("Loading data...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Loaded {len(df)} records.")

# ==========================
# TEXTUAL FEATURES
# ==========================
print("Creating textual features...")
df["text"] = (df["title"].fillna("") + " " + df["abstract"].fillna("")).str.replace("\n", " ")

tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=5)
X_text = tfidf.fit_transform(df["text"])
print(f"TF-IDF matrix shape: {X_text.shape}")

# ==========================
# CATEGORICAL FEATURES
# ==========================
print("Encoding categorical features...")
df[["category_main", "university_match"]] = df[["category_main", "university_match"]].fillna("Unknown")
ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
X_cat = ohe.fit_transform(df[["category_main", "university_match"]])
print(f"✅ One-hot matrix shape: {X_cat.shape}")

# ==========================
# NUMERIC FEATURES
# ==========================
print("Creating numeric features...")
df["num_authors"] = df["authors"].fillna("").apply(lambda s: 0 if s == "" else len([a for a in s.split(",") if a.strip()]))
df["abstract_length"] = df["abstract"].fillna("").apply(lambda t: len(t.split()))
df["title_length"] = df["title"].fillna("").apply(lambda t: len(t.split()))
df["year"] = df["year"].fillna(0)

X_num = csr_matrix(df[["num_authors", "abstract_length", "title_length", "year"]].values)
print(f" Numeric matrix shape: {X_num.shape}")

# ==========================
# COMBINE FEATURES
# ==========================
print("Combining all features...")
X_all = hstack([X_text, X_cat, X_num]).tocsr()
y = df["is_high_impact"].values
print(f"Combined matrix shape: {X_all.shape}")

# ==========================
# TRAIN-TEST SPLIT
# ==========================
print("Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

# ==========================
# MODEL TRAINING
# ==========================
print("Training LightGBM model...")
params = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5
}

# Handle imbalance
pos = y_train.sum()
neg = len(y_train) - pos
params["scale_pos_weight"] = neg / max(1, pos)

model = LGBMClassifier(**params, n_estimators=500)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[early_stopping(stopping_rounds=30)]
)

# ==========================
# PREDICTION & METRICS
# ==========================
print("Evaluating model...")
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

print(classification_report(y_test, y_pred, digits=4))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("PR AUC (avg precision):", average_precision_score(y_test, y_pred_proba))

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

metrics_df = pd.DataFrame({
    "precision": [precision],
    "recall": [recall],
    "f1": [f1],
    "roc_auc": [roc_auc_score(y_test, y_pred_proba)],
    "pr_auc": [average_precision_score(y_test, y_pred_proba)]
})
metrics_df.to_csv(OUTPUT_METRICS_FILE, index=False)
print(f"Baseline metrics saved to {OUTPUT_METRICS_FILE}")
