import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import json

df = pd.read_csv("outputs/arxiv_for_model.csv", low_memory=False)

# create metadata features
df["num_authors"] = df["authors"].fillna("").apply(lambda s: 0 if s=="" else len([a for a in s.split(",") if a.strip()]))
df["abstract_length"] = df["abstract"].fillna("").apply(lambda t: len(t.split()))
df["title_length"] = df["title"].fillna("").apply(lambda t: len(t.split()))

# keep relevant columns
cat_cols = ["category_main", "university_match"]
num_cols = ["num_authors","abstract_length","title_length","year"]

# fill missing
df[cat_cols] = df[cat_cols].fillna("Unknown")
df[num_cols] = df[num_cols].fillna(0)

# encode categorical
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(df[cat_cols])

X_num = df[num_cols].values
X = np.hstack([X_num, X_cat])
y = df["is_high_impact"].values

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# scale numeric part (optional)
scaler = StandardScaler()
X_train[:, :len(num_cols)] = scaler.fit_transform(X_train[:, :len(num_cols)])
X_test[:, :len(num_cols)] = scaler.transform(X_test[:, :len(num_cols)])

# baseline logistic regression
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

report = classification_report(y_test, y_pred, digits=4, output_dict=True)
roc_auc = roc_auc_score(y_test, y_proba)

baseline_results = {
    "precision": report["weighted avg"]["precision"],
    "recall": report["weighted avg"]["recall"],
    "f1_score": report["weighted avg"]["f1-score"],
    "roc_auc": roc_auc
}

# บันทึกผลลัพธ์เป็นไฟล์ JSON
output_path = "outputs/baseline_results.json"
with open(output_path, "w") as f:
    json.dump(baseline_results, f, indent=4)

print(classification_report(y_test, y_pred, digits=4))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
