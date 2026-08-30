import pandas as pd
import numpy as np
from scipy.stats import rankdata

ID_COL = "anonymised_id"

# --- set these from your CV/submission-confirmed weight search ---
W_RETURNING = 0.7   # weight on LR for has_history == 1
W_NEW = 0.5         # weight on LR for has_history == 0

# --- load saved test-set predictions from each model ---
lr_sub = pd.read_csv("Submissions/Main_Model.csv").rename(columns={"employed_prob": "prob_lr"})
rf_sub = pd.read_csv("Submissions/RF_model.csv").rename(columns={"employed_prob": "prob_rf"})

# --- load raw test to get has_history (needs the same engineer_features logic) ---
test = pd.read_csv("data/test.csv")
test["has_history"] = test["lag_round"].notna().astype(int)

merged = (
    test[[ID_COL, "has_history"]]
    .merge(lr_sub, on=ID_COL)
    .merge(rf_sub, on=ID_COL)
)

n = len(merged)
merged["rank_lr"] = rankdata(merged["prob_lr"]) / n
merged["rank_rf"] = rankdata(merged["prob_rf"]) / n

# --- apply segment-conditional weight ---
is_returning = merged["has_history"].values.astype(bool)
weight_lr = np.where(is_returning, W_RETURNING, W_NEW)

merged["employed_prob"] = (
    weight_lr * merged["rank_lr"] + (1 - weight_lr) * merged["rank_rf"]
)

submission = merged[[ID_COL, "employed_prob"]]
submission.to_csv("Submissions/Segment_blend.csv", index=False)
print(f"Saved Segment_blend.csv  (w_returning={W_RETURNING}, w_new={W_NEW})")

print(merged["employed_prob"].describe())
print((weight_lr + (1 - weight_lr)).describe() if hasattr(weight_lr, 'describe') else np.unique(weight_lr))

# sanity check: employed_prob should never exceed max(rank_lr, rank_rf) or go below min(rank_lr, rank_rf) for any row
check = merged[(merged["employed_prob"] > merged[["rank_lr","rank_rf"]].max(axis=1)) |
               (merged["employed_prob"] < merged[["rank_lr","rank_rf"]].min(axis=1))]
print(f"Rows where blend falls OUTSIDE the range of its two inputs: {len(check)}")
print(check.head())
