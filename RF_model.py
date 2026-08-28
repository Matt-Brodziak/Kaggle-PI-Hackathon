#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# In[15]:


RANDOM_STATE = 42
TARGET = "employed_status"
ID_COL = "anonymised_id"
N_SPLITS = 5

train = pd.read_csv("data/train.csv")
train = train.dropna(subset=["employed_status"])
test = pd.read_csv("data/test.csv")
groups_train = train[ID_COL]


# Feature Engineering: Tenure, gated by prior employment and age x employed_lag and work_readiness_score x is_first_round

# In[16]:


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["has_history"] = df["lag_round"].notna().astype(int)
    df["employed_lag_num"] = df["employed_lag"]  # 0/1/NaN
    df["employed_lag_x_recency"] = df["employed_lag_num"].fillna(0) * df["has_history"]

    # --- Tenure, gated by prior employment ---------------------------------
    # only treat tenure as a risk signal for rows that WERE employed last
    # round -- zero it out otherwise, so the coefficient isn't diluted by
    # unrelated non-employed zeros.
    df["tenure_lag_missing"] = df["tenure_lag"].isna().astype(int)
    df["tenure_lag"] = df["tenure_lag"].fillna(0)
    df["log_tenure_lag"] = np.log1p(df["tenure_lag"].clip(lower=0))
    df["log_tenure_lag_if_employed"] = df["log_tenure_lag"] * df["employed_lag_num"].fillna(0)

    df["is_first_round"] = (df["total_historical_rounds"] <= 1).astype(int)

     # --- FIX: Center age before squaring ---------------------------------
    df["age"] = df["age"].fillna(df["age"].median())
    age_mean = df["age"].mean()  # Calculate mean after imputation
    df["age_centered"] = df["age"] - age_mean  # ← NEW
    df["age_sq"] = df["age_centered"] ** 2  # ← CHANGED: square centered age

    # --- age x employed_lag --------------------------------------------
    # Age likely means something different depending on prior state: among
    # the already-employed it's closer to a tenure/seniority proxy; among
    # the not-employed it's closer to a "how long searching" proxy.
    df["age_x_employed_lag"] = df["age"] * df["employed_lag_num"].fillna(0)

    # --- work_readiness_score x is_first_round --------------------------
    # Purpose-built forward-looking score should matter most when it's the
    # ONLY forward signal available (no employed_lag / status history).
    df["work_readiness_x_first_round"] = df["work_readiness_score"].fillna(
        df["work_readiness_score"].median()
    ) * df["is_first_round"]

    return df


# In[17]:


def add_seasonality_features(df: pd.DataFrame, date_col: str = "survey_date") -> pd.DataFrame:
    """Add cyclical seasonality features: sin/cos of month."""
    df = df.copy()

    if date_col not in df.columns:
        df["month_sin"] = 0
        df["month_cos"] = 0
        df["month_sin_x_employed_lag"] = 0
        df["month_cos_x_employed_lag"] = 0
        return df

    # Extract month
    month = pd.to_datetime(df[date_col]).dt.month

    # Cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Interaction with employed_lag - season affects transitions differently
    df["month_sin_x_employed_lag"] = df["month_sin"] * df["employed_lag_num"].fillna(0)
    df["month_cos_x_employed_lag"] = df["month_cos"] * df["employed_lag_num"].fillna(0)

    return df


# In[18]:


def make_interaction_categorical(df, col_a, col_b, new_col, min_count=None,
                                  train_ref=None, other_label="Other"):
    """Combine two categorical columns into one 'A||B' categorical.
    NaNs are stringified so 'Missing' combinations are preserved as their
    own category rather than dropped."""
    a = df[col_a].astype(str).fillna("Missing")
    b = df[col_b].astype(str).fillna("Missing")
    df[new_col] = a + "||" + b

    if min_count is not None:
        ref = train_ref if train_ref is not None else df
        counts = ref[new_col].value_counts()
        keep = set(counts[counts >= min_count].index)
        df[new_col] = df[new_col].where(df[new_col].isin(keep), other_label)
    return df


# In[19]:


def add_frequency_encoding(train_df, test_df, col, new_col=None):
    new_col = new_col or f"{col}_freq"
    freq_map = train_df[col].value_counts(normalize=True)
    train_df[new_col] = train_df[col].map(freq_map).fillna(0)
    test_df[new_col] = test_df[col].map(freq_map).fillna(0)
    return train_df, test_df


def collapse_rare_categories(train_df, test_df, col, min_count=30, other_label="Other"):
    counts = train_df[col].value_counts()
    keep = set(counts[counts >= min_count].index)

    def _collapse(series):
        return series.where(series.isin(keep) | series.isna(), other_label)

    train_df[col] = _collapse(train_df[col])
    test_df[col] = _collapse(test_df[col])
    return train_df, test_df

def extract_month_from_date(df: pd.DataFrame, date_col: str = "survey_date") -> pd.Series:
    """Extract month from survey_date column."""
    if date_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_datetime(df[date_col]).dt.month


# In[20]:


def kfold_target_encode(train_df, col, target_col, groups, n_splits=N_SPLITS,
                         smoothing=20, random_state=RANDOM_STATE):
    """Leakage-safe target encoding: out-of-fold for train rows, full-train
    mapping for anything else. Smoothing shrinks small-count categories
    toward the global mean so low-n municipalities don't get noisy extreme
    rates."""
    global_mean = train_df[target_col].mean()
    oof = pd.Series(index=train_df.index, dtype=float)

    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, val_idx in gkf.split(train_df, train_df[target_col], groups):
        fold_tr = train_df.iloc[tr_idx]
        stats = fold_tr.groupby(col)[target_col].agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
            stats["count"] + smoothing
        )
        val_keys = train_df.iloc[val_idx][col]
        oof.iloc[val_idx] = val_keys.map(smoothed).fillna(global_mean).values

    # full-train mapping for use on the actual test set
    full_stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
    full_smoothed = (
        full_stats["mean"] * full_stats["count"] + global_mean * smoothing
    ) / (full_stats["count"] + smoothing)

    return oof, full_smoothed, global_mean


# In[21]:


train = engineer_features(train)
test = engineer_features(test)

train = add_seasonality_features(train)
test = add_seasonality_features(test)


# --- combined interaction categoricals -------------------------------
train = make_interaction_categorical(train, "gender", "status_broad_lag",
                                      "gender_x_status_lag")
test = make_interaction_categorical(test, "gender", "status_broad_lag",
                                     "gender_x_status_lag")

train = make_interaction_categorical(train, "education_level", "status_broad_lag",
                                      "education_x_status_lag")
test = make_interaction_categorical(test, "education_level", "status_broad_lag",
                                     "education_x_status_lag")

# race x education_level: sparser combo, so collapse rare cells using
# TRAIN-only counts to avoid leakage.
train = make_interaction_categorical(train, "race", "education_level",
                                      "race_x_education", min_count=50,
                                      train_ref=train)
test = make_interaction_categorical(test, "race", "education_level",
                                     "race_x_education")

# map test's raw combos through the same keep-set as train (anything not
# seen with min_count in train becomes "Other")
_keep_race_edu = set(train["race_x_education"].unique()) - {"Other"}
test["race_x_education"] = test["race_x_education"].where(
    test["race_x_education"].isin(_keep_race_edu), "Other"
)




# In[22]:


numeric_features = [
    "age", 
    "age_sq",
    "employed_lag_x_recency",
    "log_tenure_lag_if_employed",   # replaces log_tenure_lag
    "tenure_lag_missing",
   "total_historical_rounds",
    "has_history",
    "is_first_round",
    "work_readiness_score",
    "work_readiness_x_first_round",     # NEW
    "age_x_employed_lag",               # NEW
    "municipality_freq",
    "municipality_emp_rate",            # NEW
    "month_sin",                     # NEW
    "month_cos",                     # NEW
    "month_sin_x_employed_lag",      # NEW - interaction
    "month_cos_x_employed_lag",      # NEW - interaction
]

categorical_features = [
    "status_broad_lag",
    "gender",
    "race",
    "province",
    "education_level",
    "gender_x_status_lag",       # NEW
    "education_x_status_lag",    # NEW

]

numeric_features = [c for c in numeric_features if c in train.columns]
categorical_features = [c for c in categorical_features if c in train.columns]

X_train = train[numeric_features + categorical_features]
y_train = train[TARGET].astype(int)
X_test = test[numeric_features + categorical_features]


# PIPELINE

# In[ ]:


numeric_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
])


preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_split=100,
    min_samples_leaf=50,
    max_features='sqrt',
    min_impurity_decrease=0.0001,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)


pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", model),
])


# Cross-Validation

# In[24]:


from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
fold_aucs = []

for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups_train), 1):
    tr_df, val_df = train.iloc[tr_idx].copy(), train.iloc[val_idx].copy()

    tr_df, val_df = add_frequency_encoding(tr_df, val_df, "municipality")
    tr_df, val_df = collapse_rare_categories(tr_df, val_df, "education_level", min_count=100)

    oof_rate, full_rate_map, g_mean = kfold_target_encode(tr_df, "municipality", TARGET, tr_df[ID_COL])
    tr_df["municipality_emp_rate"] = oof_rate
    val_df["municipality_emp_rate"] = val_df["municipality"].map(full_rate_map).fillna(g_mean)

    X_tr = tr_df[numeric_features + categorical_features]
    X_val = val_df[numeric_features + categorical_features]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    pipeline.fit(X_tr, y_tr)
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    fold_aucs.append(auc)
    print(f"Fold {fold}: AUC = {auc:.5f}")

print(f"\nMean CV AUC: {np.mean(fold_aucs):.5f} (+/- {np.std(fold_aucs):.5f})")


# In[25]:


from sklearn.metrics import roc_auc_score

# Use several trailing rounds as successive cutoffs instead of one,
# to average out the "small round" noise problem.
rounds_sorted = sorted(train["current_round"].unique())
val_rounds = rounds_sorted[-3:]  # last 3 rounds as walk-forward cutoffs

fold_aucs = []
subgroup_aucs = []  # track has_history split per fold

for cutoff in val_rounds:
    tr_df = train[train["current_round"] < cutoff].copy()
    val_df = train[train["current_round"] == cutoff].copy()

    if val_df.empty or tr_df.empty:
        continue

    # all target/frequency encodings fit on tr_df ONLY (time-safe)
    tr_df, val_df = add_frequency_encoding(tr_df, val_df, "municipality")
    tr_df, val_df = collapse_rare_categories(tr_df, val_df, "education_level", min_count=100)
    oof_rate, full_rate_map, g_mean = kfold_target_encode(
        tr_df, "municipality", TARGET, tr_df[ID_COL]
    )
    tr_df["municipality_emp_rate"] = oof_rate
    val_df["municipality_emp_rate"] = val_df["municipality"].map(full_rate_map).fillna(g_mean)

    X_tr = tr_df[numeric_features + categorical_features]
    X_val = val_df[numeric_features + categorical_features]
    y_tr = tr_df[TARGET].astype(int)
    y_val = val_df[TARGET].astype(int)

    pipeline.fit(X_tr, y_tr)
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    fold_aucs.append(auc)

    # subgroup check: new entrants vs returning respondents
    has_hist = val_df["has_history"].values.astype(bool)
    sub = {}
    if has_hist.sum() > 20:
        sub["returning"] = roc_auc_score(y_val[has_hist], val_probs[has_hist])
    if (~has_hist).sum() > 20:
        sub["new_entrant"] = roc_auc_score(y_val[~has_hist], val_probs[~has_hist])
    subgroup_aucs.append(sub)

    print(f"Cutoff round {cutoff}: n_val={len(val_df)}, AUC={auc:.5f}, subgroups={sub}")

print(f"\nWalk-forward mean AUC: {np.mean(fold_aucs):.5f} (+/- {np.std(fold_aucs):.5f})")


# Fit the pipeline on the full training data and make predictions on the test set:

# In[26]:


train, test = add_frequency_encoding(train, test, "municipality")
train, test = collapse_rare_categories(train, test, "education_level", min_count=100)
oof_muni_rate, full_muni_rate_map, global_rate = kfold_target_encode(train, "municipality", TARGET, groups_train)
train["municipality_emp_rate"] = oof_muni_rate
test["municipality_emp_rate"] = test["municipality"].map(full_muni_rate_map).fillna(global_rate)

X_train_final = train[numeric_features + categorical_features]
X_test_final = test[numeric_features + categorical_features]
pipeline.fit(X_train_final, y_train)
test_probsRF = pipeline.predict_proba(X_test_final)[:, 1]

submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    "employed_prob": test_probsRF,
})
submission.to_csv("Submissions/RF_model.csv", index=False)
print("\nSaved RF_model.csv")

