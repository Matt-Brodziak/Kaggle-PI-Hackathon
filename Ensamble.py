"""LR + RF blend, full-train fit -> test prediction.

Methodology note: an OOF check (StratifiedGroupKFold, matched folds between
LR and RF) showed a single GLOBAL rank-blend weight beats per-segment
weighting on overall AUC (+0.00526 vs LR alone), even though within-segment
gains are small (+0.001ish each). The two LR submodels (returning/new) are
fit fully independently with no shared calibration, so their output scales
aren't guaranteed comparable to each other; RF is one unified model with a
consistent scale across both segments, so blending it in also fixes some of
that cross-segment miscalibration -- not just adding new discriminative
signal. See LR_model.ipynb / RF_model.ipynb for the individual pipelines
this mirrors.

W_LR is set below OOF-optimal (0.38) as a hedge: RF has a known large
CV->LB gap on this dataset (CV ~0.651 standalone, LB 0.63897), so leaning
less on it than the raw OOF search suggests is deliberate, not a mistake.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.stats import rankdata
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "employed_status"
ID_COL = "anonymised_id"
N_SPLITS = 5

W_LR = 0.55  # global blend weight on LR; (1 - W_LR) goes to RF

train = pd.read_csv("data/train.csv")
train = train.dropna(subset=["employed_status"]).reset_index(drop=True)
test = pd.read_csv("data/test.csv")

# ---------------- shared helpers (identical in both notebooks) ----------------
def add_seasonality_features(df, date_col="survey_date"):
    df = df.copy()
    if date_col not in df.columns:
        df["month_sin"] = 0; df["month_cos"] = 0
        df["month_sin_x_employed_lag"] = 0; df["month_cos_x_employed_lag"] = 0
        return df
    month = pd.to_datetime(df[date_col]).dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["month_sin_x_employed_lag"] = df["month_sin"] * df["employed_lag_num"].fillna(0)
    df["month_cos_x_employed_lag"] = df["month_cos"] * df["employed_lag_num"].fillna(0)
    return df

def make_interaction_categorical(df, col_a, col_b, new_col, min_count=None, train_ref=None, other_label="Other"):
    a = df[col_a].astype(str).fillna("Missing")
    b = df[col_b].astype(str).fillna("Missing")
    df[new_col] = a + "||" + b
    if min_count is not None:
        ref = train_ref if train_ref is not None else df
        counts = ref[new_col].value_counts()
        keep = set(counts[counts >= min_count].index)
        df[new_col] = df[new_col].where(df[new_col].isin(keep), other_label)
    return df

def add_frequency_encoding(train_df, test_df, col, new_col=None):
    new_col = new_col or f"{col}_freq"
    freq_map = train_df[col].value_counts(normalize=True)
    train_df[new_col] = train_df[col].map(freq_map).fillna(0)
    test_df[new_col] = test_df[col].map(freq_map).fillna(0)
    return train_df, test_df

def collapse_rare_categories(train_df, test_df, col, min_count=30, other_label="Other"):
    counts = train_df[col].value_counts()
    keep = set(counts[counts >= min_count].index)
    def _collapse(s): return s.where(s.isin(keep) | s.isna(), other_label)
    train_df[col] = _collapse(train_df[col])
    test_df[col] = _collapse(test_df[col])
    return train_df, test_df

def kfold_target_encode(train_df, col, target_col, groups, n_splits=N_SPLITS, smoothing=1):
    global_mean = train_df[target_col].mean()
    oof = pd.Series(index=train_df.index, dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, val_idx in gkf.split(train_df, train_df[target_col], groups):
        fold_tr = train_df.iloc[tr_idx]
        stats = fold_tr.groupby(col)[target_col].agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        val_keys = train_df.iloc[val_idx][col]
        oof.iloc[val_idx] = val_keys.map(smoothed).fillna(global_mean).values
    full_stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
    full_smoothed = (full_stats["mean"] * full_stats["count"] + global_mean * smoothing) / (full_stats["count"] + smoothing)
    return oof, full_smoothed, global_mean

def add_interactions(df, train_ref):
    df = make_interaction_categorical(df, "gender", "status_broad_lag", "gender_x_status_lag")
    df = make_interaction_categorical(df, "education_level", "status_broad_lag", "education_x_status_lag")
    df = make_interaction_categorical(df, "race", "education_level", "race_x_education", min_count=50, train_ref=train_ref)
    return df

# ---------------- LR feature engineering (matches current LR_model.ipynb) ----------------
def engineer_features_lr(df, quintile_median=None):
    df = df.copy()
    df["has_history"] = df["lag_round"].notna().astype(int)
    df["employed_lag_num"] = df["employed_lag"]
    df["employed_lag_x_recency"] = df["employed_lag_num"].fillna(0) * df["has_history"]
    df["tenure_lag_missing"] = df["tenure_lag"].isna().astype(int)
    df["tenure_lag"] = df["tenure_lag"].fillna(0)
    df["log_tenure_lag"] = np.log1p(df["tenure_lag"].clip(lower=0))
    df["log_tenure_lag_if_employed"] = df["log_tenure_lag"] * df["employed_lag_num"].fillna(0)
    df["is_first_round"] = (df["total_historical_rounds"] <= 1).astype(int)
    df["age"] = df["age"].fillna(df["age"].median())
    age_mean = df["age"].mean()
    df["age_centered"] = df["age"] - age_mean
    df["age_sq"] = df["age_centered"] ** 2
    df["age_x_employed_lag"] = df["age"] * df["employed_lag_num"].fillna(0)
    df["work_readiness_missing"] = df["work_readiness_score"].isna().astype(int)
    df["work_readiness_score_clean"] = df["work_readiness_score"].fillna(df["work_readiness_score"].median())
    return df

# ---------------- RF feature engineering (matches RF_model.ipynb) ----------------
def engineer_features_rf(df):
    df = df.copy()
    df["has_history"] = df["lag_round"].notna().astype(int)
    df["employed_lag_num"] = df["employed_lag"]
    df["employed_lag_x_recency"] = df["employed_lag_num"].fillna(0) * df["has_history"]
    df["tenure_lag_missing"] = df["tenure_lag"].isna().astype(int)
    df["tenure_lag"] = df["tenure_lag"].fillna(0)
    df["log_tenure_lag"] = np.log1p(df["tenure_lag"].clip(lower=0))
    df["log_tenure_lag_if_employed"] = df["log_tenure_lag"] * df["employed_lag_num"].fillna(0)
    df["is_first_round"] = (df["total_historical_rounds"] <= 1).astype(int)
    df["age"] = df["age"].fillna(df["age"].median())
    age_mean = df["age"].mean()
    df["age_centered"] = df["age"] - age_mean
    df["age_sq"] = df["age_centered"] ** 2
    df["age_x_employed_lag"] = df["age"] * df["employed_lag_num"].fillna(0)
    df["work_readiness_x_first_round"] = df["work_readiness_score"].fillna(df["work_readiness_score"].median()) * df["is_first_round"]
    return df

# ---------------- LR: two-model split (returning vs new entrants) ----------------
base_cat = ["gender", "race", "province", "education_level", "race_x_education"]
ret_cat = base_cat + ["status_broad_lag", "gender_x_status_lag", "education_x_status_lag"]
ret_numeric = ["age", "age_sq", "employed_lag_x_recency", "log_tenure_lag_if_employed", "tenure_lag_missing",
               "total_historical_rounds", "age_x_employed_lag", "municipality_freq", "municipality_emp_rate",
               "month_sin", "month_cos", "month_sin_x_employed_lag", "month_cos_x_employed_lag"]
new_numeric = ["age", "age_sq", "municipality_freq", "municipality_emp_rate", "month_sin", "month_cos",
               "work_readiness_missing", "work_readiness_score_clean"]
new_cat = base_cat

def build_pipeline_lr(numeric_cols, categorical_cols, C=0.1, l1_ratio=0.5):
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))])
    prep = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga",
                              l1_ratio=l1_ratio, C=C, random_state=RANDOM_STATE)
    return Pipeline([("preprocess", prep), ("clf", clf)])

# ---------------- RF: single unified model ----------------
rf_numeric = ["age", "age_sq", "employed_lag_x_recency", "log_tenure_lag_if_employed", "tenure_lag_missing",
              "total_historical_rounds", "has_history", "is_first_round", "work_readiness_score",
              "work_readiness_x_first_round", "age_x_employed_lag", "municipality_freq", "municipality_emp_rate",
              "month_sin", "month_cos", "month_sin_x_employed_lag", "month_cos_x_employed_lag"]
rf_categorical = ["status_broad_lag", "gender", "race", "province", "education_level",
                   "gender_x_status_lag", "education_x_status_lag"]

def build_pipeline_rf():
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))])
    prep = ColumnTransformer([("num", num_pipe, rf_numeric), ("cat", cat_pipe, rf_categorical)])
    clf = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_split=100, min_samples_leaf=50,
                                  max_features='sqrt', min_impurity_decrease=0.0001, class_weight='balanced',
                                  random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([("preprocess", prep), ("clf", clf)])

# ================= LR full-train fit -> test predictions =================
quintile_median = train["school_quintile"].median()
train_lr = engineer_features_lr(train, quintile_median=quintile_median)
test_lr = engineer_features_lr(test, quintile_median=quintile_median)
train_lr = add_seasonality_features(train_lr)
test_lr = add_seasonality_features(test_lr)
train_lr = add_interactions(train_lr, train_ref=train_lr)
test_lr = add_interactions(test_lr, train_ref=train_lr)
_keep_race_edu = set(train_lr["race_x_education"].unique()) - {"Other"}
test_lr["race_x_education"] = test_lr["race_x_education"].where(test_lr["race_x_education"].isin(_keep_race_edu), "Other")

train_lr, test_lr = add_frequency_encoding(train_lr, test_lr, "municipality")
train_lr, test_lr = collapse_rare_categories(train_lr, test_lr, "education_level", min_count=100)
oof_rate, full_rate_map, g_mean = kfold_target_encode(train_lr, "municipality", TARGET, train_lr[ID_COL], smoothing=1)
train_lr["municipality_emp_rate"] = oof_rate
test_lr["municipality_emp_rate"] = test_lr["municipality"].map(full_rate_map).fillna(g_mean)

mask_train_ret = (train_lr["has_history"] == 1).values
mask_train_new = (train_lr["has_history"] == 0).values
mask_test_ret = (test_lr["has_history"] == 1).values
mask_test_new = (test_lr["has_history"] == 0).values

pipe_ret = build_pipeline_lr(ret_numeric, ret_cat, C=0.1, l1_ratio=0.5)
pipe_ret.fit(train_lr[mask_train_ret], train_lr.loc[mask_train_ret, TARGET].astype(int))
pipe_new = build_pipeline_lr(new_numeric, new_cat, C=0.1, l1_ratio=0.5)
pipe_new.fit(train_lr[mask_train_new], train_lr.loc[mask_train_new, TARGET].astype(int))

prob_lr = np.zeros(len(test_lr))
prob_lr[mask_test_ret] = pipe_ret.predict_proba(test_lr[mask_test_ret])[:, 1]
prob_lr[mask_test_new] = pipe_new.predict_proba(test_lr[mask_test_new])[:, 1]
print("LR full-train fit done.")

# ================= RF full-train fit -> test predictions =================
train_rf = engineer_features_rf(train)
test_rf = engineer_features_rf(test)
train_rf = add_seasonality_features(train_rf)
test_rf = add_seasonality_features(test_rf)
train_rf = add_interactions(train_rf, train_ref=train_rf)
test_rf = add_interactions(test_rf, train_ref=train_rf)
_keep_race_edu_rf = set(train_rf["race_x_education"].unique()) - {"Other"}
test_rf["race_x_education"] = test_rf["race_x_education"].where(test_rf["race_x_education"].isin(_keep_race_edu_rf), "Other")

train_rf, test_rf = add_frequency_encoding(train_rf, test_rf, "municipality")
train_rf, test_rf = collapse_rare_categories(train_rf, test_rf, "education_level", min_count=100)
oof_rate_rf, full_rate_map_rf, g_mean_rf = kfold_target_encode(train_rf, "municipality", TARGET, train_rf[ID_COL], smoothing=20)
train_rf["municipality_emp_rate"] = oof_rate_rf
test_rf["municipality_emp_rate"] = test_rf["municipality"].map(full_rate_map_rf).fillna(g_mean_rf)

pipe_rf = build_pipeline_rf()
pipe_rf.fit(train_rf[rf_numeric + rf_categorical], train_rf[TARGET].astype(int))
prob_rf = pipe_rf.predict_proba(test_rf[rf_numeric + rf_categorical])[:, 1]
print("RF full-train fit done.")

# ================= global rank blend =================
n = len(test)
rank_lr = rankdata(prob_lr) / n
rank_rf = rankdata(prob_rf) / n
employed_prob = W_LR * rank_lr + (1 - W_LR) * rank_rf

submission = pd.DataFrame({ID_COL: test[ID_COL], "employed_prob": employed_prob})
submission.to_csv("Submissions/Blend_LR_RF.csv", index=False)
print(f"Saved Submissions/Blend_LR_RF.csv (W_LR={W_LR}, W_RF={1 - W_LR:.2f})")
print(submission["employed_prob"].describe())
