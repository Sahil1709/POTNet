import pandas as pd
import numpy as np
import smote_variants as sv
from sklearn.preprocessing import LabelEncoder
import logging
import os

# Configure logging
os.makedirs("../logs", exist_ok=True)
logging.basicConfig(
    filename='../logs/kmeans_smote_v2.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 1) load your df
logger.info("Loading dataset...")
df = pd.read_csv("../data/hf_models_withmodelcard_nov2024.csv")
logger.info(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns.")


features = ['task_group', 'author_category', 'language_category', 'location']
target = 'downloads_category'

# 2) preprocess entire df once: one-hot encode features, label‑encode target
def preprocess_data(df, features, target):
    logger.info("Preprocessing data...")
    # one-hot encode predictors
    X = pd.get_dummies(df[features], drop_first=False)
    logger.info(f"One-hot encoding completed. Resulting columns: {list(X.columns)}")
    # build mapping from feature→dummy‑cols
    onehot_mapping = {
        feat: [c for c in X.columns if c.startswith(f"{feat}_")]
        for feat in features
    }
    # label encode target
    le = None
    if df[target].dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(df[target])
        logger.info("Target column label-encoded.")
    else:
        y = df[target].values
    return X, y, onehot_mapping, le

X_full, y_full, onehot_mapping, target_le = preprocess_data(df, features, target)

# 3) decide how many rows each class should have
class_counts = pd.Series(y_full).value_counts()
desired_count = class_counts.max()
logger.info(f"Class distribution before oversampling: {class_counts.to_dict()}")
logger.info(f"Desired count per class: {desired_count}")

# prepare our oversampler
oversampler = sv.MulticlassOversampling(
    oversampler='kmeans_SMOTE',
    oversampler_params={'random_state': 5},
    oversampler_strategy='eq_1_vs_many_parallel'
)
logger.info("Initialized KMeans_SMOTE oversampler.")

# helper to invert one-hot back to categorical
def invert_one_hot(df_onehot, mapping):
    df_inv = pd.DataFrame(index=df_onehot.index)
    for feat, cols in mapping.items():
        def pick_cat(row):
            if row[cols].sum() == 0:
                return np.nan
            return row[cols].idxmax().replace(f"{feat}_", "")
        df_inv[feat] = df_onehot[cols].apply(pick_cat, axis=1)
    return df_inv

# 4) loop over each class
pieces = []
for cls_val, cls_count in class_counts.items():
    logger.info(f"Processing class {cls_val} with {cls_count} samples.")
    idx = np.where(y_full == cls_val)[0]
    X_sub = X_full.iloc[idx]
    y_sub = y_full[idx]
    
    if cls_count < desired_count:
        try:
            # compute how many synthetic samples needed
            X_res, y_res = oversampler.sample(X_sub.values.astype(int), y_sub)
            logger.info(f"Generated {len(y_res) - cls_count} synthetic samples for class {cls_val}.")
            
            # if it overshoots desired_count, truncate:
            if len(y_res) > desired_count:
                X_res = X_res[:desired_count]
                y_res = y_res[:desired_count]
                logger.info(f"Truncated oversampled data for class {cls_val} to {desired_count} samples.")
            
            X_res_df = pd.DataFrame(X_res, columns=X_sub.columns)
            inv_df = invert_one_hot(X_res_df, onehot_mapping)
            if target_le is not None:
                y_res_lab = target_le.inverse_transform(y_res)
            else:
                y_res_lab = y_res
            inv_df[target] = y_res_lab
            pieces.append(inv_df)
        except Exception as e:
            logger.error(f"Error during oversampling for class {cls_val}: {e}")
    else:
        # no resampling needed—even keep original ordering
        orig_df = df.iloc[idx][features + [target]].reset_index(drop=True)
        pieces.append(orig_df)

# 5) build the final balanced DataFrame
balanced_df = pd.concat(pieces, ignore_index=True)
logger.info("Final balanced DataFrame created.")

# verify
final_distribution = balanced_df[target].value_counts()
logger.info(f"Final class distribution: {final_distribution.to_dict()}")
print("Final distribution:\n", final_distribution)

balanced_df.to_csv("kmeans_smote_balanced.csv", index=False)