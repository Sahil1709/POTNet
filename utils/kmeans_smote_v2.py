import pandas as pd
import numpy as np
import smote_variants as sv
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('../data/hf_models_withmodelcard_nov2024.csv')[
    ['task_group','author_category','language_category','location','downloads_category']
]

# Define target and features
target   = 'downloads_category'
features = ['task_group','author_category','language_category','location']

def preprocess_data(df, features, target):
    # one‑hot encode ALL categories
    X = pd.get_dummies(df[features], drop_first=False)
    # label‑encode the target
    le = LabelEncoder()
    y  = le.fit_transform(df[target])
    return X.astype(float), y, le

def invert_one_hot(X_oh, features):
    inv = pd.DataFrame(index=X_oh.index)
    for f in features:
        # grab all columns that start with "f_"
        cols = [c for c in X_oh.columns if c.startswith(f + "_")]
        if not cols:
            continue
        # idxmax gives e.g. "task_group_Text Processing"
        s = X_oh[cols].idxmax(axis=1)
        # split on first "_" then take the second piece
        inv[f] = s.str.split("_", n=1).str.get(1)
    return inv

def apply_kmeans_smote(df_pair, features, target):
    X, y, le = preprocess_data(df_pair, features, target)
    sampler = sv.kmeans_SMOTE(random_state=42, n_clusters=21)
    X_res, y_res = sampler.sample(X.values, y)
    # back to DataFrame
    X_res_df = pd.DataFrame(X_res, columns=X.columns, index=np.arange(len(X_res)))
    y_res_sr = pd.Series(le.inverse_transform(y_res), name=target, index=X_res_df.index)
    # invert one‑hot into categories again
    X_orig = invert_one_hot(X_res_df, features)
    return pd.concat([X_orig, y_res_sr], axis=1)

# apply to each pair
outs = []
for lab in ["Low","Mid","High"]:
    print(f"--- Very Low vs {lab} ---")
    df_sub = df[df[target].isin(["Very Low", lab])]
    out = apply_kmeans_smote(df_sub, features, target)
    print(out[target].value_counts(),"\n")

    out.to_csv(f"kmeans_smote_{lab}.csv", index=False)
    outs.append(out)

# combine
final = pd.concat(outs, ignore_index=True)
print("Combined:\n", final[target].value_counts())

# save
final.to_csv("kmeans_smote_balanced.csv", index=False)

"""
# Tasks:

✅ 1. Investigage why kmeans_SMOTE is not working on Very Low vs High (High is still 15166)
Increating the number of clusters to 21 (from 10) seems to have fixed the issue.

✅ 2. The combined output distribution is Combined:
 downloads_category
Very Low    2603136  (867712 + 867712 + 867712) This shouldn't happen
Low          867712
Mid          867712
High          15166

It should be:
 downloads_category
Very Low     867712
Low          867712
Mid          867712
High          15166

(make sure this very low is same as the one in the original data)

✅ 3. Create the distribution graph for the combined data

✅ 4. Run random forest and logistic regression on this combimned data. 
    (Entire data for training and use hf_models_withmodelcard_nov2024 for testing)

Logistic regression: 0.7457
Random Forest: 0.7510

✅ 5. Train the model on entire data/generated/hf_11_24_balanced.csv and 
    use data/hf_models_withmodelcard_nov2024.csv for testing to get the accuracy

Logistic regression: 0.1707
Random Forest: 0.4530
"""