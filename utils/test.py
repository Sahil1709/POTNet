import pandas as pd
import numpy as np
import smote_variants as sv

df = pd.read_csv('../data/hf_models_withmodelcard_nov2024.csv')
print("Columns in DataFrame:", df.columns)

target = 'downloads_category'
features = ['task_group', 'author_category', 'language_category', 'location']

# Check for missing columns
for feature in features:
    if feature not in df.columns:
        print(f"Missing column: {feature}")

X, y = df[features], df[target]

# Convert categorical features to numerical using one-hot encoding
X = pd.get_dummies(X, columns=features, drop_first=True)
print("Columns after one-hot encoding:", X.columns)

# Check the distribution of the target variable
print(y.value_counts())

oversampler = sv.MulticlassOversampling(oversampler='distance_SMOTE',
                                        oversampler_params={'random_state': 5})

# X_samp and y_samp contain the oversampled dataset
X_samp, y_samp = oversampler.sample(X, y)
print(np.unique(y_samp, return_counts=True))

print("🐍" * 10)
print("Using smote_variants directly")
print("🐍" * 10)

oversampler2 = sv.distance_SMOTE()
X_np = X.values
X_samp2, y_samp2 = oversampler2.sample(X_np, y)
print(np.unique(y_samp2, return_counts=True))