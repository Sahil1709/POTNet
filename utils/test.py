import pandas as pd
import numpy as np
import smote_variants as sv

df = pd.read_csv('../data/hf_models_withmodelcard_nov2024.csv')
print(df.downloads_category.value_counts())

target = 'downloads_category'
features = ['task_group', 'author_category', 'language_category', 'location']

X, y= df[features], df[target]
# Convert categorical features to numerical using one-hot encoding
X = pd.get_dummies(X, columns=features, drop_first=True)
# Check the distribution of the target variable
print(y.value_counts())

oversampler= sv.MulticlassOversampling(oversampler='distance_SMOTE',
                                      oversampler_params={'random_state': 5})

# X_samp and y_samp contain the oversampled dataset
X_samp, y_samp= oversampler.sample(X, y)

print(np.unique(y_samp, return_counts=True))

oversampler2= sv.distance_SMOTE()
X_np = X.values
X_samp2, y_samp2= oversampler2.sample(X, y)
print(np.unique(y_samp2, return_counts=True))