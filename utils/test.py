import pandas as pd
import numpy as np
import smote_variants as sv
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../data/hf_models_withmodelcard_nov2024.csv')[['task_group', 'author_category', 'language_category', 'location', 'downloads_category']]
# for testing Very Low and Low
df = df[df.downloads_category.isin(['Very Low', 'High'])].copy()

print(df.downloads_category.value_counts())

target = 'downloads_category'
features = ['task_group', 'author_category', 'language_category', 'location']

def preprocess_data(df, features, target):
    """
    Preprocess the data:
     - One-hot encode the features to get numeric columns.
     - Label encode the target if necessary.
     - Return processed X and y.
    """
    # One-hot encode the predictor columns
    X = pd.get_dummies(df[features], drop_first=True)
    
    # Label encode the target column if it's not numeric
    if df[target].dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(df[target])
    else:
        y = df[target].values

    return X, y

# Function to invert one-hot encoding back to original categorical values
def invert_one_hot(df_onehot, mapping):
    # df_onehot: DataFrame of one-hot columns.
    # mapping: dictionary where key is feature name and value is list of dummy columns for that feature.
    df_inverted = pd.DataFrame()
    for feature, cols in mapping.items():
        def row_to_category(row):
            # Check if the row has any ones in these columns.
            if row[cols].sum() == 0:
                # If not, return NA or a default value.
                return np.nan
            # Return the category name corresponding to the column with the highest value.
            # Since it's one-hot, we can use idxmax.
            col_name = row[cols].idxmax()
            # Remove the feature_ prefix to get the original category value.
            return col_name.replace(feature + "_", "")
        df_inverted[feature] = df_onehot.apply(row_to_category, axis=1)
    return df_inverted

# Check for missing columns
for feature in features:
    if feature not in df.columns:
        print(f"Missing column: {feature}")

X, y = preprocess_data(df, features, target)
X = X.astype(float)
X = X.values

oversampler = sv.kmeans_SMOTE()

# X_samp and y_samp contain the oversampled dataset
X_samp, y_samp = oversampler.sample(X, y)
print(np.unique(y_samp, return_counts=True))