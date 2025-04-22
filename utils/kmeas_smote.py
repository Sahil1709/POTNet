import pandas as pd
import numpy as np
import smote_variants as sv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def kmeans_smote(df, features, target):
    # Preprocess using one-hot encoding (keep all categories)
    def preprocess_data(df, features, target):
        # One-hot encode features (no drop_first)
        X = pd.get_dummies(df[features], drop_first=False)
        # Also get mapping for each feature for later inversion
        onehot_mapping = {}
        for feature in features:
            # Find all dummy columns generated for this feature.
            cols = [col for col in X.columns if col.startswith(feature + "_")]
            onehot_mapping[feature] = cols
        # Label encode target if needed
        if df[target].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(df[target])
        else:
            y = df[target].values
            le = None
        return X, y, onehot_mapping, le

    X, y, onehot_mapping, target_le = preprocess_data(df, features, target)

    # Optional: Split the data if you wish to use only training data for oversampling.
    # For this example, we'll apply oversampling to the entire X and y.
    print("Original class distribution:", pd.Series(y).value_counts())

    # Apply SMOTE variant using smote_variants, e.g., KMeans_SMOTE
    oversampler = sv.MulticlassOversampling(
        oversampler='kmeans_SMOTE',
        oversampler_params={'random_state': 5}
    )
    X_samp, y_samp = oversampler.sample(X.astype(int), y)
    print("Resampled class distribution:", pd.Series(y_samp).value_counts())

    # Convert the oversampled array back into a DataFrame with the same one-hot columns:
    X_samp_df = pd.DataFrame(X_samp, columns=X.columns)

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

    # Invert one-hot encoded predictors back to their categorical values
    X_inverted = invert_one_hot(X_samp_df, onehot_mapping)

    # Optionally, also decode the target variable back to string labels if target_le exists:
    if target_le is not None:
        y_inverted = target_le.inverse_transform(y_samp)
        df_out = pd.concat([X_inverted, pd.Series(y_inverted, name=target)], axis=1)
    else:
        df_out = pd.concat([X_inverted, pd.Series(y_samp, name=target)], axis=1)

    # Display the final DataFrame with original categorical values
    print("Final oversampled DataFrame (first 10 rows):")
    print(df_out.head(10))

    return df_out



if __name__ == "__main__":
    df = pd.read_csv("../data/hf_models_withmodelcard_nov2024.csv")
    features = ['task_group', 'author_category', 'language_category', 'location']
    target = 'downloads_category'
    df_out = kmeans_smote(df, features, target)

    df_out.to_csv("oversampled_non_onehot.csv", index=False)