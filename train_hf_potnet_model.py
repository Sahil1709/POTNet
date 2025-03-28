import pandas as pd
import ast, time
from potnet import *
import argparse
import os

def train_potnet_model(model_name: str, data: pd.DataFrame):
    if 'location' not in data.columns and 'tags' in data.columns:
        data['tags'] = data['tags'].apply(ast.literal_eval)
        data['location'] = data['tags'].apply(
            lambda tags: next((tag.split(':', 1)[1] for tag in tags if tag.startswith('region:')), None)
        )
        print(data.location.value_counts())

    data = data[['task_group', 'author_category', 'language_category', 'downloads_category', 'location']]

    if 'base_model_category' in data.columns:
        data['base_model_category'] = data['base_model_category'].fillna('unknown')

    print(data.info())
    print(data.isna().sum())

    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    print(len(categorical_columns))
    print(categorical_columns)

    start_time = time.time()
    print(f"Start time: {time.ctime()}")

    potnet_model= POTNet(embedding_dim=data.shape[1],
                        categorical_cols=categorical_columns,
                        numeric_output_data_type = 'integer',
                        epochs=100,
                        batch_size=1024,
                        save_checkpoint=True,
                        checkpoint_epoch=50,
                        overwrite_checkpoint=True,
                        verbose=True
                        )

    potnet_model.fit(data)

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
    potnet_model.save(model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and save POTNet model.')
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Name of the model file to save without .pt')
    parser.add_argument('-d', '--data_path', type=str, required=True, help='Path to the CSV file containing the dataset')
    args = parser.parse_args()
    model_name = args.model_name + '.pt'
    data = pd.read_csv(args.data_path)
    if os.path.exists(model_name):
        raise FileExistsError(f"The model file '{model_name}' already exists. Please choose a different name.")
    train_potnet_model(model_name, data)