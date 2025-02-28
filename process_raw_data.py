import os
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import argparse

"""
Separate raw data into CS and SV datasets.

This script processes the raw dataset by:
1. Separating `cs` (connected sentence) and `sv` (sustained vowel) voice recordings into two datasets.
2. Saving the separated data into two CSV files for further processing or analysis.

"""

def process_data(data_path, output_dir, label_column):
    # Read CSV file
    casi_df = pd.read_csv(os.path.join(data_path, 'CASI.csv'))
    contro_df = pd.read_csv(os.path.join(data_path, 'CONTROLLI.csv'))

    # Replace path
    casi_df['path'] = casi_df['path'].map(lambda x: x.replace("audios", data_path))
    contro_df['path'] = contro_df['path'].map(lambda x: x.replace("audios", data_path))

    # Remove special symbols in the path
    casi_df['path'] = casi_df['path'].apply(remove_special_characters)
    contro_df['path'] = contro_df['path'].apply(remove_special_characters)

    # filter DataFrame
    casi_cs = casi_df[casi_df['path'].str.contains('cs')]
    casi_sv = casi_df[casi_df['path'].str.contains('sv')]
    contro_cs = contro_df[contro_df['path'].str.contains('cs')]
    contro_sv = contro_df[contro_df['path'].str.contains('sv')]

    # connected sentence or vowel DataFrame
    cs_df = pd.concat([contro_cs, casi_cs], ignore_index=True)
    sv_df = pd.concat([contro_sv, casi_sv], ignore_index=True)

    # Check if the label column exists
    if label_column not in cs_df.columns or label_column not in sv_df.columns:
        raise ValueError(f"The specified label column '{label_column}' does not exist in the dataset.")

    # Encode labels
    label_encoder = LabelEncoder()
    cs_df['label'] = label_encoder.fit_transform(cs_df[label_column])
    sv_df['label'] = label_encoder.fit_transform(sv_df[label_column])

    # save into the CSV files
    cs_df.to_csv(os.path.join(output_dir, 'cs_dataset.csv'), index=False)
    sv_df.to_csv(os.path.join(output_dir, 'sv_dataset.csv'), index=False)
    print(f"Sucess! Save cs file in {os.path.join(output_dir, 'cs_dataset.csv')} and sv file in {os.path.join(output_dir, 'sv_dataset.csv')}")

def remove_special_characters(text):
    return re.sub(r'[^A-Za-z0-9_( )./-]', '', text)

def parse_arguments():    

    parser = argparse.ArgumentParser(description="Preprocess data")

    # dataset parameters
    parser.add_argument("--data_path", type=str, required=True, default="./datasets", help="Path of dataset csv")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for model and results")
    parser.add_argument("--label_column", type=str, required=True, help="The column name to be used as labels (e.g., 's/p')")

    args = parser.parse_args()
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f'Path: {args.data_path} does not exists')
    if not os.path.exists(args.output_dir):
        raise FileNotFoundError(f'Path: {args.output_dir} does not exists') 
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    data_path = args.data_path
    output_dir = args.output_dir
    process_data(data_path, output_dir,args.label_column)

