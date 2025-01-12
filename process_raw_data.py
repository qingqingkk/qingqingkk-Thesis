import os
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import argparse

'''
Process raw data
'''
def preprocess_data(data_path, output_dir):
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

    # encode labels
    label_encoder = LabelEncoder()
    cs_df['label'] = label_encoder.fit_transform(cs_df['s/p'])
    sv_df['label'] = label_encoder.fit_transform(sv_df['s/p'])

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
    preprocess_data(data_path, output_dir)

