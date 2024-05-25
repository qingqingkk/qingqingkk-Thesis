import os
import pandas as pd
import re
import librosa
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

def load_and_preprocess_data(casi_csv, contro_csv, path):
    # Read CSV file
    casi_df = pd.read_csv(casi_csv)
    contro_df = pd.read_csv(contro_csv)

    # Replace path
    casi_df['path'] = casi_df['path'].map(lambda x: x.replace("audios/casi", path))
    contro_df['path'] = contro_df['path'].map(lambda x: x.replace("audios/controlli", path))

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
    # cs_df.to_csv('/kaggle/working/cs_df.csv', index=False)
    # sv_df.to_csv('/kaggle/working/sv_df.csv', index=False)

    return cs_df, sv_df

def remove_special_characters(text):
    return re.sub(r'[^A-Za-z0-9_( )./-]', '', text)

def preprocess_and_load_dataset(healthy_csv, diseased_csv, path, modality='both'):
    cs_df, sv_df = load_and_preprocess_data(healthy_csv, diseased_csv, path)
    
    if modality == 'cs':
        dataset = Dataset.from_pandas(cs_df)
    elif modality == 'sv':
        dataset = Dataset.from_pandas(sv_df)
    else:
        combined_df = pd.concat([cs_df, sv_df], ignore_index=True)
        dataset = Dataset.from_pandas(combined_df)
    
    dataset = dataset.map(speech_file_to_array_fn)
    return dataset

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["path"], sr=None)
    target_sampling_rate = 16000
    if sampling_rate != target_sampling_rate:
        speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=target_sampling_rate)
    batch["speech"] = np.array(speech_array)
    return batch
