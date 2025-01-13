import os
import pandas as pd
import numpy as np
import librosa
import random
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor


"""
Modules:
- load feature extractor
- AudioDataset: for single mode, concatenated dataset.
- MidFusionAudioDataset: for mid-level fusion method.
- load_csv: Splits datasets into training, validation, and testing.
- load_data: Prepares DataLoader.

"""


class AudioDataset(Dataset):
    def __init__(self, examples, feature_extractor, max_duration, sr=16000, augmentation=False, da_percentage=0):
        self.audio_paths = examples['path'].tolist()
        self.labels = examples['label'].tolist()
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration
        self.sr = sr
        self.augmentation = augmentation
        self.da_percentage = da_percentage if augmentation else 0

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        audio, _ = librosa.load(audio_path, sr=self.sr)

        if self.augmentation and random.random() < self.da_percentage:
            audio = self.apply_augmentation(audio)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.feature_extractor.feature_extractor.sampling_rate,
            return_tensors="pt",
            max_length=int(self.feature_extractor.feature_extractor.sampling_rate * self.max_duration),
            truncation=True,
            padding='max_length'
        )

        return {
            'input_values': inputs['input_values'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    @staticmethod
    def apply_augmentation(audio):
        augment_type = random.choice(['add_noise', 'time_stretch', 'pitch_shift', 'combined'])
        if augment_type == 'add_noise':
            noise = np.random.normal(0, random.uniform(0.001, 0.015), audio.shape[0])
            audio += noise
        elif augment_type == 'time_stretch':
            audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.8, 1.25))
        elif augment_type == 'pitch_shift':
            audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=random.uniform(-4, 4))
        elif augment_type == 'combined':
            noise = np.random.normal(0, random.uniform(0.001, 0.015), audio.shape[0])
            audio += noise
            audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.8, 1.25))
            audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=random.uniform(-4, 4))
        return audio


class MidFusionAudioDataset(Dataset):
    def __init__(self, cs_examples, sv_examples, feature_extractor, max_duration, sr=16000, augmentation=False, da_percentage=0):
        self.cs_paths = cs_examples['path'].tolist()
        self.sv_paths = sv_examples['path'].tolist()
        self.labels = cs_examples['label'].tolist()
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration
        self.sr = sr
        self.augmentation = augmentation
        self.da_percentage = da_percentage if augmentation else 0

    def __len__(self):
        return len(self.cs_paths)

    def __getitem__(self, idx):
        cs_audio, _ = librosa.load(self.cs_paths[idx], sr=self.sr)
        sv_audio, _ = librosa.load(self.sv_paths[idx], sr=self.sr)

        if self.augmentation and random.random() < self.da_percentage:
            cs_audio = self.apply_augmentation(cs_audio)
            sv_audio = self.apply_augmentation(sv_audio)

        cs_inputs = self.feature_extractor(
            cs_audio,
            sampling_rate=self.feature_extractor.feature_extractor.sampling_rate,
            return_tensors="pt",
            max_length=int(self.feature_extractor.feature_extractor.sampling_rate * self.max_duration),
            truncation=True,
            padding='max_length'
        )
        sv_inputs = self.feature_extractor(
            sv_audio,
            sampling_rate=self.feature_extractor.feature_extractor.sampling_rate,
            return_tensors="pt",
            max_length=int(self.feature_extractor.feature_extractor.sampling_rate * self.max_duration),
            truncation=True,
            padding='max_length'
        )

        return {
            'cs_input_values': cs_inputs['input_values'].squeeze(0),
            'cs_attention_mask': cs_inputs['attention_mask'].squeeze(0),
            'sv_input_values': sv_inputs['input_values'].squeeze(0),
            'sv_attention_mask': sv_inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    @staticmethod
    def apply_augmentation(audio):
        augment_type = random.choice(['add_noise', 'time_stretch', 'pitch_shift', 'combined'])
        if augment_type == 'add_noise':
            noise = np.random.normal(0, random.uniform(0.001, 0.015), audio.shape[0])
            audio += noise
        elif augment_type == 'time_stretch':
            audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.8, 1.25))
        elif augment_type == 'pitch_shift':
            audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=random.uniform(-4, 4))
        elif augment_type == 'combined':
            noise = np.random.normal(0, random.uniform(0.001, 0.015), audio.shape[0])
            audio += noise
            audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.8, 1.25))
            audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=random.uniform(-4, 4))
        return audio



# load dataset and split into train, valid, test dataframe
def load_csv(args):
    def read_csv(path, name):
        csv_path = os.path.join(path, name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"'{csv_path}' not found. Please verify or preprocess the data.")
        return pd.read_csv(csv_path)

    data_path = args.data_path
    SEED = args.seed

    if args.strategy == 'early':
        concat_df = read_csv(data_path, 'concat_dataset.csv')
        train_valid, test = train_test_split(concat_df, test_size=0.1, stratify=concat_df['label'], random_state=20)
        train, valid = train_test_split(train_valid, test_size=0.1111, stratify=train_valid['label'], random_state=SEED)
        return train, valid, test

    cs_df = read_csv(data_path, 'cs_dataset.csv')
    sv_df = read_csv(data_path, 'sv_dataset.csv')

    cs_train_valid, cs_test = train_test_split(cs_df, test_size=0.1, stratify=cs_df['label'], random_state=20)
    cs_train, cs_valid = train_test_split(cs_train_valid, test_size=0.1111, stratify=cs_train_valid['label'], random_state=SEED)

    sv_train_valid, sv_test = train_test_split(sv_df, test_size=0.1, stratify=sv_df['label'], random_state=20)
    sv_train, sv_valid = train_test_split(sv_train_valid, test_size=0.1111, stratify=sv_train_valid['label'], random_state=SEED)

    return (cs_train, cs_valid, cs_test), (sv_train, sv_valid, sv_test)


# Load the separated df and use the predefined Processor for processing and feature extraction
def load_data(args):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    if args.strategy == 'mid':
        (cs_train, cs_valid, cs_test), (sv_train, sv_valid, sv_test) = load_csv(args)
        return (
            MidFusionAudioDataset(cs_train, sv_train, processor, args.max_duration, augmentation=args.da, da_percentage=args.da_percentage),
            MidFusionAudioDataset(cs_valid, sv_valid, processor, args.max_duration),
            MidFusionAudioDataset(cs_test, sv_test, processor, args.max_duration)
        )
    
    elif args.strategy == 'late':
        (cs_train, cs_valid, cs_test), (sv_train, sv_valid, sv_test) = load_csv(args)
        return [
            DataLoader(AudioDataset(cs_valid, processor, args.max_duration), batch_size=args.batch_size),
            DataLoader(AudioDataset(cs_test, processor, args.max_duration), batch_size=args.batch_size)
        ], [
            DataLoader(AudioDataset(sv_valid, processor, args.max_duration), batch_size=args.batch_size),
            DataLoader(AudioDataset(sv_test, processor, args.max_duration), batch_size=args.batch_size)
        ]
    
    else:
        ### single mode or early fusion
        train, valid, test = load_csv(args)
        return (
            DataLoader(AudioDataset(train, processor, args.max_duration, augmentation=args.da, da_percentage=args.da_percentage), batch_size=args.batch_size, shuffle=True),
            DataLoader(AudioDataset(valid, processor, args.max_duration), batch_size=args.batch_size),
            DataLoader(AudioDataset(test, processor, args.max_duration), batch_size=args.batch_size)
        )

