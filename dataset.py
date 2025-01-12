from transformers import Wav2Vec2Processor
import os
import pandas as pd
import re
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from datasets import Dataset
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch
import random

from feature_extraction import load_feature_extractor

'''
Load prepared data
'''
class MidFusion_AudioDataset(torch.utils.data.Dataset):
    def __init__(self, cs_examples, sv_examples, feature_extractor, max_duration, da_percentage=0, augmentation=False):
        self.cs_path = cs_examples['path'].tolist()  # Convert paths to a list of strings
        self.sv_path = sv_examples['path'].tolist()
        self.labels = cs_examples['label'].tolist()
        self.feature_extractor = feature_extractor
        # IPV: 90% of the audio lengths in the cs dataset are 19, sv dataset are 18, concatenated 35
        # NEW: 90% of the audio lengths in the cs dataset are 26, sv dataset are , concatenated 
        self.max_duration = max_duration
        self.augmentation = augmentation
        self.da_percentage = da_percentage if augmentation else 0
        self.sr = 16_000

    def __getitem__(self, idx):
        cs_audio_path = self.cs_path[idx]
        sv_audio_path = self.sv_path[idx]
        if self.augmentation:
             # chose 30%~~100% to augment

            augment = np.random.choice([True, False], p=[self.da_percentage, 1 - self.da_percentage])
            if augment:
                # Load CS and SV audio
                cs_audio, _= librosa.load(cs_audio_path, sr=self.sr)
                sv_audio, _ = librosa.load(sv_audio_path, sr=self.sr)
                
                cs_audio = self.apply_augmentation(cs_audio)
                sv_audio = self.apply_augmentation(sv_audio)
                
                # Extract features
                cs_inputs = self.feature_extractor(
                    cs_audio.squeeze(),
                    sampling_rate=self.feature_extractor.feature_extractor.sampling_rate, 
                    return_tensors="pt",
                    max_length=int(self.feature_extractor.feature_extractor.sampling_rate * self.max_duration), 
                    truncation=True,
                    padding='max_length',
                    )
                sv_inputs = self.feature_extractor(
                    sv_audio.squeeze(),
                    sampling_rate=self.feature_extractor.feature_extractor.sampling_rate, 
                    return_tensors="pt",
                    max_length=int(self.feature_extractor.feature_extractor.sampling_rate * self.max_duration), 
                    truncation=True,
                    padding='max_length',
                    )
            else:
                cs_inputs= self.feature_extractor(
                    librosa.load(cs_audio_path, sr=self.sr)[0].squeeze(),
                    sampling_rate=self.feature_extractor.feature_extractor.sampling_rate, 
                    return_tensors="pt",
                    max_length=int(self.feature_extractor.feature_extractor.sampling_rate * self.max_duration), 
                    truncation=True,
                    padding='max_length',
                )
                sv_inputs= self.feature_extractor(
                    librosa.load(sv_audio_path, sr=self.sr)[0].squeeze(),
                    sampling_rate=self.feature_extractor.feature_extractor.sampling_rate, 
                    return_tensors="pt",
                    max_length=int(self.feature_extractor.feature_extractor.sampling_rate * self.max_duration), 
                    truncation=True,
                    padding='max_length',
                )
        else:
            cs_inputs= self.feature_extractor(
                    librosa.load(cs_audio_path, sr=self.sr)[0].squeeze(),
                    sampling_rate=self.feature_extractor.feature_extractor.sampling_rate, 
                    return_tensors="pt",
                    max_length=int(self.feature_extractor.feature_extractor.sampling_rate * self.max_duration), 
                    truncation=True,
                    padding='max_length',
                )
            sv_inputs= self.feature_extractor(
                librosa.load(sv_audio_path, sr=self.sr)[0].squeeze(),
                sampling_rate=self.feature_extractor.feature_extractor.sampling_rate, 
                return_tensors="pt",
                max_length=int(self.feature_extractor.feature_extractor.sampling_rate * self.max_duration), 
                truncation=True,
                padding='max_length',
            )
              
     
        return (cs_inputs,sv_inputs),self.labels[idx]

    def __len__(self):
        return len(self.cs_path)

    def apply_augmentation(self, audio):
        """Applies random augmentation to the audio."""
        augmentation_type = np.random.choice(['add_noise', 'time_stretch', 'pitch_shift', 'combined'])
        if augmentation_type == 'add_noise':
            noise_factor = random.uniform(0.001, 0.015)
            noise = np.random.normal(0, noise_factor, audio.shape[0])
            audio = audio + noise
        elif augmentation_type == 'time_stretch':
            audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.8, 1.25))
        elif augmentation_type == 'pitch_shift':
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=random.uniform(-4, 4))
        elif augmentation_type == 'combined':
            noise_factor = random.uniform(0.001, 0.015)
            noise = np.random.normal(0, noise_factor, audio.shape[0])
            audio = audio + noise
            audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.8, 1.25))
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=random.uniform(-4, 4))
        return audio
    
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, examples, feature_extractor, max_duration, da_percentage=0, augmentation=False):
        self.data_path = examples['path'].tolist()  # Convert paths to a list of strings
        self.labels = examples['label'].tolist()
        self.feature_extractor = feature_extractor
        # IPV: 90% of the audio lengths in the cs dataset are 19, sv dataset are 18, concatenated 35
        # NEW: 90% of the audio lengths in the cs dataset are 26, sv dataset are , concatenated 
        self.max_duration = max_duration
        self.augmentation = augmentation
        self.da_percentage = da_percentage if augmentation else 0
        self.sr = 16_000

    def __getitem__(self, idx):
        audio_path = self.data_path[idx]
        if self.augmentation:
             # chose 30%~~100% to augment

            augment = np.random.choice([True, False], p=[self.da_percentage, 1 - self.da_percentage])
            if augment:
                # Load CS and SV audio
                audio, _= librosa.load(audio_path, sr=self.sr)
                audio = self.apply_augmentation(audio)
                
                # Extract features
                inputs = self.feature_extractor(
                    audio.squeeze(),
                    sampling_rate=self.feature_extractor.feature_extractor.sampling_rate, 
                    return_tensors="pt",
                    max_length=int(self.feature_extractor.feature_extractor.sampling_rate * self.max_duration), 
                    truncation=True,
                    padding='max_length',
                    )
            else:
                inputs = self.feature_extractor(
                    audio.squeeze(),
                    sampling_rate=self.feature_extractor.feature_extractor.sampling_rate, 
                    return_tensors="pt",
                    max_length=int(self.feature_extractor.feature_extractor.sampling_rate * self.max_duration), 
                    truncation=True,
                    padding='max_length',
                    )
        else:
            inputs = self.feature_extractor(
                audio.squeeze(),
                sampling_rate=self.feature_extractor.feature_extractor.sampling_rate, 
                return_tensors="pt",
                max_length=int(self.feature_extractor.feature_extractor.sampling_rate * self.max_duration), 
                truncation=True,
                padding='max_length',
                )
        item = {
        'input_values': inputs['input_values'].squeeze(0),
        'prefix': self.prefixes[idx],
        'attention_mask': inputs['attention_mask'].squeeze(0),
            }
        return item, torch.tensor(self.labels[idx], dtype=torch.long)

    def __len__(self):
        return len(self.data_path)

    def apply_augmentation(self, audio):
        """Applies random augmentation to the audio."""
        augmentation_type = np.random.choice(['add_noise', 'time_stretch', 'pitch_shift', 'combined'])
        if augmentation_type == 'add_noise':
            noise_factor = random.uniform(0.001, 0.015)
            noise = np.random.normal(0, noise_factor, audio.shape[0])
            audio = audio + noise
        elif augmentation_type == 'time_stretch':
            audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.8, 1.25))
        elif augmentation_type == 'pitch_shift':
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=random.uniform(-4, 4))
        elif augmentation_type == 'combined':
            noise_factor = random.uniform(0.001, 0.015)
            noise = np.random.normal(0, noise_factor, audio.shape[0])
            audio = audio + noise
            audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.8, 1.25))
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=random.uniform(-4, 4))
        return audio
    
def load_csv(args):
    data_path = args.data_path
    SEED=args.seed
    cs_path = os.path.join(data_path, 'cs_dataset.csv')
    sv_path = os.path.join(data_path, 'sv_dataset.csv')
    # check if csv files exist
    if os.path.exists(cs_path):
        pass
    else:
        raise FileNotFoundError(f"Cannot find '{cs_path}' please check the path or run process_raw_data.py")
    if os.path.exists(sv_path):
        pass
    else:
        raise FileNotFoundError(f"Cannot find '{sv_path}' please check the path or run process_raw_data.py")
    
    if args.strategy == 'early':
        concat_path = os.path.join(data_path, 'concat_dataset.csv')
        if os.path.exists(sv_path):
            pass
        else:
            raise FileNotFoundError(f"Cannot find '{concat_path}' please check the path or run combined_cd_sv.py")
        
        concat_df = pd.read_csv(concat_path)
        con_train_valid_df, con_test_df = train_test_split(concat_df, test_size=0.1, stratify=concat_df['label'], random_state=20)
        con_train_df, con_valid_df = train_test_split(con_train_valid_df, test_size=0.1111, stratify=cs_train_valid_df['label'],random_state=SEED)
        
        return con_train_df, con_valid_df, con_test_df
    elif args.strategy == 'mid' or args.strategy == 'late':
        cs_df = pd.read_csv(cs_path)

        cs_train_valid_df, cs_test_df = train_test_split(cs_df, test_size=0.1, stratify=cs_df['label'], random_state=20)
        cs_train_df, cs_valid_df = train_test_split(cs_train_valid_df, test_size=0.1111, stratify=cs_train_valid_df['label'],random_state=SEED)
        sv_df = pd.read_csv(sv_path)

        sv_train_valid_df, sv_test_df = train_test_split(sv_df, test_size=0.1, stratify=sv_df['label'], random_state=20)
        sv_train_df, sv_valid_df = train_test_split(sv_train_valid_df, test_size=0.1111, stratify=sv_train_valid_df['label'],random_state=SEED)

        return [cs_train_df, cs_valid_df, cs_test_df], [sv_train_df, sv_valid_df, sv_test_df]
    else:
        if args.modality == 'cs':
            cs_df = pd.read_csv(cs_path)

            cs_train_valid_df, cs_test_df = train_test_split(cs_df, test_size=0.1, stratify=cs_df['label'], random_state=20)
            cs_train_df, cs_valid_df = train_test_split(cs_train_valid_df, test_size=0.1111, stratify=cs_train_valid_df['label'],random_state=SEED)

            return cs_train_df, cs_valid_df, cs_test_df
        
        elif args.modality == 'sv':
            sv_df = pd.read_csv(sv_path)

            sv_train_valid_df, sv_test_df = train_test_split(sv_df, test_size=0.1, stratify=sv_df['label'], random_state=20)
            sv_train_df, sv_valid_df = train_test_split(sv_train_valid_df, test_size=0.1111, stratify=sv_train_valid_df['label'],random_state=SEED)

            return sv_train_df, sv_valid_df, sv_test_df
    
def load_data(args):
    processor = load_feature_extractor()
    
    if args.strategy == 'mid':
        cs, sv = load_csv(args) #cs[0] - train, cs[1] - valid, cs[2] - test
        train_dataset = MidFusion_AudioDataset(
            cs_examples=cs[0],
            sv_examples=sv[0],
            feature_extractor=processor,
            max_duration=18,
            augmentation=args.da 
        )
        
        valid_dataset = MidFusion_AudioDataset(
            cs_examples=cs[1],
            sv_examples=sv[1],
            feature_extractor=processor,
            max_duration=18,
            augmentation=False 
        )

        test_dataset = MidFusion_AudioDataset(
            cs_examples=cs[2],
            sv_examples=sv[2],
            feature_extractor=processor,
            max_duration=18,
            augmentation=False  # No argumentation
        )
    elif args.strategy == 'late':
        cs, sv = load_csv(args)

        cs_valid_dataset = AudioDataset(
            cs_examples=cs[1],
            feature_extractor=processor,
            max_duration=18,
            augmentation=False 
        )

        cs_test_dataset = AudioDataset(
            cs_examples=cs[2],
            feature_extractor=processor,
            max_duration=18,
            augmentation=False  # No argumentation
        )

        sv_valid_dataset = AudioDataset(
            sv_examples=sv[1],
            feature_extractor=processor,
            max_duration=18,
            augmentation=False 
        )

        sv_test_dataset = AudioDataset(
            sv_examples=sv[2],
            feature_extractor=processor,
            max_duration=18,
            augmentation=False  # No argumentation
        )
        cs_valid_loader = DataLoader(cs_valid_dataset, batch_size=args.batch_size, shuffle=False)
        cs_test_loader = DataLoader(cs_test_dataset, batch_size=args.batch_size, shuffle=False)
        sv_valid_loader = DataLoader(sv_valid_dataset, batch_size=args.batch_size, shuffle=False)
        sv_test_loader = DataLoader(sv_test_dataset, batch_size=args.batch_size, shuffle=False)

        return [cs_valid_loader, cs_test_loader], [sv_valid_loader, sv_test_loader]
    else:
        train_df, valid_df, test_df = load_csv(args)
        train_dataset = AudioDataset(
            examples=train_df,
            feature_extractor=processor,
            max_duration=18,
            augmentation=args.da 
        )
        
        valid_dataset = AudioDataset(
            examples=valid_df,
            feature_extractor=processor,
            max_duration=18,
            augmentation=False 
        )


        test_dataset = AudioDataset(
            examples=test_df,
            feature_extractor=processor,
            max_duration=18,
            augmentation=False  # No argumentation
        )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader
    

