import numpy as np
from transformers import Wav2Vec2Processor

'''
basic methods
'''

def load_feature_extractor():
    return Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def extract_features(dataset, processor):
    def _extract_features(batch):
        inputs = processor(batch["speech"], sampling_rate=16000, padding=True, return_tensors="pt")
        batch["input_values"] = inputs.input_values[0]
        return batch

    dataset = dataset.map(_extract_features)
    dataset.set_format(type='torch', columns=['input_values', 'label'])
    return dataset