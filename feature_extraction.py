import numpy as np
from transformers import Wav2Vec2Processor

def load_feature_extractor(model_name):
    return Wav2Vec2Processor.from_pretrained(model_name)

def extract_features(dataset, processor):
    def _extract_features(batch):
        inputs = processor(batch["speech"], sampling_rate=16000, padding=True, return_tensors="pt")
        batch["input_values"] = inputs.input_values[0]
        return batch

    dataset = dataset.map(_extract_features)
    dataset.set_format(type='torch', columns=['input_values', 'label'])
    return dataset

