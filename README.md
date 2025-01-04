# Voice Disorder Diagnoses
## Fine-tune with cs or sv
The baseline of this paper
### how to use it
- Please make sure your dataset folder as the following structure:
```
datasets/
├── cs
│    ├──audio1_cs.wav
│    ├──audio2_cs.wav
│    └──...
├── sv
│    ├──audio1_sv.wav
│    ├──audio2_sv.wav
│    └──...
├── combined
│    ├──audio1_combined.wav
│    ├──audio2_combined.wav
│    └──...
├── cs_dataset.csv
├── sv_dataset.csv
└── concatenate.csv

```
#### Run the code
```
$python3 main.py --data_path=./datasets --output_dir=./results --modality=cs 
```

## Data argument cs or sv
### How to activate it
```
$python3 main.py --data_path=./datasets --output_dir=./results --modality=cs --da
```
#### Augument methods
todo

## Benchmark
With MLP(Multilayer Perceptron) or CNN (freeze/not freeze can be chosen)
#### Run the code
```
$python3 main.py --data_path=./datasets --output_dir=./results --strategy=benchmark
```

## Fusion model

### Fusion model details
- **Early fusion**: Concatenate two modality data
- **Mid fusion**: Combine embedding features, we have two methods, concatenate and cross-attention
- **Late fusion**: Combine output probabilities, we have two methods, simple average, and MoE method. Before you use it, make sure you've finished the fine-tuning of two independent models.

