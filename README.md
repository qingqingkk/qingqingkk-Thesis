# Voice Disorder Diagnoses

### Dataset Structure
```
raw Dataset
├── Controlli
│    ├── healthy1_cs.wav
│    ├── healthy1_sv.wav
│    └──...
├── CASI
│    ├── patient1_cs.wav
│    ├── patient1_sv.wav
│    └──...
├── Controlli.csv
├── CASI.csv
```

### Separate CS and SV Datasets
To separate the data into two modalities (CS and SV):
```bash
python process_raw_data.py --data_path='./raw Dataset' --output_dir='./separated Dataset' --label_column='s/p'
```
Resulting structure:
```
separated Dataset
├── cs_dataset.csv
├── sv_dataset.csv
```

### Concatenate CS and SV Data
To concatenate audio files for early fusion:
```bash
python concatenated_cs_sv --data_path='./raw Dataset' --output_dir='./results'
```
Resulting structure:
```
results
├── concatenated
│    ├── healthy1_combined.wav
│    ├── patient1_combined.wav
│    └──...
├── concat_dataset.csv
```

### Single Training (CS/SV or Concatenated Dataset)
Run the training script on a single modality or concatenated dataset:
```bash
python main.py --data_path=./dataset.csv --cp_path=./results --strategy=single
```

### Benchmark
Evaluate multiple models (MLP, 2D-CNN) in sequence. No separate evaluation for individual models.
```bash
python main.py --data_path=./dataset.csv --output_dir=./results --strategy=benchmark
```

#### Mid Fusion
Combine embeddings during training using concatenation or cross-attention:
```bash
python main.py \
  --data_path=./separated Dataset \
  --cp_path=./mid_results
  --cp_path1=./results/check_points1 \
  --cp_path2=./results/check_points2 \
  --strategy=mid \
  --mid_type=concate 
```

#### Late Fusion
Combine model outputs using averaging or Mixture of Experts (MoE):
```bash
python main.py \
  --data_path=./separated Dataset \
  --cp_path=./late_results
  --cp_path1=./results/check_points1 \
  --cp_path2=./results/check_points2 \
  --strategy=late \
  --late_type=average \
```

Ensure that the individual models (`cp_path1` and `cp_path2`) are fine-tuned before running late fusion.
