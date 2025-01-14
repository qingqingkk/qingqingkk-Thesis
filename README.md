# Voice Disorder Diagnoses

##### Dataset
raw Dataset
├── Controlli
│    ├──healthy1_cs.wav
│    ├──healthy1_sv.wav
│    └──...
├── CASI
│    ├──patient1_cs.wav
│    ├──patient1_sv.wav
│    └──...
├── Controlli.csv
├── CASI.csv


<!-- - Separate the data of the two modalities (For unimodal models)

```
separated Dataset
├── cs
│    ├──healthy1_cs.wav
│    ├──patient1_cs.wav
│    └──...
├── sv
│    ├──healthy1_sv.wav
│    ├──patient1_sv.wav
│    └──...
├── cs_dataset.csv
├── sv_dataset.csv
``` -->

# (Early Fusion) get concatenated audio from raw dataset
Run:
$python combined_cs_sv.py  --data_path='./raw Dataset folder' --output_dir='./results'
```
Concatenated dataset
├── healthy1_combined.wav
├── patient1_combined.wav
└── concatenate.csv
```

##### Run the code -- Single modality cs,sv or concatenated datasets (early strategies)
```
$python main.py --data_path=./dataset_cs.csv --cp_pat=./results/cs_single --strategy=single
```


## Benchmark
Continuously evaluate MLP(Multilayer Perceptron), 2D-CNN fine-tuning classification head, and 2D-CNN fine-tuning all layers. No separate evaluation of any of the models is set

#### Running example of cs mode
```
$python3 main.py --data_path=./datasets_cs.csv --output_dir=./results --strategy=benchmark
```

## Fusion model

### Fusion model details
- **Early fusion**: Concatenate two modality data
- **Mid fusion**: Combine embedding features, we have two methods, concatenate and cross-attention
- **Late fusion**: Combine output probabilities, we have two methods, simple average, and MoE method. Before you use it, make sure you've finished the fine-tuning of two independent models.


