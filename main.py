import parser
import random
import numpy as np
import os
import json
from datetime import datetime

from dataset_loader import load_data
from train import train_and_evaluate, late_fusion_val_test
from model import load_model
from benchmark import benchmark_train_test


def main(args):
    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    start_time = datetime.now()

    if args.strategy == 'late':
        cs_loader, sv_loader = load_data(args)
    else:
        train_loader, valid_loader, test_loader = load_data(args)

    # Get model
    models = load_model(args)

    if args.strategy == 'late':
        result = late_fusion_val_test(args, models, cs_loader, sv_loader)
    elif args.strategy == 'benchmark':
        result = benchmark_train_test(args)
    else:
        result = train_and_evaluate([train_loader, valid_loader, test_loader], models, start_time, args)
    
    file_path = os.path.join(args.output_dir, f"{start_time}_{args.modality}_{args.strategy}_{args.da}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4) 

if __name__ == "__main__":
    args = parser.parse_arguments()
    main(args)
