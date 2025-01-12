import argparse
import os
import torch


def parse_arguments():    

    parser = argparse.ArgumentParser(description="Audio Classification with Wav2Vec2")

    # dataset parameters
    parser.add_argument("--data_type", type=str, choices=['raw', 'pre-processed'], default="pre-processed", help="type of csv data")
    parser.add_argument("--data_path", type=str, required=True, default="./datasets", help="Path of dataset csv")
    parser.add_argument("--audio_folder", type=str,default='./audio',  help="path of audio folder")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for model and results")
    parser.add_argument("--da", action="store_true", help="Data Augmentation")
    parser.add_argument("--da_percentage", type=float, default=0.3, help="Percentage of da")
    parser.add_argument("--modality", type=str, choices=['cs', 'sv'], default='cs', help="Choose the modality to train on")
    parser.add_argument("--num_classes", type=int, default=2, help="Choose the task - binary 2 classes, multiple 8 classes")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base-960h", help="Pre-trained model name")
    parser.add_argument("--strategy", type=str, choices=['early', 'mid', 'late', 'benchmark'], default=None, help="Choose the strategy of model fine-tuning, early fusion will load data from 'both' folder")
    parser.add_argument("--mid_type", type=str, choices=['concate', 'attention'], default=None, help="Choose the mid-fusion type")
    parser.add_argument("--late_type", type=str, choices=['average', 'moe'], default=None, help="Choose the mid-fusion type")

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="early stopping")
    parser.add_argument("--seed", type=int, default=12, help="Random seed")
    parser.add_argument("--cp_name", type=str, default='best_model.pth', help="Check point model name")
    parser.add_argument("--cp_path", type=str, default='./results/check_points', help="Check point model path")

    args = parser.parse_args()

    if args.strategy == "mid" and args.mid_type == None:
        raise ValueError("You must choose a mid-fusion type between concate and attention")
    if args.strategy == "late" and args.late_type == None:
        raise ValueError("You must choose a late-fusion type between average and moe")
    
    return args