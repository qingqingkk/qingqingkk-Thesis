import argparse
import random
import numpy as np
import os

from data_processing import preprocess_and_load_dataset
from feature_extraction import load_feature_extractor, extract_features
from train import train_and_evaluate
from util import get_training_arguments

def main(args):
    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load and preprocess data
    dataset = preprocess_and_load_dataset(os.path.join(args.data_path, 'CASI.csv'), os.path.join(args.data_path, 'CONTROLLI.csv'), args.data_path, args.modality)

    # Load feature extractor
    processor = load_feature_extractor(args.model_name)

    # Extract features
    dataset = extract_features(dataset, processor)

    # Split the dataset into training and final test set
    train_val_dataset, test_dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed).values()

    # Get training arguments
    training_args = get_training_arguments(args.output_dir, args.learning_rate, args.num_train_epochs, args.batch_size)

    if args.modality == "cs":
        avg_accuracy, avg_f1_score = train_and_evaluate(train_val_dataset, args.model_name, training_args, modality='cs')
    elif args.modality == "sv":
        avg_accuracy, avg_f1_score = train_and_evaluate(train_val_dataset, args.model_name, training_args, modality='sv')
    else:
        avg_accuracy, avg_f1_score = train_and_evaluate(train_val_dataset, args.model_name, training_args)

    print(f'Cross-validation average accuracy: {avg_accuracy}')
    print(f'Cross-validation average F1 score: {avg_f1_score}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Classification with Wav2Vec2")

    # Model and dataset parameters
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base-960h", help="Pre-trained model name")
    # parser.add_argument("--casi_csv", type=str, required=True, help="Path to CASI CSV file")
    # parser.add_argument("--contro_csv", type=str, required=True, help="Path to CONTROLLI CSV file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/result/cs", help="Output directory for model and results")

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--modality", type=str, choices=['cs', 'sv', 'both'], default='both', help="Choose the modality to train on")

    args = parser.parse_args()
    main(args)
