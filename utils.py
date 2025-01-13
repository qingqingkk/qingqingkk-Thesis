
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
import torch
from transformers import Trainer, TrainingArguments

def training_args(args):
    train_args = TrainingArguments(
        output_dir=args.cp_path,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='epoch',
        save_total_limit=2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        load_best_model_at_end=True,
        greater_is_better=True,
        metric_for_best_model='accuracy',
    )
    return train_args


def create_weighted_trainer(train_dataset, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Get labels from the training dataset
    y_train = train_dataset['label']

    # Compute weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    # Convert weights to tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Class weights:", class_weights)

    # Define weighted loss function
    loss_fn = CrossEntropyLoss(weight=class_weights)

    # Define custom Trainer class
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")  # Extract labels
            outputs = model(**inputs)     # Model outputs
            logits = outputs.logits       # Get logits
            loss = loss_fn(logits, labels)  # Compute weighted loss
            return (loss, outputs) if return_outputs else loss
    return WeightedTrainer


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": accuracy, "f1_score": f1}



def get_probabilities_with_prefix(model, valid_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()

    probabilities_val = {}
    probabilities_test = {}
    # use DataLoader
    
    with torch.no_grad():
        for batch in valid_loader:
            input_values = batch["input_values"].to(device)
            prefixes = batch["prefix"]
            labels = batch["labels"]
            
            # get logits from model，compute softmax to get probability
            outputs = model(input_values).logits
            probs = F.softmax(outputs, dim=-1).cpu().numpy()

            # ensure prefix & label matching the probability
            for prefix, label, prob in zip(prefixes, labels, probs):
                key = (prefix, label.item())  # key = (prefix, label)
                probabilities_val[key] = prob  # save probability

        for batch in test_loader:
            input_values = batch["input_values"].to(device)
            prefixes = batch["prefix"]
            labels = batch["labels"]
            
            # get logits from model，compute softmax to get probability
            outputs = model(input_values).logits
            probs = F.softmax(outputs, dim=-1).cpu().numpy()

            # ensure prefix & label matching the probability
            for prefix, label, prob in zip(prefixes, labels, probs):
                key = (prefix, label.item())  # key = (prefix, label)
                probabilities_test[key] = prob  # save probability
    return probabilities_val, probabilities_test