import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, Wav2Vec2_Processor

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": accuracy, "f1_score": f1}

def train_and_evaluate(train_val_dataset, model_name, training_args, modality=None):
    # if modality:
    #     train_val_dataset = train_val_dataset.filter(lambda x: x['modality'] == modality)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []
    f1_scores = []

    for train_index, val_index in kf.split(train_val_dataset):
        train_dataset = train_val_dataset.select(train_index)
        val_dataset = train_val_dataset.select(val_index)

        model = AutoModelForAudioClassification.from_pretrained(model_name, num_labels=2)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=Wav2Vec2_Processor,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate(eval_dataset=val_dataset)

        accuracy_scores.append(eval_results["eval_accuracy"])
        f1_scores.append(eval_results["eval_f1_score"])

    return np.mean(accuracy_scores), np.mean(f1_scores)

