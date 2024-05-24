from transformers import TrainingArguments

def get_training_arguments(output_dir, learning_rate, num_train_epochs, batch_size):
    return TrainingArguments(
        gradient_checkpointing=True,
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=2e-4,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
