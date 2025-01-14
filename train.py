import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import os
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random
from sklearn.neural_network import MLPClassifier
import numpy as np
from utils import compute_metrics, training_args, create_weighted_trainer, get_probabilities
from transformers import AutoModelForAudioClassification, EarlyStoppingCallback, Trainer

def trainer(args, train_dataset,valid_dataset, test_dataset):
    seed=args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = AutoModelForAudioClassification.from_pretrained(args.model_name, num_labels=args.num_classes)

    if args.num_classes == 2:
        # Define the Trainer
        trainer = Trainer(
            model=model,
            args=training_args(args),
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)])
        
    else:
        WeightedTrainer = create_weighted_trainer(model, train_dataset)
        trainer = WeightedTrainer(
            model=model,
            args=training_args(args),
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)])
    
    print("Starting training...")    
    # Start training
    trainer.train()
    print("Training completed! Starting testing...")

    # Evaluate on the test set
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Test results:", test_results)
    return test_results


def train_Midfusion_model(train_loader, valid_loader, test_loader, fusion_model, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fusion_model.to(device)

    # Optimizer, scheduler, and loss function
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    def process_batch(data, labels):
        inputs1 = data[0]['input_values'].to(device)
        inputs2 = data[1]['input_values'].to(device)
        labels = labels.to(device)
        return inputs1, inputs2, labels

    def run_epoch(loader, is_train):
        mode = "Training" if is_train else "Validation"
        fusion_model.train() if is_train else fusion_model.eval()
        total_loss, correct, total = 0, 0, 0
        preds, labels_list = [], []

        for step, (data, labels) in enumerate(tqdm(loader, desc=mode)):
            if is_train:
                optimizer.zero_grad()

            inputs1, inputs2, labels = process_batch(data, labels)
            outputs = fusion_model(inputs1, inputs2)
            loss = criterion(outputs, labels) / (args.accumulation_steps if is_train else 1)

            if is_train:
                loss.backward()
                if (step + 1) % args.accumulation_steps == 0 or (step == len(loader) - 1):
                    optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                outputs_prob = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs_prob, dim=1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                preds.extend(predicted.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        f1 = f1_score(labels_list, preds, average='macro')
        return avg_loss, accuracy, f1

    best_valid_accuracy = 0
    patience_counter = 0

    for epoch in range(args.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")

        # Training and validation
        train_loss, train_accuracy, train_f1 = run_epoch(train_loader, is_train=True)
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")

        valid_loss, valid_accuracy, valid_f1 = run_epoch(valid_loader, is_train=False)
        print(f"Validation Loss: {valid_loss:.4f}, Accuracy: {valid_accuracy:.4f}, F1: {valid_f1:.4f}")

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            save_path = os.path.join(args.cp_path, 'best.pth')
            torch.save(fusion_model.state_dict(), save_path)
            print(f"Model saved at {save_path} with improved accuracy.")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

        scheduler.step()

    # Testing
    test_loss, test_accuracy, test_f1 = run_epoch(test_loader, is_train=False)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")

    return {
        'best_valid_accuracy': best_valid_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1
    }



def late_fusion_val_test(args, models, cs, sv):
    cs_model = models[0]
    sv_model = models[1]
    # get prob
    cs_val_p, cs_test_p = get_probabilities(cs_model, cs[0], cs[1])
    sv_val_p, sv_test_p = get_probabilities(sv_model, sv[0], sv[1])

    result = {}

    # Helper function to evaluate classification
    def evaluate_classification(true_labels, predicted_probs):
        # Convert probabilities to predicted labels
        if args.num_classes > 2:
            predicted_labels = np.argmax(predicted_probs, axis=-1)
        else:
            predicted_labels = (predicted_probs > 0.5).astype(int)

        # Ensure true_labels is consistent
        if args.num_classes > 2 and true_labels.ndim > 1:
            true_labels = np.argmax(true_labels, axis=-1)

        # Verify matching shapes
        assert true_labels.shape == predicted_labels.shape, "Shapes do not match!"

        # Compute metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='macro')

        print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        return accuracy, f1

    if args.late_type == 'average':
        # Validation loop
        true_labels = []
        fused_probs = []
        for cs_label, cs_probs in cs_val_p.items():
            if cs_label in sv_val_p:
                sv_probs = sv_val_p[cs_label]

                # Compute the average of probabilities
                fused_prob = (cs_probs + sv_probs) / 2.0
                fused_probs.append(fused_prob)

                # Add the true labels
                true_labels.append(cs_label)

        # Evaluate validation
        true_labels = np.array(true_labels)
        fused_probs = np.array(fused_probs)
        val_accuracy, val_f1 = evaluate_classification(true_labels, fused_probs)
        result['val_acc'] = val_accuracy
        result['val_f1'] = val_f1

        # Test loop
        true_labels = []
        fused_probs = []
        for cs_label, cs_probs in cs_test_p.items():
            if cs_label in sv_test_p:
                sv_probs = sv_test_p[cs_label]

                # Compute the average of probabilities
                fused_prob = (cs_probs + sv_probs) / 2.0
                fused_probs.append(fused_prob)

                # Add the true labels
                true_labels.append(cs_label)

        # Evaluate test
        true_labels = np.array(true_labels)
        fused_probs = np.array(fused_probs)
        test_accuracy, test_f1 = evaluate_classification(true_labels, fused_probs)
        result['test_acc'] = test_accuracy
        result['test_f1'] = test_f1

    elif args.late_type == 'moe':
        X_train = []
        y_train = []

        # Construct validation set features and labels
        for label, cs_prob in cs_val_p.items():
            sv_prob = sv_val_p.get(label, None)

            if sv_prob is not None:
                # Concatenate the probabilities of the two models as features
                combined_prob = np.concatenate((cs_prob, sv_prob))

                X_train.append(combined_prob)
                y_train.append(label)

        # Convert to NumPy
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Training the Gating Network
        gating_network = MLPClassifier(
            hidden_layer_sizes=(10,), max_iter=1000, random_state=42, learning_rate_init=0.001
        )

        gating_network.fit(X_train, y_train)

        # Define the internal evaluation function
        def _evaluate_fusion(cs_prob_dict, sv_prob_dict, gating_network):
            X_fused = []
            y_fused = []

            for label, cs_prob in cs_prob_dict.items():
                sv_prob = sv_prob_dict.get(label, None)

                if sv_prob is not None:
                    # Concatenate the probabilities of the two models as features
                    combined_prob = np.concatenate((cs_prob, sv_prob))

                    # Get the probabilities of the pre-trained gating network
                    gating_probs = gating_network.predict_proba(combined_prob.reshape(1, -1))[0]

                    # Converting class probabilities into model weights
                    cs_weight = np.sum(gating_probs[:args.num_classes])  # first num_classes categories are the weights of cs_prob
                    sv_weight = np.sum(gating_probs[args.num_classes:])  # last are the weights of sv_prob

                    # Normalize the weights
                    total_weight = cs_weight + sv_weight
                    cs_weight /= total_weight
                    sv_weight /= total_weight

                    # Weighted sum of the probabilities of the two models
                    final_prob = cs_weight * cs_prob + sv_weight * sv_prob

                    X_fused.append(final_prob)
                    y_fused.append(label)

            X_fused = np.array(X_fused)
            y_fused = np.array(y_fused)

            # Evaluate classification
            return evaluate_classification(y_fused, X_fused)

        # Evaluate the validation set ensemble
        valid_accuracy, valid_f1 = _evaluate_fusion(cs_val_p, sv_val_p, gating_network)
        result['val_acc'] = valid_accuracy
        result['val_f1'] = valid_f1

        # Evaluate the Test set ensemble
        test_accuracy, test_f1 = _evaluate_fusion(cs_test_p, sv_test_p, gating_network)
        result['test_acc'] = test_accuracy
        result['test_f1'] = test_f1

    return result