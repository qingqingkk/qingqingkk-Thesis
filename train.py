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
from utils import compute_metrics, training_args, create_weighted_trainer
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
        
    # Start training
    trainer.train()

    # Evaluate on the test set
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Test results:", test_results)
    return test_results


def train_and_evaluate(dataloader, model, start_time, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## Optimizer, Loss Function, and Scheduler lr=1e-5 Concatenate, lr=6e-6 attention 
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    train_loader = dataloader[0]
    valid_loader = dataloader[1]
    test_loader = dataloader[2]

    ## Training settings
    num_epochs = args.num_train_epochs
    early_stopping_patience = args.early_stopping_patience
    best_valid_accuracy = 0 
    patience_counter = 0
    accumulation_steps =  2
    result = {}
    result['best_val_acc'] = best_valid_accuracy
    result['seed'] = args.seed
    save_path = os.path.join(args.cp_path, f'{start_time}_{args.modality}_{args.strategy}_{args.da}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = os.path.join(save_path, args.cp_name)

    ### Training and evaluation loop
    ### Training and evaluation loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # ========================= Training Loop ==========================
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        all_preds, all_labels = [], []

        for step, (data, labels) in enumerate(tqdm(train_loader, desc="Training")):
            optimizer.zero_grad() 
            labels = labels.to(device)
            if args.strategy == 'mid':
                inputs1 = data[0]['input_values'].to(device)
                inputs2 = data[1]['input_values'].to(device)
                outputs = model(inputs1, inputs2)
            else:
                inputs =  data['input_values'].to(device)
                outputs = model(inputs)

            loss = criterion(outputs, labels) / accumulation_steps  # Scale loss for accumulation
            
            # Backward pass
            loss.backward()
            train_loss += loss.item() 

            # Predictions (No Grad)
            with torch.no_grad():
                outputs_prob = torch.softmax(outputs, dim=1)  # probability distribution
                predicted = torch.argmax(outputs_prob, dim=1)  # max probability as predict
                train_correct += predicted.eq(labels).sum().item()
                train_total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Perform optimizer step when accumulation is complete
            if (step + 1) % accumulation_steps == 0 or (step == len(train_loader) - 1):
                optimizer.step()

        # Compute average loss, accuracy, and F1
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}")

        # ========================= Validation Loop ==========================
        model.eval()
        valid_loss, valid_correct, valid_total = 0, 0, 0
        valid_preds, valid_labels = [], []

        with torch.no_grad():
            for data, labels in tqdm(valid_loader, desc="Validation"):
                labels = labels.to(device)
                if args.strategy == 'mid':
                    inputs1 = data[0]['input_values'].to(device)
                    inputs2 = data[1]['input_values'].to(device)
                    outputs = model(inputs1, inputs2)
                else:
                    inputs =  data['input_values'].to(device)
                    outputs = model(inputs)

                # outputs = classfier(outputs)
                outputs_prob = torch.softmax(outputs, dim=1)  # probability distribution
                predicted = torch.argmax(outputs_prob, dim=1)  # max probability as predict
                
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                valid_correct += predicted.eq(labels).sum().item()
                valid_total += labels.size(0)
                
                valid_preds.extend(predicted.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())

        # Compute average loss、accuracy and F1
        avg_valid_loss = valid_loss / len(valid_loader)
        valid_accuracy = valid_correct / valid_total
        valid_f1 = f1_score(valid_labels, valid_preds, average='macro')
        print(f"Validation Loss: {avg_valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}, Validation F1: {valid_f1:.4f}")

        # Save best model
        if valid_accuracy >= best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            result['best_val_acc'] = best_valid_accuracy
            result['best_val_loss'] = avg_valid_loss
            result['best_epoch'] = epoch
            torch.save(model.state_dict(), model_path)
            result['cp_name'] = model_path
            print("Model saved with improved.")
            patience_counter = 0 
        else:
            patience_counter += 1  


        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. No improvement in validation accuracy for {early_stopping_patience} epochs.")
            break
            
        # Update lr
        scheduler.step(best_valid_accuracy)


        # ========================= Test Loop ==========================
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        test_preds, test_labels = [], []

        with torch.no_grad():
            for data, labels in tqdm(test_loader, desc="Testing"):
                labels = labels.to(device)

                if args.strategy == 'mid':
                    inputs1 = data[0]['input_values'].to(device)
                    inputs2 = data[1]['input_values'].to(device)
                    outputs = model(inputs1, inputs2)
                else:
                    inputs =  data['input_values'].to(device)
                    outputs = model(inputs)
                    
                outputs_prob = torch.softmax(outputs, dim=1)  # probability distribution
                predicted = torch.argmax(outputs_prob, dim=1)  # max probability as predict
                
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                test_correct += predicted.eq(labels).sum().item()
                test_total += labels.size(0)
                
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        # Compute average loss、accuracy and F1
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_total
        test_f1 = f1_score(test_labels, test_preds, average='macro')
        if patience_counter == 0:
            result['best_test_acc'] = test_accuracy
            result['best_test_f1'] = test_f1
            result['test_loss'] = avg_test_loss
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")

    return result



def late_fusion_val_test(args, models, cs, sv):
    cs_model = models[0]
    sv_model = models[1]
    # get prob
    cs_val_p, cs_test_p = get_probabilities_with_prefix(cs_model, cs[0], cs[1])
    sv_val_p, sv_test_p = get_probabilities_with_prefix(sv_model, sv[0], sv[1])

    true_labels = []
    fused_probs = []
    result = {}
    if args.late_type == 'average':
        #valid loop
        for (cs_prefix, cs_label), cs_probs in cs_val_p.items():
            sv_key = (cs_prefix, cs_label)
            if sv_key in sv_val_p:
                sv_probs = sv_val_p[sv_key]
                
                # compute the average of probabilities
                fused_prob = (cs_probs + sv_probs) / 2.0
                fused_probs.append(fused_prob)
                
                # add the true labels
                true_labels.append(cs_label)

        # Convert them to tensors for batch computation
        fused_probs = torch.tensor(fused_probs)
        predicted_labels = torch.argmax(fused_probs, dim=-1).numpy()

        # Convert to NumPy
        true_labels = np.array(true_labels)

        # evaluate
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        result['val_acc'] = accuracy
        result['val_f1'] = f1
        print(f'late type: {args.late_type}, validation, acc: {accuracy}, f1: {f1}')
        #test loop
        for (cs_prefix, cs_label), cs_probs in cs_test_p.items():
            sv_key = (cs_prefix, cs_label)
            if sv_key in sv_test_p:
                sv_probs = sv_test_p[sv_key]
                
                # compute the average of probabilities
                fused_prob = (cs_probs + sv_probs) / 2.0
                fused_probs.append(fused_prob)
                
                # add the true labels
                true_labels.append(cs_label)

        # Convert them to tensors for batch computation
        fused_probs = torch.tensor(fused_probs)
        predicted_labels = torch.argmax(fused_probs, dim=-1).numpy()

        # Convert to NumPy
        true_labels = np.array(true_labels)

        # evaluate
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        result['test_acc'] = accuracy
        result['test_f1'] = f1
        print(f'late type: {args.late_type}, test, acc: {accuracy}, f1: {f1}')

    elif args.late_type == 'moe':
        '''
        Train gating networks

        Parameters:
        - cs_valid_prob: dict, model 1, {'(prefix, label)',prob}
        - sv_valid_prob: ...
        - cs_test_prob: ...
        - sv_test_prob: ...
        - num_classes: int

        Return:
        - valid_accuracy: Fusion accuracy of the validation set
        - valid_f1: Fusion F1 ..
        - test_accuracy: Fusion accuracy of the test set
        - test_f1: Fusion accuracy ..
        '''
        X_train = []
        y_train = []
        
        # Construct validation set features and labels
        for key, cs_prob in cs_val_p.items():
            prefix, label = key
            sv_prob = sv_val_p.get(key, None)
            
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
            
            for key, cs_prob in cs_prob_dict.items():
                prefix, label = key
                sv_prob = sv_prob_dict.get(key, None)
                
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
            
            # fused predictions
            predictions = np.argmax(X_fused, axis=1)
            
            #
            accuracy = accuracy_score(y_fused, predictions)
            f1 = f1_score(y_fused, predictions, average='weighted')
            
            return accuracy, f1


        # Evaluate the validation set ensemblen
        valid_accuracy, valid_f1 = _evaluate_fusion(cs_val_p, sv_val_p, gating_network)
        result['val_acc'] = valid_accuracy
        result['val_f1'] = valid_f1
        print(f"Fused Valid Accuracy: {valid_accuracy}")
        print(f"Fused Valid F1 Score: {valid_f1}")

        # Evaluate the Test set ensemble
        test_accuracy, test_f1 = _evaluate_fusion(cs_test_p, sv_test_p, gating_network)
        result['test_acc'] = test_accuracy
        result['test_f1'] = test_f1
        print(f"Fused Test Accuracy: {test_accuracy}")
        print(f"Fused Test F1 Score: {test_f1}")
        
        return result