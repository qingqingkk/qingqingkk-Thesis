import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from datasets import Dataset
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.applications import MobileNetV2

from dataset import load_csv

'''

Contains the mfcc feature extraction process and the use of three benchmark models

'''

def process_audio_dataset(args, dataset, max_duration, n_mfcc=40, sr=16000):
    mfcc_list = []  # Initialize MFCC list
    label_list = []  # Initialize label list
    max_samples = int(sr * max_duration)

    for sample in dataset:
        # Get path and label directly (no need for zip since we have single values)
        path = sample["path"]
        label = sample["label"]
        
        # Load and truncate audio
        audio, _ = librosa.load(path, sr=sr)
        audio = audio[:max_samples]
        
        # Extract MFCC and store
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_list.append(mfcc.T)  # Transpose MFCC to shape (time_steps, n_mfcc)
        
        # One-hot encode labels
        one_hot_label = np.eye(args.num_classes)[label]
        label_list.append(one_hot_label)

    # Pad MFCC sequences to the same length
    padded_mfccs = pad_sequences(mfcc_list, padding='post', dtype='float32', value=0.0)
    
    # Create padding mask (1 for real data, 0 for padding)
    padding_mask = np.array([[1] * len(mfcc) + [0] * (padded_mfccs.shape[1] - len(mfcc)) for mfcc in mfcc_list])
    
    return {
        'padded_mfcc': padded_mfccs,
        'label_1hot': np.array(label_list),
        'padding_mask': padding_mask
    }

def mlp(args, x_train, y_train, x_valid, y_valid, x_test, y_test):
    # save path
    ckp_path = os.path.join(args.output_dir, 'benchmark', f'MLP_{args.modality}_{args.num_classes}_{args.seed}.weights.h5')

    # setting parameters and layers
    learning_rate = 0.01
    batch_size = 16
    n_input = x_train[0].shape
    n_hidden_1 = 50
    n_hidden_2 = 50
    n_classes = args.num_classes

    # model structure
    inputs = tf.keras.Input(shape=n_input)
    x = tf.keras.layers.Dense(n_hidden_1, activation='relu')(inputs)
    x = tf.keras.layers.Dense(n_hidden_2, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    predictions = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    
    # choose the classify tasks
    if n_classes == 2:

        model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    else:
        
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy' if y_train.ndim == 1 else 'categorical_crossentropy',
                      metrics=['accuracy'])
    

    callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=10)
    callback_list = [
       tf.keras.callbacks.ModelCheckpoint(
            filepath = ckp_path, 
            save_freq='epoch', verbose=1, monitor='val_loss', 
            save_weights_only=True, save_best_only=True
       )]
    
    # train
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=50, validation_data=(x_valid, y_valid), callbacks=[callback_earlystop, callback_list])
    
    # load the best weights
    model.load_weights(ckp_path)

    print("------Validation------")
    val_loss, val_accuracy = model.evaluate(x_valid, y_valid, batch_size=batch_size)
    val_pred = np.argmax(model.predict(x_valid), axis=1)  # pred label
    val_true = y_valid if y_valid.ndim == 1 else np.argmax(y_valid, axis=1)  # true label
    best_val_f1 = f1_score(val_true, val_pred, average='macro')
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print(f"Best Validation F1 score: {best_val_f1:.4f}")
    
    print("------TEST------")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    test_pred = np.argmax(model.predict(x_test), axis=1)  # pred label
    test_true = y_test if y_test.ndim == 1 else np.argmax(y_test, axis=1)  # true label
    test_f1 = f1_score(test_true, test_pred, average='macro')
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 score: {test_f1:.4f}")
    
    return val_accuracy, best_val_f1, test_accuracy, test_f1


def build_lightweight_cnn(args, x_train, y_train, x_valid, y_valid, x_test, y_test, trainable=False):
    learning_rate = 0.01
    batch_size = 16

    if trainable == False:
        save_path = os.path.join(args.output_dir, 'benchmark', f'CNN_{args.modality}_{args.num_classes}_{args.seed}_freeze.weights.h5')
    else:
        save_path = os.path.join(args.output_dir, 'benchmark', f'CNN_{args.modality}_{args.num_classes}_{args.seed}.weights.h5')
    
    # Ensure the input shape matches MobileNetV2 expected input
    n_input = x_train.shape[1:]  # (num_samples, height, width, channels)
    num_classes = args.num_classes
    
    # Load MobileNetV2 model without top layers
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=n_input)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze all layers in base model
    for layer in base_model.layers:
        layer.trainable = trainable

    if num_classes == 2:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy', metrics=['accuracy'])
    else:
         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss = 'sparse_categorical_crossentropy' if y_train.ndim == 1 else 'categorical_crossentropy',
                      metrics=['accuracy'])
   
        
    callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=10)
    callback_list = [
       tf.keras.callbacks.ModelCheckpoint(
            filepath = save_path, 
            save_freq='epoch', verbose=1, monitor='val_loss', 
            save_weights_only=True, save_best_only=True
       )]

    # train
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=50, 
                        validation_data=(x_valid, y_valid), 
                        callbacks=[callback_earlystop, callback_list])

    model.load_weights(save_path)

    print("------Validation------")
    val_loss, val_accuracy = model.evaluate(x_valid, y_valid, batch_size=batch_size)
    val_pred = np.argmax(model.predict(x_valid), axis=1)  # Predicted labels
    val_true = y_valid if y_valid.ndim == 1 else np.argmax(y_valid, axis=1)  # True labels
    best_val_f1 = f1_score(val_true, val_pred, average='macro')
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print(f"Best Validation F1 score: {best_val_f1:.4f}")
    
    print("------TEST------")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    test_pred = np.argmax(model.predict(x_test), axis=1)  # Predicted labels
    test_true = y_test if y_test.ndim == 1 else np.argmax(y_test, axis=1)  # True labels
    test_f1 = f1_score(test_true, test_pred, average='macro')
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 score: {test_f1:.4f}")
    
    return val_accuracy, best_val_f1, test_accuracy, test_f1

def benchmark_train_test(args):
    result = {}
    result['seed'] = args.seed
    result['modality'] = args.modality
    out_path = os.path.join(args.output_dir, 'benchmark')

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    train_df, valid_df, test_df = load_csv(args)
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    train = process_audio_dataset(train_dataset, args.max_duration)
    valid = process_audio_dataset(valid_dataset, args.max_duration)
    test = process_audio_dataset(test_dataset, args.max_duration)

    # MLP
    x_train = np.array(train['padded_mfcc'])
    y_train = np.array(train['label_1hot'])  
    x_valid = np.array(valid['padded_mfcc'])
    y_valid = np.array(valid['label_1hot'])  
    x_test = np.array(test['padded_mfcc'])
    y_test = np.array(test['label_1hot']) 
    val_accuracy, val_f1, test_accuracy, test_f1 = mlp(args, x_train, y_train, x_valid, y_valid, x_test, y_test)
    result['mlp_val_acc'] = val_accuracy
    result['mlp_val_f1'] = val_f1
    result['mlp_test_acc'] = test_accuracy
    result['mlp_test_f1'] = test_f1

    # CNN, freeze
    x_train = np.expand_dims(x_train, axis=-1)  
    x_valid = np.expand_dims(x_valid, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # repeat the single channel 3 times -> (num_samples, height, width, 3)
    x_train = np.repeat(x_train, 3, axis=-1)  
    x_valid = np.repeat(x_valid, 3, axis=-1)
    x_test = np.repeat(x_test, 3, axis=-1)

    val_accuracy, val_f1, test_accuracy, test_f1 = build_lightweight_cnn(args, x_train, y_train, x_valid, y_valid, x_test, y_test, trainable=False)
    result['CNN_freeze_val_acc'] = val_accuracy
    result['CNN_freeze_val_f1'] = val_f1
    result['CNN_freeze_test_acc'] = test_accuracy
    result['CNN_freeze_test_f1'] = test_f1

    # not freeze
    val_accuracy, val_f1, test_accuracy, test_f1 = build_lightweight_cnn(args, x_train, y_train, x_valid, y_valid, x_test, y_test, trainable=True)
    result['CNN_nofreeze_val_acc'] = val_accuracy
    result['CNN_nofreeze_val_f1'] = val_f1
    result['CNN_nofreeze_test_acc'] = test_accuracy
    result['CNN_nofreeze_test_f1'] = test_f1

    return result
