import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
import librosa


def preprocess_audio_data(dataset, max_duration, n_mfcc=40, sr=16000, num_classes=2):
    """
    Preprocess the audio dataset to extract MFCC features and pad them.
    """
    max_samples = int(sr * max_duration)
    mfcc_list, label_list = [], []

    for sample in dataset:
        audio, _ = librosa.load(sample["path"], sr=sr)
        audio = audio[:max_samples]  # Truncate audio
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T
        mfcc_list.append(mfcc)

        # One-hot encode labels
        one_hot_label = np.eye(num_classes)[sample["label"]]
        label_list.append(one_hot_label)

    padded_mfccs = tf.keras.preprocessing.sequence.pad_sequences(
        mfcc_list, padding='post', dtype='float32', value=0.0
    )
    return np.array(padded_mfccs), np.array(label_list)


def build_model(input_shape, num_classes, model_type='mlp', trainable=False):
    """
    Build either MLP or CNN model based on `model_type`.
    """
    if model_type == 'mlp':
        inputs = tf.keras.Input(shape=input_shape)
        x = Dense(50, activation='relu')(inputs)
        x = Dense(50, activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)

    elif model_type.startswith('cnn'):
        base_model = MobileNetV2(weights='imagenet', input_shape=input_shape, include_top=False)
        for layer in base_model.layers:
            layer.trainable = trainable
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        return Model(inputs=base_model.input, outputs=outputs)


def train_and_evaluate_model(model, x_train, y_train, x_valid, y_valid, x_test, y_test, args, save_path):
    """
    Train and evaluate the model, and return validation and test metrics.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='binary_crossentropy' if args.num_classes == 2 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1
        )
    ]

    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=args.batch_size, epochs=args.num_train_epochs, callbacks=callbacks)
    model.load_weights(save_path)

    def evaluate(data_x, data_y):
        loss, acc = model.evaluate(data_x, data_y, batch_size=args.batch_size, verbose=0)
        y_pred = np.argmax(model.predict(data_x), axis=1)
        y_true = np.argmax(data_y, axis=1) if data_y.ndim > 1 else data_y
        f1 = f1_score(y_true, y_pred, average='macro')
        return loss, acc, f1

    val_metrics = evaluate(x_valid, y_valid)
    test_metrics = evaluate(x_test, y_test)
    return val_metrics, test_metrics


def benchmark_train_test(args, train_dataset, valid_dataset, test_dataset):
    """
    Train and evaluate MLP and CNN models (freeze and no-freeze).

    Results:
        {
        'mlp': {'val': (loss, acc, f1), 'test': (loss, acc, f1)},
        'cnn_freeze': {'val': (loss, acc, f1), 'test': (loss, acc, f1)},
        'cnn_no_freeze': {'val': (loss, acc, f1), 'test': (loss, acc, f1)}
        }
    """

    x_train, y_train = preprocess_audio_data(train_dataset, args.max_duration, num_classes=args.num_classes)
    x_valid, y_valid = preprocess_audio_data(valid_dataset, args.max_duration, num_classes=args.num_classes)
    x_test, y_test = preprocess_audio_data(test_dataset, args.max_duration, num_classes=args.num_classes)

    results = {}

    print('Training and evaluating MLP model......')
    # Train MLP
    save_path = os.path.join(args.output_dir, 'mlp_best_weights.weights.h5')
    mlp_model = build_model(x_train.shape[1:], args.num_classes, model_type='mlp')
    val_metrics, test_metrics = train_and_evaluate_model(mlp_model, x_train, y_train, x_valid, y_valid, x_test, y_test, args, save_path)
    results['mlp'] = {'val': val_metrics, 'test': test_metrics}

    # Prepare data for CNN (repeat channels)
    x_train = np.repeat(np.expand_dims(x_train, -1), 3, axis=-1)
    x_valid = np.repeat(np.expand_dims(x_valid, -1), 3, axis=-1)
    x_test = np.repeat(np.expand_dims(x_test, -1), 3, axis=-1)

    print('Training and evaluating Freeze CNN model......')
    # CNN freeze
    save_path = os.path.join(args.output_dir, 'cnn_freeze_best_weights.weights.h5')
    cnn_freeze_model = build_model(x_train.shape[1:], args.num_classes, model_type='cnn_freeze', trainable=False)
    val_metrics, test_metrics = train_and_evaluate_model(cnn_freeze_model, x_train, y_train, x_valid, y_valid, x_test, y_test, args, save_path)
    results['cnn_freeze'] = {'val': val_metrics, 'test': test_metrics}

    print('Training and evaluating No-Freeze CNN model......')
    # CNN no freeze
    save_path = os.path.join(args.output_dir, 'cnn_nofreeze_best_weights.weights.h5')
    cnn_nofreeze_model = build_model(x_train.shape[1:], args.num_classes, model_type='cnn_no_freeze', trainable=True)
    val_metrics, test_metrics = train_and_evaluate_model(cnn_nofreeze_model, x_train, y_train, x_valid, y_valid, x_test, y_test, args, save_path)
    results['cnn_no_freeze'] = {'val': val_metrics, 'test': test_metrics}
    print('Finished!')
    print(results)

    return results



