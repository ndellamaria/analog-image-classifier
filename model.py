import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import wandb
from wandb.integration.keras import WandbMetricsLogger,WandbModelCheckpoint

class AnalogPhotoDataset:
    def __init__(self, image_dir, labels_file):
        self.image_dir = image_dir
        self.data = pd.read_csv(labels_file)
        self.class_names = ['good', 'over_exposed', 'blurry', 'under_exposed', 'light_exposure']
        
    def prepare_datasets(self, batch_size=32, validation_split=0.1, test_split=0.2):
        """
        Prepare the datasets with proper splits:
        - test_split (0.2) takes 20% of data for final testing
        - validation_split (0.1) takes 10% of remaining 80% for validation
        - remaining ~70% used for training
        
        Args:
            batch_size: Number of samples per batch
            validation_split: Fraction of training data to use for validation
            test_split: Fraction of all data to use for testing
        """
        def process_path(img_path, label):
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = tf.keras.applications.resnet50.preprocess_input(img)
            return img, label

        # Prepare file paths and labels
        image_paths = [os.path.join(self.image_dir, fname) for fname in self.data.iloc[:, 0]]
        labels = [self.class_names.index(label) for label in self.data.iloc[:, 1]]

        # First split into train+val and test sets
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            image_paths, labels, 
            test_size=test_split,  # 20% for testing
            random_state=42, 
            stratify=labels
        )

        # Then split train+val into train and validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, 
            train_val_labels, 
            test_size=validation_split,  # 10% of remaining 80% for validation
            random_state=42, 
            stratify=train_val_labels
        )

        print(f"Dataset splits:")
        print(f"Training samples: {len(train_paths)}")
        print(f"Validation samples: {len(val_paths)}")
        print(f"Test samples: {len(test_paths)}")

        # Create TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

        # Process datasets
        train_ds = train_ds.shuffle(1000)
        train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        test_ds = test_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds, test_ds, test_paths, test_labels

def create_model():
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def train_and_evaluate_model(image_dir, labels_file, epochs=10, batch_size=32):
    # Prepare datasets
    dataset = AnalogPhotoDataset(image_dir, labels_file)
    train_ds, val_ds, test_ds, test_paths, test_labels = dataset.prepare_datasets(
        batch_size=batch_size
    )
    
    # Create and compile model
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[WandbMetricsLogger(),
        WandbModelCheckpoint("models/model.keras")]
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Get predictions for confusion matrix
    y_pred = []
    for images, _ in test_ds:
        batch_pred = model.predict(images)
        y_pred.extend(np.argmax(batch_pred, axis=1))
    
    # Generate and plot metrics
    plot_training_history(history)
    plot_confusion_matrix(test_labels, y_pred, dataset.class_names)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred, 
                              target_names=dataset.class_names))
    
    return model, history

def predict_image(model, image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    class_names = ['good', 'over_exposed', 'blurry', 'under_exposed', 'light_exposure']
    confidence = np.max(predictions[0])
    predicted_class = class_names[np.argmax(predictions[0])]
    
    return predicted_class, confidence

# Initialize a W&B run
wandb.init(
    project="analog-images",  # Replace with your project name in W&B
    config={
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
    }
)

# Train and evaluate the model
model, history = train_and_evaluate_model(
    image_dir="augmented_images",
    labels_file="labels.txt",
    epochs=10,
    batch_size=32
)

# Save the trained model with explicit .keras extension
model.save('model.keras')
wandb.save('model.keras')  # Save to wandb with correct extension

wandb.finish()
