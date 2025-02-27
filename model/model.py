import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants - optimized for both speed and accuracy
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = "dataset"

def prepare_dataset():
    # Data directories
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "validation")
    
    # Create directories if they don't exist
    for directory in [train_dir, val_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
            # Create breed subdirectories and symlinks
            for breed in os.listdir(DATA_DIR):
                breed_path = os.path.join(DATA_DIR, breed)
                if os.path.isdir(breed_path) and breed not in ["train", "validation"]:
                    breed_images = os.listdir(breed_path)
                    
                    # Create breed directory in train and validation
                    train_breed_dir = os.path.join(train_dir, breed)
                    val_breed_dir = os.path.join(val_dir, breed)
                    
                    os.makedirs(train_breed_dir, exist_ok=True)
                    os.makedirs(val_breed_dir, exist_ok=True)
                    
                    # Split 80/20
                    split_idx = int(len(breed_images) * 0.8)
                    
                    # Create links
                    for i, img in enumerate(breed_images):
                        src = os.path.abspath(os.path.join(breed_path, img))
                        if i < split_idx:
                            dst = os.path.join(train_breed_dir, img)
                        else:
                            dst = os.path.join(val_breed_dir, img)
                            
                        if not os.path.exists(dst):
                            os.symlink(src, dst)
    
    # Create data augmentation
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(0.2),
        keras.layers.RandomTranslation(0.1, 0.1),
    ])
    
    # Create preprocessing layer
    preprocess_input = keras.applications.mobilenet_v2.preprocess_input
    
    # Load datasets
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=42
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=42
    )
    
    # Performance optimization
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).cache().prefetch(tf.data.AUTOTUNE)
    
    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).cache().prefetch(tf.data.AUTOTUNE)
    
    class_names = sorted(os.listdir(train_dir))
    return train_ds, val_ds, class_names

def create_model(num_classes):
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Create an efficient model with minimal layers
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation="softmax")
    ])
    
    # Compile with efficient optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model, base_model

def train_model(model, base_model, train_ds, val_ds):
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2
        )
    ]
    
    # First phase: Train top layers
    print("Phase 1: Training top layers")
    history1 = model.fit(
        train_ds,
        epochs=5,
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    # Second phase: Fine-tune
    print("Phase 2: Fine-tuning")
    base_model.trainable = True
    
    # Freeze earlier layers
    for layer in base_model.layers[:-15]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    history2 = model.fit(
        train_ds,
        epochs=10,
        initial_epoch=5,
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    # Combine histories
    history = {}
    for key in history1.history.keys():
        history[key] = history1.history[key] + history2.history[key]
    
    return model, history

def evaluate_and_visualize(model, val_ds, class_names):
    # Get predictions
    all_labels = []
    all_preds = []
    
    for images, labels in val_ds:
        preds = model.predict(images)
        pred_classes = tf.argmax(preds, axis=1).numpy()
        true_classes = tf.argmax(labels, axis=1).numpy()
        
        all_labels.extend(true_classes)
        all_preds.extend(pred_classes)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Print classification report
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names,
        output_dict=True
    )
    
    print("\nBreed Performance:")
    for breed in class_names:
        print(f"{breed}:")
        print(f"  Precision: {report[breed]['precision']:.4f}")
        print(f"  Recall: {report[breed]['recall']:.4f}")
        print(f"  F1-score: {report[breed]['f1-score']:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    
    return accuracy, report

def predict_image(model, image_path, class_names):
    # Load and preprocess image
    img = keras.utils.load_img(
        image_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    breed = class_names[predicted_class]
    
    print(f"Image: {image_path}")
    print(f"Predicted breed: {breed}")
    print(f"Confidence: {confidence:.4f}")
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_breeds = [class_names[i] for i in top_indices]
    top_confidences = [predictions[0][i] for i in top_indices]
    
    print("\nTop 3 predictions:")
    for breed, confidence in zip(top_breeds, top_confidences):
        print(f"{breed}: {confidence:.4f}")
    
    return breed, confidence

def main():
    # Prepare dataset
    print("Preparing dataset...")
    train_ds, val_ds, class_names = prepare_dataset()
    print(f"Found {len(class_names)} breeds: {class_names}")
    
    # Create model
    print("Building model...")
    model, base_model = create_model(len(class_names))
    
    # Train model
    print("Training model...")
    model, history = train_model(model, base_model, train_ds, val_ds)
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Train")
    plt.plot(history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    
    # Evaluate model
    print("Evaluating model...")
    accuracy, report = evaluate_and_visualize(model, val_ds, class_names)
    
    # Save model in .keras format
    model.save("dog_breed_classifier.keras")
    print("Model saved as 'dog_breed_classifier.keras'")
    
    # Test prediction
    print("\nDemonstration prediction:")
    # Find a sample image
    for breed in class_names:
        val_breed_dir = os.path.join(DATA_DIR, "validation", breed)
        if os.path.exists(val_breed_dir) and len(os.listdir(val_breed_dir)) > 0:
            sample_img = os.path.join(val_breed_dir, os.listdir(val_breed_dir)[0])
            predict_image(model, sample_img, class_names)
            break

def predict_cli(image_path):
    # Load model
    model = keras.models.load_model("dog_breed_classifier.keras")
    
    # Get class names
    val_dir = os.path.join(DATA_DIR, "validation")
    class_names = sorted(os.listdir(val_dir))
    
    # Make prediction
    predict_image(model, image_path, class_names)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Predict mode
        image_path = sys.argv[1]
        predict_cli(image_path)
    else:
        # Training mode
        main()
