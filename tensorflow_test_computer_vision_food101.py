import argparse
import os
from timeit import default_timer as timer


from pathlib import Path

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from helper_functions import get_nvidia_gpu_name

try:
    import cpuinfo 
    CPU_PROCESSOR = cpuinfo.get_cpu_info().get('brand_raw').replace(" ", "_")
    print(f"[INFO] CPU Processor: {CPU_PROCESSOR}")
except Exception as e:
    print(f"Error: {e}, may have failed to get CPU_PROCESSOR name from cpuinfo, please install cpuinfo or set CPU_PROCESSOR manually") 

# Create argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--batch_sizes", default="32, 64, 128", help="Delimited list input of batch sizes to test, defaults to '32, 64, 128'", type=str)
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for, default is 3")
args = parser.parse_args()

# Convert batch_sizes to list
batch_size_args = [int(item.strip()) for item in args.batch_sizes.split(",")]

# Set constants  
GPU_NAME = get_nvidia_gpu_name()
DATASET_NAME = "FOOD101"
IMAGE_SIZE = 224
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
BATCH_SIZES = batch_size_args
MODEL_NAME = "ResNet50"
EPOCHS = args.epochs
BACKEND = "tensorflow"

print(f"[INFO] Testing model: {MODEL_NAME} on {DATASET_NAME} dataset with input shape {INPUT_SHAPE} for {EPOCHS} epochs across batch sizes: {BATCH_SIZES}")      

# Load the dataset 
# Note: This is store a ~5GB file in ./data, so make sure to delete it after if you want to free up space
print(f"[INFO] Loading {DATASET_NAME} dataset, note: this will store a ~5GB file in ./data, so make sure to delete it after if you want to free up space")
(train_data, test_data), dataset_info = tfds.load(
    'food101',
    split=['train', 'validation'],
    as_supervised=True,
    with_info=True,
    data_dir="./data"
)
print(f"[INFO] Finished loading {DATASET_NAME} dataset.")

# Create preprocess layer
preprocess_layer = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1/255.),
    tf.keras.layers.Resizing(height=IMAGE_SIZE, width=IMAGE_SIZE)
])

# Setup training
def train_and_time(batch_sizes=BATCH_SIZES,
          epochs=EPOCHS,
          train_data=train_data,
          test_data=test_data):

    batch_size_training_results = []
    for batch_size in batch_sizes:
        print(f"[INFO] Training with batch size {batch_size} for {epochs} epochs...")

        # Map preprocessing function to data and turn into batches
        train_data_batched = train_data.map(lambda image, label: (preprocess_layer(image), label)).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        # train_data = train_data.map(lambda image, label: (preprocess_layer(image), label)).shuffle(1000)
        test_data = test_data.map(lambda image, label: (preprocess_layer(image), label)) # don't shuffle test data (we're not using it anyway)

        # Print shape of first training batch
        for image_batch, label_batch in train_data_batched.take(1):
            print(f"[INFO] Training batch shape: {image_batch.shape}, label batch shape: {label_batch.shape}")

        # Create model
        model = tf.keras.applications.ResNet50(
                        include_top=True,
                        weights=None,
                        input_shape=INPUT_SHAPE,
                        classes=101,)

        # Create loss function and compile model
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

        try:
            start_time = timer()

            model.fit(train_data_batched, # batch the data dynamically
                      epochs=epochs, 
                      batch_size=batch_size,
                      validation_data=None)
                    # No validation, just testing training speed
                    # validation_data=(x_test, y_test))
                                
            end_time = timer()

            total_training_time = end_time - start_time
            avg_time_per_epoch = total_training_time / epochs

            batch_size_training_results.append({"batch_size": batch_size,
                                                "avg_time_per_epoch": avg_time_per_epoch})
            print(f"[INFO] Finished training with batch size {batch_size} for {epochs} epochs, total time: {round(total_training_time, 3)} seconds, avg time per epoch: {round(avg_time_per_epoch, 3)} seconds\n\n")
            
            save_results(batch_size_training_results)
        except Exception as e:
            print(f"[INFO] Error: {e}")
            print(f"[INFO] Failed training with batch size {batch_size} for {epochs} epochs...\n\n")
            batch_size_training_results.append({"batch_size": batch_size,
                                                "avg_time_per_epoch": "FAILED"})
            
            save_results(batch_size_training_results)
            break

    return batch_size_training_results

def save_results(batch_size_training_results, results_dir="results", target_dir="results_tensorflow_cv"):
    # Create CSV filename
    if GPU_NAME:
        csv_filename = f"{GPU_NAME.replace(' ', '_')}_{DATASET_NAME}_{MODEL_NAME}_{INPUT_SHAPE[0]}_{BACKEND}_results.csv"
    else:
        csv_filename = f"{CPU_PROCESSOR}_{DATASET_NAME}_{MODEL_NAME}_{INPUT_SHAPE[0]}_{BACKEND}_results.csv"

    # Make the target results directory if it doesn't exist (include the parents)
    target_results_dir = target_dir
    results_path = Path(results_dir) / target_results_dir
    results_path.mkdir(parents=True, exist_ok=True)
    csv_filepath = results_path / csv_filename

    # Turn dict into DataFrame
    df = pd.DataFrame(batch_size_training_results)

    # Save to CSV
    print(f"[INFO] Saving results to: {csv_filepath}")
    df.to_csv(csv_filepath, index=False)

if __name__ == "__main__":
    batch_size_training_results = train_and_time()
    print(f"[INFO] Results:\n{batch_size_training_results}")