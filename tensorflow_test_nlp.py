"""
Small script to measure training speed of a transformer model on IMDB dataset
Adapted from: https://keras.io/examples/nlp/text_classification_with_transformer/ 
"""
import argparse
import time
import os

from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from helper_functions import get_nvidia_gpu_name

try:
    import cpuinfo
    CPU_PROCESSOR = cpuinfo.get_cpu_info().get("brand_raw").replace(" ", "_")
    print(f"[INFO] CPU Processor: {CPU_PROCESSOR}")
except Exception as e:
    print(f"Error: {e}, may have failed to get CPU_PROCESSOR name from cpuinfo, please install cpuinfo or set CPU_PROCESSOR manually") 

# Create argument parser
parser = argparse.ArgumentParser()

# Misc args
parser.add_argument("--batch_sizes", type=str, default="16, 32, 64, 128", help="String delimited series of batch sizes to test training speed on, default is '16, 32, 64, 128'")
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for, default is 3")

# Model args
parser.add_argument("--embed_dim", type=int, default=128, help="Embedding size for each token, default is 128")
parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads, default is 8")
parser.add_argument("--ff_dim", type=int, default=128, help="Hidden layer size in feed forward network inside transformer, default is 128")
parser.add_argument("--num_transformer_blocks", type=int, default=1, help="Number of transformer blocks in the model, default is 1")
parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout for layers outside of Transformer blocks, default is 0.1")
parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes, default is 2")

# Data args
parser.add_argument("--maxlen", type=int, default=200, help="Maximum length of input sequence, default is 200")
parser.add_argument("--vocab_size", type=int, default=20000, help="Vocabulary size, default is 20000")

args = parser.parse_args()

# Set constants  
GPU_NAME = get_nvidia_gpu_name()
DATASET_NAME = "IMDB"
BATCH_SIZES = [int(item.strip()) for item in args.batch_sizes.split(",")] # turn batch sizes into list, e.g. "16, 32, 64, 128" -> [16, 32, 64, 128]
MODEL_NAME = "SmallTransformer"
EPOCHS = args.epochs

# Model hyperparameters
EMBED_DIM = args.embed_dim
NUM_HEADS = args.num_heads
FEEDFORWARD_DIM = args.ff_dim
NUM_TRANSFORMER_BLOCKS = args.num_transformer_blocks
DROPOUT_RATE = args.dropout_rate
NUM_OUTPUT_CLASSES = args.num_classes

# Data hyperparameters
MAXLEN = args.maxlen
VOCAB_SIZE = args.vocab_size
INPUT_SHAPE = (MAXLEN,)

# Print info
print(f"[INFO] Testing model: {MODEL_NAME} on {DATASET_NAME} dataset with sequence length {INPUT_SHAPE} for {EPOCHS} epochs across batch sizes: {BATCH_SIZES}")

### Create Transformer Model ###
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def create_transformer_model(embed_dim=EMBED_DIM,
                             num_heads=NUM_HEADS,
                             ff_dim=FEEDFORWARD_DIM,
                             num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
                             dropout_rate=DROPOUT_RATE,
                             num_classes=NUM_OUTPUT_CLASSES,
                             maxlen=MAXLEN,
                             vocab_size=VOCAB_SIZE):
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_blocks = tf.keras.Sequential([TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_transformer_blocks)])
    x = transformer_blocks(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(units=num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

model = create_transformer_model(embed_dim=EMBED_DIM,
                                 num_heads=NUM_HEADS,
                                 ff_dim=FEEDFORWARD_DIM,
                                 num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
                                 dropout_rate=DROPOUT_RATE,
                                 num_classes=NUM_OUTPUT_CLASSES)

print(f"[INFO] Model summary:\n{model.summary()}")

### Data preparation
print(f"[INFO] Loading {DATASET_NAME} dataset...")
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
x_train = keras.utils.pad_sequences(x_train, maxlen=MAXLEN)
x_val = keras.utils.pad_sequences(x_val, maxlen=MAXLEN)

print(f"[INFO] Prepared {len(x_train)} training sequences of max length: {MAXLEN} with vocab size: {VOCAB_SIZE}")
print(f"[INFO] Prepared {len(x_val)} validation sequences of max length: {MAXLEN} with vocab size: {VOCAB_SIZE}")

def train(x_train=x_train,
          y_train=y_train,
          x_val=x_val,
          y_val=y_val,
          batch_sizes=BATCH_SIZES,
          epochs=EPOCHS):

    batch_size_training_results = []
    for batch_size in batch_sizes:
        print(f"[INFO] Training with batch size {batch_size} for {epochs} epochs...")

        # Prepare the data according to tf.data best practices - https://www.tensorflow.org/guide/data_performance 
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.cache().shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Create new model
        model = create_transformer_model()

        # Compile model
        model.compile(optimizer="adam", 
                    loss="sparse_categorical_crossentropy", 
                    metrics=["accuracy"])
        
        try: 

            # Start timer 
            start_time = time.time()

            history = model.fit(train_dataset,
                epochs=epochs, 
                # Not using validation data, just testing training speed
                # validation_data=(x_val, y_val)
            )

            # End timer
            end_time = time.time()

            total_training_time = end_time - start_time
            avg_time_per_epoch = total_training_time / EPOCHS
        
            batch_size_training_results.append({"batch_size": batch_size,
                                                "avg_time_per_epoch": avg_time_per_epoch})
            print(f"[INFO] Finished training with batch size {batch_size} for {epochs} epochs, total time: {round(total_training_time, 3)} seconds, avg time per epoch: {round(avg_time_per_epoch, 3)} seconds\n\n")
        except:
            print(f"[INFO] Failed training with batch size {batch_size} for {epochs} epochs...\n\n")
            batch_size_training_results.append({"batch_size": batch_size,
                                                "avg_time_per_epoch": "FAILED"})
            break

    return batch_size_training_results

if __name__ == "__main__":
    batch_size_training_results = train()
    print(f"[INFO] Results:\n{batch_size_training_results}")

    # Create CSV filename
    if GPU_NAME:
        csv_filename = f"{GPU_NAME.replace(' ', '_')}_{DATASET_NAME}_{MODEL_NAME}_{INPUT_SHAPE[0]}_results.csv"
    else:
        csv_filename = f"{CPU_PROCESSOR}_{DATASET_NAME}_{MODEL_NAME}_{INPUT_SHAPE[0]}_results.csv"

    # Make the target results directory if it doesn't exist (include the parents)
    target_results_dir = "results_tensorflow_nlp"
    results_path = Path("results") / target_results_dir
    results_path.mkdir(parents=True, exist_ok=True)
    csv_filepath = results_path / csv_filename

    # Turn dict into DataFrame
    df = pd.DataFrame(batch_size_training_results)
    # df.head() 

    # Save to CSV
    print(f"[INFO] Saving results to: {csv_filepath}")
    df.to_csv(csv_filepath, index=False)


