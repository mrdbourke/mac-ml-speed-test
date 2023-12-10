"""
Script to test training a PyTorch NLP model on a dataset using HuggingFace's Trainer class.

Source: https://huggingface.co/docs/transformers/tasks/sequence_classification (with modifications for a focus on MPS devices + tracking)
"""

# Standard library imports
import argparse
import random
from pathlib import Path

# Third-party imports
import accelerate 
import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

# Local application/library specific imports
from helper_functions import get_nvidia_gpu_name


if __name__ == "__main__":
    CPU_PROCESSOR = None

    ### Get CPU Processor name ###
    if not CPU_PROCESSOR:
        try:
            import cpuinfo
            CPU_PROCESSOR = cpuinfo.get_cpu_info().get("brand_raw").replace(" ", "_")
            print(f"[INFO] CPU Processor: {CPU_PROCESSOR}")
        except Exception as e:
            print(f"Error: {e}, may have failed to get CPU_PROCESSOR name from cpuinfo, please install cpuinfo or set CPU_PROCESSOR manually") 

    ### Setup device ###
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[INFO] MPS device found, using device: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] CUDA device found, using device: {device}")
    else:
        device = torch.device("cpu")
        print (f"[INFO] MPS or CUDA device not found, using device: {device} (results will be much slower than using MPS or CUDA)")


    # Set random seed
    torch.manual_seed(42)

    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", default="16, 32, 64, 128, 256, 512", help="Delimited list input of batch sizes to test, defaults to '16, 32, 64, 128, 256, 512, 1024'", type=str)
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for, default is 3")
    parser.add_argument("--quick_experiment", action="store_true", help="Whether to run a quick experiment, default is False")
    parser.add_argument("--use_fp16", action="store_true", help="Whether to use fp16 precision, default is False")
    parser.add_argument("--use_torch_compile", action="store_true", help="Whether to use torch compile, default is False")
    args = parser.parse_args()

    print(args.quick_experiment)

    # Turn args.quick_experiment into boolean
    # print(f"[INFO] args.quick_experiment set to '{args.quick_experiment}', converting to boolean... ")
    # args.quick_experiment = str(args.quick_experiment) == "True"
    # print(f"[INFO] args.quick_experiment set to {args.quick_experiment}")
   
    # Convert batch_sizes to list
    batch_size_args = [int(item.strip()) for item in args.batch_sizes.split(",")]

    ### Set constants ###
    GPU_NAME = get_nvidia_gpu_name()
    BACKEND = "pytorch"
    MODEL_NAME = "distilbert-base-uncased"

    # If quick experiment, set epochs and batch size to simple values
    if args.quick_experiment:
        print(f"[INFO] args.quick_experiment set to True, setting epochs to 1 and batch_sizes to 32...")
        EPOCHS = 1
        BATCH_SIZES = [32]
    else:
        EPOCHS = args.epochs
        BATCH_SIZES = batch_size_args

    INPUT_SHAPE = (1, 512) # this is the number of tokens per sample, 512 is the max for distilbert-base-uncased
    DATASET_NAME = "IMDB"

    ### Print constants ###
    print(f"[INFO] Training {MODEL_NAME} model on {DATASET_NAME} dataset for {EPOCHS} epochs with batch sizes: {BATCH_SIZES}...")

    # Print out whether using FP16 or torch.compile
    if args.use_fp16:
        print(f"[INFO] Using fp16 precision (only available on NVIDIA GPUs, not MPS).")

    if args.use_torch_compile:
        print(f"[INFO] Using torch compile (only availabe on NVIDIA GPUs, not MPS).")

    ### Load dataset ###
    print(f"[INFO] Loading IMDB dataset...")
    imdb = load_dataset("imdb",
                        cache_dir="./data")
    rand_idx = random.randint(0, len(imdb['train']))
    print(f"[INFO] IMDB dataset loaded. Example sample:\n{imdb['train'][rand_idx]}")


    ### Load tokenizer ###
    print(f"[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    print(f"[INFO] Tokenizer loaded.")

    ### Preprocess dataset ###
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    print(f"[INFO] Preprocessing dataset...")
    tokenized_imdb = imdb.map(preprocess_function, batched=True)

    print(f"[INFO] Preprocessing complete. Example sample:\n{tokenized_imdb['train'][rand_idx]}")

    ### Create evaluation metric ### 
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        """
            Computes the accuracy of the model.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    ### Create mapping from label to id and vice versa ###
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    def count_parameters(model):
        """Helper function to count number of parameters, trainable, non-trainable and total."""
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total_parameters = trainable_parameters + non_trainable_parameters
        print(f"Trainable parameters: {trainable_parameters}")
        print(f"Non-trainable parameters: {non_trainable_parameters}")
        print(f"Total parameters: {total_parameters}")
        return trainable_parameters, non_trainable_parameters, total_parameters
    
    def save_results(batch_size_training_results):
        # Create CSV filename
        if GPU_NAME:
            csv_filename = f"{GPU_NAME.replace(' ', '_')}_{DATASET_NAME}_{MODEL_NAME}_{INPUT_SHAPE[-1]}_{BACKEND}_results.csv"
        else:
            csv_filename = f"{CPU_PROCESSOR}_{DATASET_NAME}_{MODEL_NAME}_{INPUT_SHAPE[-1]}_{BACKEND}_results.csv"

        # Make the target results directory if it doesn't exist (include the parents)
        target_results_dir = "results_pytorch_nlp"
        results_path = Path("results") / target_results_dir
        results_path.mkdir(parents=True, exist_ok=True)
        csv_filepath = results_path / csv_filename

        # Turn dict into DataFrame 
        df = pd.DataFrame(batch_size_training_results) 

        # Save to CSV
        print(f"[INFO] Saving results to: {csv_filepath}")
        df.to_csv(csv_filepath, index=False)

   
    """
    Optional? Create Data Collator (not sure if this seems to produce a warning each time or if it's AutoTokenizer?
    Seems to work though...
    """
    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ### Create model training and timing code ###
    batch_size_training_results = []
    for batch_size in BATCH_SIZES:

        print(f"[INFO] Training model with batch size: {batch_size}")

        try:
            print(f"[INFO] Instantiating DistilBert Model...")
            model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                        num_labels=2,
                                                                        id2label=id2label,
                                                                        label2id=label2id)

            if args.quick_experiment: # only train last layer
                print(f"[INFO] Quick experiment: only training last layer/using 1000 rows of data...")

                # Freeze all base layers
                for param in model.parameters():
                    param.requires_grad = False

                for param in model.classifier.parameters():
                    param.requires_grad = True
        
            else: # if not quick experiment, train last transformer layer and classifier layers
            
                # Freeze all base layers
                for param in model.parameters():
                    param.requires_grad = False
        
                # Only train last transformer layer and classifier layers
                for param in model.distilbert.transformer.layer[-1].parameters():
                    param.requires_grad = True
        
                for param in model.pre_classifier.parameters():
                    param.requires_grad = True
    
                for param in model.classifier.parameters():
                    param.requires_grad = True

            count_parameters(model)

            training_args = TrainingArguments(
                output_dir="pytorch_hf_nlp_model",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                # per_device_eval_batch_size=16, # Don't eval, just train for speed testing
                num_train_epochs=EPOCHS,
                weight_decay=0.01,
                evaluation_strategy="no",
                save_strategy="no", # don't save during training
                load_best_model_at_end=False, 
                push_to_hub=False,
                use_cpu=False, # defaults to False (will always try to use CUDA GPU or Mac MPS device if available)
                fp16=args.use_fp16, # defaults to False (will use float32 precision by default), note: not available on MPS devices
                auto_find_batch_size=False, # Note: may be something to explore in the future to automatically find batch size with `accelerate` installed
                torch_compile=args.use_torch_compile, # defaults to False, compiling may speedup thanks to PyTorch 2.0, see: https://www.learnpytorch.io/pytorch_2_intro/ (best results on NVIDIA Ampere GPUs and above, not MPS)
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_imdb["train"] if not args.quick_experiment else tokenized_imdb["train"].select(range(1000)),
                # eval_dataset=tokenized_imdb["test"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            trainer_output = trainer.train()
            trainer_metrics_dict = trainer_output.metrics
            trainer_metrics_dict["batch_size"] = batch_size

            print(f"[INFO] Trainer metrics for batch size {batch_size}:\n{trainer_metrics_dict}")

            batch_size_training_results.append(trainer_metrics_dict)
            save_results(batch_size_training_results)

            # Delete model and trainer instance, clear cache
            del model
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        except Exception as e:
            print(f"[INFO] Error: {e}")
            print(f"[INFO] Failed training with batch size {batch_size} for {EPOCHS} epochs...\n\n")

            batch_size_training_results.append({'train_runtime': "FAILED",
                                                'train_samples_per_second': "FAILED",
                                                'train_steps_per_second': "FAILED",
                                                'total_flos': "FAILED",
                                                'train_loss': "FAILED",
                                                'epoch': "FAILED",
                                                'batch_size': batch_size})
            save_results(batch_size_training_results)

            # Delete model and trainer instance, clear cache
            del model
            del trainer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            break
    
    print("[INFO] Finished training with all batch sizes.")        

    print(f"[INFO] Results:\n{batch_size_training_results}")