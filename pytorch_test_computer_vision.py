import argparse
import os
from pathlib import Path
from timeit import default_timer as timer

import torch
import torchvision
import torchvision.transforms.v2 as transforms # use v2 transforms for faster augmentations
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm.auto import tqdm

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
    parser.add_argument("--batch_sizes", default="16, 32, 64, 128, 256, 512, 1024", help="Delimited list input of batch sizes to test, defaults to '16, 32, 64, 128, 256, 512, 1024'", type=str)
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for, default is 5")
    args = parser.parse_args()

    # Convert batch_sizes to list
    batch_size_args = [int(item.strip()) for item in args.batch_sizes.split(",")]

    ### Set constants ###
    GPU_NAME = get_nvidia_gpu_name()
    BACKEND = "pytorch"
    MODEL_NAME = "resnet50"
    IMAGE_SIZE = 32
    INPUT_SHAPE = (3, IMAGE_SIZE, IMAGE_SIZE)
    NUM_WORKERS = os.cpu_count()
    EPOCHS = args.epochs
    BATCH_SIZES = batch_size_args
    DATASET_NAME = "CIFAR100"

    print(f"[INFO] Testing model: {MODEL_NAME} on {DATASET_NAME} dataset with input shape {INPUT_SHAPE} for {EPOCHS} epochs across batch sizes: {BATCH_SIZES}")


    ### Prepare Data ### 
    simple_transform = transforms.Compose([
        transforms.Resize(size=IMAGE_SIZE),
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True)
    ])

    # Get Datasets
    train_data = datasets.CIFAR10(root="data",
                                train=True,
                                transform=simple_transform,
                                download=True)

    test_data = datasets.CIFAR10(root="data",
                                train=False,
                                transform=simple_transform,
                                download=True)

    print(f"[INFO] Number of training samples: {len(train_data)}, number of testing samples: {len(test_data)}")

    # Create DataLoaders
    def create_dataloaders(batch_size, num_workers=NUM_WORKERS):
        train_dataloader = DataLoader(train_data,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=False) # note: if you pin memory, you may get "too many workers" errors when recreating DataLoaders, see: https://github.com/Lightning-AI/pytorch-lightning/issues/18487#issuecomment-1740244601

        test_dataloader = DataLoader(test_data,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=False)

        return train_dataloader, test_dataloader

    ### Train Step ###
    def train_step(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module, 
                optimizer: torch.optim.Optimizer,
                device: torch.device):
        # Put model in train mode
        model.train()
        
        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0
        
        # Loop through data loader data batches
        for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Send data to target device
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
    #         X, y = X.to(device, non_blocking=True, memory_format=torch.channels_last), y.to(device, non_blocking=True)
    #         X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item() 

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)
            
        # Adjust metrics to get average loss and accuracy per batch 
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc

    ### Test Step ###
    def test_step(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module,
                device: torch.device):
        # Put model in eval mode
        model.eval() 
        
        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0
        
        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
                # Send data to target device
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
    #             X, y = X.to(device, non_blocking=True, memory_format=torch.channels_last), y.to(device, non_blocking=True)
    #             X, y = X.to(device), y.to(device)
        
                # 1. Forward pass
                test_pred_logits = model(X)

                # 2. Calculate and accumulate loss
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                
                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
                
        # Adjust metrics to get average loss and accuracy per batch 
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc

    # 1. Take in various parameters required for training and test steps
    def train_and_test_model(model: torch.nn.Module, 
                            train_dataloader: torch.utils.data.DataLoader, 
                            test_dataloader: torch.utils.data.DataLoader, 
                            optimizer: torch.optim.Optimizer,
                            loss_fn: torch.nn.Module,
                            epochs: int,
                            device: torch.device,
                            eval: bool=False):
        
        print(f"[INFO] Training model {model.__class__.__name__} on device '{device}' for {epochs} epochs...")
        
        results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

        # Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            # Do eval before training (to see if there's any errors)
            if eval:
                test_loss, test_acc = test_step(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device)
            
            train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
            
            
            # Print out what's happening
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
            )

            if eval:
                print(
                    f"Epoch: {epoch+1} | "
                    f"test_loss: {test_loss:.4f} | "
                    f"test_acc: {test_acc:.4f} | "
                )

            # Save results to dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            if eval:
                results["test_loss"].append(test_loss)
                results["test_acc"].append(test_acc)
        
        return results

    def train_and_time(batch_sizes=BATCH_SIZES,
                       epochs=EPOCHS,
                       device=device):

        batch_size_training_results = []

        for batch_size in batch_sizes:
            print(f"[INFO] Training with batch size {batch_size} for {epochs} epochs...")
            # Create an instance of resnet50
            model = torchvision.models.resnet50(num_classes=100).to(device)
            # model = torch.compile(model) # potential way to speed up model

            # Setup loss function and optimizer
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

            # Create DataLoaders
            train_dataloader, test_dataloader = create_dataloaders(batch_size=batch_size)

            try:
                # Start the timer
                start_time = timer()

                # Train model
                model_results = train_and_test_model(model=model, 
                                                    train_dataloader=train_dataloader,
                                                    test_dataloader=test_dataloader,
                                                    optimizer=optimizer,
                                                    loss_fn=loss_fn, 
                                                    epochs=epochs,
                                                    device=device,
                                                    eval=False) # don't eval, just test training time

                # End the timer
                end_time = timer()

                total_training_time = end_time - start_time
                avg_time_per_epoch = total_training_time / epochs

                batch_size_training_results.append({"batch_size": batch_size,
                                                    "avg_time_per_epoch": avg_time_per_epoch})
                print(f"[INFO] Finished training with batch size {batch_size} for {epochs} epochs, total time: {round(total_training_time, 3)} seconds, avg time per epoch: {round(avg_time_per_epoch, 3)} seconds\n\n")

            except Exception as e:
                print(f"[INFO] Error: {e}")
                print(f"[INFO] Failed training with batch size {batch_size} for {epochs} epochs...\n\n")
                batch_size_training_results.append({"batch_size": batch_size,
                                                    "avg_time_per_epoch": "FAILED"})
                break
                
        return batch_size_training_results

    ### Train an time model ### 
    batch_size_training_results = train_and_time(batch_sizes=BATCH_SIZES,
                                                 epochs=EPOCHS,
                                                 device=device)

    print("[INFO] Finished training with all batch sizes.")        

    print(f"[INFO] Results:\n{batch_size_training_results}")

    # Create CSV filename
    if GPU_NAME:
        csv_filename = f"{GPU_NAME}_{DATASET_NAME}_{MODEL_NAME}_{INPUT_SHAPE[-1]}_{BACKEND}_results.csv"
    else:
        csv_filename = f"{CPU_PROCESSOR}_{DATASET_NAME}_{MODEL_NAME}_{INPUT_SHAPE[-1]}_{BACKEND}_results.csv"

    # Make the target results directory if it doesn't exist (include the parents)
    target_results_dir = "results_pytorch_cv"
    results_path = Path("results") / target_results_dir
    results_path.mkdir(parents=True, exist_ok=True)
    csv_filepath = results_path / csv_filename

    # Turn dict into DataFrame 
    df = pd.DataFrame(batch_size_training_results) 

    # Save to CSV
    print(f"[INFO] Saving results to: {csv_filepath}")
    df.to_csv(csv_filepath, index=False)