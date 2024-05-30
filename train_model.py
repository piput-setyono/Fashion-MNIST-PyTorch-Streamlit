# -*- coding: utf-8 -*-
"""
Created on Tue May 28 02:03:32 2024

@author: Piput Setyono
"""

import torch
import datetime
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import transforms
import time
import argparse
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect

import matplotlib.pyplot as plt
import numpy as np
from load_model_mnist import MnistMobileNetV3Small, MnistMobileNetV3Large, MnistMobileNetV2, MnistResNet50

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--model",
    type=str,
    default="MobileNetV3Small",
    # required=True,
    choices=["ResNet50", "MobileNetV2", "MobileNetV3Small", "MobileNetV3Large"],
    help="Name of the model to train.",
)
parser.add_argument(
    "--layer_image_input",
    type=str,
    default=3,
    # required=True,
    choices=[1, 3],
    help="layer input image, 1 for grayscale, 3 for RGB",
)
parser.add_argument(
    "--percentage_validation_set",
    type=int,
    default=10,
    choices=range(0, 101),
    metavar="[0-100]",
    help="Percentage of data from the training set that should be used as a validation set.",
)
parser.add_argument(
    "--patience",
    type=int,
    default=-1,
    help="Number of epochs without improvements on the validation set's accuracy before stopping the training. Use -1 to deactivate early stopping.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=60,
    choices=range(1, 101),
    metavar="[1-100]",
    help="Batch size.",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=10,
    choices=range(1, 1001),
    metavar="[1-1000]",
    help="Maximum number of epochs.",
)
args = parser.parse_args()

def get_data_loaders(train_batch_size, val_batch_size, layer_input, percentage_validation_set):
    fashion_mnist = torchvision.datasets.FashionMNIST(download=True, train=True, root="data").data.float()
    if layer_input == 3:
        train_data_transform = transforms.Compose([transforms.RandomRotation(degrees=(0,180)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                             transforms.Normalize((fashion_mnist.mean()/255,), (fashion_mnist.std()/255,))])
        
        test_data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                             transforms.Normalize((fashion_mnist.mean()/255,), (fashion_mnist.std()/255,))])
    
    elif layer_input == 1:
        train_data_transform = transforms.Compose([transforms.RandomRotation(degrees=(0,180)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((fashion_mnist.mean()/255,), (fashion_mnist.std()/255,))])
        
        test_data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((fashion_mnist.mean()/255,), (fashion_mnist.std()/255,))])
        
    
    train_dataset = torchvision.datasets.FashionMNIST(root="data", train=True, transform=train_data_transform, download=False)
    test_dataset = torchvision.datasets.FashionMNIST(download=False, root="data", transform=test_data_transform, train=False)
        
    # Define the size of the training set and the validation set
    train_set_length = int(len(train_dataset) * (100 - percentage_validation_set) / 100)
    val_set_length = int(len(train_dataset) - train_set_length)
    
    train_set, val_set = torch.utils.data.random_split(train_dataset, (train_set_length, val_set_length))
    
    train_loader = DataLoader(train_set,batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_set,batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=val_batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in f"{inspect.signature(metric_fn)}":
        return metric_fn(true_y, pred_y, average='macro', zero_division=0.0)
    else:
        return metric_fn(true_y, pred_y)
    
def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

def main():
    # --- SETUP ---
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
    
    current_datetime = datetime.datetime.now()
    timestamp = current_datetime.strftime("%y%m%d-%H%M%S")
    experiment_name = f"{args.model}_imageLayer_{args.layer_image_input}_{timestamp}"
    
    # DATALOADER
    train_loader, val_loader, test_loader = get_data_loaders(args.batch_size, args.batch_size, args.layer_image_input, args.percentage_validation_set)
    
    # MODEL
    if args.model == "ResNet50":
        model = MnistResNet50(args.layer_image_input, 10).to(device)
    elif args.model == "MobileNetV2":
        model = MnistMobileNetV2(args.layer_image_input, 10).to(device)
    elif args.model == "MobileNetV3Small":
        model = MnistMobileNetV3Small(args.layer_image_input, 10).to(device)
    elif args.model == "MobileNetV3Large":
        model = MnistMobileNetV3Large(args.layer_image_input, 10).to(device)
    
    # loss function and optimizer
    loss_function = nn.CrossEntropyLoss() # your loss function, cross entropy works well for multi-class problems
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) # Using Karpathy's learning rate constant
    
    losses = []
    batches = len(train_loader)
    val_batches = len(val_loader)
    test_batches = len(test_loader)
    best_f1 = 0.0
    best_acc = 0.0
    best_epoch = 0
    epochs = args.num_epochs
    
    start_ts = time.time()
    # loop for every epoch (training + evaluation)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}:")
        total_loss = 0
    
        print("========== Training ============")
        # progress bar (works in Jupyter notebook too!)
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)
    
        # ----------------- TRAINING  -------------------- 
        # set model to training
        model.train()
        
        for i, data in progress:
            X, y = data[0].to(device), data[1].to(device)
            
            # training step for single batch
            model.zero_grad()
            outputs = model(X)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
    
            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss
    
            # updating progress bar
            progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))
            
        # releasing unceseccary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ----------------- VALIDATION  ----------------- 
        print("========== Validation ============")
        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []
        
        # set model to evaluating (testing)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X, y = data[0].to(device), data[1].to(device)
    
                outputs = model(X) # this get's the prediction from the network
    
                val_losses += loss_function(outputs, y)
    
                predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction
                
                # calculate P/R/F1/A metrics for batch
                for acc, metric in zip((precision, recall, f1, accuracy), 
                                        (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(
                        calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                    )
              
        print(f"training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
        print_scores(precision, recall, f1, accuracy, val_batches)
        losses.append(total_loss/batches) # for plotting learning curve
        
        if sum(f1)/val_batches > best_f1:
            best_f1 = sum(f1)/val_batches
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"./models/{experiment_name}_best_f1")    
        
        if sum(accuracy)/val_batches > best_acc:
            best_acc = sum(accuracy)/val_batches
    
    print("==================================")
    print(f"Training time: {time.time()-start_ts}s")
    torch.save(model.state_dict(), f"./models/{experiment_name}")    
    print(f"Best Model F1: Epoch {best_epoch} - F1 {best_f1} - Acc {best_acc}")
    
    # ----------------- TESTING  -------------------- 
    print("========== Validation ============")
    precision, recall, f1, accuracy = [], [], [], []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            X, y = data[0].to(device), data[1].to(device)

            outputs = model(X) # this get's the prediction from the network

            predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction
            
            # calculate P/R/F1/A metrics for batch
            for acc, metric in zip((precision, recall, f1, accuracy), 
                                    (precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(
                    calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                )
          
    print_scores(precision, recall, f1, accuracy, test_batches)
    
if __name__ == "__main__":
    main()