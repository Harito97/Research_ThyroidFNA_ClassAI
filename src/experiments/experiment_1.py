# src/experiments/experiment_1.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.data_loading import load_data
from src.models.simple_cnn import SimpleCNN
from src.utils.metrics import calculate_accuracy
from src.utils.visualization import plot_training_curve

def run(config):
    print("Running Experiment A: Baseline CNN")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    
    # Load data
    train_data, val_data, test_data = load_data(
        config['data']['train_path'],
        config['data']['val_path'],
        config['data']['test_path']
    )
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['training']['batch_size'])
    test_loader = DataLoader(test_data, batch_size=config['training']['batch_size'])
    
    # Initialize model
    model = SimpleCNN(layers=config['model']['layers'], num_classes=len(config['classes']))
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Training loop
    train_losses, val_accuracies = [], []
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_accuracy = calculate_accuracy(model, val_loader)
        
        train_losses.append(train_loss / len(train_loader))
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']}, "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Val Accuracy: {val_accuracies[-1]:.4f}")
    
    # Test the model
    test_accuracy = calculate_accuracy(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Plot training curve
    plot_training_curve(train_losses, val_accuracies)
    
    # Save the model
    torch.save(model.state_dict(), f"results/experiment_a_{config['name']}.pth")
    
    print("Experiment A completed.")