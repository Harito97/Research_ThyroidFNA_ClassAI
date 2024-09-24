import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Read the CSV files
def read_csv(file_path):
    df = pd.read_csv(file_path)
    y = df['label'].values
    X = df['predicted_vector'].apply(eval).values
    return X, y

train_X, train_y = read_csv('train.csv')
val_X, val_y = read_csv('val.csv')
test_X, test_y = read_csv('test.csv')

# Step 2: Create X_flatten
def flatten_features(X):
    return np.array([np.array(x).flatten() for x in X])

X_flatten_train = flatten_features(train_X)
X_flatten_val = flatten_features(val_X)
X_flatten_test = flatten_features(test_X)

# Step 3: Prepare data for different approaches

# Approach 1: ANN with 39 features
def train_ann(X_train, y_train, X_val, y_val):
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Approach 2: Dimension reduction + ANN
def train_ann_with_pca(X_train, y_train, X_val, y_val, n_components=20):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000)
    model.fit(X_train_pca, y_train)
    return model, scaler, pca

# Approach 3: Transformer
class TransformerModel(nn.Module):
    def __init__(self, d_model=3, nhead=1, num_layers=2, num_classes=3):
        super(TransformerModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )
        self.fc = nn.Linear(d_model * 13, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_transformer(X_train, y_train, X_val, y_val, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = TransformerModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X.transpose(0, 1))
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    return model

# Step 4: Training
ann_model = train_ann(X_flatten_train, train_y, X_flatten_val, val_y)
ann_pca_model, scaler, pca = train_ann_with_pca(X_flatten_train, train_y, X_flatten_val, val_y)
transformer_model = train_transformer(train_X, train_y, val_X, val_y)

# Step 5: Validation
def evaluate_ann(model, X, y):
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)

def evaluate_ann_pca(model, scaler, pca, X, y):
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    y_pred = model.predict(X_pca)
    return accuracy_score(y, y_pred)

def evaluate_transformer(model, X, y):
    device = next(model.parameters()).device
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        outputs = model(X_tensor.transpose(0, 1))
        _, predicted = torch.max(outputs, 1)
    return accuracy_score(y, predicted.cpu().numpy())

print("ANN Accuracy:", evaluate_ann(ann_model, X_flatten_val, val_y))
print("ANN+PCA Accuracy:", evaluate_ann_pca(ann_pca_model, scaler, pca, X_flatten_val, val_y))
print("Transformer Accuracy:", evaluate_transformer(transformer_model, val_X, val_y))