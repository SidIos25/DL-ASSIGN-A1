import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
learning_rate = 1e-3
num_epochs = 10
hidden_layers = [128, 64, 32]
activation_function = nn.ReLU()
optimizer_choice = "adam"
weight_init = "xavier"
loss_function = "cross_entropy"

data_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.SVHN(root="./data", split="train", transform=data_transform, download=True)
test_data = datasets.SVHN(root="./data", split="test", transform=data_transform, download=True)

train_size = int(0.9 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation_fn, weight_init):
        super(NeuralNet, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layer = nn.Linear(prev_size, hidden_size)
            if weight_init == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            layers.append(layer)
            layers.append(activation_fn)
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

input_size = 32 * 32
output_size = 10
model = NeuralNet(input_size, hidden_layers, output_size, activation_function, weight_init).to(device)

if loss_function == "cross_entropy":
    criterion = nn.CrossEntropyLoss()
elif loss_function == "squared_error":
    criterion = nn.MSELoss()

if optimizer_choice == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
elif optimizer_choice == "momentum":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
elif optimizer_choice == "nesterov":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
elif optimizer_choice == "rmsprop":
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
elif optimizer_choice == "adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif optimizer_choice == "nadam":
    optimizer = optim.NAdam(model.parameters(), lr=learning_rate)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if loss_function == "squared_error":
                labels = nn.functional.one_hot(labels, num_classes=10).float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if loss_function == "squared_error":
                    labels = nn.functional.one_hot(labels, num_classes=10).float()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
    return train_losses, val_losses

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return all_preds, all_labels

preds, labels = evaluate_model(model, test_loader)

cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

recommendations = [
    {"hidden_layers": [128, 64], "activation": "ReLU", "optimizer": "adam", "batch_size": 32, "accuracy": None},
    {"hidden_layers": [64, 32], "activation": "ReLU", "optimizer": "rmsprop", "batch_size": 64, "accuracy": None},
    {"hidden_layers": [128, 128], "activation": "sigmoid", "optimizer": "sgd", "batch_size": 16, "accuracy": None},
]

def evaluate_mnist_recommendations():
    for config in recommendations:
        model = NeuralNet(input_size, config["hidden_layers"], output_size, nn.ReLU(), weight_init).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        _, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
        _, acc = evaluate_model(model, test_loader)
        config["accuracy"] = acc
    print("MNIST Configurations and Accuracies:", recommendations)

evaluate_mnist_recommendations()