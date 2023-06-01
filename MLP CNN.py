# -*- coding: utf-8 -*-
"""
"""

import torch
import torchvision
from torchvision.datasets import EMNIST
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset
import time

# Load the EMNIST Balanced dataset
transform = transforms.Compose([transforms.ToTensor()])

trainset = EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
testset = EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Print the number of training/testing samples
print(f"Number of training samples: {len(trainset)}")
print(f"Number of testing samples: {len(testset)}")

# Function to unnormalize and display an image with true labels and predictions
def imshow_with_predictions(img, true_labels, mlp_preds, cnn_preds):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    for i, label in enumerate(true_labels):
        plt.text(i * 32, -5, trainset.classes[label], fontsize=10, ha='center', color='k')

    for i, mlp_pred in enumerate(mlp_preds):
        plt.text(i * 32, 36, trainset.classes[mlp_pred], fontsize=10, ha='center', color='blue')

    for i, cnn_pred in enumerate(cnn_preds):
        plt.text(i * 32, 48, trainset.classes[cnn_pred], fontsize=10, ha='center', color='green')

    plt.show()
    
def cross_validate(dataset, model, criterion, optimizer, device, k=5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        trainloader = DataLoader(dataset, batch_size=100, sampler=train_subsampler, num_workers=2)
        valloader = DataLoader(dataset, batch_size=100, sampler=val_subsampler, num_workers=2)

        for epoch in range(num_epochs):
            train_evaluate(model, trainloader, criterion, optimizer, device)

        _, val_accuracy = test_evaluate(model, valloader, criterion, device)
        fold_accuracies.append(val_accuracy)
        print(f"Fold [{fold + 1}/{k}], Validation Accuracy: {val_accuracy * 100:.2f}%")

    return np.mean(fold_accuracies)

# Get some random samples from the training dataset
dataiter = iter(trainloader)
images, labels = next(dataiter)  # Use next() function
images, labels = images[:6], labels[:6]  # Choose the top six samples
# Visualize the samples along with their true labels
imshow_with_predictions(torchvision.utils.make_grid(images), labels, [], [])
print("True labels:", ' '.join('%5s' % trainset.classes[labels[j]] for j in range(6)))
# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=47):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Train and evaluate function
def train_evaluate(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / (i + 1), correct / total

def test_evaluate(model, testloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / (i + 1), correct / total
# Set the hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 1 * 28 * 28
hidden_size = 256
num_classes = 47
num_epochs = 10
batch_size = 50
learning_rate = 0.001
momentum = 0.9


# Train the MLP model with cross-validation
start_time_mlp = time.time()
mlp = MLP(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_mlp = optim.Adam(mlp.parameters(), lr=learning_rate)
avg_val_accuracy_mlp = cross_validate(ConcatDataset([trainset, testset]), mlp, criterion, optimizer_mlp, device)
print(f"MLP - Average Validation Accuracy: {avg_val_accuracy_mlp * 100:.2f}%")

# Train the CNN model with cross-validation
cnn = CNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_cnn = optim.Adam(cnn.parameters(), lr=learning_rate)
avg_val_accuracy_cnn = cross_validate(ConcatDataset([trainset, testset]), cnn, criterion, optimizer_cnn, device)
print(f"CNN - Average Validation Accuracy: {avg_val_accuracy_cnn * 100:.2f}%")



# Evaluate models on the test set
test_loss_mlp, test_accuracy_mlp = test_evaluate(mlp, testloader, criterion, device)
test_loss_cnn, test_accuracy_cnn = test_evaluate(cnn, testloader, criterion, device)

# Initialize lists to store losses and accuracies
train_losses_mlp, train_accuracies_mlp = [], []
train_losses_cnn, train_accuracies_cnn = [], []
test_losses_mlp, test_accuracies_mlp = [], []
test_losses_cnn, test_accuracies_cnn = [], []

# Initialize the start time
start_time = time.time()

# Train the MLP model and store losses and accuracies
for epoch in range(num_epochs):
    train_loss, train_acc = train_evaluate(mlp, trainloader, criterion, optimizer_mlp, device)
    test_loss, test_acc = test_evaluate(mlp, testloader, criterion, device)
    train_losses_mlp.append(train_loss)
    train_accuracies_mlp.append(train_acc)
    test_losses_mlp.append(test_loss)
    test_accuracies_mlp.append(test_acc)
    print(f"MLP - Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%")

# Calculate and print the training time
end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

# Train the CNN model and store losses and accuracies
for epoch in range(num_epochs):
    train_loss, train_acc = train_evaluate(cnn, trainloader, criterion, optimizer_cnn, device)
    test_loss, test_acc = test_evaluate(cnn, testloader, criterion, device)
    train_losses_cnn.append(train_loss)
    train_accuracies_cnn.append(train_acc)
    test_losses_cnn.append(test_loss)
    test_accuracies_cnn.append(test_acc)
    print(f"CNN - Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%")
# Function to plot loss and accuracy graphs
def plot_graphs(epochs, train_losses, test_losses, train_accuracies, test_accuracies, title):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train')
    plt.plot(range(1, epochs + 1), test_losses, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{title} - Loss vs. Epochs')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{title} - Accuracy vs. Epochs')
    
    plt.show()
def plot_combined_loss_graphs(epochs, train_losses_mlp, val_losses_mlp, train_losses_cnn, val_losses_cnn):
    plt.figure(figsize=(12, 6))

    plt.plot(range(1, epochs + 1), train_losses_mlp, label='MLP Train')
    plt.plot(range(1, epochs + 1), val_losses_mlp, label='MLP Validation')
    plt.plot(range(1, epochs + 1), train_losses_cnn, label='CNN Train', linestyle='--')
    plt.plot(range(1, epochs + 1), val_losses_cnn, label='CNN Validation', linestyle='--')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('MLP and CNN - Train and Validation Loss vs. Epochs')

    plt.show()

# Plot graphs for MLP and CNN
#plot_graphs(num_epochs, train_losses_mlp, test_losses_mlp, train_accuracies_mlp, test_accuracies_mlp, 'MLP')
#plot_graphs(num_epochs, train_losses_cnn, test_losses_cnn, train_accuracies_cnn, test_accuracies_cnn, 'CNN')
plot_combined_loss_graphs(num_epochs, train_losses_mlp, test_losses_mlp, train_losses_cnn, test_losses_cnn)


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def test_evaluate(model, testloader, device):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    return np.array(true_labels), np.array(predicted_labels)

# Get true labels and predicted labels for MLP and CNN
true_labels_mlp, predicted_labels_mlp = test_evaluate(mlp, testloader, device)
true_labels_cnn, predicted_labels_cnn = test_evaluate(cnn, testloader, device)
import seaborn as sns

def plot_confusion_matrix(true_labels, predicted_labels, title):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=trainset.classes, yticklabels=trainset.classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{title} - Confusion Matrix')
    plt.show()

def compute_metrics(true_labels, predicted_labels, title):
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"{title} - Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=trainset.classes))
def plot_graphs(epochs, train_losses, test_losses, train_accuracies, test_accuracies, model_name, title):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train')
    plt.plot(range(1, epochs + 1), test_losses, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} - {title} - Loss vs. Epochs')
    
    #plt.subplot(1, 2, 2)
    #plt.plot(range(1, epochs + 1), train_accuracies, label='Train')
    #plt.plot(range(1, epochs + 1), test_accuracies, label='Test')
    #plt.xlabel('Epochs')
    #plt.ylabel('Accuracy')
    #plt.legend()
    #plt.title(f'{model_name} - {title} - Accuracy vs. Epochs')
    
    plt.show()
# Plot accuracy graphs separately for MLP and CNN
plot_graphs(num_epochs, train_losses_mlp, test_losses_mlp, train_accuracies_mlp, test_accuracies_mlp, 'MLP', 'Accuracy vs. Epochs')
plot_graphs(num_epochs, train_losses_cnn, test_losses_cnn, train_accuracies_cnn, test_accuracies_cnn, 'CNN', 'Accuracy vs. Epochs')

#plot_accuracies(train_accuracies_mlp, test_accuracies_mlp, num_epochs)


# Plot confusion matrices and compute performance metrics for MLP and CNN
plot_confusion_matrix(true_labels_mlp, predicted_labels_mlp, "MLP")
compute_metrics(true_labels_mlp, predicted_labels_mlp, "MLP")

plot_confusion_matrix(true_labels_cnn, predicted_labels_cnn, "CNN")
compute_metrics(true_labels_cnn, predicted_labels_cnn, "CNN")
# Get some random samples from the testing dataset
test_dataiter = iter(testloader)
test_images, test_labels = next(test_dataiter)  # Use next() function
test_images, test_labels = test_images[:6], test_labels[:6]  # Choose the top six samples

# Get predictions for the top six samples for both MLP and CNN models
mlp_predicted_labels = torch.max(mlp(test_images.to(device)).data, 1)[1]
cnn_predicted_labels = torch.max(cnn(test_images.to(device)).data, 1)[1]

# Display the top six samples, true labels (black), MLP predictions (blue), and CNN predictions (green) on the image
imshow_with_predictions(torchvision.utils.make_grid(test_images), test_labels, mlp_predicted_labels.cpu().numpy(), cnn_predicted_labels.cpu().numpy())