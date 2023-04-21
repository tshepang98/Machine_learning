import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Define your MLP model here
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        # Define your layers here
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Define the forward pass through the layers here
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Define your training function here
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs.view(inputs.size(0), -1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# Define your testing function here
def test(model, dataloader, device):
    model.eval()
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs.view(inputs.size(0), -1))
            _, preds = torch.max(outputs, 1)

            correct_preds += torch.sum(preds == labels.data)
            total_preds += inputs.size(0)

    acc = correct_preds.double() / total_preds
    return acc.item()

# Define the main function here
def main():
    # Define your hyperparameters here
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 15
    input_size = 32 * 32 * 3
    hidden_size = 512
    output_size = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the transforms for the data here (if needed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Define the dataloaders for the data
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize your MLP model and send it to the device
    model = MLP(input_size, output_size).to(device)

    # Define your loss function and optimizer here
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        test_acc = test(model, test_dataloader, device)

    print(f"Epoch {epoch+1}/{num_epochs}:\n"
          f"Train loss: {train_loss:.4f}\n"
          f"Test accuracy: {test_acc:.4f}\n") 
if name == 'main':
    main()
