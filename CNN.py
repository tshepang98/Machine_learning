import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the LeNet5 CNN
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Define train function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Define test function
def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total)*100

# Define data transforms
transform_train = transforms.Compose([
    ########## data tecniques
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    #########
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# Set the device for the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the training and test datasets and data loaders
train_dataset = datasets.CIFAR10(root="data/", train=True, transform=transform_train, download=True)
test_dataset = datasets.CIFAR10(root="data/", train=False, transform=transform_test, download=True)
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
testloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Define the loss function, optimizer, and the model
criterion = nn.CrossEntropyLoss()
net = LeNet5().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Train the model for 15 epochs
for epoch in range(15):
    train_loss = train(net, trainloader, criterion, optimizer, device)
    test_acc = test(net, testloader, device)
    print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
