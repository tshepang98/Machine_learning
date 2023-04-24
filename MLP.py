import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# Define data transformations
# transform = transforms.Compose([
#     transforms.ToTensor(), # Convert the images to tensors
#     transforms.Normalize((0.5, 0.5,0.5), (0.5, 0.5,0.5)) # Normalize the pixel values
# ])
torch.manual_seed(42)
transform = transforms.Compose([

    transforms.ToTensor(),  # Convert to Tensor
    transforms.RandomHorizontalFlip(p=0.3),
    # Normalize Image to [-1, 1] first number is mean, second is std deviation
    transforms.Normalize((0.4914,0.4822,0.4465), (0.247,0.243,0.261)),
    
])
# Download the CIFAR-10 training set and apply transformations
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Download the CIFAR-10 test set and apply transformations
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# Create DataLoader objects to iterate over the datasets in batches during training and testing
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

# Define your MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the input images
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# Define your optimizer and loss function
input_size = 32*32*3 # Input size is 32x32x3 for CIFAR-10
hidden_size = 512
num_classes = 10
net = MLP(input_size, hidden_size, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Train the model for 30 epochs
start = time.time()
for epoch in range(15):
    train_loss = 0.0
    net.train() # Set the model to training mode
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
 
    # Evaluate the model on the test set
    net.eval() # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Train loss = {train_loss/len(trainloader):.4f}, Test accuracy = {test_acc:.4f}%")
end = time.time()
minutes = (end - start)/60
print("The time elapsed ",minutes)