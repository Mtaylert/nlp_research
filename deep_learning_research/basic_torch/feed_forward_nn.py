import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Create Fully Connected Netowrk
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        # calls the initialization of the parent class
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x_data):
        x_data = F.relu(self.fc1(x_data))
        x_data = self.fc2(x_data)
        return x_data


model = NN(784, 10)
x_data = torch.randn(64, 784)
print(model(x_data).shape)


# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 1

# Load Data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, download=True, transform=transforms.ToTensor()
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Init netowrk
model = NN(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train
for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device).reshape(data.shape[0], -1)
        targets = targets.to(device=device)
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()  # reset the gradients
        loss.backward()

        # gradient descent
        optimizer.step()


# Check accuracy on training & test
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("check accuracy on training data")
    else:
        print("check accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x_train, y_train in loader:
            x_train = x_train.to(device=device)
            y_train = y_train.to(device=device)
            x_train = x_train.reshape(x_train.shape[0], -1)
            scores = model(x_train)

            # 64X10
            scores.max()
            predictions = scores.max(1)[1]
            num_correct += (predictions == y_train).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )
        model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)