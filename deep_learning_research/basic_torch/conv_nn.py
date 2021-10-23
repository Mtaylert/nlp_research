import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

class CNN(nn.Module):
    #in channels = 1 because MNIST is black and white
    #if it was RGB, in_channel would be 3
    def __init__(self, in_channels=1, num_classes = 10):
        #instantiate parent class
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3),
                               stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,
                               kernel_size=(3,3), stride=(1,1),
                               padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


model = CNN()
x = torch .randn(64, 1, 28, 28)
confidence = model(x)

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
IN_CHANNEL = 1
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 2

# Load Data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, download=True, transform=transforms.ToTensor()
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


model = CNN().to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train
for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, targets) in tqdm(enumerate(train_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)
        print(targets)
        #forward
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