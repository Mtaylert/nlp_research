import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


#Create Fully Connected Netowrk
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        #calls the initialization of the parent class
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = NN(784, 10)
x = torch.randn(64, 784)
print(model(x).shape)


#Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

#Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#Init netowrk
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#Train
for epoch in range(num_epochs):
    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device=device).reshape(data.shape[0],-1)
        targets = targets.to(device=device)
        scores = model(data)
        loss = criterion(scores, targets)

        #backward
        optimizer.zero_grad() #reset the gradients
        loss.backward()

        #gradient descent
        optimizer.step()


#Check accuracy on training & test
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('check accuracy on training data')
    else:
        print('check accuracy on test data')
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            scores = model(x)

            #64X10
            scores.max()
            predictions = scores.max(1)[1]
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
        model.train()


check_accuracy(train_loader,model)