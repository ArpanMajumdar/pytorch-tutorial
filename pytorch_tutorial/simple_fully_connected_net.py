import torch
import torch.nn.functional as F
from icecream import ic
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Create fully connected network


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


class NeuralNet(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, n_classes: int) -> None:
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Set device
device = torch.device(get_device())
ic(device)

# Hyperparameters
input_size = 784
hidden_size = 50
n_classes = 10
learning_rate = 1e-3
batch_size = 64
n_epochs = 1

# Load data
train_dataset = datasets.MNIST(
    "datasets/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    "datasets/", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Initialize neural network
model = NeuralNet(input_size=input_size, hidden_size=hidden_size, n_classes=n_classes)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# Train network
for epoch in range(n_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        if batch_idx % 100 == 0:
            print(f"Training batch {batch_idx} ...")
        data = data.to(device)
        targets = targets.to(device)

        # Flatten each image of shape 1 x 28 x 28 to flat vector of size 768
        data = data.reshape(data.shape[0], -1)  #  torch.Size([64, 1, 28, 28])

        # Forward propagation
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()

        # Perform a gradient descent or Adam step
        optimizer.step()

print()


def check_accuracy(loader: DataLoader, model: NeuralNet) -> float:
    num_correct = 0
    num_samples = 0

    # Set the model to evaluation mode. This might impact how the calculations are performed
    model.eval()

    # No need to calculate gradients while evaluation
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):

            X = X.to(device)
            y = y.to(device)
            X = X.reshape(X.shape[0], -1)

            scores = model(X)
            _, predictions = torch.max(scores, 1)

            num_correct = (predictions == y).sum()
            num_samples = predictions.size(0)

            # if batch_idx % 100 == 0:
            #     ic(X.shape)
            #     ic(y.shape)
            #     ic(scores.shape)
            #     ic(predictions.shape)

        acc = float(num_correct) / float(num_samples)
        print(f"Got {num_correct}/{num_samples} correct with accuracy {(acc * 100):4f}")
        model.train()
        return acc


print("Checking accuracy on training data ...")
check_accuracy(train_loader, model)
print("Checking accuracy on test data ...")
check_accuracy(test_loader, model)
