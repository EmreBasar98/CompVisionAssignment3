import glob
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),  # to add variaton in our dataset and it increases the number of unique images
    transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors to be able to work with pytorch
    transforms.Normalize([0.5, 0.5, 0.5],  # 0-1 to [-1,1], formula x-mean /std
                         [0.5, 0.5, 0.5])
])

num_classes = 15
learning_rate = 1e-3
batch_size = 32
num_epochs = 5

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


model = torchvision.models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False


num_ftrs = model.fc.in_features

model.layer4.requires_grad_(True)
model.fc = model.fc = nn.Linear(num_ftrs, num_classes)


train_path = "dataset/train"
test_path = "dataset/test"
validation_path = "dataset/validation"

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=batch_size, shuffle=True
)

test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=batch_size, shuffle=True
)


validation_loader = DataLoader(
    torchvision.datasets.ImageFolder(validation_path, transform=transformer),
    batch_size=batch_size, shuffle=True
)


train_count = len(glob.glob(train_path+'/**/*.jpg'))
test_count = len(glob.glob(test_path+'/**/*.jpg'))
validation_count = len(glob.glob(test_path+'/**/*.jpg'))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    losses = []

    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


    print(f"Cost at epoch {epoch} is {sum(losses) / len(losses):.5f}")

def check_accuracy(loader, model):

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()

print("Train Accuracy")
check_accuracy(train_loader, model)
print("Validation Accuracy")
check_accuracy(validation_loader, model)
print("Test Accuracy")
check_accuracy(test_loader, model)
