import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib

batch_size = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = transforms.Compose([
    transforms.Resize((128, 128)),#150 den 128 e değiştirildi
    transforms.RandomHorizontalFlip(),  # to add variaton in our dataset and it increases the number of unique images
    transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors to be able to work with pytorch
    transforms.Normalize([0.5, 0.5, 0.5],  # 0-1 to [-1,1], formula x-mean /std
                         [0.5, 0.5, 0.5])
])

# dataloader

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


root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

print(classes)


# CNN

class ConvNet(nn.Module):
    def __init__(self, num_classes=15):
        super(ConvNet, self).__init__()

        # in shape = (256?,3,128,128)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(num_features=12)

        self.relu1 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)  # reduce the im size , shape (256?,12,64,64)

        # layer2

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=1)
        # shape(256?, 20, 64, 64)
        self.relu2 = nn.ReLU()
        # shape(256?, 20, 64, 64)

        # layer3

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=3, stride=1, padding=1)
        # shape(256?, 32, 64, 64)
        self.bn3 = nn.BatchNorm2d(num_features=20)
        # shape(256?, 32, 64, 64)
        self.relu3 = nn.ReLU()

        #####
        # layer4
        self.conv4 = nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.relu4 = nn.ReLU()

        # layer5
        self.conv5 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=64)
        self.relu5 = nn.ReLU()
        ######

        self.dropout = nn.Dropout(0.25)
        ##
        self.fc =nn.Linear(in_features=64*32*32, out_features=1000)

        self.fc2 = nn.Linear(in_features=1000, out_features=num_classes)

        self.non_resi_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=20),
            nn.ReLU(),

            ##
            nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            ##

            nn.Flatten(),
            # output = output.view(-1, 32 * 75 * 75)

            nn.Dropout(0.25),
            nn.Linear(in_features=64*32*32, out_features=1000),

            nn.Linear(in_features=1000, out_features=num_classes),

        ).to(device)


    def forward(self, input):
        # output = self.conv1(input)
        # output = self.bn1(output)
        # output = self.relu1(output)
        #
        # output = self.pool(output)
        #
        # output = self.conv2(output)
        # output = self.relu2(output)
        #
        # output = self.conv3(output)
        # output = self.bn3(output)
        # output = self.relu3(output)
        #
        # ##
        # output = self.conv4(output)
        # output = self.bn4(output)
        # output = self.relu4(output)
        #
        # output = self.pool(output)
        #
        # output = self.conv5(output)
        # output = self.bn5(output)
        # output = self.relu5(output)
        # ##
        #
        # output = output.view(-1, 64*32*32)
        #
        # output = self.dropout(output)
        # output = self.fc(output)
        #
        # output = self.fc2(output)
        ###############################################

        output = self.non_resi_model(input)

        ###############################################

        return output



model = ConvNet(num_classes=15).to(device)

optimizer = Adam(model.parameters(),lr=0.001, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()

num_epochs = 30

train_count = len(glob.glob(train_path+'/**/*.jpg'))
test_count = len(glob.glob(test_path+'/**/*.jpg'))
validation_count = len(glob.glob(test_path+'/**/*.jpg'))


#model train

best_accuracy = 0.0

for epoch in range(num_epochs):

    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images= Variable(images.cuda())
            labels = Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(outputs,labels)
        loss.backward()
        optimizer.step()

        train_loss+=loss.cpu().data*images.size(0)
        _,prediction = torch.max(outputs.data, 1)

        train_accuracy+= int(torch.sum(prediction==labels.data))

    train_accuracy = train_accuracy/train_count
    train_loss = train_loss/train_count


    model.eval()

    validation_accuracy = 0.0
    for i, (images, labels) in enumerate(validation_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        validation_accuracy += int(torch.sum(prediction == labels.data))


    validation_accuracy = validation_accuracy / validation_count


    test_accuracy = 0.0
    for i, (images,labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images= Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs=model(images)
        _,prediction= torch.max(outputs.data, 1)
        test_accuracy+= int(torch.sum(prediction==labels.data))

    test_accuracy = test_accuracy/test_count


    print('Epoch '+ str(epoch)+ ' Train Loss: '+ str(int(train_loss)) + ' Train Accuracy: '+str(train_accuracy)+' Validation Accuracy: '+str(validation_accuracy)+ ' Test Accuracy: '+str(test_accuracy))

    if(validation_accuracy> best_accuracy):
        torch.save(model.state_dict(),'best_checkpoint.model')
        best_accuracy=validation_accuracy


print("done")
