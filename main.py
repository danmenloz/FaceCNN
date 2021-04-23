import src.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Directories
traindir = './data/train'
validdir = './data/valid'
testdir = './data/test'

# Training and test dataset sizes per class
train_size = 500
val_size = 100
test_size = 100

# Options
new_datasets = False
resolution = (50,50)

# Parameters
batch_size = 32
workers = 0 # subprocesses to use for data loading
max_epochs = 20


if __name__ == "__main__":
    # run first 'python ./faceScrub download.py' to generate the actors folder
    
    # Create datasets with equal number of pos and neg classes
    ### The next line creates new datasets with randomly selected images from the actors/ folder
    if new_datasets:
        train_set, valid_set, test_set = utils.create_datasets(train_size,val_size,test_size,resolution)

    # Load data from folders
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.ToTensor() # rescale to [0.0, 1.0]
            # TODO: add normalization
            # TODO: add data agumentation schemes
        ])
    )
    valid_dataset = datasets.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.ToTensor() # rescale to [0.0, 1.0]
            # TODO: add normalization
        ])
    )

    # Samples count
    print('Training samples: \t%d' %(train_dataset.__len__()))
    print('Validation samples: \t%d' %(valid_dataset.__len__()))

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers)
    
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device to use is",device)
    
    # Convolutional Layers
    k1 = 5 # Kernel size cnn 1
    k2 = 13 # Kernel size cnn 2
    cnn1 = nn.Conv2d(3, 16, k1) # Stride = 1, Padding = 0
    cnn2 = nn.Conv2d(16, 32, k2) # Stride = 1, Padding = 0
    out_size = int(((resolution[0]-k1+1)/2-k2+1)/2) # Size after convolutional layers

    # Dense Layers
    fc1 = nn.Linear(32 * out_size * out_size, 256)
    fc2 = nn.Linear(256, 32)
    fc3 = nn.Linear(32, 2)

    # Xavier initialization
    nn.init.xavier_uniform_(cnn1.weight) # Xavier initialization CNN1
    nn.init.xavier_uniform_(cnn2.weight) # Xavier initialization CNN2
    nn.init.xavier_uniform_(fc1.weight) # Xavier initialization FC1
    nn.init.xavier_uniform_(fc2.weight) # Xavier initialization FC2
    nn.init.xavier_uniform_(fc3.weight) # Xavier initialization FC3

    # Batch Normalization
    batch1 = nn.BatchNorm2d(16)
    batch2 = nn.BatchNorm2d(32)
    
    # Vanilla CNN archirecture
    class CNN(nn.Module):
        def __init__(self):
            # Neural Network layers
            super(CNN, self).__init__()
            self.layer1 = nn.Sequential(
                cnn1,
                batch1,
                nn.ReLU(),
                nn.MaxPool2d(2, 2))
            self.layer2 = nn.Sequential(
                cnn2,
                batch2,
                nn.ReLU(),
                nn.MaxPool2d(2, 2))
            self.layer3 = nn.Sequential(fc1,nn.ReLU())
            self.layer4 = nn.Sequential(fc2,nn.ReLU())
            self.layer5 = nn.Sequential(fc3)
        
        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(-1, 32 * out_size * out_size)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            return out

    net = CNN().to(device) # Send CNN to GPU if available

    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0025, momentum=0.9)

    # Save training and validation history in a dictionary
    hist = {
        # Lists for train results
        'train_loss': [],         # Stores train loss at each iteration
        'train_loss_epoch': [],   # Stores train Loss per Epoch
        'train_acc': [],          # Stores train accuracy per mini batch
        'train_predict': [],      # Stores prediction labels
        # List for validation results
        'val_loss_epoch': [],     # Stores train Loss per Epoch
        'val_acc': [],            # Stores train accuracy per mini batch
        'val_predict': [],        # Stores prediction labels
        # List learning rate
        'lr_list': []
    }

    # Training and validation for a fixed number of epochs
    for epoch in range(max_epochs):
        loss_batch = 0.0
        correct = 0
        total = 0
        
        # Trainining step
        for i, (images,labels) in enumerate(train_loader, 0):
            # Copy to GPU if available 
            images, labels = images.to(device), labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # Fiting
            output = net.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            # Storing loss
            hist['train_loss'].append(loss.item())
            loss_batch += loss.item()
        
            # Storing accuracy
            _, predicted = torch.max(output.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()
            hist['train_predict'] += (predicted).tolist()
        
        # Train loss and accuracy per epoch 
        hist['train_loss_epoch'].append(loss_batch/(i+1))
        hist['train_acc'].append(correct/total)

        # Reset variables for validation
        loss_batch = 0.0
        total = 0
        correct = 0
        
        # Validation step
        for j, (images,labels) in enumerate(valid_loader):
            with torch.no_grad():
                # Copy to GPU if available
                images, labels = images.to(device), labels.to(device)

                output = net(images)
                
                # Storing Loss
                loss = criterion(output, labels)
                loss_batch += loss.item()

                # Accuracy
                _, predicted = torch.max(output.data, 1)
                total += len(labels)
                correct += (predicted == labels).sum().item()
                hist['val_predict'] += (predicted).tolist()
        
        # Validation loss and accuracy per epoch
        hist['val_loss_epoch'].append(loss_batch/(j+1))
        hist['val_acc'].append(correct/total)
        loss_batch = 0.0
        total = 0
        correct = 0
        hist['lr_list'].append(optimizer.param_groups[0]['lr'])
        
        # Print results
        print("Epoch %2d -> train_loss: %.5f, train_acc: %.5f | val_loss: %.5f, val_acc: %.5f | lr: %.5f" 
            %(epoch+1,hist['train_loss_epoch'][epoch],hist['train_acc'][epoch], \
                hist['val_loss_epoch'][epoch],hist['val_acc'][epoch],hist['lr_list'][epoch]))
    
    print('Training complete!')

    utils.save_history(hist, 'CNN_history.csv') 
    read_hist = utils.read_history('CNN_history.csv') 

    # Generate and save plots
    utils.plot_loss(read_hist['train_loss'], read_hist['train_loss_epoch'], scatter=True)
    utils.plot_loss(read_hist['train_loss_epoch'], read_hist['val_loss_epoch'])
    utils.plot_accuracy(read_hist['train_acc'], read_hist['val_acc'])