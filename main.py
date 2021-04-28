import src.utils as utils
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset

# datasets paths
traindir = './data/train'
validdir = './data/valid'
testdir = './data/test'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Face classification using a CNN.')
    
    parser.add_argument(
        '--new_datasets', help='create new train, valid and test datasets?', default=False, type=bool)
    parser.add_argument(
        '--train', help='number of images per class in training dataset', default=100, type=int)
    parser.add_argument(
        '--valid', help='number of images per class in validation dataset', default=20, type=int)
    parser.add_argument(
        '--test', help='number of images per class in testing dataset', default=20, type=int)
    parser.add_argument(
        '--res', help='image resolution when creating new datasets', default=50, type=int)
    parser.add_argument(
        '--workers', help='number of subprocesses to use for data loading', default=0, type=int)
    parser.add_argument(
        '--suffix', help='suffix for the resulting files', default='', type=str)
    parser.add_argument(
        '--batch_size', help='images batch size for training and validation', default=32, type=int)
    parser.add_argument(
        '--epochs', help='number of max epochs to train the model', required=True, type=int)
    parser.add_argument(
        '--lr', help='learning rate for SGD optimizer', default=0.0025, type=float)
    parser.add_argument(
        '--momentum', help='momentum for SGD optimizer', default=0.9, type=float)

    return parser.parse_args()



def main():
    ## run first 'python ./faceScrub download.py' to generate the actors folder... quite time consuming

    args = parse_args()

    # Create datasets with equal number of pos and neg classes
    ### The next line creates new datasets with randomly selected images from the actors/ folder
    if args.new_datasets:
        train_set, valid_set, test_set = utils.create_datasets(args.train,args.valid,args.test,(args.res,args.res))

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
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device to use is",device)
    
    # Convolutional Layers
    k1 = 5 # Kernel size cnn 1
    k2 = 13 # Kernel size cnn 2
    cnn1 = nn.Conv2d(3, 16, k1) # Stride = 1, Padding = 0
    cnn2 = nn.Conv2d(16, 32, k2) # Stride = 1, Padding = 0
    out_size = int(((args.res-k1+1)/2-k2+1)/2) # Size after convolutional layers

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
        # List for validation results
        'val_loss_epoch': [],     # Stores train Loss per Epoch
        'val_acc': [],            # Stores train accuracy per mini batch
        # List learning rate
        'lr_list': [],
        # Test accuracy
        'test_acc': None
    }

    # List to store prediction labels
    train_predict = []
    val_predict = []

    # Training and validation for a fixed number of epochs
    for epoch in range(args.epochs):
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
            train_predict += (predicted).tolist()
        
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
                val_predict += (predicted).tolist()
        
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
    
    print('Training complete!\n')

    # Generate and save plots
    utils.plot_loss(hist['train_loss'], hist['train_loss_epoch'],filename='./images/loss_train_' + args.suffix + '.png',scatter=True)
    utils.plot_loss(hist['train_loss_epoch'], hist['val_loss_epoch'],filename='./images/loss_' + args.suffix + '.png')
    utils.plot_accuracy(hist['train_acc'], hist['val_acc'],filename='./images/accuracy_' + args.suffix + '.png')


    ## Measure performance using the Test dataset
    # Load data from folder
    test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose([
            transforms.ToTensor() # rescale to [0.0, 1.0]
            # TODO: add normalization
            # TODO: add data agumentation schemes
        ])
    )
    # Samples count
    print('Test samples: \t%d' %(test_dataset.__len__()))
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    # Test data evaluation
    true_labels = test_loader.dataset.targets
    predict_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for (images,labels) in test_loader:
                # Copy to GPU if available
                images, labels = images.to(device), labels.to(device)
                output = net(images)
                # Accuracy
                _, predicted = torch.max(output.data, 1)
                total += len(labels)
                predict_labels += predicted.tolist()
                correct += (predicted == labels).sum().item()
    hist['test_acc'] = 100 * correct / total
    print('Accuracy in test set: %3.3f%%' % (hist['test_acc']))

    # Add test accuracy and save history file
    utils.save_history(hist, './images/CNN_history_' + args.suffix + '.csv') 

    # Generate report
    utils.build_report(predict_labels,true_labels)

    # Plot and save confusion matrix
    utils.plot_confusion_matrix(predict_labels,true_labels,filename='./images/conf_mtx_' + args.suffix + '.png')

    # Plot and save some mispredicted images
    utils.plot_mispredictions(predict_labels,true_labels,test_dataset,filename='./images/mispredicted_' + args.suffix + '.png')


if __name__ == "__main__":
    main()
