import src.utils as utils
import src.architecture as model
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset
from torchsummary import summary
import numpy as np

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
        '--norm', help='Apply data normalization?', default=False, type=bool)
    parser.add_argument(
        '--epochs', help='number of max epochs to train the model', required=True, type=int)
    parser.add_argument(
        '--lr', help='learning rate for SGD optimizer', default=0.01, type=float)
    parser.add_argument(
        '--momentum', help='momentum for SGD optimizer', default=0.0, type=float)
    parser.add_argument(
        '--l2', help='L2 regularizer for SGD', default=0.0, type=float)
    parser.add_argument(
        '--lr_decay_rate', help='lr decay rate in percentage', default=0.0, type=float)
    parser.add_argument(
        '--verbose', help='level of output', default=1, type=int)

    return parser.parse_args()



def main():
    ## run first 'python ./faceScrub download.py' to generate the actors folder... quite time consuming

    args = parse_args()

    # Create datasets with equal number of pos and neg classes
    ### The next line creates new datasets with randomly selected images from the actors/ folder
    if args.new_datasets:
        train_set, valid_set, test_set = utils.create_datasets(args.train,args.valid,args.test,(args.res,args.res),args.verbose-1 if args.verbose>0 else 0)

    if args.norm:
        # Approximate mean and standard deviation using the train and validation datasets
        images_mean, images_std, _ = utils.image_normalization(traindir, validdir, args.verbose)
        trans = transforms.Compose([
            transforms.ToTensor(), # rescale to [0.0, 1.0]
            transforms.Normalize(mean=images_mean, std=images_std)
            # TODO: add data agumentation schemes
        ])
    else:
        trans = transforms.Compose([
            transforms.ToTensor(), # rescale to [0.0, 1.0]
            # TODO: add data agumentation schemes
        ])

    # Load data from folders
    train_dataset = datasets.ImageFolder(
        traindir,
        trans
    )
    valid_dataset = datasets.ImageFolder(
        validdir,
        trans
    )

    # Normalized Data Visualization
    # utils.visualize_normalization(train_dataset, train_dataset_normalized, batch_size = train_size*2)

    # Samples count
    if args.verbose>1:
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
    if args.verbose>1:
        print("Device to use is",device)
    
    # Vanilla CNN aquitecture
    net = model.CNN(args.res).to(device) # Send CNN to GPU if available
    if args.verbose>2:
        print('Model summary:')
        summary(net, input_size=(3, args.res, args.res))

    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2)

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

        if args.lr_decay_rate>0:
            lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr - args.lr_decay_rate * lr
        
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
        if args.verbose:
            print("Epoch %2d -> train_loss: %.5f, train_acc: %.5f | val_loss: %.5f, val_acc: %.5f | lr: %.5f" 
                %(epoch+1,hist['train_loss_epoch'][epoch],hist['train_acc'][epoch], \
                    hist['val_loss_epoch'][epoch],hist['val_acc'][epoch],hist['lr_list'][epoch]))
    
    if args.verbose>1:
        print('Training complete!\n')

    # Generate and save plots
    if args.verbose>2:
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
    if args.verbose>1:
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
    if args.verbose>1:
        print('Accuracy in test set: %3.3f%%' % (hist['test_acc']))

    if args.verbose>2:
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
