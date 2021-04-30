import csv
import random
import shutil
import os
import numpy as np
from pathlib import Path
from PIL import Image
from random import randrange
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

path_actors_faces = './actors/faces/'
path_actors_images = './actors/images/'
path_faces_file = './actors/faces.txt'
path_save_features = './data/'
path_save_dir = './images/'


def random_crop(image_path, target_size):
        image = Image.open(image_path)

        if Path(image_path).suffix == '.png':
            image = image.convert('RGB')

        img_size = image.size
        x_max = img_size[0] - target_size[0]
        y_max = img_size[1] - target_size[1]

        random_x = randrange(0, x_max//2 + 1) * 2
        random_y = randrange(0, y_max//2 + 1) * 2

        area = (random_x, random_y, random_x + target_size[0], random_y + target_size[1])
        c_img = image.crop(area)

        return c_img



def create_directory(dataset, dataset_dir, resolution, verbose=1):
    if verbose:
        print('Creating {} directory...'.format(dataset_dir))
    Path(dataset_dir).mkdir(parents=True)
    (Path(dataset_dir) / '0').mkdir()
    (Path(dataset_dir) / '1').mkdir()
    for face in dataset:
        img1 = Image.open(face['face'])
        if verbose==2:
            print('Resizing {} '.format( str(Path(face['face']).name)) )
        if Path(face['face']).suffix != '.jpg' or Path(face['face']).suffix != '.jpeg':
            img1 = img1.convert('RGB')
        img1 = img1.resize(resolution)
        save_path = str(Path(dataset_dir) / '1' / Path(face['face']).stem ) +  '.png'
        img1.save( save_path, 'PNG')
        face['1'] = save_path
        img0 = random_crop(face['image'],resolution)
        save_path = str(Path(dataset_dir) / '0'/ Path(face['image']).stem ) +  '.png'
        img0.save( save_path, 'PNG')
        face['0'] = save_path



def create_info_file(dataset, dataset_dir, file_name, verbose=1):
    if verbose:
        print('Creating {} file...'.format(file_name))
    with open( str(Path(dataset_dir) / file_name), 'w', newline='') as file:
        fieldnames = dataset[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for face in dataset:
            writer.writerow(face)
    


def create_datasets(train_size, val_size, test_size, resolution, verbose=1):
    # Read download directory
    if verbose:
        print('\nReading actors folder...')

    faces = [] # this list contains all the download information

    # check if faces.txt file exists
    if Path(path_faces_file).is_file():
        with open(path_faces_file, newline='') as faces_file:
            faces_reader = csv.DictReader(faces_file, delimiter='\t')
            for face in faces_reader:
                faces.append(face)
    else:
        # it takes some minutes to scan the whole directory!
        for actor_entry in tqdm(Path(path_actors_faces).iterdir(), desc ="Reading faces"):
            if actor_entry.is_dir(): # read only directories
                for face_entry in actor_entry.iterdir():
                    faces.append( {'name':actor_entry.name, 'face':str(face_entry)} ) # add info to faces list

        for actor_entry in tqdm(Path(path_actors_images).iterdir(), desc ="Reading images"):
            if actor_entry.is_dir(): # read only directories
                for image_entry in actor_entry.iterdir():
                    # Search for dictionary and add key
                    for face in faces:
                        if face['face'].find(image_entry.stem)>=0: # stem -> name without sufix
                            face['image'] = str(image_entry)
        
        create_info_file(faces, str(Path(path_faces_file).parent), Path(path_faces_file).name) # create faces.txt file

    # Shuffle list
    random.shuffle(faces)

    # clear data dir
    if Path('data').is_dir():
        shutil.rmtree('data')

    if verbose:
        print('\nCreating test set...')
    # Create test set list
    test_set = []
    actors = [] # list to keep track which actors have been used
    for idx, face in enumerate(faces):
        # Examine image and dicard if not RGB, e.g, type L (b/w)
        img = Image.open(face['image'])
        if img.mode == 'RGB':
            test_set.append(face)
            actors.append(face['name'])
        if len(test_set) == test_size:
            break
    actors = list(set(actors)) # delete duplicates
    create_directory(test_set, 'data/test/', resolution, verbose=verbose)
    create_info_file(test_set, 'data/test/', 'test.txt',verbose=verbose)

    if verbose:
        print('\nCreating validation set...')
    # Create validation set list, make sure that no actor from test set is here
    validation_set = []
    for idx, face in enumerate(faces[idx+1:], start=idx+1): # continue reading from idx
        img = Image.open(face['image'])
        if face['name'] not in actors and img.mode == 'RGB':
            validation_set.append(face)
        if len(validation_set) == val_size:
            break
    create_directory(validation_set, 'data/valid/', resolution, verbose=verbose)
    create_info_file(validation_set, 'data/valid/', 'valid.txt',verbose=verbose)

    if verbose:
        print('\nCreating training set...')
    # Create training set list, make sure that no actor from test set is here
    training_set = []
    for face in faces[idx+1:]:
        img = Image.open(face['image'])
        if face['name'] not in actors and img.mode == 'RGB':
            training_set.append(face)
        if len(training_set) == train_size:
            break
    create_directory(training_set, 'data/train/', resolution, verbose=verbose)
    create_info_file(training_set, 'data/train/', 'train.txt',verbose=verbose)

    if verbose:
        print('\nDatasets created sucessfully!\n')

    return training_set, validation_set, test_set



def image_normalization(traindir, validdir, verbose=1):
    # Load data from folders
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.ToTensor() # rescale to [0.0, 1.0]
    )
    valid_dataset = datasets.ImageFolder(
        validdir, 
        transforms.ToTensor() # rescale to [0.0, 1.0]
    )

    # Combine train and valid datasets
    dataset = ConcatDataset([train_dataset, valid_dataset])
    if verbose>1:
        print('Samples for normalization: \t%d' %(dataset.__len__()))

    # Create data loaders
    # The larger the batch_size the better the mean and std approximation
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=50, shuffle=True,
        num_workers=0)

    imgs_mean = []
    imgs_std0 = []
    imgs_std1 = []

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, (images,labels) in enumerate(dataloader, 0):
        # Copy to GPU if available
        images, labels = images.to(device), labels.to(device)

        # shape (batch_size, 3, height, width)
        images = images.numpy()

        batch_mean = np.mean(images, axis=(0, 2, 3))
        batch_std0 = np.std(images, axis=(0, 2, 3))
        batch_std1 = np.std(images, axis=(0, 2, 3), ddof=1) # not sure what this std is for :(

        imgs_mean.append(batch_mean)
        imgs_std0.append(batch_std0)
        imgs_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    imgs_mean = np.array(imgs_mean).mean(axis=0)
    imgs_std0 = np.array(imgs_std0).mean(axis=0)
    imgs_std1 = np.array(imgs_std1).mean(axis=0)

    if verbose>1:
        print("Dataset Normalization: mean={}, std1={}, std2{}".format(imgs_mean,imgs_std0,imgs_std1))

    return imgs_mean, imgs_std0, imgs_std1



def visualize_normalization(dir, mean, std, batch_size = 50, filename='./images/normalized.png', show = False):
    # Create datasets
    original_data = datasets.ImageFolder(
        dir,
        transforms.ToTensor(), # rescale to [0.0, 1.0]
    )
    normalized_data = datasets.ImageFolder(
        dir,
        transforms.Compose([
            transforms.ToTensor(), # rescale to [0.0, 1.0]
            transforms.Normalize(mean=mean, std=std)
        ])
    )

    # Create data loaders
    original_DL = torch.utils.data.DataLoader(original_data, batch_size, shuffle=False, num_workers=0)
    normalized_DL = torch.utils.data.DataLoader(normalized_data, batch_size, shuffle=False, num_workers=0)

    for i, (original,normalized) in enumerate(zip(original_DL, normalized_DL)):
        original_imgs, original_labels = original #unpack imgs and lbls
        normalized_imgs, _ = normalized #unpack imgs and lbls
        if original_labels[0] == 1: # This is a batch of faces
            break

    original_imgs = original_imgs.numpy()
    normalized_imgs = normalized_imgs.numpy()

    fig, axesarr = plt.subplots(2,4)
    for row, ax in enumerate(axesarr[:, 0], start=1): #
        if (row == 1):
            ax.set_title("Original Images")
        else:
            ax.set_title("Normalized Images")

    axesarr[0, 0].imshow(original_imgs[0].transpose(1, 2, 0))
    axesarr[0, 1].imshow(original_imgs[2].transpose(1, 2, 0))
    axesarr[0, 2].imshow(original_imgs[-2].transpose(1, 2, 0))
    axesarr[0, 3].imshow(original_imgs[-1].transpose(1, 2, 0))
    axesarr[1, 0].imshow(normalized_imgs[0].transpose(1, 2, 0))
    axesarr[1, 1].imshow(normalized_imgs[2].transpose(1, 2, 0))
    axesarr[1, 2].imshow(normalized_imgs[-2].transpose(1, 2, 0))
    axesarr[1, 3].imshow(normalized_imgs[-1].transpose(1, 2, 0))

    if show:
        plt.show()

    plt.savefig(filename)
    plt.close()



def save_history(input_dict, filename = 'history.csv'):
    # Every key to data frame column
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in input_dict.items() ]))
    df.index += 1 # start index at one
    df.to_csv(filename) # write data frame to csv file



def read_history(filename='history.csv'):
    df = pd.read_csv(filename, index_col=0) # read dataframe from csv file
    output_dict = df.to_dict('list') # all columns in df to list within a dictionary
    for key in list(output_dict.keys()): # for all keys 
        output_dict[key][:] = [x for x in output_dict[key] if x == x] # remove nan values in lists
    return output_dict



def plot_loss(loss1, loss2, filename='./images/loss.png',scatter=False):
    if scatter:
        x_loss = np.arange(0,len(loss1))
        x_loss_epoch = np.linspace(0,len(loss1),len(loss2)).astype(int)
        plt.plot(x_loss, loss1, color='red', marker='o', linestyle='none', markersize=2)
        plt.plot(x_loss_epoch, loss2, linewidth=5)
        plt.title("Loss Train")
        plt.ylabel("Loss")
        plt.xlabel("Iteration")
    else:
        plt.plot(loss1, linewidth=5)
        plt.plot(loss2, linewidth=5)
        plt.title("Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train","Validation"])
    plt.ylim([0,1.0])
    plt.savefig(filename)
    plt.close()



def plot_accuracy(train_acc, val_acc, filename='./images/accuracy.png'):
    plt.plot(train_acc, linewidth=5)
    plt.plot(val_acc, linewidth=5)
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train","Validation"], loc="lower right")
    plt.ylim([0.75,1])
    plt.savefig(filename)
    plt.close()
    



def plot_confusion_matrix(score, labels, filename='./images/conf_mtx.png'):
    confM = metrics.confusion_matrix(labels, score)
    fig, ax = plt.subplots(figsize=(7,4)) 
    axis_labels = ["non-face","face"]
    sns.heatmap(confM, annot=True,cmap="Greens",fmt='g', xticklabels=axis_labels, yticklabels=axis_labels,ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(filename)
    plt.close()



def build_report(score, labels):
    print('\nClassification Report')
    print(metrics.classification_report(labels, score))



def plot_mispredictions(score,labels,dataset,num_imgs=np.inf,filename='./images/mispredicted.png'):
    correct = np.array([s==l for (s,l) in zip(score,labels)])
    mispredict = np.argwhere(correct==False).ravel()
    np.random.shuffle(mispredict)
    
    num_imgs = min(num_imgs,len(mispredict)) # cannot plot more images than the mispredicted num
    if num_imgs < 1:
        print('Not enough mispredictions to plot!')
        return

    # Plot images on one figure
    fig,axes = plt.subplots(1, num_imgs,figsize=(3*num_imgs,3*num_imgs))

    for idx, ax in enumerate(axes.ravel()):
        img_idx = mispredict[idx]
        path,label = dataset.imgs[img_idx]
        img = Image.open(path)
        title = 'Non-face' if score[img_idx]==0 else 'Face'
        ax.imshow(img)
        ax.title.set_text(title)
        ax.set_axis_off()
    
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    pass