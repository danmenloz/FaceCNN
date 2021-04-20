import src.utils as utils

# Training and test dataset sizes per class
train_size = 100
val_size = 20
test_size = 20
new_datasets = True

# Image resolution
resolution = (50,50)

if __name__ == "__main__":
    # run first 'python ./faceScrub download.py' to generate the actors folder
    
    # Create datasets with equal number of pos and neg classes
    ### The next line creates new datasets with randomly selected images from the actors/ folder
    ### Create new datasets or alternatively read the existing datasets
    if new_datasets:
        train_set, valid_set, test_set = utils.create_datasets(train_size,val_size,test_size,resolution)
    else:
        train_set, valid_set, test_set = utils.load_datasets() # read txt files
    # set = [{face: image: 1: 0:},...]

    # Read images from datasets
    pos_train_imgs, neg_train_imgs = utils.read_dataset(train_set)
    pos_valid_imgs, neg_valid_imgs = utils.read_dataset(valid_set)
    pos_test_imgs, neg_test_imgs = utils.read_dataset(test_set)

    # Samples count
    num_train_pos, num_train_neg = len(pos_train_imgs), len(neg_train_imgs)
    num_valid_pos, num_valid_neg = len(pos_valid_imgs), len(neg_valid_imgs)
    num_test_pos, num_test_neg = len(pos_test_imgs), len(neg_test_imgs)
    print('Faces count:     +train=%d  +valid=%d    +test=%d' % (num_train_pos, num_valid_pos, num_test_pos))
    print('Non-faces count: -train=%d  -valid=%d    -test=%d' % (num_train_neg, num_valid_neg, num_test_neg))
