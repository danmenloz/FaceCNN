import torch.nn as nn
from torchsummary import summary

# Define model architectures here

class CNN(nn.Module):
    def __init__(self,res):
        # Neural Network layers
        super(CNN, self).__init__()
        cnn, batch, fc = self.init_layers(res)
        self.layer1 = nn.Sequential(
            cnn[0],
            batch[0],
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            cnn[1],
            batch[1],
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(fc[0],nn.ReLU())
        self.layer4 = nn.Sequential(fc[1],nn.ReLU())
        self.layer5 = nn.Sequential(fc[2])
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(-1, 32 * self.out_size * self.out_size) # flatten
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out
    
    def init_layers(self, res):
        # Convolutional Layers
        k1 = 5 # Kernel size cnn 1
        k2 = 13 # Kernel size cnn 2
        cnn1 = nn.Conv2d(3, 16, k1) # Stride = 1, Padding = 0
        cnn2 = nn.Conv2d(16, 32, k2) # Stride = 1, Padding = 0
        self.out_size = int(((res-k1+1)/2-k2+1)/2) # Size after convolutional layers

        # Dense Layers
        fc1 = nn.Linear(32 * self.out_size * self.out_size, 256)
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
        
        return (cnn1,cnn2), (batch1,batch2), (fc1,fc2,fc3)

if __name__ == "__main__":
    # example of model summary using input image of 50x50
    model = CNN(50) 
    # print(model) # doesn't print nice-typed info
    summary(model, input_size=(3, 50, 50))
