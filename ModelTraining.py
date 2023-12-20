import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim

mnist_trainset = datasets.MNIST(
                                root='./data', 
                                train=True, download=True, 
                                transform=ToTensor())
mnist_testset = datasets.MNIST(
                            root='./data', 
                            train=False, download=True, 
                            transform=ToTensor())

train_dataloader = DataLoader(
                            mnist_trainset, 
                            batch_size=20, 
                            shuffle=True, 
                            num_workers=4)
test_dataloader = DataLoader(
                            mnist_testset, 
                            batch_size=20, 
                            shuffle=False, 
                            num_workers=4)

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = x.view(x.size(0), -1) #flattening the layer      
        output = self.out(x)
        return output
    
device = torch.device("cuda:0")
cnn_model = CNN().to(device)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(cnn_model.parameters(), lr=0.005)

def calculate_accuracy(
                        model_output, 
                        target):
    
    predictions = torch.max(model_output, 1)[1].data.squeeze()
    
    accuracy = (predictions == target).sum().item()/float(target.size(0))
    return accuracy

#In argument train_test, use "train" for training and "test" for test or evaluation
def trainTest_model(
                cnn_model, 
                loss_function, 
                optimizer, 
                train_dataloader,test_dataloader,train_test):
    if(train_test=="train"):
        num_epoch = 30
        cnn_model.train()
        
        for epoch in range(num_epoch):
            
            epoch_loss = 0
            epoch_accuracy = 0
            i = 0
            for i, (images, labels) in enumerate(train_dataloader):

                images, labels = images.to(device), labels.to(device)
                
                output = cnn_model(images)
                
                loss = loss_function(output, 
                                    labels)
                
                optimizer.zero_grad() #releasing the cache
                
                loss.backward()
                epoch_loss += loss.item()
                
                epoch_accuracy += calculate_accuracy(output, labels)
                
                optimizer.step()               
            
            print(f"Epoch: {epoch} - Loss: {epoch_loss} - Accuracy: {epoch_accuracy/(i+1)}")
    
    elif(train_test=="test"):
        cnn_model.eval()
        accuracy = 0
        i = 0
        for i, (images, labels) in enumerate(test_dataloader):

            images, labels = images.to(device), labels.to(device)
            output = cnn_model(images)
            accuracy += calculate_accuracy(output, labels)

        print(f"Test Accuracy: {accuracy/(i+1)}")

trainTest_model(
                cnn_model, 
                loss_function, 
                optimizer, 
                train_dataloader,test_dataloader,"train")

test_model(
            cnn_model, 
            loss_function, 
            optimizer, 
            train_dataloader,test_dataloader,"test")

#save the model state
torch.save(cnn_model.state_dict(), './cnn_model.pth')