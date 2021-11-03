import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Load the dataset using Pandas
data= pd.read_csv(r'https://raw.githubusercontent.com/fawazsammani/The-Complete-Neural-Networks-Bootcamp-Theory-Applications/master/diabetes.csv')
print(data.head())

# For x: Extract out the dataset from all the rows (all samples) and all columns except last column (all features).
# For y: Extract out the last column (which is the label)
# Convert both to numpy using the values method.

x = data.iloc[:,0:-1].to_numpy()
y_string = list(data.iloc[:,-1])
print(x)
print(x.dtype)
print(y_string)

# Our neural network only understand numbers, So convert the string to labels
y_int = []
for s in y_string:
    if s == 'positive':
        y_int.append(1)
    else:
        y_int.append(0)

# Now convert to an array
y = np.array(y_int, dtype= 'float64')

# Feature Normalization. All features should have the same range of values (-1,1)
sc = StandardScaler()
x = sc.fit_transform(x)
print(x)

# Now we convert the arrays to pytorch tensor
x = torch.tensor(x)
y = torch.tensor(y).unsqueeze(1) # to add the dimensionality of 1 because our binary cross entropy loss requires a row and column input
print(x.shape)
print(y.shape)

class Dataset(Dataset):

    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.x)

dataset = Dataset(x,y)

print(len(dataset))

# Load the data to your dataloader for batch processing and shuffling
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32,shuffle = True)
print(train_loader)

# Let's have a look at the data loader
print('There is {} batches in the dataset'.format(len(train_loader)))
for (x,y) in train_loader:
    print('For one iteration (batch),there is:')
    print('Data: {}'.format(x.shape))
    print('Labels: {}'.format(y.shape))
    break

class Model(nn.Module):
    def __init__(self,input_features,output_features):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(input_features,5)
        self.fc2 = nn.Linear(5,4)
        self.fc3 = nn.Linear(4,3)
        self.fc4 = nn.Linear(3,output_features)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
# Whenever you are using a binary cross entropy loss function, the activation function should be a sigmoid to scale between 0 and 1


    def forward(self,x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

# Create the network (an object of the Net Class)
net = Model(7,1)

# In Binary Cross Entropy: the input and output should have the same shape
# Size_average =True --> the losses are averaged over observations for each minibatch

criterion = torch.nn.BCELoss(size_average=True)
# SGD with momentum with a learning rate of 0.1
optimizer = torch.optim.SGD(net.parameters(),lr = 0.1,momentum= 0.9)

# Training the Neural Network.
epochs = 200
# In pytorch you need to train using a for loop
for epoch in range(200):
    for inputs,labels in train_loader:
        inputs = inputs.float()
        labels = labels.float()
        # Forward prop
        outputs = net(inputs) # the same as net.forward(). The forward function is called behind the scene.

        # Loss Calculation
        loss = criterion(outputs,labels)

        #Clear the gradient buffer( w<--w-lr*gradient)
        optimizer.zero_grad()

        # Backprop
        loss.backward()
        #update weights
        optimizer.step()

    # Accuracy Calculation
    output = (outputs>0.5).float()
    #(output==labels).sum()/output.shape[0]
    accuracy=(output == labels).float().mean()
    # Print Statistics
    print('Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}'.format(epoch+1,epochs, loss,accuracy))








