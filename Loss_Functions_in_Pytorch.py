import torch
import torch.nn as nn
import numpy as np
# Using the Mean Squared Error loss function
prediction = torch.randn(4,5)
label = torch.rand(4,5)
mse = nn.MSELoss(reduction= 'mean')

loss = mse(prediction,label)
print(loss)

print(((prediction-label)**2).mean())

# Using the Binary Cross entropy loss function
label = torch.zeros(4,5).random_(0,2)
print(label)
sigmoid = nn.Sigmoid()
bce=nn.BCELoss(reduction='mean')
print(bce(sigmoid(prediction),label))

# Using the Binary Cross entropy with Logits Loss function
bces = nn.BCEWithLogitsLoss(reduction='mean')
print(bces(prediction, label))

x = prediction.numpy()
y = label.numpy()

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = sigmoid(x)
loss_values = []
for i in range(len(y)):
    batch_loss = []
    for j in range(len(y[0])):
        batch_loss.append(-np.log(x[i][j]) if y[i][j]==1 else -np.log(1-x[i][j]))
    loss_values.append(batch_loss)
print(np.mean(loss_values))


