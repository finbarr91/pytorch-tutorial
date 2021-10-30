import torch
import torch.nn as nn

layer = nn.Linear(5,5)
print(layer.weight.data)

print(nn.init.uniform_(layer.weight, a=0.0,b=3))
print(nn.init.normal_(layer.weight,mean=0.0,std=1.0))
print(nn.init.constant_(layer.bias,0))
print(nn.init.zeros_(layer.bias))
print(nn.init.xavier_uniform_(layer.weight,gain =1.0))
