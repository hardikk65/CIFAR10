import torch 
import torch.nn as nn
import torchvision.models
from Trainer.training import training


model = torchvision.models.resnet18()
model.fc = nn.Linear(512,10,bias=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr = 0.001)


training(model,criterion,optimizer)


