import torch 
import torch.nn as nn
from Dataset.dataset import make_batches

train_loader,_ = make_batches()

def training(model,criterion,optimizer,train_loader=train_loader,epochs=10):
    training_loss = 0.0
    training_accuracy = 0
    for epoch in range(epochs):
        for i,(images,labels) in enumerate(train_loader):
            output = model(images)
            loss = criterion(output,labels)

            # training_loss += loss.item()
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            _, predicted = torch.max(output.data, 1)

            training_accuracy += (predicted == labels).sum().item()
            if ((i+1)%100 == 0):
                print(f'epoch:{epoch + 1},loss:{loss.item():.4f}')
        print(f'accuracy:{training_accuracy*100/50000}')
        training_accuracy = 0    

    torch.save('Network/model.pth',model.state_dict())

    return    
