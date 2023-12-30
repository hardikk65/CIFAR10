import torch 
import torch.nn as nn
from Dataset.dataset import make_batches

_,test_loader = make_batches()

model = torch.load('Network/model.pth')

def evaluation(model,test_loader=test_loader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        test_loss = 0.0
        test_accuracy = 0
        for i,(images, labels) in enumerate(test_loader):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')