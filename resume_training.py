import torch
import models
import torch.optim as optim

print(__file__)
exit(-1)

model = models.MyCustomResnet18()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load('.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['val_loss']
acc = checkpoint['val_acc']

model.train()

