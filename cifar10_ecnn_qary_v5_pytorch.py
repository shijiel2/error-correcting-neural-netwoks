import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from utils_ecnn_qary_pytorch import FLAGS, lr_schedule, custom_loss, ce_metric
from model_qary_pytorch import ECNNModel

cm = None  # placeholder for code matrix loading

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Automatically download CIFAR-10 if not present
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=FLAGS.batch_size, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=FLAGS.batch_size, shuffle=False)

model = ECNNModel(FLAGS.num_models, q=2, depth=20)
optimizer = optim.Adam(model.parameters(), lr=lr_schedule(0))
criterion = custom_loss(FLAGS.num_models, 2, 1.0, 'hinge')

for epoch in range(FLAGS.epochs):
    model.train()
    for imgs, labels in trainloader:
        labels = labels.unsqueeze(-1).float()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(labels, outputs)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'{FLAGS.save_dir}/model.{epoch:03d}.pt')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_schedule(epoch)

print('Training complete')
