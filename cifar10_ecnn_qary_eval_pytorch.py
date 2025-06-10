import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_qary_pytorch import ECNNModel
from utils_ecnn_qary_pytorch import FLAGS, ce_metric

transform = transforms.ToTensor()
# Automatically download CIFAR-10 if the dataset is unavailable
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=FLAGS.batch_size)

model = ECNNModel(FLAGS.num_models, q=2, depth=20)
model.load_state_dict(torch.load(f'{FLAGS.save_dir}/model.000.pt', map_location='cpu'))
model.eval()

metric = ce_metric(FLAGS.num_models, 2)
acc = 0
count = 0
with torch.no_grad():
    for imgs, labels in testloader:
        labels = labels.unsqueeze(-1).float()
        outputs = model(imgs)
        acc += metric(labels, outputs).sum()
        count += labels.size(0)
print('Accuracy:', (acc / count).item())
