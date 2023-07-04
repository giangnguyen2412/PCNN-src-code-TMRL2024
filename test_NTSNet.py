import os
from torch.autograd import Variable
import torch.utils.data
from torch.nn import DataParallel
from core import model, dataset
from torch import nn
from datasets import Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

Dataset = Dataset()

from torchvision import transforms
from PIL import Image

data_transforms = transforms.Compose([
    transforms.Resize((600, 600), interpolation=Image.BILINEAR),
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


val_data = ImageFolder(
    # ImageNet train folder
    root="/home/giang/Downloads/datasets/CUB/test0", transform=data_transforms
)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=8,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    drop_last=False
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# define model
net = model.attention_net(topN=6)
ckpt = torch.load('/home/giang/Downloads/NTS-Net/model.ckpt')

net.load_state_dict(ckpt['net_state_dict'])

# feature_extractor = nn.Sequential(*list(net.children())[:-1])  # avgpool feature
# print(net)

net.eval()
net = net.cuda()
net = DataParallel(net)
test_correct = 0
total = 0

for i, data in enumerate(tqdm(val_loader)):
    with torch.no_grad():
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        _, concat_logits, _, _, _ = net(img)
        # calculate accuracy
        _, concat_predict = torch.max(concat_logits, 1)
        total += batch_size
        test_correct += torch.sum(concat_predict.data == label.data)

test_acc = float(test_correct) / total
print('test set acc: {:.3f} total sample: {}'.format(test_acc, total))

