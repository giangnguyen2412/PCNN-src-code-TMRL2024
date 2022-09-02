import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import Normalize
import cv2
import torch

abm_transform = A.Compose(
    [A.VerticalFlip(p=1),
     A.Resize(height=256, width=256),
     A.CenterCrop(height=224, width=224),
     Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
     ToTensorV2(),
     ],

    additional_targets={'image0': 'image', 'image1': 'image'}
    # , 'image2': 'image', 'image3': 'image', 'image4': 'image',
    #                     'image5': 'image', 'image6': 'image'}
)

image = cv2.imread('/home/giang/tmp/Blue_Jay_0018_63455.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image0 = cv2.imread('/home/giang/tmp/Blue_Jay_0018_63455.jpg')
# image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
transformed = abm_transform(image=image, image0=image0)

pass

