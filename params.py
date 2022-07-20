from torchvision import transforms


class RunningParams(object):
    def __init__(self):
        self.batch_size = 256
        self.GradCAM_RNlayer = 'layer4'
        self.XAI_method = 'No-XAI'
        self.learning_rate = 0.001
        self.epochs = 25
        self.confidence_loss = False
        self.top1 = False
        self.fine_tune = False
        self.advising_network = True

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }