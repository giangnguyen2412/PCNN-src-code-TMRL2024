from torchvision import transforms


class RunningParams(object):
    def __init__(self):
        # General
        self.IMAGENET_REAL = True
        self.advising_network = True
        self.MAX_NORM = False
        self.conv_layer = 2
        self.conv_layer_size = {4: 512, 3: 256, 2: 128, 1: 64}

        # XAI methods
        self.NO_XAI = 'No-XAI'
        self.GradCAM = 'GradCAM'
        self.NNs = 'NNs'
        self.XAI_method = self.NNs

        # Training
        self.batch_size = 128
        self.learning_rate = 0.001
        self.epochs = 25
        self.query_frozen = False  # freeze?
        self.heatmap_frozen = False  # freeze?
        self.nns_frozen = False  # freeze?

        # Training heatmap
        self.GradCAM_RNlayer = 'layer4'

        # Training NNs
        self.embedding_loss = True
        self.k_value = 3

        # Data processing
        self.FFCV_loader = False
        self.DDP = False

        # Infer
        self.advising_steps = 1
        self.MODEL2_ADVISING = False

        # Visualization
        self.M2_VISUALIZATION = True




