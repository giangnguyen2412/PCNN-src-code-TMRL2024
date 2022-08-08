from torchvision import transforms


class RunningParams(object):
    def __init__(self):
        # XAI methods
        self.NO_XAI = 'No-XAI'
        self.GradCAM = 'GradCAM'
        self.NNs = 'NNs'
        self.k_value = 5
        self.embedding_loss = True

        self.batch_size = 256
        self.GradCAM_RNlayer = 'layer4'
        self.learning_rate = 0.001
        self.epochs = 25
        self.advising_network = True
        self.query_frozen = False  # freeze?
        self.heatmap_frozen = False  # freeze?
        self.nns_frozen = False  # freeze?

        self.FFCV_loader = False
        self.DDP = False
        self.MODEL2_ADVISING = True
        self.advising_steps = 3
        self.BATCH_NORM = True

        self.XAI_method = self.NNs



