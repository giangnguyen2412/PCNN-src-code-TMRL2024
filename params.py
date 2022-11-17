from torchvision import transforms


class RunningParams(object):
    def __init__(self):
        # General
        self.SIMCLR_MODEL = True
        self.IMAGENET_REAL = True
        self.advising_network = True
        self.conv_layer = 4  # ----------------------------------------- IMPORTANT PARAM --------
        self.conv_layer_size = {4: 512, 3: 256, 2: 128, 1: 64}
        if self.SIMCLR_MODEL is True:
            self.conv_layer_size = {4: 2048, 3: 1024, 2: 512, 1: 256}
        self.feat_map_size = {4: 49, 3: 196, 2: 784}
        self.CONTINUE_TRAINING = False
        self.TOP1_NN = True  # ----------------------------------------- IMPORTANT PARAM --------
        self.CUB_TRAINING = True

        # XAI methods
        self.NO_XAI = 'No-XAI'
        self.GradCAM = 'GradCAM'
        self.NNs = 'NNs'
        self.XAI_method = self.NNs  # ----------------------------------------- IMPORTANT PARAM --------
        # TODO: write script to run a set of basic experiments: No-XAI, NNs with conv2,3,4, k1,3,5

        self.USING_SOFTMAX = True  # ----------------------------------------- IMPORTANT PARAM --------
        # TODO: make the difference b/w two run Running params
        # TODO: No need to use softmax anymore at this time bcz we balanced this feature already

        # Training
        self.batch_size = 32  # ----------------------------------------- IMPORTANT PARAM --------
        self.learning_rate = 0.001
        self.epochs = 100
        self.query_frozen = True  # False = Trainable; True = Freeze? -------------------- IMPORTANT PARAM --------
        self.heatmap_frozen = False  # freeze?
        self.nns_frozen = False  # freeze?
        self.dropout = 0.0

        # Training heatmap
        self.GradCAM_RNlayer = 'layer4'

        # Training NNs
        self.embedding_loss = False
        self.k_value = 5  # ----------------------------------------- IMPORTANT PARAM --------
        self.PRECOMPUTED_NN = True
        self.CrossCorrelation = True  # ----------------------------------------- IMPORTANT PARAM --------

        # Data processing
        self.FFCV_loader = False
        self.DDP = False
        self.ALBUM = False

        # Infer
        self.advising_steps = 1
        self.MODEL2_ADVISING = False

        # Visualization
        self.M2_VISUALIZATION = False

        # Unused
        self.BATCH_NORM = True

        # Uncategorized
        self.COSINE_ONLY = False  # ----------------------------------------- IMPORTANT PARAM --------

