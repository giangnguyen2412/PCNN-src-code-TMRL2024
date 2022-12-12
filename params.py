from torchvision import transforms


class RunningParams(object):
    def __init__(self):
        # General
        self.SIMCLR_MODEL = False
        self.IMAGENET_REAL = True
        self.advising_network = True
        self.conv_layer = 4
        # self.conv_layer_size = {4: 512, 3: 256, 2: 128, 1: 64}
        # if self.SIMCLR_MODEL is True:
        self.conv_layer_size = {4: 2048, 3: 1024, 2: 512, 1: 256}
        self.feat_map_size = {4: 49, 3: 196, 2: 784}
        self.CONTINUE_TRAINING = False
        self.TOP1_NN = True  # ----------------------------------------- IMPORTANT PARAM --------
        self.CUB_TRAINING = True  # ----------------------------------------- IMPORTANT PARAM --------
        self.DOGS_TRAINING = False
        self.IMAGENET_TRAINING = False

        # XAI methods
        self.NO_XAI = 'No-XAI'
        self.GradCAM = 'GradCAM'
        self.NNs = 'NNs'
        self.XAI_method = self.NNs
        # TODO: write script to run a set of basic experiments: No-XAI, NNs with conv2,3,4, k1,3,5

        self.USING_SOFTMAX = True  # ----------------------------------------- IMPORTANT PARAM --------
        # TODO: make the difference b/w two run Running params
        # TODO: No need to use softmax anymore at this time bcz we balanced this feature already

        # Training
        if self.CUB_TRAINING is True:
            self.MODEL2_FINETUNING = False
            self.HIGHPERFORMANCE_FEATURE_EXTRACTOR = False
            self.HIGHPERFORMANCE_MODEL1 = False
            self.batch_size = 64  # ----------------------------------------- IMPORTANT PARAM --------

            self.CUB_200WAY = False

            if self.CUB_200WAY is False:
                self.batch_size = 100
        elif self.DOGS_TRAINING:
            self.batch_size = 512  # ----------------------------------------- IMPORTANT PARAM --------

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
        self.k_value = 3  # ----------------------------------------- IMPORTANT PARAM --------
        self.PRECOMPUTED_NN = True
        self.CrossCorrelation = True  # ----------------------------------------- IMPORTANT PARAM --------

        # Data processing
        self.FFCV_loader = False
        self.DDP = False
        self.ALBUM = False

        # Infer
        self.advising_steps = 4
        self.MODEL2_ADVISING = True

        # Visualization
        self.M2_VISUALIZATION = False

        # Unused
        self.BATCH_NORM = True

        # Uncategorized
        self.COSINE_ONLY = False  # ----------------------------------------- IMPORTANT PARAM --------
        self.HUMAN_AI_ANALYSIS = False

