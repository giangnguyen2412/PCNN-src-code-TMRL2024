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
        # self.feat_map_size = {4: 49, 3: 196, 2: 784}
        self.feat_map_size = {4: 196, 3: 196, 2: 784}
        self.CONTINUE_TRAINING = False
        self.TOP1_NN = True  # ----------------------------------------- IMPORTANT PARAM --------
        self.CUB_TRAINING = True  # ----------------------------------------- IMPORTANT PARAM --------
        self.DOGS_TRAINING = False
        self.IMAGENET_TRAINING = False

        if self.DOGS_TRAINING is True:
            self.conv_layer_size = {4: 512}

        # XAI methods
        self.NO_XAI = 'No-XAI'
        self.GradCAM = 'GradCAM'
        self.NNs = 'NNs'
        self.XAI_method = self.NNs
        # TODO: write script to run a set of basic experiments: No-XAI, NNs with conv2,3,4, k1,3,5

        self.BOTTLENECK = False
        if self.BOTTLENECK is True:
            self.conv_layer_size = {4: 512}

        # Training
        if self.CUB_TRAINING:
            self.MODEL2_FINETUNING = True
            self.HIGHPERFORMANCE_FEATURE_EXTRACTOR = True
            self.HIGHPERFORMANCE_MODEL1 = True

            self.CUB_200WAY = False

            if self.CUB_200WAY is True:
                self.batch_size = 64  # ----------------------------------------- IMPORTANT PARAM --------
                self.epochs = 50
            else:
                self.batch_size = 100
                self.epochs = 300

        elif self.DOGS_TRAINING:
            self.batch_size = 8  # ----------------------------------------- IMPORTANT PARAM --------
            self.epochs = 100

            self.WEIGHTED_LOSS_DOGS = True

        self.learning_rate = 1e-3
        self.query_frozen = True  # False = Trainable; True = Freeze? -------------------- IMPORTANT PARAM --------
        self.heatmap_frozen = False  # freeze?
        self.nns_frozen = False  # freeze?
        self.dropout = 0.0

        if self.CUB_TRAINING is True and self.MODEL2_FINETUNING is True:
            self.HIGHPERFORMANCE_FEATURE_EXTRACTOR = True
            self.HIGHPERFORMANCE_MODEL1 = True
            self.CONTINUE_TRAINING = True

            self.batch_size = 96
            self.epochs = 100
            self.learning_rate = 1e-3

        self.USING_SOFTMAX = False  # ----------------------------------------- IMPORTANT PARAM --------
        self.UNBALANCED_TRAINING = True
        self.EXP_TOKEN = True

        self.THREE_BRANCH = False

        self.MODEL_ENSEMBLE = False

        # Training heatmap
        self.GradCAM_RNlayer = 'layer4'

        # Training NNs
        self.embedding_loss = False
        if self.DOGS_TRAINING is True:
            self.k_value = 1
        elif self.CUB_TRAINING is True:
            self.k_value = 1  # ----------------------------------------- IMPORTANT PARAM --------
        self.PRECOMPUTED_NN = True
        self.CrossCorrelation = True  # ----------------------------------------- IMPORTANT PARAM --------

        # Data processing
        self.FFCV_loader = False
        self.DDP = False
        self.ALBUM = False

        # Infer
        self.MODEL2_ADVISING = False
        self.advising_steps = 5

        # Visualization
        self.M2_VISUALIZATION = False

        # Unused
        self.BATCH_NORM = True

        # Uncategorized
        self.COSINE_ONLY = False  # ----------------------------------------- IMPORTANT PARAM --------
        self.HUMAN_AI_ANALYSIS = False

        # Class embeddings
        self.USING_CLASS_EMBEDDING = False
        if self.USING_CLASS_EMBEDDING is True:
            self.action = 'Using'
            self.FIXED = False
            self.LEARNABLE = True

            if self.FIXED is True:
                self.optim = 'fixed'
            elif self.LEARNABLE is True:
                self.optim = 'learnable'
        else:
            self.action = 'NOT Using'