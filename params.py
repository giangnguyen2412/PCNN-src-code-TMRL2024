class RunningParams(object):
    def __init__(self):
        # Training mode
        self.CUB_TRAINING = False  # ----------------------------------------- IMPORTANT PARAM --------
        self.DOGS_TRAINING = True
        self.CARS_TRAINING = False
        self.IMAGENET_TRAINING = False

        # General
        self.SIMCLR_MODEL = False
        self.IMAGENET_REAL = True
        self.advising_network = True
        self.conv_layer = 4

        if self.CUB_TRAINING is True:
            self.feat_map_size = {4: 49}
            self.conv_layer_size = {4: 2048, 3: 1024, 2: 512, 1: 256}
        elif self.CARS_TRAINING is True:
            self.feat_map_size = {4: 49}
            self.conv_layer_size = {4: 512}
        elif self.DOGS_TRAINING is True:
            self.feat_map_size = {4: 256}
            self.conv_layer_size = {4: 512}
        else:
            exit(-1)

        # XAI methods
        self.NO_XAI = 'No-XAI'
        self.GradCAM = 'GradCAM'
        self.NNs = 'NNs'
        self.XAI_method = self.NNs
        # TODO: write script to run a set of basic experiments: No-XAI, NNs with conv2,3,4, k1,3,5

        self.BOTTLENECK = False
        if self.BOTTLENECK is True:
            self.conv_layer_size = {4: 512}

        self.HIGHPERFORMANCE_FEATURE_EXTRACTOR = True
        self.HIGHPERFORMANCE_MODEL1 = True

        # Training parameters
        if self.CUB_TRAINING is True:
            self.batch_size = 96
            self.epochs = 100
            self.learning_rate = 1e-3
            self.k_value = 1

        elif self.DOGS_TRAINING is True:
            self.batch_size = 96
            self.epochs = 100
            self.learning_rate = 1e-2
            self.k_value = 1

        elif self.CARS_TRAINING is True:
            self.batch_size = 256
            self.epochs = 100
            self.learning_rate = 1e-2
            self.k_value = 1
        else:
            exit(-1)

        self.query_frozen = True  # False = Trainable; True = Freeze? -------------------- IMPORTANT PARAM --------
        self.UNBALANCED_TRAINING = True

        self.PRECOMPUTED_NN = True

        # Infer
        self.MODEL2_ADVISING = False
        self.advising_steps = 0

        # Visualization
        self.M2_VISUALIZATION = False

        # Unused
        self.BATCH_NORM = True
        self.GradCAM_RNlayer = 'layer4'