import os

class RunningParams(object):
    def __init__(self):
        # Training mode
        self.CUB_TRAINING = True  # ----------------------------------------- IMPORTANT PARAM --------
        self.DOGS_TRAINING = False
        self.CARS_TRAINING = False

        # TODO
        self.parent_dir = '/home/giang/Downloads'
        self.prj_dir = '/home/giang/Downloads/advising_network'

        self.dropout = 0.0

        self.wandb_sess_name = f'nov-8-23-1-bs128-p0_5-dropout-{self.dropout}'

        if self.CARS_TRAINING is True:
            self.resnet = 50

        # General
        self.conv_layer = 4

        if self.CUB_TRAINING is True:
            self.feat_map_size = {4: 49}
            self.conv_layer_size = {4: 2048, 3: 1024, 2: 512, 1: 256}

        elif self.CARS_TRAINING is True:
            self.feat_map_size = {4: 49}
            if self.resnet == 50:
                self.conv_layer_size = {4: 2048}
            elif self.resnet == 34:
                self.conv_layer_size = {4: 512}
            elif self.resnet == 18:
                self.conv_layer_size = {4: 512}
            else:
                print('Not supported architecture! Exiting...')
                exit(-1)
        else:
            print('Wrong params! Exiting...')
            exit(-1)

        # XAI methods
        self.NNs = 'NNs'
        self.XAI_method = 'NNs'
        # TODO: write script to run a set of basic experiments: No-XAI, NNs with conv2,3,4, k1,3,5

        self.BOTTLENECK = False
        if self.BOTTLENECK is True:
            self.conv_layer_size = {4: 512}

        self.HIGHPERFORMANCE_FEATURE_EXTRACTOR = True

        # Training parameters
        if self.CUB_TRAINING is True:
            self.batch_size = 128 #TODO: 64, 32 if not enough memory
            self.epochs = 100
            self.learning_rate = 1e-3
            self.k_value = 1

            self.in_features = 2048

            # Determine if you want to use 1st, 2nd or 3rd NNs (in each class) to pair with your input to train AdvNet.
            self.negative_order = 1

            # Retrieving NNs and sample positive and negative pairs
            # Set it when you extract the NNs. data_dir is the folder containing query images for KNN retrieval
            self.set = 'train'
            if self.set == 'test':
                self.data_dir = f'{self.parent_dir}/datasets/CUB/test0'  # CUB test folder
            else:
                self.data_dir = f'{self.parent_dir}/datasets/CUB/advnet/{self.set}'  # CUB train folder

            self.QK = 10  # Q and K values for building positives and negatives
            self.faiss_npy_file = '{}/faiss/cub/top{}_k{}_enriched_NeurIPS_Finetuning_faiss_{}5k7_top1_HP_MODEL1_HP_FE_NN{}th.npy'. \
                format(self.prj_dir, self.QK, self.k_value, self.set, self.negative_order)
            self.aug_data_dir = self.data_dir +'_all' + f'_top{self.QK}_NN{self.negative_order}th_{self.wandb_sess_name}'


            self.N = 4  # Depth of self-attention
            self.M = 4  # Depth of cross-attention
            self.L = 2  # Depth of transformer
            self.extension = '.jpg'

        elif self.CARS_TRAINING is True:
            self.batch_size = 256
            self.epochs = 100
            self.learning_rate = 1e-2
            self.k_value = 1

            # Retrieving NNs and sample positive and negative pairs
            # Set it when you extract the NNs. data_dir is the folder containing query images for KNN retrieval
            self.set = 'train'
            self.data_dir = f'{self.parent_dir}/Cars/Stanford-Cars-dataset/' + set

            self.QK = 7  # Q and K values for building positives and negatives
            self.faiss_npy_file = '{}/faiss/cars/top{}_k{}_enriched_NeurIPS_Finetuning_faiss_{}_top1.npy'. \
                format(self.prj_dir, self.QK, self.k_value, self.set)
            self.aug_data_dir = os.path.join(self.data_dir + f'_top{self.QK}_rn{self.resnet}')

            self.N = 3
            self.M = 3
            self.L = 3
            self.extension = '.jpg'

        else:
            exit(-1)

        # Visualization
        self.M2_VISUALIZATION = False
        self.VISUALIZE_TRANSFORMER_ATTN = False