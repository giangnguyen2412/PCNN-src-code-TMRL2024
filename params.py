import os

class RunningParams(object):
    def __init__(self):
        # Training mode
        self.CUB_TRAINING = True  # ----------------------------------------- IMPORTANT PARAM --------
        self.DOGS_TRAINING = False
        self.CARS_TRAINING = False

        self.TRANSFORMER_ARCH = True
        self.resnet = 50
        self.RN50_INAT = True

        # if self.CUB_TRAINING is True and self.RN50_INAT is True and self.resnet == 50:
        #     self.NTSNET = True
        # else:
        self.NTSNET = False

        self.VisionTransformer = False
        # Retrieving NNs and sample positive and negative pairs
        # Set it when you extract the NNs. data_dir is the folder containing query images for KNN retrieval
        # Set it when you run train/test
        self.set = 'test'

        self.parent_dir = '/home/giang/Downloads'
        self.prj_dir = '/home/giang/Downloads/advising_network'

        self.dropout = 0.0
        self.trivial_aument_p = 0.0

        # General
        self.conv_layer = 4

        self.feat_map_size = {self.conv_layer: 49} # 7x7
        if self.resnet == 50:
            self.conv_layer_size = {4: 2048}
        elif self.resnet == 34:
            self.conv_layer_size = {4: 512}
        elif self.resnet == 18:
            self.conv_layer_size = {4: 512}
        else:
            print('Not supported architecture! Exiting...')
            exit(-1)

        # XAI methods
        self.NNs = 'NNs'
        self.XAI_method = 'NNs'
        # TODO: write script to run a set of basic experiments: No-XAI, NNs with conv2,3,4, k1,3,5

        self.BOTTLENECK = False

        self.HIGHPERFORMANCE_FEATURE_EXTRACTOR = True

        # Training parameters
        if self.CUB_TRAINING is True:
            self.batch_size = 256
            self.epochs = 200
            self.learning_rate = 3e-4
            self.k_value = 1

            # Determine if you want to use 1st, 2nd or 3rd NNs (in each class) to pair with your input to train AdvNet.
            self.negative_order = 1

            if self.set == 'test':
                self.data_dir = f'{self.parent_dir}/datasets/CUB/test0'  # CUB test folder
            else:
                self.data_dir = f'{self.parent_dir}/datasets/CUB/advnet/{self.set}'  # CUB train folder

            self.QK = 10  # Q and K values for building positives and negatives
            self.faiss_npy_file = '{}/faiss/cub/INAT_{}_top{}_k{}_enriched_rn{}_{}_NN{}th.npy'. \
                format(self.prj_dir, self.RN50_INAT, self.QK, self.k_value, self.resnet, self.set, self.negative_order)

            self.wandb_sess_name = f'INAT_{self.RN50_INAT}_cub_rn{self.resnet}_bs{self.batch_size}-p{self.trivial_aument_p}-dropout-{self.dropout}-NNth-{self.negative_order}'

            self.aug_data_dir = self.data_dir +'_all' + f'_top{self.QK}_rn{self.resnet}_NN{self.negative_order}th_{self.wandb_sess_name}_INAT_{self.RN50_INAT}'

            if self.VisionTransformer is True:
                self.faiss_npy_file = '{}/faiss/cub/ViT_top{}_k{}_enriched_{}_NN{}th.npy'. \
                    format(self.prj_dir, self.QK, self.k_value, self.set,
                           self.negative_order)

                self.wandb_sess_name = f'ViT_bs{self.batch_size}-p{self.trivial_aument_p}-dropout-{self.dropout}-NNth-{self.negative_order}'

                self.aug_data_dir = self.data_dir + '_all' + f'_top{self.QK}_ViT_NN{self.negative_order}th_{self.wandb_sess_name}'

                # self.wandb_sess_name = f'ViT_bs{self.batch_size}-p{self.trivial_aument_p}-dropout-{self.dropout}-NNth-{self.negative_order}-trainViT-SGD-no-trivialaugment'

            self.N = 4  # Depth of self-attention
            self.M = 4  # Depth of cross-attention
            self.L = 2  # Depth of transformer
            self.extension = '.jpg'

        elif self.CARS_TRAINING is True:
            self.batch_size = 256
            self.epochs = 100
            self.learning_rate = 1e-2
            self.k_value = 1

            self.negative_order = 1

            # Retrieving NNs and sample positive and negative pairs
            # Set it when you extract the NNs. data_dir is the folder containing query images for KNN retrieval
            self.data_dir = f'{self.parent_dir}/Cars/Stanford-Cars-dataset/' + self.set

            self.QK = 10  # Q and K values for building positives and negatives
            self.faiss_npy_file = '{}/faiss/cars/top{}_k{}_enriched_{}_Finetuning_faiss_{}_top1.npy'. \
                format(self.prj_dir, self.QK, self.k_value, self.resnet, self.set)

            self.wandb_sess_name = f'cars_rn{self.resnet}_bs{self.batch_size}-p{self.trivial_aument_p}-dropout-{self.dropout}-NNth-{self.negative_order}'

            self.aug_data_dir = os.path.join(self.data_dir + f'_top{self.QK}_rn{self.resnet}')

            self.N = 3
            self.M = 3
            self.L = 3
            self.extension = '.jpg'

        elif self.DOGS_TRAINING is True:
            self.batch_size = 256
            self.epochs = 50
            self.learning_rate = 1e-3
            self.k_value = 1

            self.negative_order = 1

            # Retrieving NNs and sample positive and negative pairs
            # Set it when you extract the NNs. data_dir is the folder containing query images for KNN retrieval
            self.data_dir = f'{self.parent_dir}/Stanford_Dogs_dataset/' + self.set

            self.QK = 10  # Q and K values for building positives and negatives
            self.faiss_npy_file = '{}/faiss/dogs/top{}_k{}_enriched_{}_Finetuning_faiss_{}_top1.npy'. \
                format(self.prj_dir, self.QK, self.k_value, self.resnet, self.set)

            # self.wandb_sess_name = f'dogs_rn{self.resnet}_bs{self.batch_size}-p{self.trivial_aument_p}-dropout-{self.dropout}-NNth-{self.negative_order}'

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

