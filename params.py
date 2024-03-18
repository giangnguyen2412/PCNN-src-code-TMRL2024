import os

global_training_type = 'CUB'

class RunningParams:
    def __init__(self, training_type=None):
        # Use the global training_type if not specified
        if training_type is None:
            training_type = global_training_type

        self.set_active_training(training_type)

        self.TRANSFORMER_ARCH = True
        self.VisionTransformer = False
        self.resnet = 50

        self.k_value = 1

        # Retrieving NNs and sample positive and negative pairs
        # Set it when you extract the NNs. data_dir is the folder containing query images for KNN retrieval
        # Set it when you run train/test
        self.set = 'train'
        self.PRODUCT_OF_EXPERTS = True

        self.parent_dir = '/home/giang/Downloads'
        self.prj_dir = '/home/giang/Downloads/advising_network'
        self.model_dir = f'{self.prj_dir}/pretrained_models'

        # General
        self.conv_layer = 4

        self.feat_map_size = {self.conv_layer: 49}  # 7x7
        if self.resnet == 50:
            self.conv_layer_size = {4: 2048}
        elif self.resnet == 34:
            self.conv_layer_size = {4: 512}
        elif self.resnet == 18:
            self.conv_layer_size = {4: 512}
        else:
            print('Not supported architecture! Exiting...')
            exit(-1)

        # Visualization
        self.VISUALIZE_COMPARATOR_CORRECTNESS = False
        self.VISUALIZE_COMPARATOR_HEATMAPS = False

        # Set training-specific parameters
        self.set_training_params()

    def set_active_training(self, training_type):
        # Set all training modes to False
        self.CUB_TRAINING = False
        self.DOGS_TRAINING = False
        self.CARS_TRAINING = False

        # Activate the chosen training mode
        if training_type == 'CUB':
            self.CUB_TRAINING = True
        elif training_type == 'DOGS':
            self.DOGS_TRAINING = True
        elif training_type == 'CARS':
            self.CARS_TRAINING = True
        else:
            raise ValueError("Invalid training type. Please choose 'CUB', 'DOGS', or 'CARS'.")

    def set_training_params(self):
        # Set parameters specific to the active training mode
        if self.CUB_TRAINING:
            # Set parameters for CUB training
            self.train_path = 'datasets/CUB/train1'
            self.test_path = 'datasets/CUB/test0'
            self.combined_path = 'datasets/CUB/combined'
            self.batch_size = 256
            self.epochs = 100

            self.learning_rate = 1e-3

            self.RN50_INAT = True
            self.NTSNET = False

            # Determine if you want to use 1st, 2nd or 3rd NNs (in each class) to pair with your input to train AdvNet.
            self.negative_order = 1

            if self.set == 'test':
                self.data_dir = f'{self.parent_dir}/{self.test_path}'  # CUB test folder
            else:
                self.data_dir = f'{self.parent_dir}/datasets/CUB/advnet/{self.set}'  # CUB train folder

            self.Q = 10  # Q values for building positives and negatives
            self.faiss_npy_file = f'{self.prj_dir}/faiss/cub/INAT_{self.RN50_INAT}_top{self.Q}_k{self.k_value}_rn{self.resnet}_{self.set}-set_NN{self.negative_order}th.npy'

            self.aug_data_dir = f'{self.data_dir}_INAT_{self.RN50_INAT}_top{self.Q}_rn{self.resnet}_NN{self.negative_order}th'

            if self.VisionTransformer is True:
                self.faiss_npy_file = f'{self.prj_dir}/faiss/cub/ViT_top{self.Q}_k{self.k_value}_{self.set}-set_NN{self.negative_order}th.npy'

                self.aug_data_dir = f'{self.data_dir}_ViT_INAT_{self.RN50_INAT}_top{self.Q}_rn{self.resnet}_NN{self.negative_order}th'

            self.N = 4  # Depth of self-attention
            self.M = 4  # Depth of cross-attention
            self.L = 2  # Depth of transformer
            self.extension = '.jpg'

        elif self.CARS_TRAINING:
            # Set parameters for CARS training
            self.train_path = 'Cars/Stanford-Cars-dataset/BACKUP/train'
            self.test_path = 'Cars/Stanford-Cars-dataset/test'
            self.batch_size = 256
            self.epochs = 100

            self.learning_rate = 1e-2
            self.negative_order = 1

            # Retrieving NNs and sample positive and negative pairs
            # Set it when you extract the NNs. data_dir is the folder containing query images for KNN retrieval
            self.data_dir = f'{self.parent_dir}/Cars/Stanford-Cars-dataset/' + self.set

            self.Q = 10  # Q values for building positives and negatives
            self.faiss_npy_file = f'{self.prj_dir}/faiss/cars/top{self.Q}_k{self.k_value}_rn{self.resnet}_{self.set}-set_top1.npy'

            self.aug_data_dir = f'{self.data_dir}_top{self.Q}_rn{self.resnet}'

            self.N = 3
            self.M = 3
            self.L = 3
            self.extension = '.jpg'

        elif self.DOGS_TRAINING:
            # Set parameters for DOGS training
            self.train_path = 'Stanford_Dogs_dataset/train'
            self.test_path = 'Stanford_Dogs_dataset/test'
            self.batch_size = 256
            self.epochs = 50

            self.learning_rate = 1e-3
            self.negative_order = 1

            # Retrieving NNs and sample positive and negative pairs
            # Set it when you extract the NNs. data_dir is the folder containing query images for KNN retrieval
            self.data_dir = f'{self.parent_dir}/Stanford_Dogs_dataset/' + self.set

            self.Q = 10  # Q values for building positives and negatives
            self.faiss_npy_file = f'{self.prj_dir}/faiss/dogs/top{self.Q}_k{self.k_value}_rn{self.resnet}_{self.set}-set_top1.npy'

            self.aug_data_dir = os.path.join(self.data_dir + f'_top{self.Q}_rn{self.resnet}')

            self.N = 3
            self.M = 3
            self.L = 3
            self.extension = '.jpg'

        else:
            raise ValueError("No valid training mode set.")


