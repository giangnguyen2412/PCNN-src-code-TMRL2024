# Learning to Distinguish Correct and Wrong Classifications via Nearest-Neighbor Explanations
## How to run the training for image-comparator network?

Download pretrained models of ResNets for CUB-200 and Cars-196 at [this link](https://drive.google.com/drive/folders/1pC_5bEi5DryDZCaKb51dzCE984r8EnqW?usp=sharing).

## Training

For CUB-200,
> sh cub_run.sh

For Cars-196,
> sh cars_run.sh

Notes:

a. Must config CUB_TRAINING or CARS_TRAINING in params.py before anything below.

b. Must config `parent_dir` in params.py

c. Must config `prj_dir` in params.py

d. Must config `set` in params.py


## Testing
We provide two scripts to run two applications of AdvisingNets (a) correct/wrong prediction classification and (b) multi-way image classification using the Top-Class Reranking algorithm.
To run (a),

> python cub_infer.py

To run (b),

> python cub_extract_feature_adv_process.py

> python cub_advising_process
## How to visualize qualitative figures for AdvisingNets
1. Corrections of AdvisingNets
> python cub_visualize_corrections.py
2. Training pairs of AdvisingNets
> cub_visualize_training_nns.py
3. Failures of AdvisingNets (please change M2_VISUALIZATION in params.py to True)
> python cub_infer.py
The same steps for Cars-196.


### RN50 image comparator improving NTSNet
Set NTSNET in params.py to True. Then set MODEL1_RESNET in cub_extract_feature_adv_process.py to False then run:
> python cub_extract_feature_adv_process.py

Then set MODEL1_RESNET in cub_advising_process.py to False then run:
> python cub_advising_process.py

You can also set PRODUCT_OF_EXPERTS in cub_advising_process.py to True/False to use/not use the product of experts.

