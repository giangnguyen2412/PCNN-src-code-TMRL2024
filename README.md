# Learning to Distinguish Correct and Wrong Classifications via Nearest-Neighbor Explanations
## How to run the training for image-comparator network?

Download pretrained models of ResNets at [this link](https://drive.google.com/drive/folders/1pC_5bEi5DryDZCaKb51dzCE984r8EnqW?usp=sharing).

## Training

For CUB-200,
> sh cub_run.sh

Notes:

a. Must config CUB_TRAINING in params.py before anything below.

b. Must config `parent_dir` in params.py

c. Must config `prj_dir` in params.py

d. Must config `set` in params.py


## Test
We provide two scripts to run two applications of S (a) correct/wrong prediction classification and (b) multi-way image classification using the Top-Class Reranking algorithm.
To run (a),

> python cub_infer.py

To run (b),

> python cub_extract_feature_adv_process.py

> python cub_advising_process

Same with other datasets.

## How to visualize qualitative figures
1. Corrections of CxS
> python cub_visualize_corrections.py
2. Training pairs of S
> cub_visualize_training_nns.py
3. Failures of S (please change M2_VISUALIZATION in params.py to True)
> python cub_infer.py

The same steps for other datasets.



