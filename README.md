# PCNN: Probable-Class Nearest-Neighbor Explanations Improve Fine-Grained Image Classification Accuracy for AIs and Humans


```markdown
Nearest neighbors are traditionally used to compute final decisions, e.g., in Support Vector Machines or 
-NN classifiers and to provide users with supporting evidence for the model's decision. In this paper, we show a novel use of nearest neighbors: To improve predictions of an existing pretrained classifier \classifier. We leverage an image comparator \comparator that (1) compares the input image with nearest-neighbor images from the top-
 most probable classes; and (2) weights the confidence scores of \classifier (like a Product of Experts). Our method consistently improves fine-grained image classification accuracy on CUB-200, Cars-196, and Dogs-120. Furthermore, a human study finds that showing layusers our probable-class nearest neighbors (PCNN) improves their decision-making accuracy over showing only the top-1 class examples (as in prior work).
```

## How to run the training for image-comparator network?

Download pretrained models for CUB-200, Cars-196, and Dogs-120 at [this link](https://drive.google.com/drive/folders/1pC_5bEi5DryDZCaKb51dzCE984r8EnqW?usp=sharing).

## Training image comparator network


For CUB-200,

Step 1: Set `global_training_type = 'CUB'` and `self.set = 'train'` in `params.py`

Step 2:
> sh train_cub.sh

For Cars-196,

Step 1: Set `global_training_type = 'CARS'` and `self.set = 'train'` in `params.py`

Step 2:
> sh train_cars.sh


For Dogs-120,

Step 1: Set `global_training_type = 'DOGS'` and `self.set = 'train'` in `params.py`

Step 2:
> sh train_dogs.sh

## Testing (Binary classification and Reranking)

For CUB-200,

Step 1: Set `global_training_type = 'CUB'` and `self.set = 'test'` in `params.py`

Step 2:
> sh test_cub.sh

For Cars-196,

Step 1: Set `global_training_type = 'CARS'` and `self.set = 'test'` in `params.py`

Step 2:
> sh test_cars.sh

For Dogs-120,

Step 1: Set `global_training_type = 'DOGS'` and `self.set = 'test'` in `params.py`

Step 2:
> sh test_dogs.sh

## How to visualize qualitative figures
For CUB-120,

Step 1: Set `global_training_type = 'CUB'` and `self.set = 'test'` in `params.py`

Step 2:

1. Corrections of S
> python cub_visualize_corrections.py
2. Training pairs of S
> cub_visualize_training_nns.py
3. Failures of S (please change `VISUALIZE_COMPARATOR_CORRECTNESS` in `params.py` to `True`)
> python cub_infer.py

Same steps for Cars-196 and Dogs-120.
