# PCNN: Probable-Class Nearest-Neighbor Explanations Improve Fine-Grained Image Classification Accuracy for AIs and Humans


```markdown
Nearest neighbors are traditionally used to compute final decisions, e.g., in Support Vector Machines or 
-NN classifiers and to provide users with supporting evidence for the model's decision. In this paper, we show a novel use of nearest neighbors: To improve predictions of an existing pretrained classifier \classifier. We leverage an image comparator \comparator that (1) compares the input image with nearest-neighbor images from the top-
 most probable classes; and (2) weights the confidence scores of \classifier (like a Product of Experts). Our method consistently improves fine-grained image classification accuracy on CUB-200, Cars-196, and Dogs-120. Furthermore, a human study finds that showing layusers our probable-class nearest neighbors (PCNN) improves their decision-making accuracy over showing only the top-1 class examples (as in prior work).
```

## How to run the training for image-comparator network?

Download pretrained models for CUB-200, Cars-196, and Dogs-120 at [this link](https://drive.google.com/drive/folders/1pC_5bEi5DryDZCaKb51dzCE984r8EnqW?usp=sharing).

## Sampling &rightarrow; Training &rightarrow; Binary classify

For CUB-200,
> sh cub_run.sh

For Cars-196,
> sh cars_run.sh

For Dogs-120,
> sh dogs_run.sh

Notes: Set `global_training_type` to  = `'CUB'` or `'CARS'` or `'DOGS'`


## Reranking

> python cub_extract_feature_for_reranking.py

> python cub_reranking.py

## How to visualize qualitative figures
1. Corrections of S
> python cub_visualize_corrections.py
2. Training pairs of S
> cub_visualize_training_nns.py
3. Failures of S (please change `VISUALIZE_COMPARATOR_CORRECTNESS` in `params.py` to `True`)
> python cub_infer.py

Same steps for Cars-196 and Dogs-120.

[//]: # ()
[//]: # (### RN50 image comparator improving NTSNet)

[//]: # (Set NTSNET in params.py to True. Then set MODEL1_RESNET in cub_extract_feature_adv_process.py to False then run:)

[//]: # (> python cub_extract_feature_adv_process.py)

[//]: # ()
[//]: # (Then set MODEL1_RESNET in cub_advising_process.py to False then run:)

[//]: # (> python cub_advising_process.py)

[//]: # ()
[//]: # (You can also set PRODUCT_OF_EXPERTS in cub_advising_process.py to True/False to use/not use the product of experts.)

[//]: # ()
