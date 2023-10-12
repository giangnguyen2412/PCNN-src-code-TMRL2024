# AdvisingNets: Learning to Distinguish Correct and Wrong Classifications via Nearest-Neighbor Explanations

## How to run the training for AdvisingNets?

In general, for CUB-200, you need to follow these steps:
1. Sampling positive and negative pairs for training AdvisingNets

> python cub_extract_feature.py
> python copy_and_rename_topk.py

2. Run the training

> python cub_model_training.py

The same for Cars-196.

## How to run the inference for AdvisingNets

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

The more details for reproducibility will be made publicly available.

