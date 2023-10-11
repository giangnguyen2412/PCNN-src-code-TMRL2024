# AdvisingNets: Learning to Distinguish Correct and Wrong Classifications via Nearest-Neighbor Explanations
## How to run the training for AdvisingNets?
In general, for CUB-200, you need to follow these steps:
1. Sampling positive and negative pairs for training AdvisingNets using a target classifier C and its feature extractor. 
> python cub_extract_feature.py
2. Rename images and put them into proper `class` folders
> python copy_and_rename_topk.py
3. Run the training
> python cub_model_training.py

In details:

a. Must set CUB_TRAINIG|DOGS|IMAGENET in params.py before anything below.

b. Retrieving correct NNs of top1 — Done

c. Check if correct images having NNs in the same category

d.Check if in training, the query and the 1st are the same?

e. When running feature extraction, notice if FAISS index file exists or not?

f. Check the model is working correctly in training —> the acc for MODEL1 should be ~50:50 for both training and validation.

g. Check if the numbers of training and val loaded for training are expected

h. Convert the accuracy to exactly 50:50

i. Change the backbone model to the good pretrained model — 

k. Verify the confidence score? How can we do this?

l. In inference, please remember to ensure that faiss_dataset path in infer.py is the same with faiss_dataset in extract_feature.py


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
The same steps for Cars-196.
