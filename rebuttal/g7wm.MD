We sincerely thank the Reviewer for the thoughtful and constructive feedback. 
Below, we address each of the Reviewer's questions and suggestions.

> 1. (Questions 1) How does the proposed method perform on tasks other than fine-grained image classification, such as object detection or semantic segmentation (the paper is 47 pages long so hopefully I didn’t miss any results)?

Thank you for this very interesting question!

To tackle it, we first conduct a literature review on the use of nearest neighbors (NNs) for [object detection](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C1&as_ylo=2020&q=%22object+detection%22+%2B+%22nearest+neighbor+samples%22&btnG=) and [semantic segmentation](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C1&q=%22semantic+segmentation%22+%2B+%22nearest+neighbor%22&btnG=)
to see how PCNN could be useful.

What we found is that literature has not directly used nearest-neighbor examples to either train a model or perform re-ranking or refinement in the context of object detection and semantic segmentation tasks.
Instead, nearest-neighbor samples often play a **supplementary** role or are used in specific subcomponents (e.g. papers [1](https://openaccess.thecvf.com/content/CVPR2023W/GCV/papers/Lin_Explore_the_Power_of_Synthetic_Data_on_Few-Shot_Object_Detection_CVPRW_2023_paper.pdf),[2](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8967762)) rather than being the primary algorithm.

Therefore, we see a big gap between our method vs. the existing literature in these tasks and think that the proposed method would need extensive adaptation or further development to be applied to object detection or semantic segmentation tasks.


> 2. (Questions 2) Can the re-ranking algorithm be extended to handle multi-label classification problems?

Yes, it can!

First, let's revisit the definition of multi-label classification for clarity.

**Multi-label classification**: is a type of machine learning problem where each instance (or example) can 
be assigned multiple labels from a set of possible labels, rather than just one. 
This differs from the traditional single-label classification tasks, where each instance is 
associated with only one label.

We took the below Figure from [C-Tran paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lanchantin_General_Multi-Label_Image_Classification_With_Transformers_CVPR_2021_paper.pdf) and will base on it to give examples.
In the Figure, the based classifier are predicting the presence of a `person` and an `umbrella` in the input image.
The threshold to determine presence is set to 0.5 (specified in `Sec. 4.1` of [C-Tran paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lanchantin_General_Multi-Label_Image_Classification_With_Transformers_CVPR_2021_paper.pdf)).

Figure: [An example of prediction in multi-label image classification](https://drive.google.com/file/d/1ltlCz-pIVm1e7s8133XMeCGWfkgP-aRn/view?usp=sharing)


In multi-label classification, the ground-truth y is a binary vectors indicating the presence of C classes `{y1, y2, ..., yc}, yi ∈ {0, 1}`.

Then, for each label, the two classes to be re-ranked here are the presence (1) and absence (0) of the object (i.e. `person`).
To extend our re-ranking algorithm to handle multi-label classification problems, we compare the input image with the nearest neighbors from the two classes (presence and absence) using the image comparator S.

For example, S will compare the input image with the nearest neighbors from the `person` class (1) and the `no-person` class (0) from the training data.
As we know that S returns a similarity score in the range `[0, 1]` indicating the likelihood of both the two images containing person.
The PoE will work by multiplying the output similarity score with the predicted probability (`0.83`) of the base classifier C-Tran for the `person` class.
The final predictions will be thresholded at a certain value (0.5) to determine the presence of the object.

Another problem to concern is the number of classes to be re-ranked. 
We do not want to re-rank all the possible classes because it could be computationally expensive.

In our submission, we propose to re-rank the **top-10** classes predicted by the base classifier C.
Yet, in multi-label classification, the re-ranking algorithm could only just care about the classes that 
are suggested by the pretrained classifier (e.g. classes having >= 0.5 confidence score, `person` and `umbrella` in the [C-Tran Figure](https://drive.google.com/file/d/1ltlCz-pIVm1e7s8133XMeCGWfkgP-aRn/view?usp=sharing).
As a result, re-ranking could help reduce the False Positive rate in multi-label classification tasks, where the model predicts the presence of an object that is not in the test image.

> 3. (Questions 3) How does the performance of the PoE model scale with the number of classes or the size of the dataset?

First, we want to confirm that the reviewer refers to the `test-time accuracy` of the PoE model when mentioning "performance".

We think that there are three main factors that primarily affect the performance of the PoE model:

- (a) The number of classes (mentioned by the Reviewer)
- (b) The size of the dataset (mentioned by the Reviewer)
- (c) The performance of the base classifier C because the PoE model is a combination of the base classifier C and the image comparator S.

(a) When the number of classes **increases** (e.g. from 200 in CUB to 1000), the performance of the PoE model will likely **decrease**.
This is because having more classes will make the image comparator S more likely sees visually similar objects from different classes (e.g. `Tern species` in `Fig.2` in PCNN submission look very similar),
making the re-ranking weights more uniform across the classes, and thus reducing the effect of the image comparator S on the final predictions.
Please can revisit the `Fig. 2` linked below.

Figure: [PoE Re-ranking algorithm from PCNN submission](https://drive.google.com/file/d/1nGgf18xVLF_i26omjy0_M8l3WBdGuNsZ/view?usp=sharing)

It is important to note that the final PoE performance will be also affected by the performance of the base classifier C, which will be changed if the number of classes increases
(as shown in `Fig. 6` in PCNN submission).


[//]: # (If the reviewer has any specific experiments in mind, we are more than happy to run.)


(b) When the size of the dataset **increases**, the performance of the PoE model will likely **increase** because the image comparator S will have more data to learn from and 
combat with variations better. 
For example, in CUB-200 datasets, within a class, there are many variations in the appearance of the same bird ([males, females, and juveniles](https://www.allaboutbirds.org/guide/Painted_Bunting/)).
Having more data will make the comparator S more robust to these variations, thus improving the PoE model `CxS`.
In addition, having more data will help the base classifier C to yield better performance, which will also influence the impact of PoE (as shown in `Fig. 6` in PCNN submission).

Regarding (c), we found that when C is low-performing, adding S (via PoE) will be very useful.
However, when C is already high-performing, adding S will not help much.
These two mentioned cases are respective to ImageNet-pretrained ResNet18 and NTS-Net in `Fig. 6` in PCNN submission linked below.

Figure: [The impact of the base classifier C to the PoE performance](https://drive.google.com/file/d/1IU-_sjf_uUgOcWHIOt-aBuBaVAyMnqT1/view?usp=sharing)

Finally, we are willing to run any requested experiments to address your concern in scaling PCNN.

> 4. (Question 4) What is the impact of using different distance metrics or similarity measures for finding nearest neighbours on the performance of the PoE model?

Thank you for the insightful question! We want to break down the answer into two parts:

a. When the NNs are found by `L2` distance through the `IndexFlatL2` function of `faiss` library, 
we tested using different functions to compute the similarity between the input image and the NNs for re-ranking (See `Tab. 7` in PCNN submission) or the below Table.


| NN-th | Top-1 Acc | Top-1 Acc       | Top-1 Acc       | Top-1 Acc     | Top-1 Acc        |
|-------|----------|-----------------|--------|------|------------------|
|       | RN50 × S | Our trained S** | cosine | EMD  | 4-layer MLP only |
| 1st   | 88.59    | 87.72           | 60.20  | 54.83 | 83.76            |
| 2nd   | 88.06    | 87.34           | 58.84  | 57.05 | 84.33            |
| 3rd   | 88.21    | 87.43           | 57.47  | 57.14 | 83.93            |

** S = shared feature extractor + 4-layer MLP for similarity function

We found our comparator S works the best for the re-ranking algorithm.
When combining the scores of RN50 and S via PoE (RN50 × S), we even found it more effective than using S alone.

b. When we fix the re-ranking method to PoE (RN50 × S), we tested using different distance metrics to find the nearest neighbors as the Reviewer suggested.
Here we assess the PoE performance with `cosine` similarity and [DreamSim](https://dreamsim-nights.github.io/) instead of `L2`.
Compared to using `L2` as reported in PCNN submission, `cosine` similarity measure the global similarity between two images while `DreamSim`
is a human-aligned similarity measure.

| Distance function                              | CUB-200 PoE perf (%) | Dogs-120 PoE perf (%) |
|------------------------------------------------|----------------------|-----------------------|
| Random                                         | 87.95%               | 85.89%                |
| L2                                             | 88.59%               | 86.31%                |
| Cosine                                         | 88.33%               | 86.25%                |
| [DreamSim](https://dreamsim-nights.github.io/) | 88.38%               | 86.56%                |


We first test on CUB-200, and we found that different measures to retrieve NNs have a small to no impact on the PoE model performance.
We suppose that because the number of samples per class is fairly small (`30` images per class on average), then when doing retrieval with different distance metrics, 
the nearest neighbors are **not very different**.

Then, we move from CUB-200 to Dogs-120 that has more samples per class (`100` images per class on average).
Interestingly, we also found that different distance metrics have a small to no impact on the PoE model performance (see the above Table).

We also test using the **Random nearest neighbors** (i.e. randomly picking one sample from each of the probable classes for re-ranking). 
Yet, it drops the PoE model performance from `88.59` to `87.95` on CUB-200 and from `86.31` to `85.89` on Dogs-120, respectively.

Overall, we found that changing the distance metrics for finding nearest neighbors has a small to no impact on the performance of the PoE model.
In addition, the Random baseline confirms the advantages of using the most similar images over the less-similar ones. 


> 5. (Question 5) Are there any theoretical insights or principles that can guide the design and training of the image comparator S?

**The task of the image comparator** S is inspired by the [distinction task](https://arxiv.org/pdf/2112.03184) where we give human users two images 
and ask them to judge if the two images are of the same class or not.

**To compare the two images**, we are inspired by [Siamese neural networks](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) where the feature extractor
is shared between two branches and the similarity function could be learnable, as shown in the following Figure below:

Figure: [Siamese neural networks](https://drive.google.com/file/d/1xXblL8rHRqz1FWr6ulDbo0gKgKoP7yBG/view?usp=sharing)

In the experiment in `Tab. 7` in PCNN submission, we found that this architecture (shared feature extractor + 4-layer MLP for similarity function) works the best for PoE model.


| NN-th | Top-1 Acc | Top-1 Acc       | Top-1 Acc       | Top-1 Acc     | Top-1 Acc        |
|-------|----------|-----------------|--------|------|------------------|
|       | RN50 × S | Our trained S** | cosine | EMD  | 4-layer MLP only |
| 1st   | 88.59    | 87.72           | 60.20  | 54.83 | 83.76            |
| 2nd   | 88.06    | 87.34           | 58.84  | 57.05 | 84.33            |
| 3rd   | 88.21    | 87.43           | 57.47  | 57.14 | 83.93            |

** S = shared feature extractor + 4-layer MLP for similarity function

**In training of S**, we are inspired by a long line of work in [contrastive learning](https://arxiv.org/pdf/2002.05709) that aims to learn discriminative feature representations via 
`same-class` samples and `different-class` samples.
Yet, simple sampling techniques, where the negative samples were randomly picked from the non-groundtruth classes, **is not optimal** for training S as shown in `Sec. B.3` in PCNN submission.
We innovate by proposing a sampling strategy that selects the negative samples based on the softmax scores of a pretrained classifier 
(i.e. improving PoE acc from `86.55%` → `88.59%` on CUB-200, see `Sec. B.3` in PCNN submission).

> 6. (Requested Changes 6) In section 4.3 you say “First, we test using cosine similarity in the pretrained feature space of a CUB-200 classifier” – why do you call CUB-200 a classifier? CUB-200 is mentioned 80+ across the paper.

Sorry for the confusion. We revised the writing and changed `CUB-200 classifier` to `classifier trained on CUB-200` in the paper to avoid any misunderstanding.
The change can be seen in the latest revision of the paper.

> 7. (Potential Changes 1) Explore techniques to reduce the computational overhead introduced by the re-ranking process, such as efficient nearest neighbour search or pruning strategies.

Thanks for bringing up this point, and it really encourages us to optimize the runtime of our method!

We agree that the re-ranking process can be computationally expensive, especially when the dataset is large.
The computational overhead comes from (a) finding the nearest neighbors or (b) querying the image comparator S for re-ranking weights as shown in `Table. 11` in PCNN submission.

(a) We found that **nearest neighbor retrieval** can be easily sped up by leveraging the speedup options of as listed [here](https://github.com/facebookresearch/faiss/wiki/How-to-make-Faiss-run-faster).
For example, as we are currently use CPU for faiss indexing (`L137` in the submitted [code](https://anonymous.4open.science/r/nearest-neighbor-XAI-FF2E/cub-200/cub_extract_feature_for_reranking.py)), 
using GPU by `GpuIndexFlatL2` from faiss can provide significant speedups, especially for larger datasets.

(b) For **image comparator S**, we can reduce the number of queries to the image comparator S by ignoring less probable labels.
Currently, we are setting this to `K = 10` as we always examine the **top-10** most probable classes.
Yet, there always exists classes receiving `< 1%` by ResNet50 (see Fig. 5 in PCNN submission), which are likely never to be the top-1 after re-ranking.
Reducing the number of `K` can save a lot of computation at a minimal cost of accuracy.

To verify this, we run an experiment for CUB-200 where we instead of re-ranking the whole **top-10**, we only re-rank the classes that have a probability `>= 1%` assigned by the base classifier C.
We found that:
- The PoE model accuracy on CUB-200 drops very marginally by `only 0.08%` (from `88.59%` → `88.51%`).
- However, the number of queries to the image comparator S was reduced by approx. `4x` (from `10` to just about `2.5` queries/image).
This leads to a `2.5x` speedup in the overall runtime of the PoE model (from `64.55` seconds to `28.95` seconds), as shown in the following Table.

| Model                        |   Time (s)   | Top-1 Acc (%) |
|:----------------------------:|:------------:|:-------------:|
| RN50 xS                      | 64.55 ± 0.35 |     88.59     |
| **RN50 xS (with threshold)** | 28.95 ± 0.11 |     88.51     |


> 8. (Potential Changes 2) Extend the human study to other datasets and domains to further evaluate the interpretability and usefulness of PCNN explanations.

We agree that extending the human study to other datasets and domains can provide more insights into the interpretability and usefulness of PCNN explanations.
Therefore, we **repeat** the human study that compares top-1 nearest neighbors and PCNN on `Stanford Dogs-120` dataset.

- Samples: Similar to CUB-200 study detailed in `Sec. H` in PCNN submission, we select 300 correctly classified and 300
misclassified query samples determined by the RN50 × S classifier, amounting to a total of 600 images for the study.

- Participants: We recruit `30` participants for the study, with `16` participants for the **top-1** and `14` participants for **PCNN** experiments. 
The participants are encouraged to perform responsibly, and we consider only the data from those who complete all the 30 test trials.

- Findings: We report the results from the human study in the following Table and Figure.

Figure: [User accuracy on Stanford Dogs](https://drive.google.com/file/d/1Fh3GrO6blL3muijc2-hGTnp-NwCUoYAU/view?usp=sharing)

| Explanation | AI Correctness | mean (%) | std (%) | Numb. of Samples |
|------------|----------------|----------|---------|------------------|
| Top-1      | AI is Wrong    | 34.86    | 24.46   | 225              |
| Top-1      | AI is Correct  | 89.07    | 9.10    | 255              |
| Overall    | ---            | 63.66    | 27.05   | 480              |
| PCNN       | AI is Wrong    | 52.74    | 18.63   | 192              |
| PCNN       | AI is Correct  | 82.55    | 9.80    | 228              |
| Overall    | ---            | 68.92    | 14.85   | 420              |

**We observe that presenting human participants with PCNN explanations improves their mean accuracy by 5.26%.**

When the AI is correct, participants achieve a lower mean accuracy of `82.55%` (± 9.80) with PCNN explanations, compared to `89.07%` (± 9.10) with top-1 nearest neighbors. 
However, when the AI is incorrect, participants with PCNN explanations achieve a significantly higher mean accuracy of `52.74%` (± 18.63) vs. `34.86%` (± 24.46) accuracy of those with top-1 nearest neighbors.
These findings suggest that PCNN explanations provide more informative cues for humans to recognize the correctness of AI predictions, particularly in cases where the AI is incorrect. 
The results from the Stanford Dogs-120 dataset corroborate the findings from the CUB-200 study, further demonstrating the usefulness of PCNN explanations across different domains.

The reviewer can try the human study on Stanford Dogs-120 via this [link](https://huggingface.co/spaces/xairesearch2023-advnet/HumanStudy-Dogs).

> 9. (Potential Changes 3) Provide a theoretical analysis or insights into the effectiveness of the proposed approach, potentially drawing connections to existing theories or principles in machine learning or cognitive science.

Please see our response in <Questions 5>.

> 10. (Weaknesses 2) The method relies on the availability of a well-trained comparator S, which may not be feasible or practical in certain scenarios.

Our method has been proven to work very well on small-size, fine-grained datasets (e.g. CUB-200, Dogs-120) where the comparator S can be trained successfully.

For large-scale, general domains, we can leverage existing state-of-the-art image similarity metrics like [DreamSim](https://arxiv.org/pdf/2306.09344) or [DINO](https://arxiv.org/abs/2104.14294) that were trained on large-scale and contrastive data.
In the Related Work section of [DreamSim paper](https://arxiv.org/pdf/2306.09344), you can find several other alternative metrics for general domains (e.g. ImageNet).
Adapting these metrics in our PoE model is straightforward as what we did with cosine or Earth Mover's Distance in Tab. 7 in the paper.

Again, we thank the Reviewer for the great suggestions, and we have included them in the latest revision of the paper.
