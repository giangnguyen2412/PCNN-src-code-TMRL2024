We sincerely thank the Reviewer for the thoughtful and constructive feedback. 
Below, we address each of the Reviewer's questions and suggestions.

> 1. (Questions 1) How does the proposed method perform on tasks other than fine-grained image classification, such as object detection or semantic segmentation (the paper is 47 pages long so hopefully I didn’t miss any results)?

Thank you for bringing this interesting question!

To answer this question, we first do a literature review on the use of nearest neighbors (NNs) for [object detection](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C1&as_ylo=2020&q=%22object+detection%22+%2B+%22nearest+neighbor+samples%22&btnG=) and [semantic segmentation](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C1&q=%22semantic+segmentation%22+%2B+%22nearest+neighbor%22&btnG=).

What we found is that literature has not directly used nearest-neighbor examples to either train a model or perform re-ranking or refinement in the context of object detection and semantic segmentation tasks.
Instead, nearest-neighbor samples often play a supplementary role or are used in specific subcomponents (e.g. papers [1](https://openaccess.thecvf.com/content/CVPR2023W/GCV/papers/Lin_Explore_the_Power_of_Synthetic_Data_on_Few-Shot_Object_Detection_CVPRW_2023_paper.pdf),[2](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8967762)) rather than being the primary algorithm.

Therefore, we see a big gap between our method vs. the existing literature and think that the proposed method would need extensive adaptation or further development to be applied to object detection or semantic segmentation tasks.

> 2. (Questions 2) Can the re-ranking algorithm be extended to handle multi-label classification problems?

First, we want to reiterate the definition of multi-label classification to smoothen the discussion:

**Multi-label classification**: is a type of machine learning problem where each instance (or example) can 
be assigned multiple labels from a set of possible labels, rather than just one. 
This differs from the traditional single-label classification tasks, where each instance is 
associated with only one label.

We took the below Figure from [C-Tran paper](https://arxiv.org/pdf/2011.14027) and will base on it to give examples.

![](images/person%200.83.png)

Here is an example of prediction in multi-label image classification. 
The ground-truth y is a binary vectors indicating the presence of ℓ classes {y1, y2, ..., yℓ}, yi ∈ {0, 1}. {y1, y2, ..., yℓ}, yi ∈ {0, 1}.

Then, for each label, the two classes to be re-ranked here are the presence (1) and absence (0) of the object (i.e. person).
To extend our re-ranking algorithm to handle multi-label classification problems, we compare the input image with the nearest neighbors from the two classes, presence and absence,
using the image comparator S.

For example, S will compare the input image with the nearest neighbors from the person class (1) and the no-person class (0) from the training data.
As we know that S returns a similarity score in the range of [0, 1] indicating the likelihood of both the two images containing person.
The PoE will work by multiplying the output similarity score with the predicted probability (0.83) of the base classifier C-Tran for the person class.
The final predictions will be thresholded at a certain value (e.g. 0.5) to determine the presence of the object.

Another problem to concern is the number of classes to be re-ranked. 
In our submission, we propose to re-rank the top-10 classes predicted by the base classifier C.
Yet, in multi-label classification, the re-ranking algorithm could only care about the classes that 
are suggested by the pretrained classifier (e.g. classes having >= 0.5 confidence score, person and umbrella in the above Figure).

This could help reduce the False Positive rate in multi-label classification tasks, where the model predicts the presence of an object that is not in the image.

> 3. (Questions 3) How does the performance of the PoE model scale with the number of classes or the size of the dataset?

This is a very interesting question!
First, we want to confirm that the reviewer refers to the test-time accuracy of the PoE model when mentioning "performance".

We think that there are three main factors that primarily affect the performance of the PoE model:

- a. The number of classes (mentioned by the reviewer)
- b. The size of the dataset (mentioned by the reviewer)
- c. The performance of the base classifier C because the PoE model is a combination of the base classifier C and the image comparator S.

When the number of classes increases (a) (e.g. from 200 in CUB to 1000), the performance of the PoE model will likely decrease.
This is because having more classes will make the image comparator S more likely sees visually similar objects from different classes (e.g. Tern species in Fig.2 look very similar),
making the re-ranking weights more uniform across the classes, and thus reducing the effect of the image comparator S on the final predictions.
However, the final PoE performance will be also affected by the performance of the classifier C, which will be changed accordingly to the change in the number of classes as shown in Fig. 6.
If the reviewer has any specific experiments in mind, we are more than happy to run.

![](images/img_3.png)

When the size of the dataset increases (b), the performance of the PoE model will likely increase because the image comparator S will have more data to learn from and 
combat with variations better. 
For example, in CUB-200 datasets, within a class, there are many variations in the appearance of the same bird ([males, females, and juveniles](https://www.allaboutbirds.org/guide/Painted_Bunting/)).
Having more data will help the comparator S more robust to these variations, thus improving the PoE model CxS.
In addition, having more data will help the base classifier C to yield better performance, which will also influence the impact of PoE as shown in Fig. 6.

![](images/img.png)

Regarding (c), we found that when C is low-performing, adding S (via PoE) will be very useful.
However, when C is already high-performing, adding S will not help much.
These two mentioned cases are respective to ImageNet-pretrained ResNet18 and NTS-Net in Fig. 6.

Finally, we are more than happy to run any requested experiments to address your concern in scaling PCNN.

> 4. (Questions 4) What is the impact of using different distance metrics or similarity measures for finding nearest neighbours on the performance of the PoE model?

Thank you for the insightful question! We suppose the reviewer refers to the test-time accuracy of the PoE model when he mentions the performance.

We want to break down the answer into two parts:

a. Given the NNs being found, we tested using different functions to measure the similarity between the input image and the NNs for reranking (See Tab. 7 in page 19).
![](images/img_1.png)
We found our comparator S works the best for the reranking algorithm.

b. Fixing the similarity function as S, we tested using different distance metrics to find the nearest neighbors for reranking.
Here we assess the PoE performance on cosine similarity and [DreamSim](https://dreamsim-nights.github.io/).
Compared to using L2 as reported in our paper, cosine similarity measure the global similarity between two images while DreamSim
is a human-aligned similarity measure.

| Distance function                              | CUB-200 PoE perf (%) | Dogs-120 PoE perf (%) |
|------------------------------------------------|----------------------|-----------------------|
| Random                                         | 87.95%               | 85.89%                |
| L2                                             | 88.59%               | 86.31%                |
| Cosine                                         | 88.33%               | 86.25%                |
| [DreamSim](https://dreamsim-nights.github.io/) | 88.38%               | 86.56%                |

[//]: # (| Random &#40;top-5 of DreamSim&#41;                     | 88.37%               | 86.34%                |)

We first test on CUB-200 and we found that different measures to retrieve NNs have a small to no impact on the PoE model performance.
We suppose that because the number of samples per class is fairly small (30 images per class on average), then when doing retrieval with different distance metrics, 
the nearest neighbors are not very different.

Then, we move from CUB-200 to Dogs-120 that has more samples per class (100 images per class on average).
Interestingly, we also found that different distance metrics have a small to no impact on the PoE model performance (see the above Table).

[//]: # (Our second hypothesis is that the image comparator S is robust to changes in the nearest-neighbor images it receives.)

[//]: # (This robustness is from the data augmentation technique &#40;[TrivialAugment]&#40;https://arxiv.org/pdf/2103.10158&#41;&#41; we applied during training &#40;See Sec. B7 in paper&#41;.)

[//]: # (To test this hypothesis, we first retrieve NNs using DreamSim and then randomly pick one sample from the top-5 NNs to do reranking.)

[//]: # (We found that the PoE model is robust to changes in the nearest-neighbor images it receives, that can explain why with different measures &#40;L2, cosine, DreamSim&#41; the PoE model performance is similar.)

We also test using the **Random nearest neighbors** (i.e. randomly picking one sample from each predicted class for re-ranking). 
Yet, it drops the PoE model performance from 88.59 to 87.95 on CUB-200 and from 86.31 to 85.89 on Dogs-120, respectively.

Overall, we found that changing the distance metrics for finding nearest neighbors has a small to no impact on the performance of the PoE model.
In addition, the Random baseline confirms the advantages of using the most similar images over the less-similar ones. 


> 5. (Questions 5) Are there any theoretical insights or principles that can guide the design and training of the image comparator S?

**The task of the image comparator** S is inspired by the [distinction task](https://arxiv.org/pdf/2112.03184) where we give human users two images 
and ask them to judge if the two images are of the same class or not.

**To compare the two images**, we are inspired by [Siamese neural networks](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) where the feature extractor
is shared between two branches and the similarity function could be learnable (see the Figure below).
![](images/img_5.png)

In the experiment in Tab. 7, we found that this architecture (shared feature extractor + 4-layer MLP for similarity function) works the best for PoE model.
![](images/img_2.png)


**In training of S**, we are inspired by a long line of work in [contrastive learning](https://arxiv.org/pdf/2002.05709) that aims to learn discriminative feature representations via 
same-class samples and different-class samples.
Yet, simple sampling techniques is not optimal for training S in an experiment shown in Sec. B.3.
The negative samples were randomly picked from the non-groundtruth classes.
We innovate by proposing a sampling strategy that selects the negative samples based on the softmax similarity of a pretrained classifier 
(i.e. improving PoE acc from 86.55% → 88.59% on CUB-200). 

> 6. (Requested Changes 6) In section 4.3 you say “First, we test using cosine similarity in the pretrained feature space of a CUB-200 classifier” – why do you call CUB-200 a classifier? CUB-200 is mentioned 80+ across the paper.

Sorry for the confusion. We revised the writing and changed "CUB-200 classifier" to "classifier trained on CUB-200" in the paper to avoid any misunderstanding.

> 7. (Potential Changes 1) Explore techniques to reduce the computational overhead introduced by the re-ranking process, such as efficient nearest neighbour search or pruning strategies.

Thanks for bringing up this point, and it really encourages us to optimize the runtime of our method! 
We agree that the re-ranking process can be computationally expensive, especially when the dataset is large.
The computational overhead comes from a) finding the nearest neighbors or b) querying the image comparator S for reranking weights.

We found that a) can be easily done by leveraging speedup options of faiss.
For example, we can reduce the dimensionality of embeddings vectors from 2048 to 1024 or 512 by using earlier layers of the ResNet-50 for feature extraction.
Also, as we are currently use CPU for faiss indexing (L137 in this [file](https://anonymous.4open.science/r/nearest-neighbor-XAI-FF2E/cub-200/cub_extract_feature_for_reranking.py)), using GPU by GpuIndexFlatL2 from faiss can provide significant speedups, especially for larger datasets.
Another option is to adjust the precision of computations (e.g., using floating-point 16 bits instead of 32 bits) which can increase speed at the cost of some accuracy.

[//]: # ([quantify how much speedup is])

For b), we can reduce the number of queries to the image comparator S by using a threshold on the reranking weights.
Currently, we are setting this to 10 as we always examine the top-10 most probable classes.
Yet, there always exists classes receiving < 1% by ResNet50 (see Fig. 5), which are likely never to be the top-1 after reranking.
Reducing the number of K can save a lot of computation at a minimal cost of accuracy.

For that, we run an experiment for CUB-200 where we instead of re-ranking the whole top-10, we only re-rank the classes that have a probability >= 1% by the base classifier C.
By doing so, we found:
- The PoE model performance on CUB-200 drops very marginally by 0.08% (from 88.59% to 88.51%).
- However, the runtime of the re-ranking process is significantly improved by 77.32% (from 46.04 seconds to only 10.44 seconds), as reflected in the below Table.

| Model                    | Time (s)        |
|--------------------------|-----------------|
| RN50                     | 8.81 ± 0.14     |
| kNN                      | 9.70 ± 0.32     |
| S                        | 46.04 ± 0.04    |
| **S (with threshold)**       | 10.44 ± 0.04    |
| RN50 xS                  | 64.55 ± 0.35    |
| **RN50 xS (with threshold)** | 28.95 ± 0.35    |

Again, thanks for the suggestion, and we will include this in the revised version of the paper.

> 8. (Potential Changes 2) Extend the human study to other datasets and domains to further evaluate the interpretability and usefulness of PCNN explanations.

We agree that extending the human study to other datasets and domains can provide more insights into the interpretability and usefulness of PCNN explanations.
Therefore, we repeat the human study that compares top-1 nearest neighbors and PCNN explanations on Stanford Dogs-120 dataset.

- Samples: Similar to CUB-200 study detailed in Section H, we select 300 correctly classified and 300
misclassified query samples determined by the RN50 × S classifier, amounting to a total of 600 images for the study.

- [ ] Participants:
- [ ] Findings

The reviewer can try out the human study on Stanford Dogs-120 via this [link](TBA).

[need to anonymize the demo -e.g. video or anonymous link]

> 9. (Potential Changes 3) Provide a theoretical analysis or insights into the effectiveness of the proposed approach, potentially drawing connections to existing theories or principles in machine learning or cognitive science.

Please see our response in <Questions 5>.

> 10. (Weaknesses 2) The method relies on the availability of a well-trained comparator S, which may not be feasible or practical in certain scenarios.

Our method has been proven to work very well on small-size, fine-grained datasets (e.g. CUB-200, Dogs-120) where the comparator S can be trained successfully.

For large-scale, general domains, we can leverage existing state-of-the-art image similarity metrics like [DreamSim](https://arxiv.org/pdf/2306.09344) or [DINO](https://arxiv.org/abs/2104.14294) that were trained on large-scale and contrastive data.
In the Related Work section of [DreamSim paper](https://arxiv.org/pdf/2306.09344), you can find several other alternative metrics for general domains (e.g. ImageNet).
Using these metrics in our PoE model is straightforward as what we did with cosine or Earth Mover's Distance in Tab. 7 in the paper.

