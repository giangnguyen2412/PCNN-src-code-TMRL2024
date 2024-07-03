We sincerely thank the Reviewer for the valuable and insightful feedback!
Below, please see our responses to your comments and questions:

> (Weaknesses 1 in Presentation) The message comprises two disconnected pieces: 1) showing explanations for the top-K classes aids human decision making, 2) reranking predictions using NNs helps machine performance. This is not a big deal, but a more streamlined message would have helped. 

Thank you for the suggestion.
We want to note that a unified take-away from our work is that `showing PCNN explanations for the top-K classes improves both humans and AI accuracy`.
Reranking is a novel method to leverage these PCNN explanations to improve AI accuracy. 
More specifically, PCNN explanations help train the image comparator S (the AI), which is later used in the re-ranking algorithm CxS to improve the overall top-1 classification accuracy.

After re-reading the paper, we agree that the message could be more streamlined.
To connect the two pieces, we added a message in the beginning of Sec 4. Results (page 6) of the latest revision that:

`
In this section, we will show that PCNN explanations improve both AI and human accuracy. Regarding
improved AI accuracy, we demonstrate that PCNN explanations can be used to both train fine-grained image
comparator S and to re-rank then correct wrong predictions of a pretrained classifier C. For improved human
accuracy, we show that when shown PCNN explanations, humans improve their accuracy in distinguishing
between correct and incorrect predictions by almost +10 points
`

> (Weaknesses 2 Presentation) Another issue is the method name is not used consistently throughout the text. For instance, it is sometimes called C \times S, sometimes PoE; sometimes PCNN is an architecture, sometimes a new type of explanation; this is a bit confusing. I would prefer if the authors used PCNN everywhere, for simplicity.

We agree with this point and to reduce the confusion, we will use the term `CxS` consistently throughout the text.

For PCNN, indeed, it is a variant of nearest-neighbor explanations. After revising the whole paper, we have made sure to use the term `PCNN` consistently to refer to the nearest-neighbor explanations.
We just noted that PCNN can be confused with CNN, which is convolutional neural networks, and indeed an architecture.
If the Reviewer sees this as a potential issue, we can consider adding a footnote to clarify this.

> (Requested Changes 1) The authors are upfront about the fact that PCNN requires training data at test time (like all kNN-based predictors), but do not list this as an actual limitation in Sec 5, while I think it is.
There they mention run-time of PCNN is longer than other competitors, but the main issue is PCNN requires the training data to be available in entirety -- or, at least, the experiments do not study the impact of reducing the sice of the training data on inference-time performance. This is a clear downside compared to, say, ProtoPNets, which memorize relevant (part) prototypes instead.
So there exists a clear trade-off between space and time requirements and prediction improvements (which is substantial but not huge to begin with, usually in the order of 2-3% top-1 accuracy over the runner-ups, at least according to Tables 3-5).

Thank you for bringing up this excellent point! Indeed, requiring the whole training data at test time is a significant downside.
The comment from the Reviewer encouraged us to investigate the impact of reducing the size of the training data on inference-time performance.
We present the experimental data on CUB-200 and Dogs-120 below.

| Dataset   | % Data | Samples per Class | Top-1 Acc(%) | Runtime (s) |
|-----------|--------|-------------------|--------------|-------------|
| CUB-200   | 100%   | 30                | 88.43        | 64.55       |
| CUB-200   | 50%    | 15                | 88.26        | 59.70       |
| CUB-200   | 33%    | 10                | 88.19        | 58.08       |
| Dogs-120  | 100%   | 100               | 86.27        | 87.18       | 
| Dogs-120  | 50%    | 50                | 86.32        | 71.02       |
| Dogs-120  | 33%    | 33                | 86.42        | 65.52       |

**_The top-1 accuracy and runtime of CxS on CUB-200 and Dogs-120 for different sizes of training data during inference. Runtime was computed over 1000 samples, similar to the setup in Appendix E._**

We found that reducing the size of the training data has little-to-no impact on the inference-time performance.
When keeping the same accuracy, we can reduce the runtime by 10% on CUB-200 and 24.9% on Dogs-120 by reducing the training set to 33% of the original size.


Regarding the runtime of PCNN being longer than other competitors, we also attempted to reduce the overhead introduced by the re-ranking process by reducing the number of queries to the image comparator S.
Currently, we are always examining the top-`K` (with `K=10`) most probable classes.
Yet, there always exists classes assigned a `< 1%` probability by ResNet50 (`Fig. 5` in PCNN submission) and are not in the top-1 after re-ranking.
Reducing the number of `K` can save a lot of computation at a minimal cost of accuracy.

To verify this, we run an experiment for CUB-200 where we instead of re-ranking the whole **top-10**, we only re-rank the classes that have a probability `>= 1%` assigned by the base classifier C.
We found that:
- The PoE model performance on CUB-200 drops very marginally by `only 0.08%` (from `88.59%` → `88.51%`).
- However, the number of queries to the image comparator S was reduced by approx. `4x` (from `10` to just about `2.5` queries/image).
This leads to a `2.5x` speedup in the overall runtime of the PoE system (from `64.55` seconds to `28.95` seconds per 1000 images), as shown in the following Table.

| Model                        |   Time (s)   | Top-1 Acc (%) |
|:----------------------------:|:------------:|:-------------:|
| RN50 xS (before)                      | 64.55 ± 0.35 |     88.59     |
| RN50 xS (after) | 28.95 ± 0.11 |     88.51     |

**_The run-time of PoE on 1,000 queries on one Nvidia V100 GPU._**

We added these details in Sec. E of the latest revision.

> (Requested Changes 2) I don't think the results of the user study are reported appropriately in the introduction.
Bottom line: showing more NNs to users makes them more skeptical, meaning they end up underestimating machine perforance. 
This should be clearly stated in the introduction, at the bare minimum. 
Instead, the authors focus on the benefits of PCNN only, and write: "A 60-user study finds that, compared to showing top-1 class examples, PCNN improves user performance on the distinction task by almost +10 points (pp) (54.55% vs. 64.58%) on CUB-200 (Sec. 4.6)." I don't think this is entirely fair and the text should be amended. This should also be listed in the Limitations section.

Thank you for this thoughtful comment!

We agree with the Reviewer that the introduction should clearly state the bottom line of the user study.
We attribute the improvement in user accuracy to the reduced over-reliance (the established term) [1] where users are more skeptical of the machine predictions when shown PCNN explanations.
This observation also aligns with the findings of Buçinca et al. [2] and Bansal et al. [3] that reduced over-reliance helps improve AI-assisted decision-making.

We added this bottom-line message to the latest version of our submission:

In Abstract:
`
Also, a human study finds that showing lay users our probable-class nearest neighbors (PCNN)
reduces over-reliance on AI, thus improving their decision accuracy over prior work which
only shows only the top-1 class examples.
`

In Introduction:
`
A 60-user study finds that PCNN explanations, compared with top-1 class examples, reduce over-
reliance on AI, thus improving user performance on the distinction task by almost 10 points (54.55%
vs. 64.58%) on CUB-200 (Sec. 4.6).
`

In Sec 4.6. (page 12):
`
Our finding aligns with the literature that showing explanations helps users reduce
over-reliance on machine predictions Buçinca et al. (2021); Schemmer et al. (2023); Chen et al. (2023a).
`

[1] Appropriate reliance on AI advice: Conceptualization and the effect of explanations
[2] To trust or to think: cognitive forcing functions can reduce overreliance on AI in AI-assisted decision-making.
[3] Does the whole exceed its parts? the effect of ai explanations on complementary team performance.

> (Requested Changes 3) The construction of the training set for S assumes C is already reasonably high-quality: is this always a reasonable assumption? Please add this to the Limitations section too.

Our answer is: Classifiers C is NOT strictly required to be `reasonably high-quality`.
We tested three cases: (1) using high-performing classifiers C (e.g. >= 80% top-1 acc), (2) low-performing classifiers C (e.g. ~ 60% top-1 acc), and (3) excluding classifiers C from sampling process.

(1) We perform sampling using high-performing classifiers C (e.g. ResNet-50 scores 85.83% on CUB-200, 89.73% on Cars-196, and 85.82% on Dogs-120) to train S and report numbers in Table 1 of the latest revision.
We often see improvements from 1-3 points when comparing CxS to C.

(2) We also test the case where we use low-performing classifiers C (e.g. ResNet classifiers pretrained on ImageNet score 60% to 63% on CUB-200) and report numbers in Table 1.
It is interesting that we see even much bigger improvements in this case, e.g. up to almost 12 points on CUB-200.

(3) We also show that we can exclude C in sampling. Please refer to the paragraph `Hard negatives are more useful than easy, random negatives in training S` in Sec 4.2.
In this experiment, the negatives are randomly sampled from non-ground-truth classes, and the positives are the ground-truth class, defined by training annotations.
This sampling still yield positive improvements in the top-1 accuracy for a CUB-200 iNaturalist-pretrained ResNet-50 from 85.85% → 86.55% (+0.70%).


> (Requested Changes 4) An analysis of errors introduced by the reranking step would have been useful.

Thank you for this suggestion!

To analyze where reranking failed, we visualized 700 CUB-200 samples that were misclassified by our re-ranking method (CxS 88.59% in Table. 1). You can also access them [here](https://drive.google.com/drive/folders/1CJRAmuoBSLQQ2ES0dmCbiTDTCWZvewzM?usp=sharing).

After manually inspecting the samples, we found that the majority of the misclassifications were due to the following reasons:
1. Inter-class similarities: The species from different classes look very similar to each other. 
2. Intra-class variations: The species from the same class have large variations in appearance because of  lighting or angles. This is expected because the birds are captured in the wild.
3. Low initial confidence scores: The initial classifier C assigns too low confidence scores to the correct class because of using softmax. Therefore, even if the comparator S assigns high confidence scores, the CxS score is not sufficient to be recognized as the top-1.

We believe that both (1) and (2) are inherent to the dataset and not specific to our method. However, a possible remedy is data augmentation. We did not explore data augmentation extensively in this work, but we showed it can benefit S in Sec. 4.2.
(3) could be addressed by re-normalizing the confidence scores of C or more advanced re-ranking methods.

If the Reviewer has any specific experiments in mind, we would be happy to run them and report the results!

> (Requested Changes 5) p 4: "We empirically test K = {1, 3, 5, 10} and find K = 10 to be optimal." The fact that the optimal value is at the very end of the spectrum begs the question whether increasing K could improve performance further. Did the authors evaluate what happens for larger values of K? Clearly, increasing K would not be ideal for human decision making, but it should not consistute for PCNNs proper.

Thank you for this valuable comment!
We re-run our method with different values of K and present the results below.

| K  | CUB Perf (%)  | CUB Runtime (s) | Dogs Perf (%)         | Dogs Runtime(s) |
|----|---------------|-----------------|-----------------------|-----------------|
| 1  | 85.83         | 8.81            | 85.82                 | 8.81            |
| 2  | 87.95 (+2.12) | 27.72           | 86.06 (+0.24)         | 50.35           |
| 3  | 88.28 (+2.45) | 32.32           | 86.03 (+0.21)         | 54.96           |
| 5  | 88.28 (+2.45) | 41.53           | 85.91 (+0.09)         | 64.16           |
| 10 | 88.42 (+2.59) | 64.55           | 86.27 (+0.45)         | 87.18           |
| 15 | 88.00 (+2.17) | 87.57           | 85.86 (+0.04)         | 110.20          |

**_The top-1 accuracy and runtime of CxS on CUB-200 and Dogs-120 for different K values. Runtime was computed over 1000 samples, similar to the setup in Appendix E._**

We found that increasing K does not only hurt the classification accuracy but also increases the runtime.
Still, using K = 10 strikes the optimal balance between accuracy and runtime.

> (Requested Changes 6) p 2 onwards: The authors say their model is a Product of Experts (PoE), based on the definition of Hinton, 1999. Reading through this reference, however, I get the impression that in PoEs the various distributions are conditionally independent given the input (for instance, in p 4 of Hinton '99, they state "the hidden states of different experts are conditionally independent given the data"). The same (conditional) independence assumption seems to be instrumental in more recent research on PoEs, see:
Gordon et al., "Identifiability of Product of Experts Models." 2024.
To me, conditional independence seems necessary to reinterpret the product of distributions as a factorization of a more complex joint distribution, which lies at the heart of PoEs.
However, independence does not appear to hold for the CSP model. I would appreciate if the authors could clarify this point, and -- if independence is indeed a prerequisite of PoEs -- changed their wording accordingly.

Thank you for this insightful comment!

First, let us reiterate the definition of PoE: 
`A PoE model combines multiple probability distributions (experts) to form a more complex joint distribution, where each expert captures different aspects of the data, and their product forms the overall model.`
Indeed, we agree with the Reviewer that conditional independence lies at the heart of PoEs because this enables factorization of distributions.

Let's re-visit our CxS model:

Model C: A pre-trained classifier producing a probability distribution over classes for a given input image.
Model S: Image comparator, which compares an input image with its nearest neighbors and generates confidence scores.

To verify if C and S can be considered experts in a PoE framework, we need to determine if their outputs are conditionally independent given the input data.
Conditional independence implies that the probability distribution over classes from C should not influence the probability distribution from S, once we know the input image.

However, the P(C) and P(S) are not conditionally independent given an input image. 
This is because C produces a class distribution, and S refines this distribution based on nearest neighbors. The confidence score S assigns can be influenced by the initial ranking from C.
Due to the dependency between C and S, our model does not strictly follow the PoE framework's requirement.

Given that, we changed the wording throughout in the latest version. We truly appreciate the Reviewer for pointing out this discrepancy!

We also added the connection between our model to PoE into Sec. 2 (Related Work) at page 3 that:
`
A key difference from standard PoE and boosting techniques is that we leverage training-set examples (PCNN) at the test time of S, improving CxS model accuracy further over the baseline C.
Also, our model does not strictly follow the PoE framework's requirement of conditional independence between experts because the confidence scores that image compartor S assigns to most-probable classes can be influenced by the initial ranking from C.
`

> (Broader Impact) Broader impact should briefly discuss the potential impact of manipulating (in the case of PCNN, lowering) user confidence in machine predictions in, especially in time critical high-stakes scenarios.

Thank you for the suggestion.
Indeed, trust calibration is a critical aspect of human-AI interaction, especially in high-stakes scenarios per the Reviewer's comment.
We included a discussion on the potential impact of manipulating user confidence in machine predictions in the Sec 6. Discussion and Conclusion in the latest revision.