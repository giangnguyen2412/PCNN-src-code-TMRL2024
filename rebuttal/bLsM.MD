Thank you very much for your valuable and insightful feedback!
Below, please see our responses to your comments.

> (Weaknesses 1 in Presentation) The message comprises two disconnected pieces: 1) showing explanations for the top-K classes aids human decision making, 2) reranking predictions using NNs helps machine performance. This is not a big deal, but a more streamlined message would have helped. 

Thank you for the suggestion.
We want to note that a unified take-away from our work is that `showing explanations for the top-K classes improves both humans and AI performance`.

Yet, after re-reading the paper, we agree that the message could be more streamlined.
In particular, we see the message that `showing explanations for the top-K classes` improves AI performance can be clearer.
We plan to elaborate in the next revision that: PCNN explanations help the AI model during training (i.e. training image comparator S) → eventually improve the overall top-1 classification accuracy of AI via re-ranking.

> (Weaknesses 2 Presentation) Another issue is the method name is not used consistently throughout the text. For instance, it is sometimes called C \times S, sometimes PoE; sometimes PCNN is an architecture, sometimes a new type of explanation; this is a bit confusing. I would prefer if the authors used PCNN everywhere, for simplicity.

We agree with this point and to reduce the confusion, we will use the term `CxS` consistently throughout the text.

For PCNN, indeed, it is a variant of nearest-neighbor explanations, and we will make sure to clarify this in the text.

> (Requested Changes 1) The authors are upfront about the fact that PCNN requires training data at test time (like all kNN-based predictors), but do not list this as an actual limitation in Section 5, while I think it is.
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
Regarding the rune-time of PCNN being longer than other competitors, we also attempted to reduce the overhead introduced by the re-ranking process (per suggested by Reviewer `g7wm`) in this [response](https://openreview.net/forum?id=OcFjqiJ98b&noteId=XuB3bY6d9q).

TL;DR: We can speed up the inference time by `2.5x` times without sacrificing the accuracy.

> (Requested Changes 2) I don't think the results of the user study are reported appropriately in the introduction.
Bottom line: showing more NNs to users makes them more skeptical, meaning they end up underestimating machine perforance. 
This should be clearly stated in the introduction, at the bare minimum. 
Instead, the authors focus on the benefits of PCNN only, and write: "A 60-user study finds that, compared to showing top-1 class examples, PCNN improves user performance on the distinction task by almost +10 points (pp) (54.55% vs. 64.58%) on CUB-200 (Sec. 4.6)." I don't think this is entirely fair and the text should be amended. This should also be listed in the Limitations section.

Thank you for this thoughtful comment. 

We agree on that `showing more NNs to users makes them more skeptical` (and we will revise accordingly in the next revision), but stating the bottom line is `human users end up underestimating machine performance` is strong and not supported by our data.

This skeptical behavior can be attributed to other possible reasons.
For example, we tried the human experiments ourselves and found that the reason we tend to accept less and reject more is that
we see more similar species in classes top-2, top-3, etc. than in top-1.
In this case, the bottom line might be `PCNN provides richer and contrastive information` rather than `humans ending up underestimating machine performance`.

While the concern about the bottom line is very interesting and indeed important; honestly, we did not collect this perception data to give a definite answer.
We will make sure to clarify this in the paper.

> (Requested Changes 3) The construction of the training set for S assumes C is already reasonably high-quality: is this always a reasonable assumption? Please add this to the Limitations section too.

Our answer is: Classifiers C is NOT strictly required to be `reasonably high-quality`.
We tested three cases: (1) using high-performing classifiers C (e.g. >= 80% top-1 acc), (2) low-performing classifiers C (e.g. ~ 60% top-1 acc), and (3) excluding classifiers C from sampling process.

(1) We perform sampling using high-performing classifiers C (e.g. ResNet-50 scores 85.83% on CUB-200, 89.73% on Cars-196, and 85.82% on Dogs-120) to train S and report numbers in Table 1.

(2) We also test the case where we use low-performing classifiers C (e.g. ResNet classifiers pretrained on ImageNet).

(3) We also show that we can exclude C in sampling. Please refer to the paragraph `Hard negatives are more useful than easy, random negatives in training S` in Sec 4.2.
In this experiment, the negatives are randomly sampled from non-ground-truth classes, and the positives are the ground-truth class, defined by training annotations.
This sampling still yield positive improvements in the top-1 accuracy for a CUB-200 iNaturalist-pretrained ResNet-50 from 85.85% → 86.55% (+0.70%).


> (Requested Changes 4) An analysis of errors introduced by the reranking step would have been useful.

Thank you for the suggestion.

We visualized 700 CUB-200 samples that were misclassified by our re-ranking method (CxS 88.59% in Table. 1) and shared with you [here](https://drive.google.com/drive/folders/1CJRAmuoBSLQQ2ES0dmCbiTDTCWZvewzM?usp=sharing).
After manually inspecting the samples, we found that the majority of the misclassifications were due to the following reasons:
1. Inter-class similarities: The species from different classes look very similar to each other. 
2. Intra-class variations: The species from the same class have large variations in appearance because of  lighting or angles. This is expected because the birds are captured in the wild.
3. Low initial confidence scores: The initial classifier C assigns too low confidence scores to the correct class because of using softmax. Therefore, even if the comparator S assigns high confidence scores, the CxS score is not sufficient to be recognized as the top-1.

We believe that both (1) and (2) are inherent to the dataset and not specific to our method. However, a possible remedy is data augmentation. We did not explore data augmentation extensively in this work, but we showed it can benefit S in Sec. 4.2.
(3) can be addressed by re-normalizing the confidence scores of C or more advanced re-ranking methods.

If the Reviewer has any specific experiments in mind, we would be happy to run them and report the results!

> (Requested Changes 5) p 4: "We empirically test K = {1, 3, 5, 10} and find K = 10 to be optimal." The fact that the optimal value is at the very end of the spectrum begs the question whether increasing K could improve performance further. Did the authors evaluate what happens for larger values of K? Clearly, increasing K would not be ideal for human decision making, but it should not consistute for PCNNs proper.

Thank you for this valuable comment!
We re-run our method with diffrent values of K and present the results below.

| K  | CUB Perf (%)  | CUB Runtime (s) | Dogs Perf (%)         | Dogs Runtime(s) |
|----|---------------|-----------------|-----------------------|-----------------|
| 1  | 85.83         | 8.81            | 85.82                 | 8.81            |
| 2  | 87.95 (+2.12) | 27.72           | 86.06 (+0.24)         | 50.35           |
| 3  | 88.28 (+2.45) | 32.32           | 86.03 (+0.21)         | 54.96           |
| 5  | 88.28 (+2.45) | 41.53           | 85.91 (+0.09)         | 64.16           |
| 10 | 88.42 (+2.59) | 64.55           | 86.27 (+0.45)         | 87.18           |
| 15 | 88.00 (+2.17) | 87.57           | 85.86 (+0.04)         | 110.20          |

**_The top-1 accuracy and runtime of CxS on CUB-200 and Dogs-120 for different K values. Runtime was computed over 1000 samples, similar to the setup in Appendix E._**

We found that increasing K does not only hurt the classification performance but also increases the runtime.
Still, using K = 10 strikes the optimal balance between performance and runtime.

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

Given that, we will change the wording in the next revision. We truly appreciate the Reviewer for pointing out this discrepancy!

> (Broader Impact) Broader impact should briefly discuss the potential impact of manipulating (in the case of PCNN, lowering) user confidence in machine predictions in, especially in time critical high-stakes scenarios.

Thank you for the suggestion.
Indeed, trust calibration is a critical aspect of human-AI interaction, especially in high-stakes scenarios per the Reviewer's comment.
We plan to add a section in the Broader Impact in the next revision of the paper. We are also open to suggestions on how to mitigate 
the potential negative impact of lowering user confidence in machine predictions.