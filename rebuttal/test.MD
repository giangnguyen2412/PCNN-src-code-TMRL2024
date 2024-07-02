\subsec{Task 1: Single-label, many-way image classification}
Let \classifier be a frozen, pretrained image classifier that takes in an image $x$ and outputs a softmax probability distribution over all $c$ possible classes, e.g., $c=200$ for CUB-200 \cite{wah2011caltech}.
Let \comparator be an image comparator
that takes in two images and outputs a sigmoid score predicting whether they belong to the same class (Fig.~\ref{fig:architecture}).
Our goal is to improve the final classification accuracy without changing \classifier by leveraging a separate image comparator network and PCNN (\cref{fig:re-ranking_algo}).


\subsec{Task 2: Distinction task for humans}
Following \citet{nguyen2021effectiveness,taesiri2022visual}, we provide each user with the input query, the top-1 prediction given by \classifier, and an explanation (e.g., five PCNN images; \cref{fig:accuracy_breakdown_maintext}) and ask the user to accept or reject the top-1 predicted label.

In this section, we will show that PCNN explanations improve both AI and human accuracy.
Regarding improved AI accuracy, we demonstrate that PCNN explanations can be used to both train fine-grained image comparator \comparator and to re-rank then correct wrong predictions of a pretrained classifier \classifier.
For improved human accuracy, we show that when shown PCNN explanations, humans improve their accuracy in distinguishing between correct and incorrect predictions by almost \increasenoparent{10} points.