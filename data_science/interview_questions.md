# Interview Questions
## General Machine Learning
### What is the trade-off between bias and variance?
The bias of a supervised machine learning model corresponds to its propensity to underfit, or to fail to pick up on "signal" in the training data. Models with "high bias" tend to not be able to represent the optimal prediction function given the training data. A linear regression model, for example, can only represent linear functions; therefore any time the relationship between the x- and y-variables is not linear, this model will provide predictions that are far from the optimal ones.

The variance of a model, on the other hand, is how susceptible it is to small changes in the training data. When using a high-variance model, a small change to the training set may result in large differences in learned parameter values and therefore predictions. High-variance models tend to overfit and pick up on noise in the training data.

The bias-variance trade-off says that models with high bias tend to have low variance and vice-versa. A linear regression model (high bias) will not tend to change much with small modifications to the training data (low variance). A neural network which has huge numbers of parameters, however, can represent highly nonlinear functions (low bias), but can change drastically across training sets.

### What is gradient descent?
Gradient descent is a numerical method to minimize a function, typically used when analytical methods are too difficult for some reason. Intuitively, it corresponds to a ball (where you're at at some iteration of the algorithm) rolling down a hill (the function you're optimizing). The advantages of GD are:
- When dealing with large datasets on typical loss functions, you can optimize using "mini-batches" rather than the whole thing at once. This reduces memory requirements while still maintaining good algorithm performance. It also has the benefit of adding some stochasticity to the training process, helping the algorithm escape local minima, saddle points, etc.
- Gradient descent is a first-order method; it only requires first derivatives. This is helpful when second derivatives require too much computing power or memory or are otherwise unavailable.

### Explain over- and under-fitting and how to combat them.
A supervised ML model overfits the data when it takes into account not just the "signal", representing the true relationship between x and y, but also "noise" that is present: randomness that corrupts the training data due to the measurement procedure or just the stochasticity of the real world. Underfitting is when the model fails to capture all of the signal present in the data. The goal is therefore for a model to neither overfit nor underfit; we would like the model to capture all of the signal and none of the noise. This would lead to optimal predictions (good approximations of the conditional expectation of y | x).

Underfitting is caused by using a model without a high enough capacity: That is, the model cannot represent the ideal prediction function. This is why linear regression, for example, tends to underfit. Choosing a more flexible model/class of functions should allow the algorithm to capture more of the signal.

Overfitting happens when the model has too much capacity or can represent too many different kinds of functions, including those which pick up on noise in the training set. The typical cure for this is regularization, which corresponds to restricting the number of possible functions (the hypothesis space) that can be represented by the model.

### How do you combat the curse of dimensionality?
Reduce the dimensionality of the features by using PCA, or perhaps nonlinear dimensionality reduction techniques (multidimensional scaling, isomap, etc.). Manual, domain-specific "feature engineering" can also reduce the dimensionality. Unsupervised "neural" methods like RBMs and autoencoders can also be used to reduce dimensionality.

### What is regularization and why do we use it? Give some examples of common methods.
Regularization is usually used to restrict the number of possible functions a model can represent and thereby combat overfitting and high variance. Some regularization methods also combat overfitting by introducing noise into the training process. Common methods include:
- L1 regularization, which encourages sparsity in model parameters.
- L2 regularization, which encourages smaller model parameters in general.
- Dropout regularization in NNs, which adds noise to the training process and requires successive NN layers to be robust to failures in previous layers.

### What is data normalization, and why do we need it?
Data normalization is used to shift and rescale values (perhaps to make feature means 0 and sample variances 1, for example). It is useful because it ensures that training is invariant to feature units (feet vs. meters, for instance) and weighs features solely based on their predictive power rather than unit specifics. It also can positively impact optimization procedures by affecting the conditioning of the problem. Finally, in NNs with sigmoid units, normalization can, early on in training, ensure that activation values remain close to 0 and therefore preserve gradient flow (this insight is what led to batch normalization, which is a more advanced version of the standard data normalization setup).

### Explain dimensionality reduction, where it's used, and its benefits.
Dimensionality reduction is most frequently used to reduce the number of features (or, equivalently, the length of the feature vector) used to represent each example in a dataset. This is useful because large numbers of features leads to the curse of dimensionality, where all of the training examples are far from each other in feature space. The CoD is a problem because supervised methods all fundamentally operate on the principle that examples close to each other in feature space are likely to have similar target values. Dimensionality reduction helps us get around this problem.

In practice, this is frequently done by finding a lower-dimensional manifold, whether linear or nonlinear and projecting the data onto it to get the new feature representations. This is what PCA does, for example.

Dimensionality reduction can also be used simply for data compression, without any particular aim toward improving supervised method performance.

(Note that DR also leads to faster computing and may allow us to plot datasets that were previously unplottable.)

### How do you know which ML model you should use?
First, make sure that the class of problem the model solves is the same as the class of problem you have. Don't try to use a "regression" algorithm for a classification problem (usually). Similarly, consider your loss function and make sure it's appropriate; perhaps L1 loss is preferable to L2 loss if outlier predictions shouldn't be penalized extra-hard, for instance. Use training and validation loss curves to monitor the training and detect over- and underfitting. In the underfitting case, use a model with greater capacity; when overfitting, use regularization to shrink the model. Use validation error or cross-validation to choose hyperparameter values; in extreme cases, the "hyperparameter" chosen may include the "model" itself. When possible, take advantage of additional information in the structure of the problem: a convolutional layer is frequently a good idea when doing image classification, for instance, because it takes advantage of the structure of image data.

From a statistical modeling standpoint, there are many parallels. In general, make sure that the assumptions made in the model are correct, or at least not clearly false unless you have a good reason to use them.

## Data Munging
### How do you handle missing or corrupted data in a dataset?
1. Get it from the source.
2. Use a method with built-in missing data handling.
3. Drop the offending examples or fields.
4. Use a default value.
5. (Categorical only) add "missing" as one of the categories.
6. Use a variant of the EM algorithm to dodge the problem.

### How to do EDA?
0. Throughout this process, be careful that your test set data remain completely separate from the model that you're training. Otherwise, they might end up in your training procedure and worsen your estimated generalization error.
1. Try to understand the provenance of the data (where they come from and how they've changed).
2. Format the data properly, especially dates.
3. Check for outliers and nonsense values (plots are helpful here).
4. Determine a strategy for missing data and the outliers and nonsense values. This is frequently related to step (1).
5. Make plots to "get a feel" for what the data look like. This frequently leads to insights that are later used in feature engineering. Relationships between variables can be plotted using SPLOMs, bar charts, etc., and are usually quite enlightening.
6. Simple modeling may be appropriate here, but this frequently goes beyond the "exploratory" phase of the data analysis.

## Specific Algorithms (Excluding NNs)
### Explain PCA.
PCA is an unsupervised method typically used for dimensionality reduction. This method finds the directions of greatest variation in feature space and uses those to (frequently) more efficiently represent the input data. Mechanically, this is done by using the singular value decomposition and choosing the directions corresponding to the largest *d* singular values, where d is the number of dimensions of the resulting feature vector (this is chosen by, perhaps, a scree plot). The original feature vectors are then projected onto the subspace defined by those directions to get the new dimension-reduced features.

## Neural Networks
### Why is ReLU better and more often used than the sigmoid activation function in NNs?
The sigmoid activation function *saturates*, which means that gradient flow can be interrupted during backpropagation, leading to slow training. This is a result of the tails of the sigmoid function, which have small derivatives, and how that affects derivatives further back through multiplication thanks to the chain rule.

It is also computationally faster to compute ReLU gradients than sigmoid gradients, leading to more speed benefits.

ReLU has also been found, in general, to simply perform better than sigmoid activations in many situations; its adoption has mostly been driven by empirical, rather than theoretical, benefits.

### Why use convolutions for images rather than just FC layers?
1. Less parameters -> less variance.
2. The modeling assumption encoded in the convolution operation, that pixels closer together should be more related than those far apart, is intuitively sensible for image data.
3. Because of (1) and (2), we expect that reducing the parameters in this way will not have a large negative effect on the algorithm's performance. That is, we're getting less bias AND less variance!
4. Convolutional layers add *shift-invariance* to the algorithm. A picture of a cat is just as much a picture of a cat if the cat moves two pixels to the left; this shift should have no effect on the prediction made by the algorithm.

### What makes CNNs translation invariant?
In convolutional (pooling) layers, the same kernels (pooling operations) are applied to each region of the image, with the same weights each time (not applicable to pooling layers). This means that insofar as the prediction is concerned, "all regions of the image are created equal". A cat right here will be detected just as well as a cat over there.

---
## Question Sources
1. [DS and ML Interview Questions](https://towardsdatascience.com/data-science-and-machine-learning-interview-questions-3f6207cf040b)
