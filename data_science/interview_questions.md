# Interview Questions
## Machine Learning
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

### Explain PCA.
PCA is an unsupervised method typically used for dimensionality reduction. This method finds the directions of greatest variation in feature space and uses those to (frequently) more efficiently represent the input data. Mechanically, this is done by using the singular value decomposition and choosing the directions corresponding to the largest *d* singular values, where d is the number of dimensions of the resulting feature vector (this is chosen by, perhaps, a scree plot). The original feature vectors are then projected onto the subspace defined by those directions to get the new dimension-reduced features.

### Why is ReLU better and more often used than the sigmoid activation function in NNs?
The sigmoid activation function *saturates*, which means that gradient flow can be interrupted during backpropagation, leading to slow training. This is a result of the tails of the sigmoid function, which have small derivatives, and how that affects derivatives further back through multiplication thanks to the chain rule.

It is also computationally faster to compute ReLU gradients than sigmoid gradients, leading to more speed benefits.

ReLU has also been found, in general, to simply perform better than sigmoid activations in many situations; its adoption has mostly been driven by empirical, rather than theoretical, benefits.

## References
1. [DS and ML Interview Questions](https://towardsdatascience.com/data-science-and-machine-learning-interview-questions-3f6207cf040b)
