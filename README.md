# Character-Level GPT

A character-level GPT model pre-trained on the enwik8 dataset. The aim was twofold; to code a transformer-based language model with minimal assistance from PyTorch; and to test training methods that were potentially effective with fewer iterations given the limited resources available, namely the 1cycle learning rate policy outlined in _Super-Convergence - Very Fast Training of Neural Networks Using Large Learning Rates_ (Smith & Topin, 2018). A character-level model was chosen for simplicity.

The model architecture was largely based on GPT-2, as described in _Language Models are Unsupervised Multitask Learners_ (Radford et al, 2019), as well as the character-level model described in _Character-Level Language Modeling with Deeper Self-Attention_ (Al-Rfou et al, 2018). More details on the properties that were adopted from each of these models is given below. 

The model achieved a result of 1.64bpc on the test set, with a next-character prediction accuracy of 68.37%. This seems a reasonable result compared to the 1.18bpc of the equivalent model in _Character-Level Language Modeling with Deeper Self-Attention_ (on the same dataset), considering that that model was trained to 8 milion steps - 167 times that of this model.

This repository contains the model code (model.py), the main training and inference functions (train.py), as well as the trained model's state dictionary  and the train/val/test split of enwik8 used (data.zip). There is also a simple accuracy test in main.py that can be run to test the model's next-character prediction accuracy over a chosen number of randomly selected test set examples.


## Model Architecture

The underlying transformer architecture is largely my own code, based on the original _Attention Is All You Need_ paper from 2017. Howeever basic PyTorch modules such as the Linear, Embedding, and Parameter modules were used for convenience. Early versions included an entirely self-coded attention module, however for computation efficiency it became necessary to use PyTorch's scaled dot product attention function, as well a some other code rewrites influenced by [nanoGPT](https://github.com/karpathy/nanoGPT).

The model's configuration parameters (eg. context length, hidden vector dimension) were adopted from _Character-Level Language Modeling with Deeper Self-Attention_, as was the per-layer positional encoding, choice of ReLU as the activation, and the application of dropout. All other architecture choices were based on GPT-2, for example weight initialization, and the location of layer normalization.


## Dataset

The model was trained on the [enwik8](http://prize.hutter1.net/index.htm) dataset, in the manner of the unsupervised pre-training used for GPT-1, as described in _Improving Language Understanding by Generative Pre-Training_ (Radford et al, 2018). It consists of the first 100 millon characters of the English Wikipedia in XML format, from which sequences of characters were randomly chosen as contexts.


## Training

The model was trained over 48000 iterations using a T4 GPU (via Google Colab). Much of the training methodology was motivated by the limited computing resources available, the most significant example being the use of the 1cycle training policy. This attempts to make use of "super-convergence" by linearly increasing the learning rate to a value significantly higher than would usually be effective, and then decreasing over an equal number of iterations to the starting point. Usually the learning rate would then be decreased further at a slower rate, however this was omitted as it was not showing strong improvements when initally attempted.

Stochastic gradient descent with momentum was used as the optimization algorthm, mainly to align with the research on the 1cycle policy. Momentum was varied in opposition to the learning rate, ie. decreased and then increased.

The large learning rate reached during training acts as a regularizer, and so with dropout also adding regularization, a relatively small weight decay was used.

The hyperparameter values were:
learning rate   [0.0001, 0.1]
momentum        [0.9, 0.8]
weight decay    0.00001

Appropriate values for the minimum and maximum learning rates were determined with the learning rate range test as described in _Cyclical Learning Rates for Training Neural Networks_ (Smith, 2017), which requires training over a number of iterations while linearly increasing the learning rate between minimum and maximum values that are expected to be too extreme. Suitable minimum and maximum learning rates are determined by tracking the validation accuracy; as the point at which the accuracy begins to increase is a suitable minimum and the point at which the accuracy curve flattens is a suitable maximum.

Momentum and weight decay values were determined by monitoring the validation loss and accuracy in the early stages of training over a range of values for each. It was clear early that relatively lower momentum values were needed, however the performance remained fairly even at a variety of weight decay values well into the training process, meaning multiple models were trained until it was clear that any of the remaining vlaues for weight decay were acceptable.



