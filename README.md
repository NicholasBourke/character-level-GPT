# Character-Level GPT

A character-level GPT model pre-trained on the enwik8 dataset. The aim was twofold; to code a transfomer-based language model with minimal assistance from PyTorch; and to test training methods that were potentially effective with fewer iterations given the limited resources available, namely the 1cycle learning rate policy outlined in _Super-Convergence - Very Fast Training of Neural Networks Using Large Learning Rates_ (Smith & Topin, 2018). A character-level model was chosen for simplicity.

The model architecture was largely based on GPT-2, as described in _Language Models are Unsupervised Multitask Learners_ (Radford et al, 2019), as well as the character-level model described in _Character-Level Language Modeling with Deeper Self-Attention_ (Al-Rfou et al, 2018). A more detailed accounting of the properties that were adopted from each of these models is given is the model architecture section.


## Model Architecture

The underlying transformer architecture is largely my own code, based on the original _Attention Is All You Need_ paper from 2017. Howeever basic PyTorch modules such as the Linear, Embedding, and Parameter modules were used for convenience. Early versions included an entirely self-coded attention module, however for computation efficiency it became necessary to use PyTorch's scaled dot product attention function, as well a some other code rewrites influenced by [nanoGPT](https://github.com/karpathy/nanoGPT).

The model's configuration parameters (eg. context length, hidden vector dimension) were adopted from _Character-Level Language Modeling with Deeper Self-Attention_, as was the per-layer positional encoding, choice of ReLU as the activation, and the application of dropout. All other architecture choices were based on GPT-2, for example weight initialization, and the location of layer normalization.


## Training
