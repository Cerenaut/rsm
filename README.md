# Recurrent Sparse Memory
The codebase for the Recurrent Sparse Memory (RSM) project.

## Dependencies
- [PAGI Framework](https://github.com/ProjectAGI/pagi) >= 0.1

## Getting Started
Ensure that you have `pagi` installed and that its accessible via the command-line by running `pagi --help`. Clone this
repository and install the package using `pip install -e .`

## Experiments
In addition to the hyperparameters available below (supplemental to the paper), the JSON definition files for these experiments that can be run with our framework are also available under the [definitions](definitions) directory.

### Distant cause & effect (Embedded Reber Grammar task)

#### Hyperparameters

| Hyperparameter  | Value |
| ------------- | ------------- |
| Batch size  | 400 |
| `g` (groups) | 200 |
| `c` (cells per group) | 6 |
| `k` (sparsity) | 25 |
| `gamma` (inhibition decay rate) | 0.98 |
| `epsilon` (recurrent input decay rate) | 0 |
| Classifier hidden layer size | 500 |

### Partially observable sequences (MNIST images)

#### Hyperparameters

| Hyperparameter  | Value |
| ------------- | ------------- |
| Batch size  | 300 |
| `g` (groups) | 200 |
| `c` (cells per group) | 6 |
| `k` (sparsity) | 25 |
| `gamma` (inhibition decay rate) | 0.5 |
| `epsilon` (recurrent input decay rate) | 0 |
| Classifier hidden layer size | 1200 |

### Generative modelling (Bouncing Balls video prediction)

### Demonstration
We have uploaded a video of the generated dynamics from the RSM 2-layer + GAN algorithm compared to RTRBM [1]. Unfortunately, there is no available video of the generative dynamics from the PGN method [2]. In the video, our algorithm is first primed with 50 frames and then switched to self-looping mode for 150 frames, feeding the GAN-rectified prediction back into the RSM. The video is located at: [https://www.youtube.com/watch?v=-lBFW1gbokg](https://www.youtube.com/watch?v=-lBFW1gbokg)

#### Hyperparameters

| Hyperparameter  | Value |
| ------------- | ------------- |
| Batch size  | 256 |
| **ConvRSM Layer 1** | |
| `g` (groups) | 64 |
| `c` (cells per group) | 8 |
| `k` (sparsity) | 3 |
| `gamma` (inhibition decay rate) | 0 |
| `epsilon` (recurrent input decay rate) | 0 |
| receptive field | 5x5 |
| pool size | 2 |
| strides | 2 |
| **ConvRSM layer 2** | |
| `g` (groups) | 128 |
| `c` (cells per group) | 8 |
| `k` (sparsity) | 5 |
| `gamma` (inhibition decay rate) | 0 |
| `epsilon` (recurrent input decay rate) | 0 |
| receptive field | 3x3 |
| strides | 2 |
| **GAN Generator [convolutional AE]** | |
| Enc. receptive field | 5x5 |
| Enc. filters | 64, 128, 256 |
| Enc. strides | 1, 2, 1 |
| Enc. nonlinearity | Leaky ReLU |
| Dec. receptive field | 5x5 |
| Dec. filters | 128, 64, 1 |
| Dec. strides | 2, 1, 1 |
| Dec. hidden nonlinearity | Leaky ReLU |
| Dec. output nonlinearity | Sigmoid |
| **GAN Discriminator [fully connected]** | |
| Hidden layer size | 128 |
| Hidden layer nonlinearity | Leaky ReLU |
| Output layer nonlinearity | Sigmoid |

### Language modelling (next word prediction, PTB corpus)

#### Hyperparameters

| Hyperparameter  | Value |
| ------------- | ------------- |
| Batch size  | 300 |
| `g` (groups) | 600 |
| `c` (cells per group) | 8 |
| `k` (sparsity) | 20 |
| `gamma` (inhibition decay rate) | 0.8 |
| `epsilon` (recurrent input decay rate) | 0.85 |
| Classifier hidden layer size | 1200 |


## References
1. I.  Sutskever,  G.  E.  Hinton,  and  G.  W.  Taylor,  “The  recurrent  tempo-ral  restricted  boltzmann  machine,”  inAdvances  in  neural  informationprocessing systems, 2009, pp. 1601–1608.
2. W.  Lotter,  G.  Kreiman,  and  D.  Cox,  “Unsupervised  learning  of  vi-sual  structure  using  predictive  generative  networks,”arXiv  preprintarXiv:1511.06380, 2015.
