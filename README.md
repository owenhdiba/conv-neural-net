# conv-neural-net
# Contents
- [Overview](#Overview)
- [Installation](#Installation)
- [Usage](#Usage)
- [Further Information](#Further-Information)
- [References](#References)
## Overview
This repo includes a module (`cnn.py`) I wrote to build and train a convolutional neural network (CNN) without recourse to dedicated machine learning libraries. It also includes a Jupyter notebook, `cnn_mnist.ipynb`, where I test the module on the MNIST dataset, using two different CNN architectures (one quite simple, and another a bit more complex), and then benchmark these results with a PyTorch CNN model. The file `trainer.py` includes a class which trains and tests a PyTorch neural net and records the history of the training for comparison with my own neural net framework. The simple network is a 3-layer CNN with a single convolutional layer and two dense layers, altogether having 232662 learnable coefficients:

<div align="center">
	
|                                 |Layer 1|Layer 2|Layer 3|
|---------------------------------|-------|-------|-------|
|Layer type                       | conv  |dense  | dense |
|Number of filters/output channels|      2|      -|      -|
|Filter rows and columns          |  (5x5)|      -|      -|
|Convolution output image size (pixels)| 24|      -|      -|
|Stride                           |      1|      -|      -|
|Number of hidden units           |   1152|    200|     10|
|Number of parameters  (including bias)            |52   |230600    |2010|

_Table 1_: architecture for the simple CNN model.
</div>

The more complex architecture is a 5-layer CNN with 122270 parameters:

<div align="center" >
	
|                                 |Layer 1|Layer 2|Layer 3|Layer 4|Layer 5|
|---------------------------------|-------|-------|-------|-------|-------|
|Layer type                       | conv  | conv  | conv  |dense  | dense |
|Number of filters/output channels|      4|      8|     12|      -|      -|
|Filter rows and columns          |  (5x5)|  (5x5)|  (4x4)|      -|      -|
|Convolution output image size (pixels)| 28|    14 |    7  |      -|      -|
|Stride                           |      1|      2|      2|      -|      -|
|Number of hidden units           |   3136|   1568|    588|    200|     10|
|Number of parameters  (including bias)            |104    |808    |1548   | 117800|   2010|

_Table 2_: architecture for the more complex CNN model. Adapted from Example 6.3 in [[1]](#References).
</div>

An example plot from the notebook, showing the loss during training of the second model, is shown below	
<p align="center">
  <img
  src="/example_loss.png" width="65%">	
</p>
<p align="center">	
  <em> Figure 1: </em> loss of the CNN during training.
</p> 



## Installation
1. Ensure you have Conda and Jupyter installed.
2. Clone the repository
   ```sh
   git clone https://github.com/owenhdiba/conv-neural-net.git
   ```
3. Move into the newly created directory of the repo. 
4. Install the conda environment:
   ```sh
   conda env create -f cnn.yaml
   ```
5. Activate the environment with 
   ```sh
   conda activate cnn
   ```
6. Add a Jupyter kernel for this environment
   ```sh
   python -m ipykernel install --user --name=cnn --display-name cnn
   ```
	
## Usage 

The Jupyter notebook `cnn_mnist.ipynb` downloads the MNIST data, trains both models described in the introduction using both my module and PyTorch, and then plots the loss and error during the training process. To run this notebook, first, ensure you are still in the cloned repo directory, then activate whichever conda environment has Jupyter installed. Now open the notebook in JupyterLab
```sh
jupyter lab cnn_mnist.ipynb
```
Check that the kernel is set to `cnn`. Now you can run the notebook and train the models. Alternatively, you can train the second model by running a python script:
```sh
python cnn_mnist.py
```
this goes through the same steps as in the notebook and then displays the plots of loss and error.

Using a Macbook Pro (2021) with an M1 chip, 15 epochs of training, and testing the validation datset every 100th iteration, it takes 6 minutes to train the first model and 20 minutes to train the second model using my CNN module. In comparison, using PyTorch and Apple's [MPS backend](https://developer.apple.com/metal/pytorch/) it takes roughly 2 minutes to train both models.

## Further Information

The original purpose of this project was to reproduce Example 6.3 in [_Machine Learning - A First Course for Engineers and Scientists_](http://smlbook.org) [[1]](#References). The architecture in table 2 and the optimization parameters I use are  taken from this example. 

Whilst _Machine Learning - A First Course..._ gives a good overview of CNNs,  it does not provide a detailed explanation of the back-propagation algorithm. I spent some time deriving the forwards and backwards equations myself. If you are interested I have included my notes on the derivation in (`cnn_notes.pdf`)[/cnn_notes.pdf].

## References

[1] Lindholm, A., Wahlström, N., Lindsten, F., & Schön, T. B. (2022). Machine learning—A first course for engineers and scientists. Cambridge University Press. https://smlbook.org

