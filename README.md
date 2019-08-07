# PySNIP
Unofficial implementation of SNIP (ICLR 19) in PyTorch.
SNIP is a single shot neural network prunning technique which prunes the network before training based on sensitivity of connections of the randomly initialized weights.

## Usage
```python
from snip_prunner import Prunner
from model import my_model
from loss_func import my_loss

prunner = Prunner(my_model, my_loss, train_dataloader)
prunned_model, masks = prunner.prun(compression_factor=0.9, num_batch_sampling=1)

"""
Now continue training prunned_model 
as you would do in normal setup
"""
```
Refer test_mnist.ipynb for experiments on MNIST

## MNIST Results
| Parameters / Batches |    1    |    10    |
|----------------------|---------|----------|
| 90%                  |  97.74  |  97.70   |
| 75%                  |  97.79  |  97.79   |
| 50%                  |  97.74  |  97.67   |
| 10%                  |  96.69  |  96.69   |
|  2%                  |  93.01  |  93.69   |

### ToDo
Run experiments using ResNet Model on CIFAR 10

## Paper
[SNIP](https://openreview.net/pdf?id=B1VZqjAcYX)
