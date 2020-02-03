# Coding up a Neural Network classifier from scratch

<p align="center">
<img src="https://github.com/ankonzoid/NN-from-scratch/blob/master/images/NN.png" width="50%">
</p>
 
We train a multi-layer fully-connected neural network from scratch to classify the seeds dataset (https://archive.ics.uci.edu/ml/datasets/seeds). 
An L2 loss function, sigmoid activation, and no bias terms are assumed. 
The weight optimization is gradient descent via the delta rule.

### Usage

Run:
```
python3 NN_scratch.py
```

The output should look like:

```
Reading 'data/seeds_dataset.csv'...
 -> X.shape = (210, 7), y.shape = (210,), n_classes = 3

Neural network model:
 input_dim = 7
 hidden_layers = [5]
 output_dim = 3
 eta = 0.1
 n_epochs = 400
 n_folds = 4
 seed_crossval = 1
 seed_weights = 1

Cross-validating with 4 folds...
 Fold 1/4: acc_train = 98.10%, acc_valid = 94.23% (n_train = 158, n_valid = 52)
 Fold 2/4: acc_train = 98.10%, acc_valid = 98.08% (n_train = 158, n_valid = 52)
 Fold 3/4: acc_train = 98.73%, acc_valid = 96.15% (n_train = 158, n_valid = 52)
 Fold 4/4: acc_train = 98.73%, acc_valid = 94.23% (n_train = 158, n_valid = 52)
  -> acc_train_avg = 98.42%, acc_valid_avg = 95.67%
```

### Libraries required:

* numpy, pandas