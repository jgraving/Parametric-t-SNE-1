from par_tsne import *
from scipy.io import load mat

train_mnist = loadmat('mnist_train.mat')
test_mnist = loadmat('mnist_test.mat')
layers=[250,250,2000,2]
network,err=train_par_tsne(train_mnist['train_X',train_mnist['train_labels'],layers,"CD1")
