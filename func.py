import os
import numpy as np
from importlib import reload
import tensorflow as tf


def read_gene_dataset(fname):
    """
    read a dataset and return number of samples for class0, class1 and amount of features

    input: fname

    returns:
    ns, P - number of samples in class0, class1 and amount of features
    X - dataset
    """

    try:
        with open(fname) as f:
            s = f.readline()
            P = np.int(f.readline().strip())
            a = f.readlines()

        ns = s.strip().split(",")
        ns = [np.int(x) for x in ns]
        n = np.sum(ns)

        X = np.empty((P, n), dtype=np.float)

        for i, x in enumerate(a):
            # print("i is {}, x is ".format(i), x)
            X[i, :] = np.array([np.float(v) for v in x.strip().split('\t')])

    except Exception as e:
        raise e

    return ns, P, X


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def normalize_data(X):
    """
    normalize data (X-mean(X))/std(X)
    """
    A = X.transpose().copy()
    return ((A - np.mean(A, 0))/np.std(A, 0)).transpose()
