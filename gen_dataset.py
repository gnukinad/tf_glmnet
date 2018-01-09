import numpy as np
import os
import tensorflow as tf
from importlib import reload
from pprint import pprint
import pandas as pd
from sklearn.model_selection import train_test_split


def read_gene_dataset(fname):
    # read gene dataset provided by Prof. Amin Zollanvari
    with open(fname) as f:
        NS = [np.int32(x) for x in f.readline().strip().split(',')]
        P = np.int32(f.readline())
        a = f.readlines()


    xr = []
    for row in a:
        xr.append(np.array([np.float64(x) for x in row.strip().split('\t')]))


    return NS, P, np.array(xr)


def normalize_data(x):
    """normalize data"""
    for i in range(x.shape[0]):
        row = x[i,:]
        x[i, :] = (row - row.mean())/(row.max()-row.min())

    return x


def standardize_data(x):
    """normalize data to standard normal distribution"""
    for i in range(x.shape[0]):
        row = x[i,:]
        x[i, :] = (row - row.mean())/row.std()

    return x

def bias_variable(shape, name=None):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(1.1, dtype=np.float64, shape=[shape])
  return tf.Variable(initial, name=name)


def weight_variable(shape, name=None):
  """weight_variable generates a weight variable of a given shape."""
  # initial = tf.truncated_normal(shape, stddev=0.1)
  initial = tf.Variable(np.ones(shape, dtype=np.float64))
  return tf.Variable(initial, name=name)


def dnn_layer(x_input, wsiz, name=None):
    """simple dnn layer"""

    if name is not None:
        wname = "{}{}".format(name, '/weights')
        bname = "{}{}".format(name, '/bias')
    else:
        wname = None
        bname = None

    weights = weight_variable(wsiz)
    bias = bias_variable(wsiz[1])
    return tf.matmul(x_input, weights) + bias


if __name__ == "__main__":

    fname = "YeohLeukemia2002_5077x248.txt"
    # fname = "ChenLiver2004_10237x157.txt"
    # fname = "Rosenwald2002_5013x203.txt"
    fname = os.path.join("datasets", fname)

    print("reading {}".format(fname))

    NS, P, X_raw = read_gene_dataset(fname)

    N0, N1 = NS
    R = N0/N1
    N = N0 + N1

    all_lams = np.arange(1,10)/10
    # all_lams = [1]

    X = standardize_data(X_raw).transpose()
    y = np.concatenate((np.zeros(N0), np.ones(N1)))

    train_dataX, test_dataX, train_datay, test_datay = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)

    NREP = 5000

    n = train_dataX.shape[0]
    n_tst = test_dataX.shape[0]
    n0 = np.argwhere(train_datay == 0).shape[0]
    n1 = np.argwhere(train_datay == 1).shape[0]

    # input placeholders
    X_input = tf.placeholder(tf.float64, [None, P])
    y_input = tf.placeholder(tf.float64, [None])

    # test placeholders
    test_sample = tf.placeholder(tf.float64, [None, P])
    test_label = tf.placeholder(tf.float64, [None])

    # dropout
    hold_prob = tf.placeholder(tf.float64)

    # glmnet vars
    # lam = tf.placeholder(tf.float64, name='lambda')
    lam = tf.Variable(0.6, dtype=np.float64)
    alpha = tf.Variable(0.9, name='alpha', dtype=np.float64)

    # nn vars
    # nn_size = [P, P, P, P, P, 1]
    nn_size = [P, P, P, 1]


    """
    with tf.name_scope('layer0'):
        mm1 = dnn_layer(X_input, [nn_size[0], nn_size[1]])
        # act1 = tf.sigmoid(mm1, name='actfun')
        act1 = tf.sigmoid(mm1, name='actfun')

    act = act1

    for i in range(1, len(nn_size)-1):
        lname = 'layer{}'.format(i)
        with tf.name_scope(lname):
            mm = dnn_layer(act, [nn_size[i-1], nn_size[i]])
            act = tf.sigmoid(mm, name='actfun')

    with tf.name_scope('layer1'):
        mm2 = dnn_layer(act1, [nn_size[1], nn_size[2]])
        act2 = tf.sigmoid(mm2, name='actfun')


    with tf.name_scope('layer2'):
        mm3 = dnn_layer(act2, [nn_size[0], nn_size[1]])
        act3 = tf.sigmoid(mm3, name='actfun')
    """

    # beta0 = tf.Variable(0.1, name='beta0', trainable=False)
    beta0 = tf.Variable(0.1, dtype=np.float64, name='beta0', trainable=True)
    beta = tf.constant(np.ones((P,1), dtype=np.float64), name='beta')
    mmlast = tf.matmul(X_input, beta) + beta0

    ldrop = tf.nn.dropout(x=mmlast, keep_prob=hold_prob, name='dropout')


    # glmnet multiplying first data by weights
    # xbeta = tf.matmul(X_input, beta) + beta0
    # lossfun = -tf.reduce_sum(y_input * xbeta - tf.log(1 + tf.exp(xbeta))) + lam * ((1-alpha)/2*tf.norm(beta,2) + alpha * tf.norm(beta,1))

    # glmnet
    xbeta = mmlast
    lossfun = -tf.reduce_sum(y_input * xbeta - tf.log(1 + tf.exp(xbeta))) + lam * ((1-alpha)/2*tf.square(tf.norm(beta,2)) + alpha * tf.norm(beta,1))
    # lossfun = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input, logits=tf.squeeze(mmlast))

    # accuracy function
    xbeta_test = beta0 + tf.matmul(test_sample, beta)
    y_pred = tf.reshape(tf.round(1/(1+tf.exp(-xbeta_test))), [-1])
    # accuracyfun = tf.metrics.accuracy(tf.cast(test_label, tf.int32), tf.cast(y_pred, tf.int32))
    # accuracyfun = tf.metrics.accuracy(test_label, y_pred)
    accuracyfun = tf.reduce_mean(tf.cast(tf.equal(test_label, y_pred), tf.float64))

    # optim = tf.train.GradientDescentOptimizer(0.0001).minimize(lossfun)
    optim = tf.train.AdamOptimizer(0.001)
    optim_grads = optim.compute_gradients(lossfun)
    optim_upd = optim.apply_gradients(optim_grads)

    init = tf.global_variables_initializer()

    accs = pd.DataFrame({'lam': all_lams,
                         'acc': np.zeros(len(all_lams), dtype=object)})
    accs = accs.set_index('lam')


    # for clam in all_lams:
    for clam in range(1):

      a = []

      with tf.Session() as sess:
          sess.run(init)

          mbeta0, mbeta = sess.run([beta0, beta])
          print("before nn")
          print('beta0 is ', mbeta0)
          print('beta is ', mbeta)

          for i in range(NREP):
            x_train, x_test, y_train, y_test = train_test_split(train_dataX, train_datay, test_size=0.3, shuffle=True, stratify=train_datay)
            
            fd = {X_input: x_train,
                  y_input: y_train,
                  hold_prob: 0.8}

            """
            fd = {X_input: x_train,
                  y_input: y_train,
                  hold_prob: 0.8,
                  lam: clam}
            """

            # l1m, l1a = sess.run([mm1, act1], feed_dict=fd)
            # print("l1m is ", l1m)
            # print("l1a is ", l1a)
# 
            # mbeta0, mbeta = sess.run([beta0, beta])
            # print("after layer1")
            # print('beta0 is ', mbeta0)
            # print('beta is ', mbeta)

            # lm, la = sess.run([mm, act], feed_dict=fd)
            # print("lm is ", lm)
            # print("la is ", la)

            # mbeta0, mbeta = sess.run([beta0, beta])
            # print("after layers")
            # print('beta0 is ', mbeta0)
            # print('beta is ', mbeta)
# 

            # mbeta0, mbeta = sess.run([beta0, beta])
            # print("before optimizer")
            # print('beta0 is ', mbeta0)
            # print('beta is ', mbeta)


            mbeta0, mbeta, mxbeta = sess.run([beta0, beta, xbeta], feed_dict=fd)
            print('beta0 is ', mbeta0)
            print('beta is ', mbeta)
            print('xbeta is ', mxbeta)

            # mbeta0 = sess.run(beta0, feed_dict=fd)
            # print("before optimization beta0 is {}".format(mbeta0))

            print(sess.run([optim_upd, optim_grads], feed_dict=fd))

            # mbeta0 = sess.run(beta0, feed_dict=fd)
            # print("after optimization beta0 is {}".format(mbeta0))

            # mbeta0, mbeta = sess.run([beta0, beta])
            mbeta0, mbeta, mxbeta = sess.run([beta0, beta, xbeta], feed_dict=fd)
            print("after optimizer")
            print('beta0 is ', mbeta0)
            print('beta is ', mbeta)
            print('xbeta is ', mxbeta)

            # if i % 100 == 0 or i == 1:
            if i < NREP:

              fd = {test_sample: x_test,
                    test_label: y_test,
                    hold_prob: 1.00}

              """
              fd = {test_sample: x_test,
                    test_label: y_test,
                    hold_prob: 1.00,
                    lam: clam}
              """
              
              # with lam as a constant
              # mbeta0, mbeta, xxx, elab, acc = sess.run([beta0, beta, xbeta_test, y_pred, accuracyfun], feed_dict=fd)

              # with lam as a variable
              mbeta0, mbeta, xxx, elab, mlam, acc = sess.run([beta0, beta, xbeta_test, y_pred, lam, accuracyfun], feed_dict=fd)

              a.append(acc)


              print("i is {}, lam is {}, acc is {}".format(i, mlam, acc))
              print("elab is ", elab[:5], "tlab is ", y_test[:5])
              # print('xbeta_test is ', xxx)
              print('beta0 is ', mbeta0)
              # print('beta is ', mbeta)

      accs.at[clam, 'acc'] = np.array(a,dtype=object)
