import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

from gen_dataset import read_gene_dataset, standardize_data, weight_variable, bias_variable

# trying gradients

"""
P = 10

y_input = tf.placeholder(tf.float64, [None])
# x_input = tf.placeholder(tf.float64, [None, P])

lam = tf.Variable(0.3)
alpha = tf.Variable(0.9)

beta = tf.Variable(np.ones((P, 1), dtype=np.float64)) # weights
beta0 = tf.Variable(0.9) # bias
# xbeta = tf.matmul(x_input, beta) + beta0

lossfun = -tf.reduce_sum(y_input * beta0 - tf.log(1 + tf.exp(beta0)))# + lam * ((1-alpha)/2*tf.square(tf.norm(beta,2)) + alpha * tf.norm(beta,1))

gfun = tf.gradients(lossfun, [beta0])

init = tf.global_variables_initializer()

optim = tf.train.GradientDescentOptimizer(0.001).minimize(lossfun)

with tf.Session() as sess:

    sess.run(init)

    for i in range(100):

        feed_dict = {y_input: np.ones(P, dtype=np.float64)}
        print(sess.run(gfun,feed_dict=feed_dict))
        sess.run(optim, feed_dict=feed_dict)
        print(sess.run(gfun,feed_dict=feed_dict))
"""


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
X_input = tf.placeholder(tf.float64, [None, P], name='x_input')
y_input = tf.placeholder(tf.float64, [None], name='y_input')

# test placeholders
test_sample = tf.placeholder(tf.float64, [None, P], name='x_test')
test_label = tf.placeholder(tf.float64, [None], name='y_test')

# dropout
hold_prob = tf.placeholder(tf.float64)

# glmnet vars
# lam = tf.placeholder(tf.float64, name='lambda')
lam = tf.Variable(0.6)
alpha = tf.Variable(0.9, name='alpha')

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
# lossfun = -tf.reduce_sum(y_input * xbeta - tf.log(1 + tf.exp(xbeta))) + lam * ((1-alpha)/2*tf.square(tf.norm(beta,2)) + alpha * tf.norm(beta,1))
# lossfun = -tf.reduce_sum(- tf.log(1 + tf.exp(xbeta)))
lossfun = tf.exp(xbeta)
# lossfun = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input, logits=tf.squeeze(mmlast))

# accuracy function
xbeta_test = beta0 + tf.matmul(test_sample, beta)
y_pred = tf.reshape(tf.round(1/(1+tf.exp(-xbeta_test))), [-1])
# accuracyfun = tf.metrics.accuracy(tf.cast(test_label, tf.int32), tf.cast(y_pred, tf.int32))
# accuracyfun = tf.metrics.accuracy(test_label, y_pred)
accuracyfun = tf.reduce_mean(tf.cast(tf.equal(test_label, y_pred), tf.float64))


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    x_train, x_test, y_train, y_test = train_test_split(train_dataX, train_datay, test_size=0.3, shuffle=True, stratify=train_datay)
    
    fd = {X_input: x_train,
          y_input: y_train,
          hold_prob: 0.8}
    

    vbeta, exp_xbeta, vgrad = sess.run([xbeta, tf.exp(xbeta), tf.gradients(lossfun, [beta0])], feed_dict=fd)

    print("vbeta is {}, exp_beta, vgrad is {}".format(vbeta, exp_xbeta, vgrad))
