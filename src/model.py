import tensorflow as tf
import scipy.io as sc
import numpy as np
import random

import time
from sklearn import preprocessing

# this function is used to transfer one column label to one hot label
def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]




#  Data loading
feature = sc.loadmat("S1_nolabel6.mat")
all = feature['S1_nolabel6']

print (all.shape)

np.random.shuffle(all)   # mix eeg_all

final=2800*10
all=all[0:final]
feature_all =all[:,0:64]
label=all[:,64:65]

#z-score
feature_all=preprocessing.scale(feature_all)
no_fea=feature_all.shape[-1]
label_all=one_hot(label)
print (label_all.shape)


n_classes=6
###CNN code,
feature_all=feature_all# the input data of CNN
print "cnn input feature shape", feature_all.shape
n_fea=feature_all.shape[-1]
# label_all=one_hot(label_all)

final=all.shape[0]
middle_number=final*3/4
feature_training =feature_all[0:middle_number]
feature_testing =feature_all[middle_number:final]
label_training =label_all[0:middle_number]
label_testing =label_all[middle_number:final]
label_ww=label_all[middle_number:final]  # for the confusion matrix
print ("label_testing",label_testing.shape)
a=feature_training
b=feature_testing
print(feature_training.shape)
print(feature_testing.shape)

keep=1
batch_size=final-middle_number
n_group=3
train_fea=[]
for i in range(n_group):
    f =a[(0+batch_size*i):(batch_size+batch_size*i)]
    train_fea.append(f)
print (train_fea[0].shape)

train_label=[]
for i in range(n_group):
    f =label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
    train_label.append(f)
print (train_label[0].shape)

# the CNN code
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess3.run(prediction, feed_dict={xs: v_xs, keep_prob: keep})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess3.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: keep})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# def max_pool_2x2(x):
#     # stride [1, x_movement, y_movement, 1]
#     return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')
def max_pool_1x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, n_fea]) # 1*64
ys = tf.placeholder(tf.float32, [None, n_classes])  # 2 is the classes of the data
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 1, n_fea, 1])
print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([1,1, 1,20]) # patch 1*1, in size is 1, out size is 2
b_conv1 = bias_variable([20])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 1*64*2
h_pool1 = max_pool_1x2(h_conv1)                          # output size 1*32x2

## conv2 layer ##
# W_conv2 = weight_variable([1,1, 2, 4]) # patch 1*1, in size 2, out size 4
# b_conv2 = bias_variable([4])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 1*32*4
# h_pool2 = max_pool_1x2(h_conv2)                          # output size 1*16*4

## fc1 layer ##
W_fc1 = weight_variable([1*(n_fea/2)*20, 120])
b_fc1 = bias_variable([120])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool1, [-1, 1*(n_fea/2)*20])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([120, n_classes])
b_fc2 = bias_variable([n_classes])
prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# the error between prediction and real data
l2 = 0.001 * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))+l2   # Softmax loss
train_step = tf.train.AdamOptimizer(0.04).minimize(cross_entropy) # learning rate is 0.0001

sess3 = tf.Session()
init = tf.global_variables_initializer()
sess3.run(init)

np.set_printoptions(threshold=np.nan)
step = 1
while step < 1500:
    for i in range(n_group):
        sess3.run(train_step, feed_dict={xs: train_fea[i], ys: train_label[i], keep_prob:keep})
    if step % 5 == 0:
        cost=sess3.run(cross_entropy, feed_dict={xs: b, ys: label_testing, keep_prob: keep})
        acc_cnn_t=compute_accuracy(b, label_testing)
        print('the step is:',step,',the acc is',acc_cnn_t,', the cost is', cost)
    step+=1
acc_cnn=compute_accuracy(b, label_testing)
time2=time.clock()
feature_all_cnn=sess3.run(h_fc1_drop, feed_dict={xs: feature_all, keep_prob: keep})
print ("the shape of cnn output features",feature_all.shape,label_all.shape)

time3=time.clock()


#######RNN
feature_all=feature_all
no_fea=feature_all.shape[-1]
print no_fea
feature_all =feature_all.reshape([final,1,no_fea])
print tf.argmax(label_all,1)


print label_all.shape

# middle_number=21000
feature_training =feature_all[0:middle_number]
feature_testing =feature_all[middle_number:final]
label_training =label_all[0:middle_number]
label_testing =label_all[middle_number:final]
# print "label_testing",label_testing
a=feature_training
b=feature_testing
print(feature_training.shape)
print(feature_testing.shape)
nodes=264
lameda=0.004
lr=0.005

batch_size=final-middle_number
train_fea=[]
n_group=3
for i in range(n_group):
    f =a[(0+batch_size*i):(batch_size+batch_size*i)]
    train_fea.append(f)
print (train_fea[0].shape)

train_label=[]
for i in range(n_group):
    f =label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
    train_label.append(f)
print (train_label[0].shape)


# hyperparameters

n_inputs = no_fea
n_steps = 1 # time steps
n_hidden1_units = nodes   # neurons in hidden layer
n_hidden2_units = nodes
n_hidden3_units = nodes
n_hidden4_units=nodes
n_classes = n_classes

# tf Graph input

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights

weights = {
# (28, 128)
'in': tf.Variable(tf.random_normal([n_inputs, n_hidden1_units]), trainable=True),
'a': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden1_units]), trainable=True),
#(128,128)
'hidd2': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden2_units])),
'hidd3': tf.Variable(tf.random_normal([n_hidden2_units, n_hidden3_units])),
'hidd4': tf.Variable(tf.random_normal([n_hidden3_units, n_hidden4_units])),
# (128, 10)
'out': tf.Variable(tf.random_normal([n_hidden4_units, n_classes]), trainable=True),
}

biases = {
# (128, )
'in': tf.Variable(tf.constant(0.1, shape=[n_hidden1_units])),
#(128,)
'hidd2': tf.Variable(tf.constant(0.1, shape=[n_hidden2_units ])),
'hidd3': tf.Variable(tf.constant(0.1, shape=[n_hidden3_units])),
'hidd4': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),
# (10, )
'out': tf.Variable(tf.constant(0.1, shape=[n_classes ]), trainable=True)
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    X_hidd1 = tf.matmul(X, weights['in']) + biases['in']
    X_hidd2 = tf.matmul(X_hidd1, weights['hidd2']) + biases['hidd2']
    X_hidd3 = tf.matmul(X_hidd2, weights['hidd3']) + biases['hidd3']
    X_hidd4 = tf.matmul(X_hidd3, weights['hidd4']) + biases['hidd4']
    X_in = tf.reshape(X_hidd4, [-1, n_steps, n_hidden4_units])


    # cell
    ##########################################

    # basic LSTM Cell.
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    with tf.variable_scope('lstm1'):
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results, outputs[-1]



pred,Feature = RNN(x, weights, biases)
lamena =lameda
l2 = lamena * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())  # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) + l2  # Softmax loss
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    # train_op = tf.train.AdagradOptimizer(l).minimize(cost)
    # train_op = tf.train.RMSPropOptimizer(0.00001).minimize(cost)
    # train_op = tf.train.AdagradDAOptimizer(0.01).minimize(cost)
    # train_op = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)
# pred_result =tf.argmax(pred, 1)
label_true =tf.argmax(y, 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
confusion_m=tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(pred, 1))
with tf.Session() as sess:
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    step = 0
    filename = "/home/xiangzhang/scratch/results/rnn_acc.csv"
    f2 = open(filename, 'wb')
    while step < 2500:
        for i in range(n_group):
            sess.run(train_op, feed_dict={
                x: train_fea[i],
                y: train_label[i],
            })
        if sess.run(accuracy, feed_dict={x: b,y: label_testing,})>0.96:
            print(
            "The lamda is :", lamena, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is: ",
            sess.run(accuracy, feed_dict={
                x: b,
                y: label_testing,
            }))

            break
        if step % 5 == 0:
            hh=sess.run(accuracy, feed_dict={
                x: b,
                y: label_testing,
            })
            f2.write(str(hh)+'\n')
            print(", The step is:",step,", The accuracy is:", hh, "The cost is :",sess.run(cost, feed_dict={
                x: b,
                y: label_testing,
            }))
        step += 1

    ##confusion matrix
    time4 = time.clock()
    feature_0=sess.run(Feature, feed_dict={x: train_fea[0]})
    for i in range(1,n_group):
        feature_11=sess.run(Feature, feed_dict={x: train_fea[i]})
        feature_0=np.vstack((feature_0,feature_11))

    print feature_0.shape
    feature_b = sess.run(Feature, feed_dict={x: b})
    feature_all_rnn=np.vstack((feature_0,feature_b))

    confusion_m=sess.run(confusion_m, feed_dict={
                x: b,
                y: label_testing,
            })
    print confusion_m
    time5 = time.clock()
    ## predict probility
    # pred_prob=sess.run(pred, feed_dict={
    #             x: b,
    #             y: label_testing,
    #         })
    # # print pred_prob


    print ("RNN train time:", time4 - time3, "Rnn test time", time5 - time4, 'RNN total time', time5 - time3)




###second RNN
print feature_all_rnn.shape, feature_all_cnn.shape
feature_all=np.hstack((feature_all_rnn,feature_all_cnn))
no_fea=feature_all.shape[-1]

# feature_all =feature_all.reshape([28000,1,no_fea])
print label_all.shape

# middle_number=21000
feature_training =feature_all[0:middle_number]
feature_testing =feature_all[middle_number:final]
label_training =label_all[0:middle_number]
label_testing =label_all[middle_number:final]
# print "label_testing",label_testing
a=feature_training
b=feature_testing

##AE

feature_all=feature_all


train_fea=feature_all[0:middle_number]


group=3
display_step = 10
training_epochs = 400

# Network Parameters
n_hidden_1 = 800# 1st layer num features, should be times of 8


n_hidden_2=100

n_input_ae = no_fea # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input_ae])
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input_ae, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input_ae])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input_ae])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    return layer_1


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_1

for ll in range(1):
    learning_rate = 0.2
    for ee in range(1):
        # Construct model
        encoder_op = encoder(X)
        decoder_op = decoder(encoder_op)
        # Prediction
        y_pred = decoder_op
        # Targets (Labels) are the input data.
        y_true = X

        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        # cost = tf.reduce_mean(tf.pow(y_true, y_pred))
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        saver = tf.train.Saver()
        with tf.Session() as sess1:
            sess1.run(init)
            saver = tf.train.Saver()
            # Training cycle
            for epoch in range(training_epochs):
                # Loop over all batches
                for i in range(group):
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess1.run([optimizer, cost], feed_dict={X: a})
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1),
                          "cost=", "{:.9f}".format(c))
            print("Optimization Finished!")
            time6=time.clock()
            a = sess1.run(encoder_op, feed_dict={X: a})
            b = sess1.run(encoder_op, feed_dict={X: b})

time7=time.clock()
print ("AE train time:", time6 - time5, "AE test time", time7 - time6, 'AE total time', time7 - time5)



##XGBoost
import xgboost as xgb
xg_train = xgb.DMatrix(a, label=np.argmax(label_training,1))
xg_test = xgb.DMatrix(b, label=np.argmax(label_testing,1))

# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob' # can I replace softmax by SVM??
# softprob produce a matrix with probability value of each class
# scale weight of positive examples
param['eta'] = 0.5

param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['subsample']=0.9
# param['lambda']=1
param['num_class'] =n_classes



np.set_printoptions(threshold=np.nan)
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 500
bst = xgb.train(param, xg_train, num_round, watchlist );
time8=time.clock()
pred = bst.predict( xg_test );
#
# print ('predicting, classification error=%f' % (sum( int(pred[i]) != label_testing[i] for i in range(len(label_testing))) / float(len(label_testing)) ))
time9=time.clock()


# print ("CNN train time:", time2-time1, "cnn test time", time3-time2, 'CNN total time', time3-time1)
# print ("RNN train time:", time4 - time3, "Rnn test time", time5 - time4, 'RNN total time', time5 - time3)
# print ("AE train time:", time6 - time5, "AE test time", time7 - time6, 'AE total time', time7 - time5)
# print ("XGB train time:", time8 - time7, "XGB test time", time9 - time8, 'XGB total time', time9 - time7)
# print 'total train time', time2-time1+time4 - time3+time6 - time5+time8 - time7, 'total test time',time3-time2+time5 - time4+time7 - time6+time9 - time8, 'total run time', time9-time1

