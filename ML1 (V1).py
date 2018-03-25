import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp/data/", one_hot =True)

l1_filter = 5
l2_filter = 5
l3_filter = 5

l1_filterNum = 32
l2_filterNum = 100
l3_filterNum = 256

fc1_nodeCount = 100
fc2_nodeCount = 32

n_classes = 10 #For one_hot definition
batch_size = 32 #m

x = tf.placeholder('float', [None,784])
y = tf.placeholder('float')

def new_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.zeros(shape = [length]))

def conv2d(x,W):
     return tf.nn.conv2d(x,W,strides = [1,1,1,1], padding = 'SAME')
 
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def flatten(x):
    shapeX = x.get_shape()
    featureN = shapeX[1:4].num_elements()
    flat = tf.reshape(x,[-1,featureN])
    return flat

def conv_network_model(x):
    
    x = tf.reshape(x, shape=[-1,28,28,1])
    
    W1 = new_weights([l1_filter, l1_filter, 1, l1_filterNum])
    W2 = new_weights([l2_filter, l2_filter, l1_filterNum, l2_filterNum])
    W3 = new_weights([l3_filter, l3_filter, l2_filterNum, l3_filterNum])
    
    b1 = new_biases(l1_filterNum)
    b2 = new_biases(l2_filterNum)
    b3 = new_biases(l3_filterNum)
    
    layer_one = conv2d(x,W1)+b1
    layer_one = maxpool2d(layer_one)
    
    layer_two = conv2d(layer_one,W2)+b2
    layer_two = maxpool2d(layer_two)
    
    layer_three = conv2d(layer_two,W3)+b3
    layer_three = maxpool2d(layer_three)
    
    flattened = flatten(layer_three)
    
    fc1 = tf.layers.dense(flattened, fc1_nodeCount, activation = tf.nn.relu)
    fc1 = tf.layers.dropout(fc1, 0.5)
    fc2 = tf.layers.dense(fc1, fc2_nodeCount, activation = tf.nn.relu)
    output = tf.layers.dense(fc2, n_classes)
    
    return output

def train_neural_network(x):
 prediction = conv_network_model(x)
 cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))   # v1.0 changes
  # optimizer value = 0.001, Adam similar to SGD
 optimizer = tf.train.AdamOptimizer().minimize(cost)
 epochs_no = 100
 
 with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())   # v1.0 changes
  
  # training
  for epoch in range(epochs_no):
   epoch_loss = 0
   epoch_x, epoch_y = mnist.train.next_batch(batch_size)
   _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
     # code that optimizes the weights & biases
   epoch_loss += c
   if epoch % 10 == 0:
       print('Epoch', epoch, 'completed out of', epochs_no, 'loss:', epoch_loss)
  
  # testing
  correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
  print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)
 
