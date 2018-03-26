import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp/data/", one_hot =True)

tf.reset_default_graph()

best_accuracy = 0.0

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
x_image = tf.reshape(x,shape = [-1,28,28,1])
y_true = tf.placeholder('float', shape = [None, 10])
y_true_cls = tf.argmax(y_true, dimension = 1)

def new_weights(shape, name):
    return tf.Variable(tf.random_normal(shape,stddev=0.05), name)

def new_biases(length, name):
    return tf.Variable(tf.zeros(shape = [length]), name)

def conv2d(x,W):
     return tf.nn.conv2d(x,W,strides = [1,1,1,1], padding = 'SAME')
 
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def flatten(x):
    shapeX = x.get_shape()
    featureN = shapeX[1:4].num_elements()
    flat = tf.reshape(x,[-1,featureN])
    return flat

def get_weights(layer):
    
    with tf.variable_scope(layer, reuse = True):
        variable = tf.get_variable('weights')  
    return variable
    
W1 = new_weights([l1_filter, l1_filter, 1, l1_filterNum], 'w1')
W2 = new_weights([l2_filter, l2_filter, l1_filterNum, l2_filterNum], 'w2')
W3 = new_weights([l3_filter, l3_filter, l2_filterNum, l3_filterNum], 'w3')
    
b1 = new_biases(l1_filterNum, 'b1')
b2 = new_biases(l2_filterNum, 'b2')
b3 = new_biases(l3_filterNum, 'b3')
    
layer_one = conv2d(x_image,W1)+b1
layer_one = maxpool2d(layer_one)
    
layer_two = conv2d(layer_one,W2)+b2
layer_two = maxpool2d(layer_two)
    
layer_three = conv2d(layer_two,W3)+b3
layer_three = maxpool2d(layer_three)
    
flattened = flatten(layer_three)
    
fc1 = tf.contrib.layers.fully_connected(flattened, fc1_nodeCount, activation_fn = tf.nn.relu)
fc2 = tf.contrib.layers.fully_connected(fc1, fc2_nodeCount, activation_fn = tf.nn.relu)
output = tf.contrib.layers.fully_connected(fc2, n_classes, activation_fn = None)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_true)) 
optimizer = tf.train.AdamOptimizer().minimize(cost)

y_pred = tf.nn.softmax(output)
y_pred_cls= tf.argmax(y_pred, dimension = 1)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.train.Saver()

sess = tf.Session()
 
saver.restore(sess, "my_net/save_net.ckpt")

#sess.run(tf.global_variables_initializer()) - if initialized during the data restoration, variable restoration will fail
batch_size = 32

def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)


    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred

def predict_cls_test():
    return predict_cls(images = mnist.test.images, labels = mnist.test.labels, cls_true = np.argmax(mnist.test.labels,axis = 1))
    
def cls_accuracy(correct):
    totalN = len(correct)
    correct = correct.sum()
    acc = (float(correct)/totalN)
    return acc, totalN

def print_tAccuracy():
    
    correct, cls_pred = predict_cls_test()
    
    acc, num = cls_accuracy(correct)
    
    num_correct = acc*num
    
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num))

def train_neural_network(epochs_no):
 
 global best_accuracy  
  # training
 for epoch in range(epochs_no):
   epoch_loss = 0
   epoch_x, epoch_y = mnist.train.next_batch(batch_size)
   feed_dict_train = {x: epoch_x, y_true: epoch_y}
   _, c = sess.run([optimizer, cost], feed_dict = feed_dict_train)
     # code that optimizes the weights & biases
   epoch_loss += c
   if epoch % 10 == 0:
       acc_train = sess.run(accuracy, feed_dict = feed_dict_train)
       if (acc_train > best_accuracy):
           best_accuracy = acc_train
           saver.save(sess, 'my_net/save_net.ckpt')
       msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}"
       print(msg.format(epoch + 1, acc_train))
       print('Epoch loss', epoch_loss)

#train_neural_network(100)

print_tAccuracy()

