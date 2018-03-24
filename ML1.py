import tensorflow as tf
import numpy as np
import time
from datetime import timedelta

from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('data/MNIST/', one_hot=True)

data.test.cls = np.argmax(data.test.labels, axis=1)

filter1 = 5
numfilter1 = 20
filter2 = 5
numfilter2 = 40
fc_size = 128
fc2_size = 50

img_size = 28
img_size_flattened = img_size*img_size

num_channels = 1

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.zeros(shape = [length]))

def new_conv_layer(input, num_input_channels, filter_size,num_filters, use_pooling = True):
    
    shape = [filter_size,filter_size, num_input_channels,num_filters]
    biases = new_biases(length = num_filters)
    weights = new_weights(shape=shape)
    layer = tf.nn.conv2d(input=input, filter = weights,strides = [1,1,1,1], padding = 'SAME')
    
    layer += biases
    
    if use_pooling:
        
        layer = tf.nn.max_pool(layer, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    
    layer = tf.nn.relu(layer)
    
    return layer, weights

def flatten(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    
    return layer_flat, num_features

def new_fc_layer(input, num_inputs, node_count, use_relu = True):
    
    weights = new_weights(shape=[num_inputs,node_count])
    bias = new_biases(length = node_count)
    layer = tf.add(tf.matmul(input,weights),bias)
    
    if (use_relu):
        layer = tf.nn.relu(layer)
    

    return layer

#Model
    
x = tf.placeholder(tf.float32, shape = [None, img_size_flattened], name = 'x')

x_image = tf.reshape(x,[-1,img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape = [None, 10], name = 'y_true')

y_true_cls = tf.argmax(y_true,dimension=1)

#(conv, pooling) -> (conv, pooling) -> (flatten) -> (fc layer + relu) -> (fc layer + relu) -> (fc layer) -> (softmax)

layer_conv1, weights_conv1 = new_conv_layer(input = x_image,num_input_channels= num_channels, filter_size = filter1, num_filters = numfilter1, use_pooling = True)
layer_conv2, weights_conv2 = new_conv_layer(input = layer_conv1,num_input_channels = numfilter1, filter_size = filter2, num_filters = numfilter2, use_pooling = True)
layer_flat, num_features = flatten(layer_conv2)
layerOut = tf.layers.dense(layer_flat, fc_size, activation = tf.nn.relu)
layerOut2 = tf.layers.dense(layerOut, fc2_size, activation = tf.nn.relu)
layerOut3 = tf.layers.dense(layerOut, 10)
y_pred = tf.nn.softmax(layerOut3)
y_pred_cls = tf.argmax(y_pred, dimension = 1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = layerOut3, labels = y_true))
optimizer = tf.train.AdamOptimizer().minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

total_iterations = 0
mini_batch_size = 45

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    def optimize(num_iterations):
        
        global total_iterations
        
        start_time = time.time()
        
        for i in range(total_iterations, total_iterations + num_iterations):
        
            x_batch, y_true_batch = data.train.next_batch(mini_batch_size)
        
            feed_dict_train = {x: x_batch, y_true: y_true_batch}
        
            sess.run(optimizer, feed_dict = feed_dict_train)
        
            if i % 100 == 0:
            
                acc = sess.run(accuracy, feed_dict = feed_dict_train)
            
                print (sess.run(cost, feed_dict = feed_dict_train))
            
                msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
                
                print(msg.format(i+1,acc))
            
        total_iterations += num_iterations
        
        end_time = time.time()
        
        time_dif = end_time - start_time
        
        print("Time usage: " +str(timedelta(seconds=int(round(time_dif)))))
    
    test_batch_size = 256

#print_test_accuracy Derived from Siraj Raval

    def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
        num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
        i = 0

        while i < num_test:
        # The ending index for the next batch is denoted j.
            j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
            images = data.test.images[i:j, :]

        # Get the associated labels.
            labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
            feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
            cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
            i = j
        
    # Convenience variable for the true class-numbers of the test-set.
        cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
        correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
        acc = float(correct_sum) / num_test

    # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, correct_sum, num_test))
        
    optimize(num_iterations = 1000)
    print_test_accuracy()