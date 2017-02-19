import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from features import Features

class Network:

    def __init__(self, feature_dim, classes, hidden_units=[], learn_rate=0.01):
        self.features_dim = feature_dim
        self.classes = classes
        self.hidden_units = hidden_units
        length = len(hidden_units)
        stddev = 1/np.sqrt(feature_dim)
        self.input_type = tf.placeholder(tf.float32, [None, feature_dim])
        self.output_type = tf.placeholder(tf.float32, [None, classes])
        self.layers = [[[], [], []] for i in range(0, length + 1)]
        self.layers[0][0] = tf.Variable(tf.random_normal([
            feature_dim, hidden_units[0]], mean=0, stddev=stddev))
        self.layers[0][1] = tf.Variable(tf.random_normal([hidden_units[0]], mean=0, stddev=stddev))
        self.layers[0][2] = tf.nn.tanh(
            tf.matmul(self.input_type, self.layers[0][0]) + self.layers[0][1])
        for i in range(1, length):
            self.layers[i][0] = tf.Variable(tf.random_normal([
                hidden_units[i-1], hidden_units[i]], mean=0, stddev=stddev))
            self.layers[i][1] = tf.Variable(tf.random_normal([
                hidden_units[i]], mean=0, stddev=stddev))
            self.layers[i][2] = tf.nn.sigmoid(tf.matmul(
                self.layers[i-1][2], self.layers[i][0]) + self.layers[i][1])
        self.layers[length][0] = tf.Variable(tf.random_normal([
            hidden_units[length-1], classes], mean=0, stddev=stddev))
        self.layers[length][1] = tf.Variable(
            tf.random_normal([classes], mean=0, stddev=stddev))
        self.layers[length][2] = tf.nn.softmax(
            tf.matmul(self.layers[length-1][2], self.layers[length][0]) + self.layers[length][1])
        self.init = tf.global_variables_initializer()
        self.cost_fxn = tf.reduce_sum(self.output_type * tf.log(self.layers[length][2]))
        self.optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.cost_fxn)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.layers[length][2], 1), tf.argmax(self.output_type, 1)), tf.float32))

    def train(self, train, test, epochs=5000):
        cost_history = np.empty(shape=[1], dtype=float)
        y_true, y_pred = None, None
        print 'Training begins'
        with tf.Session() as sess:
            sess.run(self.init)
            for epoch in range(epochs):            
                _,cost = sess.run([self.optimizer, self.cost_fxn], feed_dict={self.input_type:train.features, self.output_type:train.labels})
                cost_history = np.append(cost_history, cost)
                print 'Epoch ' + str(epoch) + '->' + str(cost)
    
            y_pred = sess.run(tf.argmax(self.layers[len(self.layers)][2], 1), feed_dict={self.input_type: test.features})
            y_true = sess.run(tf.argmax(test.labels, 1))
            print 'Test accuracy: ', round(sess.run(self.accuracy, feed_dict={self.input_type: test.features, self.output_type: test.labels}), 3)

            fig = plt.figure(figsize=(10, 8))
            plt.plot(cost_history)
            plt.axis([0, epochs, 0, np.max(cost_history)])
            plt.show()
        
    def __str__(self):
        network_str = 'Neural Network ->'
        network_str += '\nClasses -> ' + str(self.classes)
        network_str += '\nFeatures Dimensions -> ' + str(self.features_dim)
        network_str += '\nHidden layer neurons -> '
        for i in range(0, len(self.hidden_units)):
            network_str += '\n\t' + str(i+1)+ ' -> ' + str(self.hidden_units[i])
        network_str += '\nLayers ->'
        for i in range(0, len(self.layers)):
            type_str = 'sigmoid'
            if i == 0:
                input_str = str(self.features_dim)
                type_str = 'tanh'
            else:
                input_str = str(self.hidden_units[i-1])
            if i == len(self.layers) - 1:
                output_str = str(self.classes)
                type_str = 'softmax'
            else:
                output_str = str(self.hidden_units[i])
            network_str += '\n\t Layer ' + str(i+1) + ' -> '
            network_str += '[ Input -> ' + input_str
            network_str += ', Ouput -> ' + output_str
            network_str += ', Type -> ' + type_str + ']'
        return network_str

    def __repr__(self):
        network_str = 'Neural Network ->'
        network_str += '\nClasses -> ' + str(self.classes)
        network_str += '\nFeatures Dimensions -> ' + str(self.features_dim)
        network_str += '\nHidden layer neurons -> '
        for i in range(0, len(self.hidden_units)):
            network_str += '\n\t' + str(i+1)+ ' -> ' + str(self.hidden_units[i])
        network_str += '\nLayers ->'
        for i in range(0, len(self.layers)):
            type_str = 'sigmoid'
            if i == 0:
                input_str = str(self.features_dim)
                type_str = 'tanh'
            else:
                input_str = str(self.hidden_units[i-1])
            if i == len(self.layers) - 1:
                output_str = str(self.classes)
                type_str = 'softmax'
            else:
                output_str = str(self.hidden_units[i])
            network_str += '\n\t Layer ' + str(i+1) + ' -> '
            network_str += '[ Input -> ' + input_str
            network_str += ', Ouput -> ' + output_str
            network_str += ', Type -> ' + type_str + ']'
        return network_str
