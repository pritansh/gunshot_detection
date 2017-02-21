import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from features import Features

class Layer:
    ''''''
    def __init__(self, input_type, output_type, type_h, mean, stddev, type_fxn=''):
        self.weight = tf.Variable(tf.random_normal([
            input_type, output_type], mean=mean, stddev=stddev))
        self.biases = tf.Variable(tf.random_normal([output_type], mean=mean, stddev=stddev))
        h = tf.matmul(type_h, self.weight) + self.biases
        if type_fxn == 'tanh':
            self.h = tf.nn.tanh(h)
        elif type_fxn =='sigmoid':
            self.h = tf.nn.sigmoid(h)
        elif type_fxn == 'softmax':
            self.h = tf.nn.softmax(h)


class Evaluator:
    ''''''
    def __init__(self, output_type, output_h, learn_rate):
        self.cost_fxn = tf.reduce_sum(output_type * tf.log(tf.clip_by_value(output_h, 1e-10, 1.0)))
        self.optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.cost_fxn)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(output_h, 1), tf.argmax(output_type, 1)), tf.float32))


class Network:
    ''''''
    def __init__(self, feature_dim, classes, hidden_units=[], learn_rate=0.01):
        self.features_dim = feature_dim
        self.classes = classes
        self.hidden_units = hidden_units
        length = len(hidden_units)
        stddev = 1/np.sqrt(feature_dim)
        self.input_type = tf.placeholder(tf.float32, [None, feature_dim])
        self.output_type = tf.placeholder(tf.float32, [None, classes])
        self.layers = []
        self.layers.append(Layer(input_type=self.features_dim, output_type=self.hidden_units[0], 
                  type_h=self.input_type, mean=0, stddev=stddev, type_fxn='tanh'))
        for i in range(1, length):
            self.layers.append(Layer(input_type=self.hidden_units[i-1], output_type=self.hidden_units[i], 
                      type_h=self.layers[i-1].h, mean=0, stddev=stddev, type_fxn='sigmoid'))
        self.layers.append(Layer(input_type=self.hidden_units[length-1], output_type=self.classes, 
                  type_h=self.layers[length-1].h, mean=0, stddev=stddev, type_fxn='softmax'))
        self.eval = Evaluator(output_type=self.output_type, output_h=self.layers[length].h, learn_rate=learn_rate)

    def train(self, train, test, epochs=5000, batch_size=10):
        cost_history = np.empty(shape=[1], dtype=float)
        y_true, y_pred = None, None
        batches = train.to_batch()
        print 'Training begins'
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                for i in range(0, np.shape(train.features)[0]):
                    _,cost = sess.run([self.eval.optimizer, self.eval.cost_fxn], feed_dict={self.input_type:train.features[i], self.output_type:train.labels[i]})
                    cost_history = np.append(cost_history, cost)
                    print np.shape(cost_history), cost
    
            y_pred = sess.run(tf.argmax(self.layers[len(self.layers)-1].h, 1), feed_dict={self.input_type: test.features})
            y_true = sess.run(tf.argmax(test.labels, 1))
            print 'Test accuracy: ', round(sess.run(self.eval.accuracy, feed_dict={self.input_type: test.features, self.output_type: test.labels}), 3)

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
