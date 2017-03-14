import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from ml.features import AudioFeatures
from ml.progress import print_progress

class Layer:
    ''''''
    def __init__(self, input_type, output_type, type_out, mean, stddev):
        self.weight = tf.Variable(tf.random_normal([
            input_type, output_type], mean=mean, stddev=stddev))
        self.biases = tf.Variable(tf.random_normal([output_type], mean=mean, stddev=stddev))
        self.logits = tf.matmul(type_out, self.weight) + self.biases
        self.out = tf.nn.relu(self.logits)


class Evaluator:
    ''''''
    def __init__(self, output_type, output, learn_rate):
        self.cost_fxn = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=output_type))
        self.optimizer = tf.train.AdamOptimizer(learn_rate).minimize(self.cost_fxn)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(output_type, 1)), tf.float32))

class Trainer:
    ''''''
    def __init__(self, input_type, output_type, layers, train, test, eval, epochs=5000):
        cost_history = np.empty(shape=[1], dtype=float)
        print 'Training ->'
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                _, cost = sess.run(
                    [eval.optimizer, eval.cost_fxn],
                    feed_dict={input_type: train.features, output_type: train.labels})
                cost_history = np.append(cost_history, cost)
                print_progress(iteration=epoch, total=epochs)
            '''pred_label = sess.run(
                tf.argmax(layers[len(layers)-1].out, 1),
                feed_dict={input_type: test.features})
            true_label = sess.run(tf.argmax(test.labels, 1))'''

            self.test_accuracy = round(sess.run(
                eval.accuracy,
                feed_dict={input_type: test.features, output_type: test.labels}), 3)
            print '\n\nTest accuracy: ', self.test_accuracy

            '''cnf_max = confusion_matrix(y_true=true_label, y_pred=pred_label)
            print cnf_max

            print true_label
            print pred_label

            p = Plotter(size=(10, 8))
            p.plot(data=cost_history, axis=[0,epochs, 0, cost_history])'''

    def getAccuracy(self):
        ''''''
        return self.test_accuracy


class Plotter:
    
    def __init__(self, size):
        plt.figure(figsize=size)
    
    def plot(self, data, axis):
        plt.plot(data)
        plt.axis(axis)
        plt.show()


class MLP:
    ''''''
    def __init__(self, feature_dim, classes, hidden_units=[280, 300], learn_rate=0.01):
        self.features_dim = feature_dim
        self.classes = classes
        self.hidden_units = hidden_units
        length = len(hidden_units)
        stddev = 1/np.sqrt(self.features_dim)
        self.input_type = tf.placeholder(tf.float32, [None, self.features_dim])
        self.output_type = tf.placeholder(tf.float32, [None, self.classes])
        self.layers = []
        self.layers.append(Layer(
            input_type=self.features_dim, output_type=self.hidden_units[0],
            type_out=self.input_type, mean=0, stddev=stddev))
        for i in range(1, length):
            self.layers.append(Layer(
                input_type=self.hidden_units[i-1], output_type=self.hidden_units[i],
                type_out=self.layers[i-1].out, mean=0, stddev=stddev))
        self.layers.append(Layer(
            input_type=self.hidden_units[length-1], output_type=self.classes,
            type_out=self.layers[length-1].out, mean=0, stddev=stddev))
        self.eval = Evaluator(
            output_type=self.output_type, output=self.layers[length].logits,
            learn_rate=learn_rate)

    def train(self, train=AudioFeatures, test=AudioFeatures, epochs=5000):
        ''''''
        return Trainer(input_type=self.input_type, output_type=self.output_type, layers=self.layers,
                       train=train, test=test, eval=self.eval, epochs=epochs).getAccuracy()

    def __str__(self):
        network_str = 'Multilayer Perceptron ->'
        network_str += '\nClasses -> ' + str(self.classes)
        network_str += '\nInput -> ' + str(self.features_dim)
        network_str += '\nHidden layer neurons -> '
        for i in range(0, len(self.hidden_units)):
            network_str += '\n\t' + str(i+1)+ ' -> ' + str(self.hidden_units[i])
        return network_str

    def __repr__(self):
        return str(self)
