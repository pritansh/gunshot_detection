import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from ml.features import AudioFeatures
from ml.progress import print_progress

class NetworkConfig:
    ''''''
    def __init__(self, features_dim, classes, hidden_units, learn_rate=0.01):
        self.features_dim = features_dim
        self.classes = classes
        self.hidden_units = hidden_units
        self.learn_rate = learn_rate
        self.mean = 0
        self.stddev = 1/np.sqrt(features_dim)

    def __str__(self):
        net_config_str = '\nNetwork Config'
        net_config_str += '\n\tInput -> ' + str(self.features_dim)
        net_config_str += '\n\tOutput -> ' + str(self.classes)
        net_config_str += '\n\tLearning rate -> ' + str(self.learn_rate)
        net_config_str += '\n\t((Mean, ' + str(self.mean) + '),'
        net_config_str += '(Standard deviation, ' + str(self.stddev) + '))'
        net_config_str += '\n\tHidden layer neurons -> '
        for i in range(0, len(self.hidden_units)):
            net_config_str += '\n\t\t' + str(i+1)+ ' -> ' + str(self.hidden_units[i])
        return net_config_str

    def __repr__(self):
        return str(self)


class Layer:
    ''''''
    def __init__(self, input_type, output_type, type_out, net_config):
        self.weight = tf.Variable(tf.random_normal([
            input_type, output_type], mean=net_config.mean, stddev=net_config.stddev))
        self.biases = tf.Variable(tf.random_normal(
            [output_type], mean=net_config.mean, stddev=net_config.stddev))
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


class MLP:
    ''''''
    def __init__(self, net_config=NetworkConfig(
            features_dim=1, classes=1, hidden_units=[200], learn_rate=0.01)):
        length = len(net_config.hidden_units)
        self.net_config = net_config
        self.input_type = tf.placeholder(tf.float32, [None, net_config.features_dim])
        self.output_type = tf.placeholder(tf.float32, [None, net_config.classes])
        self.layers = []
        self.layers.append(Layer(
            input_type=net_config.features_dim, output_type=net_config.hidden_units[0],
            type_out=self.input_type, net_config=net_config))
        for i in range(1, length):
            self.layers.append(Layer(
                input_type=net_config.hidden_units[i-1], output_type=net_config.hidden_units[i],
                type_out=self.layers[i-1].out, net_config=net_config))
        self.layers.append(Layer(
            input_type=net_config.hidden_units[length-1], output_type=net_config.classes,
            type_out=self.layers[length-1].out, net_config=net_config))
        self.eval = Evaluator(
            output_type=self.output_type, output=self.layers[length].logits,
            learn_rate=net_config.learn_rate)

    def train(self, train, test, epochs=5000):
        ''''''
        cost_history = np.empty(shape=[1], dtype=float)
        print 'Training ->'
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                _, cost = sess.run(
                    [self.eval.optimizer, self.eval.cost_fxn],
                    feed_dict={self.input_type: train.features,
                               self.output_type: train.labels})
                cost_history = np.append(cost_history, cost)
                print_progress(iteration=epoch, total=epochs)
            print test.features, test.labels
            pred_label = sess.run(
                tf.argmax(self.layers[len(self.layers)-1].out, 1),
                feed_dict={self.input_type: test.features})
            true_label = sess.run(tf.argmax(test.labels, 1))

            test_accuracy = round(sess.run(
                self.eval.accuracy,
                feed_dict={self.input_type: test.features, self.output_type: test.labels}), 3)
            print '\n\nTest accuracy: ', test_accuracy

            '''cnf_max = confusion_matrix(y_true=true_label, y_pred=pred_label)
            print cnf_max'''

            print true_label
            print pred_label
            '''
            p = Plotter(size=(10, 8))
            p.plot(data=cost_history, axis=[0,epochs, 0, cost_history])'''

    def __str__(self):
        network_str = 'Multilayer Perceptron ->'
        return network_str + str(self.net_config)

    def __repr__(self):
        return str(self)
