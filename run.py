import tensorflow as tf
import numpy as np

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

x_train=np.random.normal(0, 1, [50,10])
y_train=np.random.randint(0, 10, [50])
y_train_onehot = dense_to_one_hot(y_train, 10)

x_test=np.random.normal(0, 1, [50,10])
y_test=np.random.randint(0, 10, [50])
y_test_onehot = dense_to_one_hot(y_test, 10)

#  A number of features, 4 in this example
#  B = 3 species of Iris (setosa, virginica and versicolor)

A=10
B=10
tf_in = tf.placeholder("float", [None, A]) # Features
tf_weight = tf.Variable(tf.zeros([A,B]))
tf_bias = tf.Variable(tf.zeros([B]))
tf_softmax = tf.nn.softmax(tf.matmul(tf_in,tf_weight) + tf_bias)

tf_bias = tf.Print(tf_bias, [tf_bias], "Bias: ")
tf_weight = tf.Print(tf_weight, [tf_weight], "Weight: ")
tf_in = tf.Print(tf_in, [tf_in], "TF_in: ")
matmul_result = tf.matmul(tf_in, tf_weight)
matmul_result = tf.Print(matmul_result, [matmul_result], "Matmul: ")
tf_softmax = tf.nn.softmax(matmul_result + tf_bias)

# Training via backpropagation
tf_softmax_correct = tf.placeholder("float", [None,B])
tf_cross_entropy = -tf.reduce_sum(tf_softmax_correct*tf.log(tf_softmax))

# Train using tf.train.GradientDescentOptimizer
tf_train_step = tf.train.GradientDescentOptimizer(0.01).minimize(tf_cross_entropy)

# Add accuracy checking nodes
tf_correct_prediction = tf.equal(tf.argmax(tf_softmax,1), tf.argmax(tf_softmax_correct,1))
tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))

print tf_correct_prediction
print tf_accuracy

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1):
    print "Running the training step"
    sess.run(tf_train_step, feed_dict={tf_in: x_train, tf_softmax_correct: y_train_onehot})
    #print y_train_onehot
    #saver.save(sess, 'trained_csv_model')

    print "Running the eval step"
    ans = sess.run(tf_softmax, feed_dict={tf_in: x_test})
    print ans