import ml
import numpy as np

folder = './features/'
names = ['tr_mean_f', 'te_mean_f', 'tr_mean_pca_f', 'te_mean_pca_f',
         'tr_var_f', 'te_var_f', 'tr_var_pca_f', 'te_var_pca_f',
         'tr_iqr_f', 'te_iqr_f', 'tr_iqr_pca_f', 'te_iqr_pca_f']

features = np.empty((0, 2))

for i in range(0, len(names), 2):
    tr = ml.AudioFeatures()
    te = ml.AudioFeatures()
    tr.load(folder + names[i])
    te.load(folder + names[i+1])
    features = np.vstack([features, [tr, te]])

layers = [e for e in range(1, 11)]
neurons = [e for e in range(50, 350, 50)]
conf = []

for i in range(0, len(layers)):
    for j in range(0, len(neurons)):
        conf.append([neurons[j] for e in range(0, layers[i])])

print str(len(conf)) + ' network(MLP) configurations'

accuracy = []

for i in range(0, len(conf)):
    print 'Configuration ' + str(i+1)
    temp = []
    for j in range(0, np.shape(features)[0]):
        net = ml.MLP(features[j][0].features_dim, 2, conf[i])
        print 'Network ' + str(j+1)
        temp.append(net.train(train=features[j][0], test=features[j][1], epochs=300))
    accuracy.append(temp)

print accuracy

final_accuracy = np.array(accuracy)
np.save('./features/accuracy.npy', final_accuracy)