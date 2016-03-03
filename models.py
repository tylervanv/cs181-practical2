import cPickle as pickle
import numpy as np

from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sknn.mlp import Classifier, Layer
#import skflow

import util

ids, X, t, features = pickle.load(open('train_data_new.p', 'r'))
print X.shape

# add binary indicator variables for every nontrivial variable in the original data
#X = X.T
#for i in range(len(X)):
#    if X[i].any() and not X[i].all():
#        X = np.vstack((X, np.array(map(int, X[i] != 0))))
#X = X.T
#print X.shape

train_list = np.random.choice(range(len(X)), size = 0.7 * len(X), replace=False)
train_mask = np.array([i in train_list for i in range(len(X))])
valid_mask = ~train_mask

#clf = svm.SVC()
#clf.fit(X[train_mask], t[train_mask])
#predictions = clf.predict(X[valid_mask])
#print float(sum(np.array(predictions) == t[valid_mask])) / len(t[valid_mask])

#clf = svm.SVC()
#print np.mean(cross_val_score(clf, X, t))

#clf = RandomForestClassifier(min_samples_split=1)
#print np.mean(cross_val_score(clf, X, t))

clf = ExtraTreesClassifier(min_samples_split=1)
print np.mean(cross_val_score(clf, X, t))
clf.fit(X, t)
import matplotlib.pyplot as plt
plt.plot(range(X.shape[1]), clf.feature_importances_)
print list(enumerate(np.array(features)[np.argsort(clf.feature_importances_)]))
#plt.hist(X[:,features.index('totaltime')], bins=np.arange(0, 2000, 10))

#clf = DecisionTreeClassifier(min_samples_split=1)
#print np.mean(cross_val_score(clf, X, t))

#clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
#print cross_val_score(clf, X, t)

#print [np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=n_neighbors), X, t)) for n_neighbors in range(1,21)]



#clf = skflow.TensorFlowLinearClassifier()
#clf.fit(X[train_mask], t[train_mask])
#predictions = clf.predict(X[valid_mask])
#print float(sum(np.array(predictions) == t[valid_mask])) / len(t[valid_mask])



#from sklearn.neural_network import MLPClassifier



#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras.optimizers import SGD
#model = Sequential()
#model.add(Dense(64, input_dim=X.shape[1], init='uniform'))
#model.add(Activation('tanh'))
#model.add(Dropout(0.5))
#model.add(Dense(64, init='uniform'))
#model.add(Activation('tanh'))
#model.add(Dropout(0.5))
#model.add(Dense(10, init='uniform'))
#model.add(Activation('softmax'))
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)
#model.fit(X[train_mask], t[train_mask], nb_epoch=20, batch_size=16, show_accuracy=True)
#score = model.evaluate(X[valid_mask], t[valid_mask], batch_size=16)


#mlp = Classifier(layers = [Layer('Sigmoid', units=128), Layer('Tanh', units=64), Layer(type='Softmax')])
#mlp.fit(X[train_mask], t[train_mask])
#predictions = mlp.predict(X[valid_mask])[:,0]
#print float(np.sum(np.array(predictions) == t[valid_mask])) / len(t[valid_mask])