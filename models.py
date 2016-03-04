import cPickle as pickle
import numpy as np

import util

from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import MultinomialNB
import itertools

from sknn.mlp import Classifier, Layer
#import skflow

import matplotlib.pyplot as plt


ids, X, t, features = pickle.load(open('data_matrix.p', 'r'))
print X.shape
X = X[t != -1]
t = t[t != -1]
print X.shape

train_list = np.random.choice(range(len(X)), size = 0.7 * len(X), replace=False)
train_mask = np.array([i in train_list for i in range(len(X))])
valid_mask = ~train_mask

def validate(clf):
    cvscore = np.mean(cross_val_score(clf, X, t))
    clf.fit(X, t)
    try:
        feature_importances = list(reversed(np.array(features)[np.argsort(clf.feature_importances_)]))
    except:
        feature_importances = None
    selection_results = {'mean' : dict(), 'median' : dict()}
    scalings = [0, 0.25, 0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2]
    for scaling in scalings:
        X_new = SelectFromModel(clf, threshold=str(scaling)+'*mean', prefit=True).transform(X)
        selection_results['mean'][scaling] = np.mean(cross_val_score(clf, X_new, t))
        X_new = SelectFromModel(clf, threshold=str(scaling)+'*median', prefit=True).transform(X)
        selection_results['median'][scaling] = np.mean(cross_val_score(clf, X_new, t))
    best_select = max(itertools.product(['mean', 'median'], scalings), key = lambda (m,s) : selection_results[m][s])
    model = SelectFromModel(clf, threshold=str(best_select[1]) + '*' + best_select[0], prefit=True)
    X_new = model.transform(X)
    feature_mask = model.get_support()
    cvscore_selected = np.mean(cross_val_score(clf, X_new, t))
    best_model = clf.fit(X_new, t)
    return cvscore, feature_importances, best_select, feature_mask, cvscore_selected, best_model

def valid_accuracy(clf):
    clf.fit(X[train_mask], t[train_mask])
    predictions = clf.predict(X[valid_mask])
    return float(sum(np.array(predictions) == t[valid_mask])) / len(t[valid_mask])

results = dict()


clf = svm.SVC()
accuracy = valid_accuracy(clf)#np.mean(cross_val_score(clf, X, t))
clf.fit(X, t)
results['SVC'] = (clf, range(X.shape[1]), accuracy)
print 'SVC:', accuracy

clf = svm.LinearSVC()
accuracy = valid_accuracy(clf)#np.mean(cross_val_score(clf, X, t))
clf.fit(X, t)
results['LinearSVC'] = (clf, range(X.shape[1]), accuracy)
print 'LinearSVC:', accuracy
#clf.feature_importance_
#_, _, _, feature_mask, accuracy, clf = validate(clf)
#results['LinearSVC'] = (clf, accuracy)

clf = RandomForestClassifier(min_samples_split=1)
_, _, _, feature_mask, _, clf = validate(clf)
accuracy = valid_accuracy(clf)
results['RandomForest'] = (clf, feature_mask, accuracy)
print 'Random Forest:', accuracy
#print list(enumerate(reversed(np.array(features)[np.argsort(clf.feature_importances_)])))
#X_new = SelectFromModel(clf, threshold=str(best_select[1]) + '*' + best_select[0], prefit=True).transform(X)
#pickle.dump(clf, open('clf.p', 'w'))

clf = ExtraTreesClassifier(min_samples_split=1)
_, _, _, feature_mask, _, clf = validate(clf)
accuracy = valid_accuracy(clf)
results['ExtraTrees'] = (clf, feature_mask, accuracy)
print 'Extra Random Trees:', accuracy

clf = DecisionTreeClassifier(min_samples_split=1)
_, _, _, feature_mask, _, clf = validate(clf)
accuracy = valid_accuracy(clf)
results['DecisionTree'] = (clf, feature_mask, accuracy)
print 'Decision Tree:', accuracy

clf = LogisticRegression()
accuracy = valid_accuracy(clf)#np.mean(cross_val_score(clf, X, t))
clf.fit(X, t)
results['LogisticRegression'] = (clf, range(X.shape[1]), accuracy)
print 'Logistic Regression:', accuracy

clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
accuracy = valid_accuracy(clf)#np.mean(cross_val_score(clf, X, t))
clf.fit(X, t)
results['LogisticRegressionMultinomial'] = (clf, range(X.shape[1]), accuracy)
print 'Logistic Regression (Multinomial):', accuracy

clf = MultinomialNB()
accuracy = valid_accuracy(clf)#np.mean(cross_val_score(clf, X, t))
clf.fit(X, t)
results['MultinomialNB'] = (clf, range(X.shape[1]), accuracy)
print 'Multinomial NB:', accuracy

clf = KNeighborsClassifier(n_neighbors=20, weights='distance')
accuracy = valid_accuracy(clf)#np.mean(cross_val_score(clf, X, t))
clf.fit(X, t)
results['kNN (k=20)'] = (clf, range(X.shape[1]), accuracy)
print 'k-Nearest Neighbors (k=20):', accuracy
#print [np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=n_neighbors), X, t)) for n_neighbors in range(1,21)]


best_model_name = max(results, key = lambda k : results[k][2])
best = results[best_model_name]
clf = best[0]
feature_mask = best[1]
print 'Best model: %s (accuracy %f)' % (best_model_name, results[best_model_name][2])
pickle.dump((feature_mask, clf), open('classifier.p', 'w'))
pickle.dump(results, open('classifier_results.p', 'w'))


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