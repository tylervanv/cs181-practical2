import cPickle as pickle
from sample_code import create_data_matrix
import csv

ids, X, t, used_features = pickle.load(open('data_matrix.p', 'r'))
feature_mask, clf = pickle.load(open('classifier.p', 'r'))

X_test = X[t == -1, feature_mask]
ids_test = ids[t == -1]
predictions = clf.predict(X_test)
predictions = zip(ids_test, predictions)

with open('predictions.csv', 'wb') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(('Id', 'Prediction'))
    for tup in predictions:
        writer.writerow(tup)

#### CAN PROBABLY DELETE THE REST, AS LONG AS EVERYTHING ABOVE THIS WORKS

#X_test, t_test, ids_test, features = create_data_matrix(100)
#feature_indices = [features.index(feature) for feature in used_features]
#X_test = X_test[:, feature_indices]
#predictions = clf.predict(X_test)
#predictions = zip(ids_test, predictions)

#with open('predictions.csv', 'wb') as output_file:
#    writer = csv.writer(output_file)
#    writer.writerow(('Id', 'Prediction'))
#    for tup in predictions:
#        writer.writerow(tup)