import cPickle as pickle
from sample_code import create_data_matrix
import csv

ids, X, t, used_features = pickle.load(open('train_data_new.p', 'r'))

clf = pickle.load(open('clf.p', 'r'))
X_test, t_test, ids_test, features = create_data_matrix(0, 10000, 'test')
feature_indices = [features.index(feature) for feature in used_features]
X_test = X_test[:, feature_indices]
predictions = clf.predict(X_test)
predictions = zip(ids_test, predictions)

with open('predictions.csv', 'wb') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(('Id', 'Prediction'))
    for tup in predictions:
        output_file.write(predictions)