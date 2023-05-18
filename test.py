import pickle
import gzip

# Load the dataset
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    
x_train = train_set[0]
y_train = train_set[1]

x_test = test_set[0]
y_test = test_set[1]