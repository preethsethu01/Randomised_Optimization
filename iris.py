from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import mlrose

# Load the Iris dataset
data = load_iris()

# Get feature values
print('The feature values for Obs 0 are: ', data.data[0])

# Get feature names
print('The feature names are: ', data.feature_names)

# Get target value of first observation
print('The target value for Obs 0 is:', data.target[0])

# Get target name of first observation
print('The target name for Obs 0 is:', data.target_names[data.target[0]])

# Get minimum feature values
print('The minimum values of the four features are:', np.min(data.data, axis = 0))

# Get maximum feature values
print('The maximum values of the four features are:', np.max(data.data, axis = 0))

# Get unique target values
print('The unique target values are:', np.unique(data.target))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,test_size = 0.2, random_state = 3)

# Normalize feature data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()

y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

# Initialize neural network object and fit object
nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', algorithm = 'genetic_alg', max_iters = 200,bias = True, is_classifier = True, learning_rate = 0.0001,early_stopping = True, clip_max = 5, max_attempts =100,random_state = 3)

nn_model1.fit(X_train_scaled, y_train_hot)

# Predict labels for train set and assess accuracy
y_train_pred = nn_model1.predict(X_train_scaled)

y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

print('Training accuracy: ', y_train_accuracy)

# Predict labels for test set and assess accuracy
y_test_pred = nn_model1.predict(X_test_scaled)

y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

print('Test accuracy: ', y_test_accuracy)


# Initialize neural network object and fit object
nn_model2 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu',algorithm = 'gradient_descent', max_iters = 200,bias = True, is_classifier = True, learning_rate = 0.0001,early_stopping = True, clip_max = 5, max_attempts =100,random_state = 3)
nn_model2.fit(X_train_scaled, y_train_hot)
# Predict labels for train set and assess accuracy
y_train_pred = nn_model2.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
print("Gradient Descent")
print(y_train_accuracy)
# Predict labels for test set and assess accuracy
y_test_pred = nn_model2.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
print(y_test_accuracy)
