import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]) 
y = np.array([0, 1, 1, 1])  

clf = MLPClassifier(hidden_layer_sizes=(), activation='identity', solver='lbfgs', random_state=42)


clf.fit(X, y)


predictions = clf.predict(X)


print("Input X:")
print(X)
print("\nTrue outputs y (OR operation):")
print(y)
print("\nPredicted outputs:")
print(predictions)
print("\nPredictions match true outputs:", np.array_equal(predictions, y))


print("\nLearned weights:", clf.coefs_[0])
print("Learned bias:", clf.intercepts_[0]) 