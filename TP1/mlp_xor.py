import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])  
y = np.array([0, 1, 1, 0])  

def test_mlp(clf, case_name):
    clf.fit(X, y)
    
    predictions = clf.predict(X)
    
    print(f"\n=== Case {case_name} ===")
    print("Input X:")
    print(X)
    print("\nTrue outputs y (XOR operation):")
    print(y)
    print("\nPredicted outputs:")
    print(predictions)
    print("\nPredictions match true outputs:", np.array_equal(predictions, y))
    return np.array_equal(predictions, y)

# Case a: No hidden layers, linear activation
print("\nCase (a): No hidden layers, linear activation")
clf_a = MLPClassifier(hidden_layer_sizes=(), 
                     activation='identity',
                     solver='lbfgs',
                     random_state=42)
success_a = test_mlp(clf_a, "a")

# Case b: Two hidden layers (4,2), linear activation
print("\nCase (b): Two hidden layers (4,2), linear activation")
clf_b = MLPClassifier(hidden_layer_sizes=(4, 2),
                     activation='identity',
                     solver='lbfgs',
                     random_state=42)
success_b = test_mlp(clf_b, "b")

# Case c: Two hidden layers (4,2), tanh activation, multiple trials
print("\nCase (c): Two hidden layers (4,2), tanh activation, multiple trials")
n_trials = 5
successes = 0

for i in range(n_trials):
    clf_c = MLPClassifier(hidden_layer_sizes=(4, 2),
                         activation='tanh',
                         solver='lbfgs',
                         random_state=i) 
    success = test_mlp(clf_c, f"c (trial {i+1})")
    if success:
        successes += 1

print(f"\nCase (c) succeeded in {successes} out of {n_trials} trials") 