import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model

X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :])

shape_X = X.shape
shape_Y = Y.shape

m = shape_X[1]

print("The shape of X is: " + str(shape_X))
print("The shape of Y is: " + str(shape_Y))
print("I have m = %d training examples." % m)

clf = sklearn.linear_model.LogisticRegressionCV()

clf.fit(X.T, Y.T)

plt.title("Logistic Regression")
LR_predictions = clf.predict(X.T)
print("Accuracy of logistic regression: %d" %
      float((np.dot(Y, LR_predictions))))
