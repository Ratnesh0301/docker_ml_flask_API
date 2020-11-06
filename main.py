from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)

model = RandomForestClassifier(n_estimators=10)

model.fit(X_train, y_train)

predicted = model.predict(X_test)

print(accuracy_score(predicted, y_test))
