import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart
disease/processed.cleveland.data" 
 
column_names = [ 
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target' 
] 
 
try: 
    data = pd.read_csv(url, header=None, names=column_names, na_values='?') 
    for col in column_names: 
        data[col] = pd.to_numeric(data[col], errors='coerce') 
    data['target'] = data['target'].apply(lambda x: 0 if x == 0 else 1) 
    data = data.dropna() 
except Exception as e: 
    print(f"Error loading data: {e}") 
    exit() 
 
X = data.drop('target', axis=1).values  
                                                                                                                       
 
y = data['target'].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42) 
 
epsilon = 1e-8 
X_train_mean = np.mean(X_train, axis=0) 
X_train_std = np.std(X_train, axis=0) 
X_train = (X_train - X_train_mean) / (X_train_std + epsilon) 
X_test = (X_test - X_train_mean) / (X_train_std + epsilon) 
 
class LogisticRegressionModel:  
    def __init__(self, learning_rate=0.01, num_iterations=1000):  
        self.learning_rate = learning_rate  
        self.num_iterations = num_iterations  
        self.weights = None  
        self.bias = None  
     
    def _sigmoid(self, z):  
        z = np.clip(z, -500, 500) 
        return 1 / (1 + np.exp(-z))  
 
    def fit(self, X, y):  
        n_samples, n_features = X.shape  
        self.weights = np.zeros(n_features)  
        self.bias = 0  
 
                                                                                                                       
 
        for iteration in range(self.num_iterations):  
            linear_model = np.dot(X, self.weights) + self.bias  
            y_predicted = self._sigmoid(linear_model)  
 
            cost = -np.mean(y * np.log(y_predicted + 1e-8) + (1 - y) * np.log(1 - 
y_predicted + 1e-8)) 
 
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))  
            db = (1/n_samples) * np.sum(y_predicted - y)  
 
            self.weights -= self.learning_rate * dw  
            self.bias -= self.learning_rate * db  
 
            if (iteration+1) % 100 == 0:  
                print(f"Iteration {iteration+1}/{self.num_iterations}, Cost: 
{cost:.4f}")  
 
    def predict(self, X):  
        linear_model = np.dot(X, self.weights) + self.bias  
        y_predicted = self._sigmoid(linear_model)  
        y_cls = [1 if i > 0.5 else 0 for i in y_predicted]  
        return np.array(y_cls)  
 
model = LogisticRegressionModel(learning_rate=0.01, num_iterations=1000)  
model.fit(X_train, y_train)  
y_pred = model.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred)  
                                                                                                                       
 
print("\nModel Accuracy on Heart Disease Test Set:", accuracy)
