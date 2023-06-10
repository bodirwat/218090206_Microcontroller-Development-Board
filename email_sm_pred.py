
import pandas as pd
import numpy as np 
data = pd.read_csv("emails.csv")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X = data.iloc[:, 1:] 
y = data['Prediction'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = LogisticRegression(max_iter=1000) 
from sklearn.preprocessing import StandardScaler

feature_names = X.columns.tolist()

#Train the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

# Training the model on the scaled data
classifier.fit(X_train_scaled, y_train)


y_pred = classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)

# Print evaluation metrics
print("====Accuracy===\n")
print("Accuracy:", accuracy)



import pickle



model_fname = 'binary_class_model.pkl' 
with open(model_fname, 'wb') as file:
    pickle.dump(classifier, file)

