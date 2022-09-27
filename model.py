import pandas as pd # To manage data as data frames
import numpy as np # To manipulate data as arrays
from sklearn.linear_model import LogisticRegression



# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# loading the iris dataset
iris = pd.read_csv('./fd_change-83.csv' , usecols=['Ad Supported','Fraud',	'In App Purchases',	'Rating','review'])
group1=['Ad Supported']
group2 =['In App Purchases','Rating','review']
group3 = ['Fraud']
new_col = group1+group2+group3
df2 = iris[new_col]
df2.dropna(inplace=True)
variety_mappings = {0: 'Safe', 1: 'Fraud'}
# X -> features, y -> label
X = df2.iloc[:, 0:-1] # Extracting the features/independent variables

y = df2.iloc[:, -1] # Extracting the target/dependent variable

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)

from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, dtree_predictions))



def classify(a, b, c, d):
    arr = np.array([a, b, c, d]) # Convert to numpy array
    arr = arr.astype(np.float64) # Change the data type to float
    query = arr.reshape(1, -1) # Reshape the array
    prediction = variety_mappings[dtree_model.predict(query)[0]] # Retrieve from dictionary
    return prediction
