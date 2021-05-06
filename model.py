#                                        Diabetes Predictor

# Importing Libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import scipy as sp
import pickle

# Importing and visualizing the data

data = pd.read_csv('diabetes.csv')

data.head()

## Finding the total healty and diabetic people

plt.figure(figsize=(10, 5))
sns.countplot(x='Outcome', data=data)
plt.show()

data.info()

data.describe()

data.corr()

data.isnull().sum()

##No null values found

print(sp.mean(data))

print(sp.median(data))

plt.figure(figsize=(7, 10), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber <= 9:  # as there are 9 columns in the data
        ax = plt.subplot(3, 3, plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column, fontsize=10)
    plotnumber += 1
plt.show()

## We can see that there are lots of zeros in some features and pregnancies,insulin and age are not normalised.
## All the rest features are ok

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=data, width=0.5, ax=ax, fliersize=3)


## We can see outliers and we need to treat them as well upto a upper point

# Feature Engineering

##Restricting the outliers upto a certain limit i.e. upper_limit and lower_limit


def limit_imputer(value):
    if (value > upper_limit):
        return upper_limit
    elif (value < lower_limit):
        return lower_limit
    else:
        return value


mean = data["Insulin"].mean()
std = data["Insulin"].std()
upper_limit = mean + 3 * std
lower_limit = mean - 3 * std

data['Insulin'] = data['Insulin'].apply(limit_imputer)

mean = data["Pregnancies"].mean()
std = data["Pregnancies"].std()
upper_limit = mean + 3 * std
lower_limit = mean - 3 * std
data['Pregnancies'] = data['Pregnancies'].apply(limit_imputer)

mean = data["BloodPressure"].mean()
std = data["BloodPressure"].std()
upper_limit = mean + 3 * std
lower_limit = mean - 3 * std
data["BloodPressure"] = data["BloodPressure"].apply(limit_imputer)

mean = data["SkinThickness"].mean()
std = data["SkinThickness"].std()
upper_limit = mean + 3 * std
lower_limit = mean - 3 * std
data["SkinThickness"] = data["SkinThickness"].apply(limit_imputer)

mean = data["BMI"].mean()
std = data["BMI"].std()
upper_limit = mean + 3 * std
lower_limit = mean - 3 * std
data["BMI"] = data["BMI"].apply(limit_imputer)

mean = data['DiabetesPedigreeFunction'].mean()
std = data['DiabetesPedigreeFunction'].std()
upper_limit = mean + 3 * std
lower_limit = mean - 3 * std
data['DiabetesPedigreeFunction'] = data['DiabetesPedigreeFunction'].apply(limit_imputer)

mean = data["Age"].mean()
std = data["Age"].std()
upper_limit = mean + 3 * std
lower_limit = mean - 3 * std
data["Age"] = data["Age"].apply(limit_imputer)

data.describe()

plt.figure(figsize=(7, 10), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber <= 9:  # as there are 9 columns in the data
        ax = plt.subplot(3, 3, plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column, fontsize=10)
    plotnumber += 1
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=data, width=0.5, ax=ax, fliersize=3)

##Now we can see that the outliers are bit less but they are still present so we will proceed with this

## Replacing the zeros with the mean values


data['BMI'] = data['BMI'].replace(0, data['BMI'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0, data['BloodPressure'].mean())
data['Glucose'] = data['Glucose'].replace(0, data['Glucose'].mean())
data['Insulin'] = data['Insulin'].replace(0, data['Insulin'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].mean())

plt.figure(figsize=(7, 10), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber <= 9:  # as there are 9 columns in the data
        ax = plt.subplot(3, 3, plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column, fontsize=10)
    plotnumber += 1
plt.show()

## Now we can see that insulin and glucose

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=data, width=0.5, ax=ax, fliersize=3)

# Standardization of data and splitting it into train and test sets

X = data.drop(columns=['Outcome'])
y = data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=192)

# Variance Inflation Factor

vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
vif["Features"] = X.columns
print(vif)

# Logistic Regression

log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(x_test)

accuracy_logistic = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", accuracy_logistic)

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", conf_mat)

true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]

Precision = true_positive / (true_positive + false_positive)
print("Precision:", Precision)

Recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
print("Recall:", Recall)

# Using Decission Tree

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

print('Decission Tree Accuracy:', clf.score(x_test, y_test))

grid_param = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(2, 32, 1),
    'min_samples_leaf': range(1, 10, 1),
    'min_samples_split': range(2, 10, 1),
    'splitter': ['best', 'random']

}

from sklearn.model_selection import train_test_split, GridSearchCV

grid_search = GridSearchCV(estimator=clf,
                           param_grid=grid_param,
                           cv=5,
                           n_jobs=-1)

grid_search.fit(x_train, y_train)

best_parameters = grid_search.best_params_
print(best_parameters)
print(grid_search.best_score_)

clf = DecisionTreeClassifier(criterion='gini', max_depth=14, min_samples_leaf=8, min_samples_split=8, splitter='random')
clf.fit(x_train, y_train)

print('Decission Tree Accuracy after hyperparameter tuning:', clf.score(x_test, y_test))

accuracy_decissionTree = clf.score(x_test, y_test)

y_pred = clf.predict(x_test)
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", conf_mat)
Recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
print("Recall:", Recall)

# Random Forest

rand_clf = RandomForestClassifier(random_state=6)
rand_clf.fit(x_train, y_train)
y_pred = rand_clf.predict(x_test)

conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", conf_mat)
Recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
print("Recall:", Recall)

accuracy_randomForest = rand_clf.score(x_test, y_test)

# XGBoost

xgb = XGBClassifier(objective='binary:logistic')
xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy_xgboost = accuracy_score(y_test, predictions)

conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", conf_mat)
Recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
print("Recall:", Recall)

# KNN

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy_knn = knn.score(x_test, y_test)

conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", conf_mat)
Recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
print("Recall:", Recall)

# Bagging using KNN

bag_knn = BaggingClassifier(KNeighborsClassifier(n_neighbors=5),
                            n_estimators=10, max_samples=0.5,
                            bootstrap=True, random_state=3, oob_score=True)
bag_knn.fit(x_train, y_train)
accuracy_bagging = bag_knn.score(x_test, y_test)
y_pred = bag_knn.predict(x_test)

conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", conf_mat)
Recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
print("Recall:", Recall)

# SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train)
y_pred = sgd_clf.predict(x_test)
accuracy_sgdclassifier = sgd_clf.score(x_test, y_test)

conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", conf_mat)
Recall = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0])
print("Recall:", Recall)

# Printing the accuracy of all the models to find the best one

print('Logistic Regression:', accuracy_logistic)
print('Decission Tree:', accuracy_decissionTree)
print('Random Forest:', accuracy_randomForest)
print('XGBoost:', accuracy_xgboost)
print('KNN:', accuracy_knn)
print('Bagging:', accuracy_bagging)
print('SGDClassifier:', accuracy_sgdclassifier)

accuracy = [accuracy_logistic, accuracy_decissionTree, accuracy_randomForest, accuracy_xgboost, accuracy_knn,
            accuracy_bagging, accuracy_sgdclassifier]
models = ['Logistic', 'DT', 'Random Forest', 'XGBoost', 'KNN', 'Bagging', 'SGDClassifier']

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(models, accuracy)
plt.show()

# Conclusion:So we can clearly see that we are getting more accuracy in Logistic Regression and Random Forest but recall value in Logistic Regression
# is high then Random Forest and as this is a disease predictor so here recal can be given a priority so we will choose Logistic as our final model

# Model dumping

# Dumping our final model in a pickle file
pickle.dump(log_reg, open('model.pkl', 'wb'))