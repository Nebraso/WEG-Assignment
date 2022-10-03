
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
#import socket

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Import the dataset from Github https://raw.githubusercontent.com/Nebraso/DEdataset/main/loan_data_set.csv
url = "loan_data_set.csv"
# data=pd.read_csv('/content/loan_data_set.csv') # the old way
data = pd.read_csv(url)
# End of Importing


def preprocessing(data):
    data['Dependents'].replace('3+', 3, inplace=True)
    data['Loan_Status'].replace('N', 0, inplace=True)
    data['Loan_Status'].replace('Y', 1, inplace=True)
    data['Gender'].replace('Male', 1, inplace=True)
    data['Gender'].replace('Female', 0, inplace=True)
    data['Married'].replace('Yes', 1, inplace=True)
    data['Married'].replace('No', 0, inplace=True)
    data['Education'].replace('Graduate', 1, inplace=True)
    data['Education'].replace('Not Graduate', 0, inplace=True)
    data['Self_Employed'].replace('Yes', 1, inplace=True)
    data['Self_Employed'].replace('No', 0, inplace=True)

    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
    data['Married'].fillna(data['Married'].mode()[0], inplace=True)
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
    data['Credit_History'].fillna(
        data['Credit_History'].mode()[0], inplace=True)
    data['Loan_Amount_Term'].fillna(
        data['Loan_Amount_Term'].mode()[0], inplace=True)
    data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
    data = data.drop('Loan_ID', axis=1)
    X = data.drop('Loan_Status', axis=1)
    y = data.Loan_Status.values
    return X, y


def predictResult_DecisionTree(X_train, y_train):
    DecisionTree = DecisionTreeClassifier()
    DecisionTree.fit(X_train, y_train)
    global DTfilename
    DTfilename = 'DecisionTree_model.pkl'  # .sav?
    pickle.dump(DecisionTree, open(DTfilename, 'wb'))
    # y=DecisionTree.predict(X)
    # if y==0 :return "DecisionTree: NO"
    # else : return "DecisionTree: YES"


def predictResult_naivebayes(X_train, y_train):
    naivebayes = GaussianNB()
    naivebayes.fit(X_train, y_train)
    global GBfilename
    GBfilename = 'naivebayes_model.pkl'  # .sav?
    pickle.dump(naivebayes, open(GBfilename, 'wb'))
    # y=naivebayes.predict(X)
    # if y==0 :return "naivebayes: NO"
    # else : return "naivebayes: YES"


def predictResult_Logistic_Regression(X_train, y_train):
    Logistic_Regression = LogisticRegression()
    Logistic_Regression.fit(X_train, y_train)
    global LRfilename
    LRfilename = 'Logistic_Regression_model.pkl'  # .sav?
    pickle.dump(Logistic_Regression, open(LRfilename, 'wb'))
    # y=Logistic_Regression.predict(X)
    # if y==0 :return "Logistic_Regression: NO"
    # else : return "Logistic_Regression: YES"


def Classes_distrbution(y, text):
    one = sum(1 for item in y if item == (1))
    two = sum(1 for item in y if item == (0))
    import matplotlib.pyplot as plt
    plt.title(text)
    slices = [one, two]
    channels = ['1', '0']
    cols = ['deepskyblue', 'orange']
    plt.pie(slices, labels=channels, colors=cols, autopct='%.2f%%')
    plt.show()


X, y = preprocessing(data)
#Classes_distrbution(y,'Orginal Classes distrbution')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
#Classes_distrbution(y,'New Classes distrbution')
# =================================================================================================


def predictResult(X):
    loaded_model = pickle.load(open(DTfilename, 'rb'))
    result1 = loaded_model.predict(X)
    if result1 == 1:
        ans1 = 'Yes'
    else:
        ans1 = 'No'
    # print(result1) #loaded_model.score(X_test, Y_test)
    loaded_model = pickle.load(open(GBfilename, 'rb'))
    result2 = loaded_model.predict(X)
    if result2 == 1:
        ans2 = 'Yes'
    else:
        ans2 = 'No'
    # print(result2)
    loaded_model = pickle.load(open(LRfilename, 'rb'))
    result3 = loaded_model.predict(X)
    if result3 == 1:
        ans3 = 'Yes'
    else:
        ans3 = 'No'
    return f"DecisionTree Prediction is:  {ans1}  AND   Naivebayes Prediction is: {ans2}   AND   Logistic Regression is :{ans3}"


# adding training part
isTrained = False


def trainAndTest(X):
    global isTrained
    if isTrained == False:
        predictResult_DecisionTree(X_train, y_train)
        predictResult_naivebayes(X_train, y_train)
        predictResult_Logistic_Regression(X_train, y_train)
        isTrained = True
    return predictResult(X)


def fromUserToBackend(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
    X = [[a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]]
    return trainAndTest(X)


# print(fromUserToBackend(1,1,2,1,0,4616,0,134,360,1)) No YES YES
# print(fromUserToBackend(0,0,0,1,0,4230,0,112,360,1)) No YES YES
# print(fromUserToBackend(0,0,0,1,0,3086,0,120,360,1)) YES YES YES
