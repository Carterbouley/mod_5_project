import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import f1_score


def plot_feature_importances(model, X_train):
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), X_train.columns.values) 
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    
def print_metrics(y_true, y_pred):
    print("Precision Score: {}".format(precision_score(y_true, y_pred)))
    print("Recall Score: {}".format(recall_score(y_true, y_pred)))
    print("Accuracy Score: {}".format(accuracy_score(y_true, y_pred)))
    print("F1 Score: {}".format(f1_score(y_true, y_pred)))
    print("AUC Score: {}".format(roc_auc_score(y_true, y_pred)))
    
def scorer():
    f1_scorer = make_scorer(score_func=f1_score,greater_is_better=True)
    return f1_scorer  

def ConfusionMatrix(y_test, pred):
    print('Confusion Matrix \n')
    print(pd.crosstab(y_test, pred, rownames = ['True'], colnames = ['Predicted'], margins = True))
    
def matrix_classification_report(y_test, pred):
    con_mat = ConfusionMatrix(y_test, pred)
    clas_rep = classification_report(y_test, pred)

    print(con_mat, '\n\n','Classification Report \n\n', clas_rep)    
    
def DecisionTree(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier(random_state=123)
    
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    
    matrix_classification_report(y_test, pred)
    
    plot_feature_importances(tree, X_train)
    
    return tree

def BaggedTree(X_train, X_test, y_train, y_test):
    bagged_tree = BaggingClassifier(random_state=123)
    
    bagged_tree.fit(X_train, y_train)
    pred = bagged_tree.predict(X_test)
    
    matrix_classification_report(y_test, pred)

    # Training accuracy score
    print("Training Accuracy for Bagging Tree Classifier: {:.4}%".format(bagged_tree.score(X_train, y_train) * 100))
    print("Testing Accuracy for Bagging Tree Classifier: {:.4}%".format(bagged_tree.score(X_test, y_test) * 100))
    
    return bagged_tree

def RandomForrest(X_train, X_test, y_train, y_test):
    forest = RandomForestClassifier()
    
    forest.fit(X_train, y_train)
    pred = forest.predict(X_test)
    
    matrix_classification_report(y_test, pred)
    
    print("Training Accuracy for Random Forest Classifier {:.4}%".format(forest.score(X_train, y_train) * 100))
    print("Testing Accuracy for Random Forest Classifier: {:.4}%".format(forest.score(X_test, y_test) * 100))
    
    return forest
