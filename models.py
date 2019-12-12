import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import models as md

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
from sklearn.metrics import make_scorer, f1_score, roc_curve, auc

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
    
def print_AUC(y_true, y_pred):
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
    prob = tree.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, prob[:,1])
    roc_auc = auc(fpr, tpr)
    
    print("Training Accuracy for Decision Tree Classifier: {:.4}%".format(tree.score(X_train, y_train) * 100))
    print("Testing Accuracy for Decision Tree Classifier: {:.4}%".format(tree.score(X_test, y_test) * 100))
    print("\n")
          
    matrix_classification_report(y_test, pred)
    
    plot_feature_importances(tree, X_train)
    
    

def PlotDecisionTree(X_train, X_test, y_train, y_test):
    
    tree = DecisionTreeClassifier(max_depth=4, random_state=123)
    
    tree.fit(X_train, y_train)
    
    pred = tree.predict(X_test)
    
    dot_data = StringIO()
    
    export_graphviz(tree, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names=X_train.columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    
    image = Image(graph.create_png())
    
    return image

def PlotRocCurve(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier(random_state=123)
    
    tree.fit(X_train, y_train)

    prob = tree.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, prob[:,1])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr,tpr)
    
    print('---------')
    print("AUC Score: {}".format(roc_auc))
    print('---------')
    
def BaggedTree(X_train, X_test, y_train, y_test):
    tree = BaggingClassifier(DecisionTreeClassifier(random_state=123))
    
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    prob = tree.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, prob[:,1])
    roc_auc = auc(fpr, tpr)
    
    matrix_classification_report(y_test, pred)

    print("Training Accuracy for Bagging Tree Classifier: {:.4}%".format(tree.score(X_train, y_train) * 100))
    print("Testing Accuracy for Bagging Tree Classifier: {:.4}%".format(tree.score(X_test, y_test) * 100))
    print("\n") 

    print('---------')
    print("AUC Score: {}".format(roc_auc))
    print('---------')
   
    plt.plot(fpr,tpr)
    

def RandomForest(X_train, X_test, y_train, y_test):
    tree = RandomForestClassifier(random_state=123)
    
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    prob = tree.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, prob[:,1])
    roc_auc = auc(fpr, tpr)
    
    matrix_classification_report(y_test, pred)
    
    print("Training Accuracy for Random Forest Classifier {:.4}%".format(tree.score(X_train, y_train) * 100))
    print("Testing Accuracy for Random Forest Classifier: {:.4}%".format(tree.score(X_test, y_test) * 100))
    print("\n") 

    print('---------')
    print("AUC Score: {}".format(roc_auc))
    print('---------')
   
    plt.plot(fpr,tpr)
    