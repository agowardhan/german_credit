import matplotlib.pyplot as plt 
import xgboost as xgb
from xgboost import XGBClassifier

## sklearn 

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import precision_score, confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.utils import resample

from sklearn.ensemble import RandomForestClassifier,  VotingClassifier, GradientBoostingClassifier,  AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

# from imbalance 

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from collections import Counter

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

def plot_confusion_matrix(y_true, y_pred, classes=['good', 'bad'],
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    y_true, y_pred, classes = ['good', 'bad'],title=None, normalize=False,cmap=blues
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def downsample(df, n_samples = 300): 
    
    """
    Input dataframe, returns downsampled dataframe 
    """
    
    df_majority = df[df['class']==1]
    df_minority = df[df['class']==0]
 
# Downsample majority class

    df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=n_samples,     # to match minority class
                                 random_state=10) # reproducible results
 
# Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
# Display new class counts
    
    df_downsampled['class'].value_counts()
    return df_downsampled
    
    
def upsample(df): 
    
    """
    Input dataframe, returns downsampled dataframe 
    """
    
    df_majority = df[df['class']==1]
    df_minority = df[df['class']==0]
    
    indxs = np.random.randint(0, 300, 700)
    df_minority_upsampled = df_minority.iloc[indxs, :]
    df_out  = pd.concat([df_majority, df_minority_upsampled])
    return df_out


def cost_loss_func(y_true, y_pred):
    
    """
    y_true, y_pred 
    
    Returns custom cost based on cost metric
        
    """
    
    diff = y_true - y_pred
    fn = sum(diff ==1)
    t = sum(diff ==0)
    fp = sum(diff ==-1)
    loss = 5 * fp + fn 
    return loss/len(y_true) 


def plot_df(df): 
    
    """
    Plots all columns in dataframe 
    
    """
    
    for column in df.columns[:-1]:
        if column=='credit_amount' : 
            df[df['class']==0]['credit_amount'].hist(alpha=0.5)
            df[df['class']==1]['credit_amount'].hist(alpha=0.5)
        else: 
            df_temp = df.groupby(['class', column])[column].count().unstack('class')
            df_temp.plot(kind='bar')        
  

def label_encode(df): 
    
    LE = LabelEncoder()

    for column in df.columns:
        if df.dtypes[column] == 'object': 
            df[column] = LE.fit_transform(df[column])
    return df 

def allclassifiers(x_train, y_train, x_val, y_val): 
    
    names = ["Nearest Neighbors", "Gaussian Process", 
             "Decision Tree", "Random Forest", "GradientBoosting", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA", "XGB"]#, "LogisticRegression"]

    classifiers = [
        KNeighborsClassifier(3),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=200, max_features=1),
        GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                         max_depth=1, random_state=0), 
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(), 
        XGBClassifier()]
        #LogisticRegression(random_state=5, solver='lbfgs', max_iter=1000,multi_class='multinomial')]

    # iterate over classifiers
    
    scores_precision = []
    scores_cust= []
    
    for name, clf in zip(names, classifiers):
        clf.fit(x_train, y_train)
        y_pred_val  = clf.predict(x_val)
        score_precision = precision_score(y_val, y_pred_val)
        score_cust = cost_loss_func(y_pred=y_pred_val, y_true=y_val)

        #print(name, "Precision: %0.2f" % score_precision)
        #print(name, "Cost function: %0.2f" % score_cust)
        scores_precision.append(score_precision)
        scores_cust.append(score_cust)
        
    return names, classifiers, np.array(scores_precision), np.array(scores_cust)
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].

def run_classifiers(x,y, cv = 10): 
    
    """
    dataframes x_training, y_training, cv=10
    
    Function to perform cv-fold cross validation with oversampling, implemented correctly 
    Uses SMOTE to oversample 
    """
    
    n = len(x)
    # cross validation loop 
    interval = int(n/cv) 
    allscores_precision = []
    allscores_cust = []
    
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    for i in range(cv-1): 
        
        indxs = np.arange(n)[i*interval:(i+1)*interval] # the rows are already shuffled, no need to shuffle again 
        x_val, y_val = x.iloc[indxs].values, y.iloc[indxs].values
        x_0, y_0 = x.drop(indxs).values, y.drop(indxs).values
        
    # oversample the training data 
    
        sm = SMOTE(random_state=42)
        x_train, y_train = sm.fit_resample(x_0, y_0)
        names, classifiers, scores_precision, scores_cust = allclassifiers(x_train, y_train, x_val, y_val)
        
        allscores_precision.append(scores_precision)
        allscores_cust.append(scores_cust)
    
    return names, classifiers, allscores_precision, allscores_cust

    # cross validation loop 
