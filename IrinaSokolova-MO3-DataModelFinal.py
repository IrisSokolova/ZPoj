#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:22:39 2019

@author: vasilisa
"""
"""
"Adult" has such attributes:

age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

Most missing values located in nonnumeric columns (object) and not necessarily to delete or replace them. Data has no outliers and all numbers are appropriate.

What was done:
    ***
- changed the header and assigned reasonable column names
- checked each column to determinate where are the outliers and missing values
- imputed and assigned median values for missing numeric values
- data has no outliers, all numbers are appropriate
- created a histogram of a numeric variables
- created a scatterplot
- created a pairplot
- determinate the standard deviation of all numeric variables
  ***
- Normalize numeric values 'capital-gain' and 'capital-loss'
- Bin numeric variables ("Age" + add new column "AgeGroup")
- Decode categorical data
- Impute missing categories
- Consolidate categorical data 
- One-hot encode categorical data 
- Remove obsolete columns
- Clustering (age and fnlwgt) 
- Apply K-Means
  *** 
- Split data set into training and testing sets
- Create and apply a classification model for the expert label
- Logistic regression to predict Target 
- Determine accuracy rate
- Calculate the ROC curve and it's AUC using sklearn

"""
##########################
# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression #classication
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, roc_auc_score, recall_score, precision_score, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split
import matplotlib

##########################
def zNormalize(df, colname):
    df1 = df
    offset = np.mean(df1[colname])
    spread = np.std(df1[colname])
    df1['znorm'+colname] = (df[colname] - offset)/spread
    return df1

def KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2):
    PointsNorm = Points.copy()
    ClusterCentroids = ClusterCentroidGuesses.copy()
    if NormD1:
        # Determine mean of 1st dimension
        mu1 = np.mean(Points.loc[:,0])
        # Determine standard deviation of 1st dimension
        sigma1 = np.std(Points.loc[:,0])
        # Normalize 1st dimension of Points
        PointsNorm.loc[:,0] = (Points.loc[:,0] - mu1)/sigma1
        # Normalize 1st dimension of ClusterCentroids
        ClusterCentroidGuesses.loc[:,0] = (ClusterCentroidGuesses.loc[:,0] - mu1)/sigma1
    if NormD2:
        # Determine mean of 2nd dimension
        mu2 = np.mean(Points.loc[:,1])
        # Determine standard deviation of 2nd dimension
        sigma2 = np.std(Points.loc[:,1])
        # Normalize 2nd dimension of Points
        PointsNorm.loc[:,1] = (Points.loc[:,1] - mu2)/sigma2
        # Normalize 2nd dimension of ClusterCentroids
        ClusterCentroidGuesses.loc[:,1] = (ClusterCentroidGuesses.loc[:,1] - mu2)/sigma2
    # Do actual clustering
    kmeans = KMeans(n_clusters=3, init=ClusterCentroidGuesses, n_init=1).fit(PointsNorm)
    Labels = kmeans.labels_
    ClusterCentroids = pd.DataFrame(kmeans.cluster_centers_)
    if NormD1:
        # Denormalize 1st dimension
        ClusterCentroids.loc[:,0] = ClusterCentroids.loc[:,0]*sigma1 + mu1
    if NormD2:
        # Denormalize 2nd dimension
        ClusterCentroids.loc[:,1] = ClusterCentroids.loc[:,1]*sigma2 + mu2
    return Labels, ClusterCentroids


def Plot2DKMeans(Points, Labels, ClusterCentroids, Title):
    for LabelNumber in range(max(Labels)+1):
        LabelFlag = Labels == LabelNumber
        color =  ['c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y'][LabelNumber]
        marker = ['s', 'o', 'v', '^', '<', '>', '8', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'][LabelNumber]
        plt.scatter(Points.loc[LabelFlag,0], Points.loc[LabelFlag,1],
                    s= 100, c=color, edgecolors="black", alpha=0.3, marker=marker)
        plt.scatter(ClusterCentroids.loc[LabelNumber,0], ClusterCentroids.loc[LabelNumber,1], s=200, c="black", marker=marker)
    plt.title(Title)
    plt.show()

##############
# Download the data
# Read data as a pandas data frame
url ='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
Adult = pd.read_csv(url,sep=',', header=None)

Adult.columns = ['age','workclass','fnlwgt','education','education-num','marital-status',
                 'occupation','relationship','race','sex','capital-gain','capital-loss',
                 'hours-per-week','native-country','50k']
#Get df info
print (Adult.head())
print (Adult.info())
print (Adult.describe())

# Check the data types
print (Adult.dtypes)
##############

#What is unic value?
Adult.loc[:, 'capital-loss'].unique()
Adult.loc[:, 'capital-gain'].unique()
Adult.loc[:, 'education-num'].unique()
Adult.loc[:, 'fnlwgt'].unique()
Adult.loc[:, 'age'].unique()

# Corece to numeric 
Adult.loc[:, 'age'] = pd.to_numeric(Adult.loc[:, 'age'], errors='coerce')
HasNan = np.isnan(Adult.loc[:,'age'])
Adult.loc[HasNan, 'age'] = np.nanmedian(Adult.loc[:,'age'])

Adult.loc[:, 'fnlwgt'] = pd.to_numeric(Adult.loc[:, 'fnlwgt'], errors='coerce')
HasNan = np.isnan(Adult.loc[:,'fnlwgt'])
Adult.loc[HasNan, 'fnlwgt'] = np.nanmedian(Adult.loc[:,'fnlwgt'])

Adult.loc[:, 'education-num'] = pd.to_numeric(Adult.loc[:, 'education-num'], errors='coerce')
HasNan = np.isnan(Adult.loc[:,'education-num'])
Adult.loc[HasNan, 'education-num'] = np.nanmedian(Adult.loc[:,'education-num'])

Adult.loc[:, 'capital-gain'] = pd.to_numeric(Adult.loc[:, 'capital-gain'], errors='coerce')
HasNan = np.isnan(Adult.loc[:,'capital-gain'])
Adult.loc[HasNan, 'capital-gain'] = np.nanmedian(Adult.loc[:,'capital-gain'])

Adult.loc[:, 'capital-loss'] = pd.to_numeric(Adult.loc[:, 'capital-loss'], errors='coerce')
HasNan = np.isnan(Adult.loc[:,'capital-loss'])
Adult.loc[HasNan, 'capital-loss'] = np.nanmedian(Adult.loc[:,'capital-loss'])

Adult.loc[:, 'hours-per-week'] = pd.to_numeric(Adult.loc[:, 'hours-per-week'], errors='coerce')
HasNan = np.isnan(Adult.loc[:,'hours-per-week'])
Adult.loc[HasNan, 'hours-per-week'] = np.nanmedian(Adult.loc[:,'hours-per-week'])

print ('\nDistribution of numerical variables')
## Let's do a pairplot with seaborn to get a sense of the numeric variables in this data set
sns.pairplot(Adult)

# #What is categorical value?
Adult.loc[:, 'workclass'].unique()
Adult.loc[:, 'education'].unique()
Adult.loc[:, 'marital-status'].unique()
Adult.loc[:, 'occupation'].unique()
Adult.loc[:, 'relationship'].unique()
Adult.loc[:, 'race'].unique()
Adult.loc[:, 'sex'].unique()
Adult.loc[:, 'native-country'].unique()

# normalize numeric values 'capital-gain' and 'capital-loss' to avoid proxies delete parents columns
Adult = zNormalize(Adult, 'capital-gain')
Adult = zNormalize(Adult, 'capital-loss')
Adult = Adult.drop('capital-gain', axis=1)
Adult = Adult.drop('capital-loss', axis=1)

#Bin numeric variables, bin age by for AgeGroups
Adult.loc[:, 'age']
MaxBin1 = 30
MaxBin2 = 50
MaxBin3 = 70
labeled = np.empty(32561, dtype=str)     
labeled[(Adult.loc[:, 'age'] > -float("inf")) & (Adult.loc[:, 'age'] <= MaxBin1)]      = "1"
labeled[(Adult.loc[:, 'age'] > MaxBin1)       & (Adult.loc[:, 'age'] <= MaxBin2)]      = "2"
labeled[(Adult.loc[:, 'age'] > MaxBin2)       & (Adult.loc[:, 'age'] <= MaxBin3)]      = "3"
labeled[(Adult.loc[:, 'age'] > MaxBin3)       & (Adult.loc[:, 'age'] <= float("inf"))] = "4"

Adult.loc[:, 'AgeGroup'] = labeled.astype(int)

Adult.loc[:, 'AgeGroup'].value_counts().plot(kind='bar', title='The age group')
plt.show()

# Specify all the locations that have a missing value. Impute missing categories.
Adult.loc[Adult.loc[:, 'workclass'] == ' ?', 'workclass'] = "Private"
Adult.loc[:,"workclass"].value_counts().plot(kind='bar', title='WorkClass')
plt.show()

Adult.loc[:,'occupation'].value_counts()
Adult.loc[Adult.loc[:, 'occupation'] == ' ?', 'occupation'] = ' Prof-specialty'
Adult.loc[:,'occupation'].value_counts().plot(kind='bar')
plt.show()

Adult.loc[:,'native-country'].value_counts()
Adult.loc[Adult.loc[:, 'native-country'] == ' ?', 'native-country'] = ' United-States'
Adult.loc[:,'native-country'].value_counts().plot(kind='pie', title='Native-Country')
plt.show()

# Consolidate categorical data
Adult.loc[:, 'workclass'].value_counts()
Adult.loc[Adult.loc[:, 'workclass'] == ' State-gov', 'workclass'] = 'gov'
Adult.loc[Adult.loc[:, 'workclass'] == ' Federal-gov', 'workclass'] = 'gov'
Adult.loc[Adult.loc[:, 'workclass'] == ' Local-gov', 'workclass'] = 'gov'
Adult.loc[Adult.loc[:, 'workclass'] == ' Without-pay', 'workclass'] = ' Self-emp-not-inc'
Adult.loc[Adult.loc[:, 'workclass'] == 'Private', 'workclass'] = ' Private'
Adult.loc[Adult.loc[:, 'workclass'] == ' Self-emp-not-inc', 'workclass'] = ' Self-emp'
Adult.loc[Adult.loc[:, 'workclass'] == ' Self-emp-inc', 'workclass'] = ' Self-emp'

Adult.loc[:, 'workclass'].value_counts().plot(kind='bar')
plt.show()

# Convert to binary "sex" column
Adult.loc[:, 'sex'].unique()
Adult.loc[Adult.loc[:, "sex"] == ' Female', "sex"] = 1
Adult.loc[Adult.loc[:, "sex"] == ' Male', "sex"] = 0

# Consolidate categorical data
print('Before decoding....')
print(Adult['50k'].value_counts(dropna=False))

# Decode Columns
# Binary-choice question:  to predict whether the income of a person exceeds 50K per year ? Yes or Not

Adult.loc[:, 'Target'] = Adult.loc[:, "50k"]
Adult = Adult.drop('50k', axis=1)

Replace = Adult.loc[:, "Target"].str.strip() == "<=50K"
Adult.loc[Replace, "Target"] = 0

Replace = Adult.loc[:, "Target"].str.strip() == ">50K"
Adult.loc[Replace, "Target"] = 1

# Let's extract the response variable into a numpy array and drop it from the dataframe
y_all = Adult["Target"].values

print (Adult.describe())

# Clustering (age and hours-per-week) and visualisation
Points = pd.DataFrame()
Points.loc[:,0] = Adult.loc[:, "age"]
Points.loc[:,1] = Adult.loc[:, 'hours-per-week']

# Create initial cluster centroids
ClusterCentroidGuesses = pd.DataFrame()
ClusterCentroidGuesses.loc[:,0] = [100, 200, 0]
ClusterCentroidGuesses.loc[:,1] = [2, 1, 0]

# Cluster with both dimensions normalized
NormD1=True
NormD2=True
Labels, ClusterCentroids = KMeansNorm(Points, ClusterCentroidGuesses, NormD1, NormD2)
Title = 'Normalization in both dimensions, age & hours per week'
Plot2DKMeans(Points, Labels, ClusterCentroids, Title)

# K-Means will be performed on some of the data.  The data may not include
# the label, which is "Target"
SomeOfTheData = Adult.loc[:, ['hours-per-week', "age", "education-num", 'sex']]
# Cluster centroid guesses are some of the rows from the data set.
ClusterCentroidGuesses = SomeOfTheData.iloc[[0,5,6],:]
kmeans = KMeans(n_clusters=3, init=ClusterCentroidGuesses, n_init=1).fit(SomeOfTheData)

# Add the labels to the data set.  We may want to use the labels as inputs
# to the supervised learning
Adult.loc[:, 'Labels'] = kmeans.labels_

print ('\nWhat is the data type of Labels?')
print(Adult.loc[:, 'Labels'].dtype)

# Cluster labels should be categories even though they are numbers (int32).
# It would be a mistake to leave the labels as numbers.
# The labels should be decoded to categories but we can skip the decoding
# The Label numbers can be directly one-hot encoded without decoding
Adult.loc[:, "Label_1"] = (Adult.loc[:, "Labels"] == 1).astype(int)
Adult.loc[:, "Label_2"] = (Adult.loc[:, "Labels"] == 2).astype(int)
Adult.loc[:, "Label_3"] = (Adult.loc[:, "Labels"] == 3).astype(int)
Adult = Adult.drop("Labels", axis=1)

pd.value_counts(pd.Series(y_all))

# One-hot encode categorical data using pd.get_dummies()
Adult = pd.get_dummies(Adult, columns=[
    'workclass', 'occupation'])

# Delete some non-numeric variables 
Adult = Adult.drop('education', axis=1) # dublicate, Adult dataset has columns education-num
Adult = Adult.drop('marital-status', axis=1)
Adult = Adult.drop('native-country', axis=1)
Adult = Adult.drop("relationship", axis=1)
Adult = Adult.drop('race', axis=1)
Adult = Adult.drop('fnlwgt', axis=1)


##############
print ('\nVerify that all variables are numeric for Supervised Learning')
print(Adult.dtypes)

##############
print ('\nDetermine Model Accuracy')

# I choose a test fraction of 0.3 
TestFraction = 0.3
print ("Test fraction is chosen to be:", TestFraction)

# Split data into test and train sets
print ('\nsklearn accurate split:')
TrainSet, TestSet = train_test_split(Adult, test_size=TestFraction)
print ('Test size should have been ', 
       TestFraction*len(Adult), "; and is: ", len(TestSet))

##############
print ('\n Use logistic regression to predict Target (whether the income of a person exceeds 50K per year?) from other variables in Adult')
Target = "Target"
Inputs = list(Adult.columns)
Inputs.remove(Target)
clf = LogisticRegression(solver='liblinear')
clf.fit(TrainSet.loc[:,Inputs], TrainSet.loc[:,Target])
BothProbabilities = clf.predict_proba(TestSet.loc[:,Inputs])
probabilities = BothProbabilities[:,1]

##############
print ('\nConfusion Matrix and Metrics')
# A probability threshold of 0.6 
Threshold = 0.6 # Some number between 0 and 1
print ("Probability Threshold is chosen to be:", Threshold)
predictions = (probabilities > Threshold).astype(int)
CM = confusion_matrix(TestSet.loc[:,Target], predictions)
tn, fp, fn, tp = CM.ravel()
print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(TestSet.loc[:,Target], predictions)
print ("Accuracy rate:", np.round(AR, 2))
P = precision_score(TestSet.loc[:,Target], predictions)
print ("Precision:", np.round(P, 2))
R = recall_score(TestSet.loc[:,Target], predictions)
print ("Recall:", np.round(R, 2))

# Lets visualise metrics
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(CM), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

##############
 # False Positive Rate, True Posisive Rate, probability thresholds
fpr, tpr, th = roc_curve(TestSet.loc[:,Target], probabilities)
AUC = auc(fpr, tpr)

plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'family' : 'DejaVu Sans', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()

##############

# summary
print('\nLogistic Regression technique to analyze Adult.dataset:' + '\n' + '\nI have got a Accuracy rate of 83% (all correct prediction). ' + '\n' + '\nPrecision: when a model makes a prediction, how often it is correct.' + '\n' + '\nIn 80% it will be correct prediction.' + '\n' + '\nRecall: 40% - Net prediction potential.')
