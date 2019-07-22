# Pandas is used for data manipulation
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

#Predicting whether a match came from the WTA or the ATP tour from match stats
#This script learns a random forest classifier using sklearn

# Read in data
df = pd.read_csv('tour_dataset.csv')

# Split into training and test set
np.random.seed(2)
chosen_idx = np.random.choice(len(df), replace=False, size=round(0.75*len(df)))

df_train = df.iloc[chosen_idx]
df_test = df.drop(chosen_idx, errors="ignore")

# Create and train classifier
clf = RandomForestClassifier(n_estimators=500)

X_train = df_train[['w_ace', 'w_df', 'w_1stPct', 'w_1stPctWon', 'w_2ndPctWon', \
            'l_ace', 'l_df', 'l_1stPct', 'l_1stPctWon', 'l_2ndPctWon']]  # Features
y_train = df_train['is_wta']  # Labels

X_test = df_test[['w_ace', 'w_df', 'w_1stPct', 'w_1stPctWon', 'w_2ndPctWon', \
            'l_ace', 'l_df', 'l_1stPct', 'l_1stPctWon', 'l_2ndPctWon']]  # Features
y_test = df_test['is_wta']  # Labels

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test) # Predict labels
y_pred_prob = clf.predict_proba(X_test) # Predict class probabilities

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

confMat = metrics.confusion_matrix(y_test, y_pred)

#compute precision, recall and F-scores
precision1 = confMat[1,1]/(sum(confMat[:,1]))
recall1 = confMat[1,1]/(sum(confMat[1,:]))
fscore1 = 2*precision1*recall1/(precision1 + recall1)
precision0 = confMat[0,0]/(sum(confMat[:,0]))
recall0 = confMat[0,0]/(sum(confMat[0,:]))
fscore0 = 2*precision0*recall0/(precision0 + recall0)
