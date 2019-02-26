import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("C:/Users/Raghu Tata/Downloads/bank-additional/bank-additional-full.csv", sep=';')

data_dict = data.T.to_dict().values()

bro = DictVectorizer()
array_signal = bro.fit_transform(data_dict).toarray()
feature_names = bro.get_feature_names()

df = pd.DataFrame(array_signal,columns=feature_names)


X = array_signal[:,:-2]
X = np.hstack((X[:,:14],X[:,15:]))
y = array_signal[:,-1]
# Build a model and compute the feature importances
model = RandomForestClassifier(n_estimators=150,
                              random_state=0)

model.fit(X, y)
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
raghu = np.argsort(importances)[::-1]

# Print the feature ranking
print("Ranking of Fetaures is given below:")

for f in range(X.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, feature_names[raghu[f]], importances[raghu[f]]))
    
df.loc[(df['campaign'] >15) & (df['y=yes']==1)]
sum(df['y=yes'])/sum(df['campaign'])

print ("Nth Call \t Efficiency")
for i in range(1,30):
    goo = sum(df.loc[df['campaign']==i]['y=yes']) / float(df.loc[df['campaign'] >= i].shape[0])
    print (str((i))+" \t\t "+str(goo))
    
    
print("For age upto 30")
print ("Kth Call --- Efficiency")
for i in range(1,30):
    num = float(df[(df['age'] <= 30) & (df['campaign']==i) & (df['y=yes']==1)].shape[0])
    den = 1+float(df[(df['age'] <= 30) & (df['campaign'] >= i)].shape[0])
    print (str((i))+" \t\t "+str(num/den))
    
    
print("For age between 30-40")
print ("Kth Call --- Efficiency")
for i in range(1,30):
    num = float(df[(df['age'] <= 40) & (df['age'] > 30) & (df['campaign']==i) & (df['y=yes']==1)].shape[0])
    den = 1+float(df[(df['age'] <= 40) & (df['age'] > 30) & (df['campaign'] >= i)].shape[0])
    print (str((i))+" \t\t "+str(num/den))
    
    
print("For age between 40-50")
print ("Kth Call --- Efficiency")
for i in range(1,30):
    num = float(df[(df['age'] <= 50) & (df['age'] > 40) & (df['campaign']==i) & (df['y=yes']==1)].shape[0])
    den = 1+float(df[(df['age'] <= 50) & (df['age'] > 40) & (df['campaign'] >= i)].shape[0])
    print (str((i))+" \t\t "+str(num/den))
    
    
    
print("For age between 50-60")
print ("Kth Call --- Efficiency")
for i in range(1,30):
    num = float(df[(df['age'] <= 60) & (df['age'] > 50) & (df['campaign']==i) & (df['y=yes']==1)].shape[0])
    den = 1+float(df[(df['age'] <= 60) & (df['age'] > 50) & (df['campaign'] >= i)].shape[0])
    print (str((i))+" \t\t "+str(num/den))
    
    
    
    

print("For age above 60")
print ("Kth Call --- Efficiency")
for i in range(1,30):
    num = float(df[(df['age'] > 60) & (df['campaign']==i) & (df['y=yes']==1)].shape[0])
    den = float(df[(df['age'] > 60) & (df['campaign'] >= i)].shape[0])+1
    print (str((i))+" \t\t "+str(num/den))
    
    

calls = sum(df['campaign'])
print(calls)



extra_c = sum(df[df['campaign']>6]['campaign']) - 6*df[df['campaign']>6].shape[0]
print(extra_c)


# Calculate reduction in marketing cost
reduce=100*extra_c/calls
print(reduce)



sales=float(df[df['y=yes']==1].shape[0])
print(sales)


less_costly=float(df[(df['campaign'] <= 6) & (df['y=yes']==1)].shape[0])
print(less_costly)



percent=100*less_costly/sales
print(percent) 