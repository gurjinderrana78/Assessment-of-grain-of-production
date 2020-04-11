import numpy as np
import pandas as pd
from random import shuffle
data = pd.read_csv("Data.csv")
print(data.head())
clay1=np.repeat('clay',1200)
clay1=clay1.reshape(1200,1)
clay1
print(type(clay1))
sandy1=np.repeat('sandy',1300)
sandy1=sandy1.reshape(1300,1)
print(type(sandy1))
good1=np.repeat('good_bio_cond',1300)
good1=good1.reshape(1300,1)
print(type(good1))
al1=np.repeat('alkaline',1300)
al1=al1.reshape(1300,1)
excellent=np.repeat('excellent_bio_cond',1100)
excellent=excellent.reshape(1100,1)
print(type(excellent))
chalky1=np.repeat('chalky ',177)
chalky1=chalky1.reshape(177,1)
poor=np.repeat('poor_bio_cond',1577)
poor=poor.reshape(1577,1)
print(type(poor))
soil1=np.concatenate((clay1,sandy1, al1, chalky1),axis=0)
bio_cond = np.concatenate((good1,excellent, poor),axis=0)
print(type(soil1))
print(type(bio_cond))
print(soil1)
soil1
print(type(soil1))
np.random.shuffle(soil1)
np.random.shuffle(bio_cond)
print(soil1)
data['soil']=soil1
print(data.head())
df = pd.DataFrame(data)
one = pd.get_dummies(df['soil'])
one['alkaline']
df['alkaline']=one['alkaline']
df['sandy']=one['sandy']
df['chalky'] = one['chalky ']
df['clay']=one['clay']
df.drop(['soil'],axis=1)
print(df)
data['bio']=bio_cond
df = pd.DataFrame(data)
one = pd.get_dummies(df['bio'])
print(df)
print(one)
df['excellent_bio_cond']=one['excellent_bio_cond']
df['poor_bio_cond']=one['poor_bio_cond']
df['good_bio_cond'] = one['good_bio_cond']
data['excellent']=df['excellent_bio_cond']
data['poor']=df['poor_bio_cond']
data['good']=df['good_bio_cond']
print(data)
data.drop(['bio','soil'],axis=1)
print(data.head())
print(data[0:1])
print(data[0:1]["Moisture"])
df2 = data.values
print(type(df2))
print(df2[0,1])
print(df2.shape)
data['yield'] = 0
print(data.head())
a= 0
b = 0
c= 0
d = 0
arr = np.zeros((3977,1))
for i in range(3977):
    if(((df2[i, 0] < 12.8)  or (df2[i, 1] > 0.005)) and (df2[i, 2] <60) and (df2[i, 3] > 55) and ((df2[i, 10] == 1.0) or (df2[i, 8] == 1.0))):
        b = b + 1
        arr[i,0] = 2
        
for i in range(3977):
    if(((df2[i, 0] < 12.8)  or (df2[i, 1] > 0.5)) and (df2[i, 2] <50) and (df2[i, 3] > 45) and ((df2[i, 10] == 1.0) or (df2[i, 9] == 1.0) or (df2[i, 8] == 1.0))):
        #print i
        c= c + 1
        arr[i,0] = 3
        
for i in range(3977):
    if(((df2[i, 0] < 12.8)  or (df2[i, 1] == 0.0)) and (df2[i, 2] <90) and (df2[i, 3] < 90) and ((df2[i, 10] == 1.0) or (df2[i, 9] == 1.0) or (df2[i, 7] == 1.0) or (df2[i, 8] == 1.0))):
        #print i
        d= d + 1
        arr[i,0] = 4
        
for i in range(3977):
    if(((df2[i, 0] > 12.6 and df2[i, 0] < 13.5)  or (df2[i, 1] > 0.000001)) and (df2[i, 2] > 40 and df2[i, 2] < 100) and (df2[i, 3] > 52 and df2[i, 3] < 100) and (df2[i, 10] == 1.0)  ):
        #print i
        a = a + 1
        arr[i,0] = 1
        
print ('3 ' + str(a))
print('second ' + str(b))
print('3   ' + str(c))
print('4  ' + str(d))
print(arr)
arr.reshape(3977,1)
data['millet yield']=arr
print(data)
data = data.drop(['excellent_bio_cond','poor_bio_cond','good_bio_cond'],axis=1)
data = data.drop(['soil','bio','excellent','poor','good','yield'],axis = 1)
a = np.unique(data['millet yield'])
print(a)
data.to_csv('FinalYield.csv')
from sklearn.preprocessing import OneHotEncoder
dt = pd.read_csv('FinalYield.csv')
X = dt.drop(['millet yield'],axis=1)
Y = dt['millet yield']
Y= Y.to_frame()
print(Y)
Y = Y.values
print(type(Y))
onehot_encoder = OneHotEncoder(sparse=False)
Y = Y.reshape(len(Y), 1)
onehot_encoded = onehot_encoder.fit_transform(Y)
print(onehot_encoded)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.2, 
                                                    random_state=0)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=140)
rf.fit(X_train,y_train)
print(rf.score(X_test,y_test))
test_vector = np.reshape(np.asarray([12.737998,0.026821,61,56,70,42,1.0,0.0,0.0,0.0,0.0]),(1,11))
p = int(rf.predict(test_vector)[0])
yield_list = ['Poor Yield','Below Average','Average','Good Yield','Excellent Yield']
print (yield_list[p])







