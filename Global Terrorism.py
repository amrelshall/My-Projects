#import libraries
import pandas as pd
import numpy as np




# read and prepare  data
#original data
path = 'D://Programing//Global Terrorism.csv'
Glob_terr_original = pd.read_csv(path)
# print(Glob_terr_original)
# print(Glob_terr_original.shape)
# #number of rows and features
print('number of rows =',Glob_terr_original.shape[0])

print('number of features =',Glob_terr_original.shape[1])
Glob_terr_original.head()

#modified data
Glob_terr = pd.read_csv('test.csv')
# print(Glob_terr)
# print(Glob_terr.shape)
# #number of rows and features
print('number of rows =',Glob_terr.shape[0])

print('number of features =',Glob_terr.shape[1])
Glob_terr.head()

# #convert text features into numbers
from sklearn.preprocessing import LabelEncoder
#labelencode region_txt 
enc  = LabelEncoder()
enc.fit(Glob_terr['region_txt'])


# print('classed found : ' , list(enc.classes_))

# print('equivilant numbers are : ' ,enc.transform(Glob_terr['region_txt']) )

Glob_terr['region_txt'] = enc.transform(Glob_terr['region_txt'])

print('Update data is : \n' ,Glob_terr )
#labelencode country_txt 
enc  = LabelEncoder()
enc.fit(Glob_terr['country_txt'])


# print('classed found : ' , list(enc.classes_))

# print('equivilant numbers are : ' ,enc.transform(Glob_terr['country_txt']) )

Glob_terr['country_txt'] = enc.transform(Glob_terr['country_txt'])

print('Updates Data : \n' ,Glob_terr )

# Turn Data to matrix
col = Glob_terr.shape[1]
# print(col)
X = Glob_terr.iloc[ : , :col-1]
# print(X)
y = Glob_terr.iloc[ : , col-1:col]
# print(y)

X = np.array(X)
print('X is \n' , X)
y = np.array('y is \n' , y)
print(y)
#handel with missing data by sk
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan , strategy = 'mean')
X = imputer.fit_transform(X)
# print(X)

#splittig the dataset to traning set and test set
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.25 , random_state = 0,shuffle =True)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Apply Keras Model
import tensorflow as tf
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(256, activation=	tf.nn.relu),
                                    tf.keras.layers.Dense(64, activation=	tf.nn.relu),

                                    tf.keras.layers.Dense(1, activation=tf.nn.softmax)
                                   ])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100)
model.predict(X_test)
test_loss, test_acc = model.evaluate(X,y)
print('Test accuracy:', test_acc)
