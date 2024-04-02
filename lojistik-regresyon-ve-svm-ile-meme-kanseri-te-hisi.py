# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

dataset=pd.read_csv('/kaggle/input/breast-cancer-3csv/breast-cancer (3).csv')
dataset.shape
dataset.head()

from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

#diognosis sütunumuzu LabelEncoder kullanarak 0-1 kategori numaralarına çevirelim. Çünkü makine öğrenmesi çalışabilmemiz için sayısal değerler gerekiyor. 
dataset['diagnosis']= labelencoder.fit_transform(dataset['diagnosis'].values) #.values bize bir view sağlar. values kullanarak ana tablodaki değişiklikleri direkt yeni view tablomuzda görebiliriz.

#datamızı train ve test olarak ayıralım

from sklearn.model_selection import train_test_split

train, test=train_test_split(dataset, test_size=0.3)

X_train=train.drop('diagnosis', axis=1)
y_train=train.loc[:, 'diagnosis']
X_test=test.drop('diagnosis', axis=1)
y_test=test.loc[:, 'diagnosis']

#kategorik verilerde kullandığımız lojistik regresyon modelini kurup datamıza uygulayalım

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X_train, y_train)

#tahmin yapalım
predictions=model.predict(X_test)
predictions

#confusion matrix: sınıflandırılmış modellerde modelin doğruluğunu ölcebileceğimiz bir matrix 

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions) # 99 TN + 68 TP = 167 accuracy değerimiz. yanlış tahminler ise şöyle: 1 FP + 3 FN 

#sınıflandırma raporumuza bakalım 

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))


#aynı işlemi SVM metoduyla çözelim. 

from sklearn.svm import LinearSVC 

model_2=LinearSVC()

model_2.fit(X_train, y_train)

#tahmin yapalım 

predictions=model.predict(X_test)
predictions 

confusion_matrix(y_test, predictions)  #yeni durumda accuracy 60+108=168 diğerine göre daha iyi 

print(classification_report(y_test, predictions))