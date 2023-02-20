#!/usr/bin/env python
# coding: utf-8

# In[100]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[101]:


biomarkers_df = pd.read_csv("/Users/akkaedoodle/Downloads/debernardi_full_plasmaCA19_control_malignant.csv")


# In[102]:


biomarkers_df


# In[103]:


biomarkers_df.info()


# In[104]:


sex = {'F':0, 'M':1}
biomarkers_df.sex = [sex[item] for item in biomarkers_df.sex]


# In[105]:


from sklearn.model_selection import train_test_split


# Drop Stuff

# In[106]:


biomarkers_df.drop('sample_id', axis=1, inplace=True)
biomarkers_df.drop('patient_cohort', axis=1, inplace=True)
biomarkers_df.drop('sample_origin', axis=1, inplace=True)
biomarkers_df.drop('benign_sample_diagnosis', axis=1, inplace=True)
#biomarkers_df.drop('plasma_CA19_9', axis=1, inplace=True)
biomarkers_df.drop('REG1A', axis=1, inplace=True)


# In[107]:


biomarkers_df.drop('sex', axis=1, inplace=True)
biomarkers_df.drop('age', axis=1, inplace=True)


# In[108]:


biomarkers_df.drop('creatinine_log', axis=1, inplace=True)
biomarkers_df.drop('LYVE_log', axis=1, inplace=True)
biomarkers_df.drop('REG1B_log', axis=1, inplace=True)
biomarkers_df.drop('TFF1_log', axis=1, inplace=True)


# In[109]:


biomarkers_df.drop('stage', axis=1, inplace=True)


# In[110]:


biomarkers_df


# In[111]:


X = biomarkers_df.drop('diagnosis', axis=1)
y = biomarkers_df['diagnosis']


# In[112]:


X.describe()


# In[113]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)


# In[114]:


from sklearn.tree import DecisionTreeClassifier


# In[115]:


dtree = DecisionTreeClassifier()


# In[116]:


dtree.fit(X_train,y_train)


# In[117]:


predictions = dtree.predict(X_test)


# In[118]:


from sklearn.metrics import classification_report,confusion_matrix


# In[119]:


print(classification_report(y_test,predictions))


# In[120]:


print(confusion_matrix(y_test,predictions))


# # Random Forest Classifier

# In[121]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[122]:


rfc_pred = rfc.predict(X_test)


# In[123]:


print(confusion_matrix(y_test,rfc_pred))


# In[124]:


print(classification_report(y_test,rfc_pred))


# In[125]:


y.value_counts()


# In[126]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)


# In[127]:


X_train.describe()


# In[128]:


forest = RandomForestClassifier()


# In[129]:


forest.fit(X_train, y_train)


# In[130]:


y_pred_test = forest.predict(X_test)


# In[131]:


y_pred_test


# In[132]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[133]:


accuracy_score(y_test, y_pred_test)


# ### random forest Confusion Matrix

# In[134]:


confusion_matrix(y_test, y_pred_test)


# In[135]:


matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10}, cmap=plt.cm.Blues, linewidths = 0.2)

class_names = ['Benign', 'Cancerous']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()


# # Support Vector Classifier

# In[38]:


from sklearn.svm import SVC


# In[39]:


model = SVC()


# In[40]:


model.fit(X_train, y_train)


# In[41]:


predictions = model.predict(X_test)


# In[42]:


from sklearn.metrics import classification_report, confusion_matrix


# In[43]:


print(confusion_matrix(y_test,predictions))


# In[45]:


print(classification_report(y_test,predictions))


# #### SVC improved drastically after removing all the actual protein/creatinine values (and only keeping the logs)

# In[46]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[47]:


from sklearn.model_selection import GridSearchCV


# In[48]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# In[49]:


grid.fit(X_train,y_train)


# In[50]:


grid.best_params_


# In[51]:


grid.best_estimator_


# In[52]:


grid_predictions = grid.predict(X_test)


# In[53]:


print(confusion_matrix(y_test,grid_predictions))


# In[54]:


print(classification_report(y_test,grid_predictions))


# # Logistic Regression

# In[55]:


from sklearn.linear_model import LogisticRegression


# In[56]:


classifier = LogisticRegression(random_state = 1)


# In[57]:


classifier.fit(X_train, y_train)


# In[58]:


y_train.describe()


# In[59]:


X_train.describe()


# In[60]:


Y_pred = classifier.predict(X_test)


# In[61]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred)


# In[62]:


cm


# In[63]:


print(classification_report(y_test,Y_pred))


# # K Nearest Neighbors

# In[64]:


from sklearn.neighbors import KNeighborsClassifier
Kclassifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
Kclassifier.fit(X_train, y_train)


# In[65]:


Y_pred = Kclassifier.predict(X_test)


# In[66]:


from sklearn.metrics import confusion_matrix
Kcm = confusion_matrix(y_test, Y_pred)


# In[67]:


Kcm


# In[68]:


print(classification_report(y_test,Y_pred))


# # Naive Bayes

# In[69]:


from sklearn.naive_bayes import GaussianNB
NBclassifier = GaussianNB()
NBclassifier.fit(X_train, y_train)


# In[70]:


Y_pred = classifier.predict(X_test)


# In[71]:


from sklearn.metrics import confusion_matrix
NBcm = confusion_matrix(y_test, Y_pred)


# In[72]:


NBcm


# In[73]:


print(classification_report(y_test,Y_pred))

