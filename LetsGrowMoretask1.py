#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[8]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[31]:



df = pd.read_csv(r"C:\Users\prajw\OneDrive\Desktop\Let's Grow More\iris.csv")


# In[48]:


df.info()


# In[ ]:





# In[47]:


df. describe()


# In[46]:


df.head() #top 5 values


# In[45]:


df.tail() #last 5 values


# In[44]:


df.isnull() 


# In[42]:


df.shape #no of rows and columns


# In[41]:


df.isnull().sum() #returns no of missing values in the data sets


# In[40]:


df.describe() #used to view the basic structural details


# In[39]:


df.columns


# In[38]:


df.nunique() #returns the unique elements in the oblject


# In[37]:


df.max()


# In[36]:


df.min()


# In[35]:


#drop the value of id form dataset
df.drop('Id',axis=1,inplace=True)
df.head()


# In[32]:


df.Species.nunique()


# In[49]:


df.Species.value_counts()


# In[50]:


#The boxplot plot is reated with the boxplot() method. The example below loads the iris flower data set.(a box-and-whisker plot)
#Then the presented boxplot shows the minimum, maximum,median, 1st quartile and 3rd quartile
sns.boxplot(x="Species", y='PetalLengthCm', data=df ) 
plt.show()


# In[53]:


sns.boxplot(x = "Species", y = "SepalLengthCm", data = df)


# In[54]:


sns.boxplot(x = "Species", y = "PetalWidthCm", data = df)


# In[56]:


sns.boxplot( y="SepalLengthCm" , data=df);
plt.show()


# In[58]:


sns.boxplot( y="SepalWidthCm" , data=df);
plt.show()


# In[60]:


sns.boxplot( y="PetalLengthCm" , data=df);
plt.show()


# In[62]:


sns.boxplot( y="PetalWidthCm" , data=df);
plt.show()


# In[64]:


sns.pairplot(df,hue = 'Species') # A pairplot plot a pairwise relationships in a dataset.


# In[66]:


# heatmap uses to show 2D data in graphical format.Each data value represents in a matrix and it has a special color. 
#‘True‘ value to annot then the value will show on each cell of the heatmap
# we change the color of seaborn heatmap but center parameter will change cmap according to a given value by the creator.
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(),annot=True,cmap="seismic") 
plt.show()


# In[70]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # LabelEncoder can be used to normalize labels.


# In[72]:


df['Species'] = le.fit_transform(df['Species'])  # fit_transform: Fit label encoder and return encoded labels.
df.head()


# In[74]:


X = df.drop(columns=['Species']) # Drop column
y = df['Species'] 
X[:5] # # Return list from beginning until index 5


# In[76]:


y[:5]


# In[79]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)


# In[ ]:


#Selecting the Models and Metrics(Supervised Machine Learning Models)


# In[81]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 


# In[83]:


lr = LogisticRegression()
knn = KNeighborsClassifier()
svm = SVC()
nb = GaussianNB()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()


# In[84]:


#Training and Evaluating the Models


# In[86]:


models = [lr, knn, svm, nb, dt, rf]
scores = []

for model in models:
  model.fit(X_train, y_train) # LogisticRegression.fit(X_train, y_train) # Fitting Support Vector Classifer to the Training set
  y_pred = model.predict(X_test) #LogisticRegression.predict(X_test) # Predicting the Test set results
  scores.append(accuracy_score(y_test, y_pred)) # Accuracy on the Test set results  
  print("Accuracy of " + type(model).__name__ + " is", accuracy_score(y_test, y_pred))


# In[87]:


results = pd.DataFrame({
    'Models': ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine', 'Naive Bayes', 'Decision Tree', 
               'Random Forest'],'Accuracy': scores})

results = results.sort_values(by='Accuracy', ascending=False)
print(results)

