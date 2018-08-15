
# coding: utf-8

# In[ ]:



import gc
gc.collect()
# import libraries


# In[ ]:


from scipy import stats
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt


# In[ ]:


application_train = pd.read_csv("//home//mgwarada//Desktop//Ruvimbo//application_train.csv")
#application_train.head()


# In[ ]:


#application_train.shape


# In[ ]:


# add debt to income ratio a key measure of capacity in credit risk management
application_train['DEBT_TO_INCOME'] = application_train['AMT_CREDIT']/ application_train['AMT_INCOME_TOTAL']
TO ADD REPAYMENT_TO_INCOME
NUMBER OF PREVIOUS DEFAULTS> 90 DAYS
NUMBER OF CURRENT ACTIVE LOANS
#application_train.head()


# In[ ]:


bureau = pd.read_csv("/home/mgwarada/Desktop/Ruvimbo/bureau.csv")


#bureau.shape


# In[ ]:


# bureau data has duplicated customer ids, aggregating variables would help when we mere data

bureau_grouped = bureau.drop(['SK_ID_BUREAU'],axis=1).groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean','max' ,'sum']).reset_index()
#bureau_grouped.head()
bureau_grouped.name= 'bureau_grouped'


# In[ ]:


# to change code ASAP

def format_columns(df):
    columns = ['SK_ID_CURR']

# Iterate through the variables names
    for var in df.columns.levels[0]:
    # Skip the id name
        if var != 'SK_ID_CURR':
        
        # Iterate through the stat names
           for stat in df.columns.levels[1][:-1]:
            # Make a new column name for the variable and stat
                columns.append(str(df.name) + " " + var + " " +str(stat))
    return columns


# In[ ]:


columns = format_columns(bureau_grouped)
bureau_grouped.columns= columns 
#bureau_grouped.head()


# In[ ]:


#df for active loans for each client
#active_loans = bureau[bureau['CREDIT_ACTIVE']=='Active']
#active_loans.head()


# In[ ]:


#active_loans.shape


# In[ ]:


#active_grouped = active_loans.drop(['SK_ID_BUREAU'],axis=1).groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean','max', 'sum']).reset_index()
#active_grouped.head()


# In[ ]:


#active_grouped.columns = columns 
#active_grouped.head()


# In[ ]:


bureau_balance = pd.read_csv("//home/mgwarada/Desktop/Ruvimbo/bureau_balance.csv")
#bureau_balance.head()


# In[ ]:


#bureau_balance.info()


# In[ ]:


# bureau balance max is 0 as debts are recorded as negative number , the min taken insted to represent the maximum loan advanced to a client 
bureau_balance_grouped = bureau_balance.drop(['STATUS'], axis= 1).groupby('SK_ID_BUREAU', as_index = False).agg(['count', 'mean', 'max','min', 'sum']).reset_index()
#bureau_balance_grouped.head()
bureau_balance_grouped.name= 'bureau_balance_grouped'


# In[ ]:


columns = ['SK_ID_BUREAU']

# Iterate through the variables names
for var in bureau_balance_grouped.columns.levels[0]:
    # Skip the id name
    if var != 'SK_ID_BUREAU':
        
        # Iterate through the stat names
        for stat in bureau_balance_grouped.columns.levels[1][:-1]:
            # Make a new column name for the variable and stat
            columns.append('bureau_balance_%s_%s' % (var, stat))


# In[ ]:



bureau_balance_grouped.columns = columns
#bureau_balance_grouped.head()


# In[ ]:


customer_id_lookup= bureau [['SK_ID_BUREAU','SK_ID_CURR']]
#customer_id_lookup.head()


# In[ ]:


bureau_balance_grouped = pd.merge(bureau_balance_grouped, customer_id_lookup, how='left',left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU')
#bureau_balance_grouped.head()


# In[ ]:




bureau_balance_customer = bureau_balance_grouped.drop(['SK_ID_BUREAU'],axis=1).groupby('SK_ID_CURR', as_index = False).agg(['mean', 'max', 'sum']).reset_index()
#bureau_balance_customer.head()
bureau_balance_customer.name='bureau_balance_customer'


# In[ ]:


columns= format_columns(bureau_balance_customer)
bureau_balance_customer.columns= columns
#bureau_balance_customer.head()


# In[ ]:


POS_CASH_balance = pd.read_csv("//home/mgwarada/Desktop/Ruvimbo/POS_CASH_balance.csv")
#POS_CASH_balance.head()


# In[ ]:


#POS_CASH_balance.info()


# In[ ]:


# bureau balance max is 0 as debts are recorded as negative number , the min taken insted to represent the maximum loan advanced to a client 
POS_CASH_grouped = POS_CASH_balance.drop(['SK_ID_PREV'],axis=1).groupby('SK_ID_CURR', as_index = False).agg([ 'mean', 'max', 'sum']).reset_index()
POS_CASH_grouped.name= 'POS_CASH_grouped'


# In[ ]:


columns= format_columns(POS_CASH_grouped)
POS_CASH_grouped.columns= columns
#POS_CASH_grouped.head()


# In[ ]:


credit_card_balance = pd.read_csv("/home/mgwarada/Desktop/Ruvimbo/credit_card_balance.csv")
#credit_card_balance.head()


# In[ ]:


credit_card_balance.info()


# In[ ]:


credit_card_balance_grouped = credit_card_balance.drop( ['SK_ID_PREV'], axis =1).groupby('SK_ID_CURR', as_index = False).agg(['mean', 'max', 'sum']).reset_index()
credit_card_balance_grouped.name= 'credit_card_balance_grouped'


# In[ ]:


columns = format_columns(credit_card_balance_grouped)
credit_card_balance_grouped.columns= columns 
#credit_card_balance_grouped.head()


# In[ ]:


previous_application = pd.read_csv("//home/mgwarada/Desktop/Ruvimbo/previous_application.csv")
#previous_application.head()


# In[ ]:


previous_application.info()


# In[ ]:


previous_application_grouped = previous_application.drop(['SK_ID_PREV'],axis=1).groupby('SK_ID_CURR', as_index = False).agg([ 'mean', 'max', 'sum']).reset_index()
previous_application_grouped.name= 'previous_application_grouped'


# In[ ]:


columns = format_columns(previous_application_grouped)
previous_application_grouped.columns= columns
#previous_application_grouped.head()


# In[ ]:


installments_payments = pd.read_csv("/home/mgwarada/Desktop/Ruvimbo/installments_payments.csv")
#installments_payments.head()


# In[ ]:


installments_payments.info()


# In[ ]:



installments_payments_grouped = installments_payments.drop( ['SK_ID_PREV'],axis=1 ).groupby('SK_ID_CURR', as_index = False).agg(['mean', 'max', 'sum']).reset_index()
installments_payments_grouped.name ='installments_payments_grouped'


# In[ ]:


columns = format_columns(installments_payments_grouped)
installments_payments_grouped.columns = columns 
#installments_payments_grouped.head()


# In[ ]:


#free up memory
del credit_card_balance
del POS_CASH_balance
del previous_application
del bureau_balance
del bureau
del installments_payments 

import gc
gc.collect()


# In[ ]:


#merging datasets DO NOT DELETE
train_data_v0 = application_train.merge(bureau_grouped, on= 'SK_ID_CURR',how='left').merge(credit_card_balance_grouped, on= 'SK_ID_CURR',how='left').merge(installments_payments_grouped,on = 'SK_ID_CURR',how='left').merge(POS_CASH_grouped, on ='SK_ID_CURR',how='left').merge(previous_application_grouped,on ='SK_ID_CURR',how='left').merge(bureau_balance_customer, on = 'SK_ID_CURR',how='left')
train_data_v0.shape


# In[ ]:


train_data_v1 =train_data_v0
#train_data_v1.head()


# In[ ]:


categorical_list = ['SK_ID_CURR']
numerical_list = []
for i in train_data_v1.columns.tolist():
    if train_data_v1[i].dtype=='object':
        categorical_list.append(i)
    else:
        numerical_list.append(i)
print('Number of categorical features:', str(len(categorical_list)))
print('Number of numerical features:', str(len(numerical_list)))
#numerical_list


# In[ ]:


numeric_train = train_data_v1 [numerical_list]
#numeric_train.head()


# In[ ]:


categorical_train = train_data_v1 [categorical_list]
#categorical_train.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in categorical_train:
    if categorical_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(categorical_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(categorical_train[col])
            # Transform both training and testing data
            categorical_train[col] = le.transform(categorical_train[col])
            
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)


# In[ ]:


#create dummies for categorical data
categorical = pd.get_dummies(categorical_train.select_dtypes('object'))
categorical['SK_ID_CURR'] = categorical_train['SK_ID_CURR']
categorical.head()


# In[ ]:


categorical_grouped = categorical.groupby('SK_ID_CURR',as_index = False).agg(['count', 'mean'])
categorical_grouped.name = 'categorical_grouped'
categorical_grouped.head()


# In[ ]:



# List of column names
columnsc = []

# Iterate through the variables names
for var in categorical_grouped.columns.levels[0]:
    # Skip the id name
    if var != 'SK_ID_CURR':
        
        # Iterate through the stat names
        for stat in categorical_grouped.columns.levels[1][:]:
            # Make a new column name for the variable and stat
            columnsc.append('categorical_grouped_%s_%s' % (var, stat))


# In[ ]:


#columns = format_columns(categorical_grouped)
categorical_grouped.columns= columnsc
#categorical_grouped.head()


# In[ ]:




#calculate missing values for each column
cat_percent_missing = (categorical_grouped.isnull().sum(axis = 0)/len(categorical_grouped ))*100
#round(abs(cat_percent_missing),1).sort_values(ascending=False)


# In[ ]:


num_percent_missing = abs((numeric_train.isnull().sum(axis = 0)/len(numeric_train))*100)
#num_percent_missing.sort_values(ascending=False)


# In[ ]:


num_percent_missing = num_percent_missing.index[num_percent_missing> 0.75]


# In[ ]:


#remove variables with more than 75% of data missing
numeric_train = numeric_train.drop(columns = num_percent_missing)


# In[ ]:


train_data_v1 = numeric_train.merge(categorical_grouped, left_on = 'SK_ID_CURR', right_index = True, how = 'left')
train_data_v1.head()


# In[ ]:


#free up memory
#del categorical_grouped
del bureau_grouped
del bureau_balance_grouped
del installments_payments_grouped
del credit_card_balance_grouped
del previous_application_grouped
del POS_CASH_grouped
del num_percent_missing
del cat_percent_missing
gc.collect()


# In[ ]:


train= train_data_v1.drop(['SK_ID_CURR','TARGET'], axis =1)


# In[ ]:


train = train.fillna(train.median())


# In[ ]:


#identify multicolinearity
threshold = 0.8

# Absolute value correlation matrix
corr_matrix = train.corr().abs()
#corr_matrix.head()


# In[ ]:


threshold = 0.8

# Empty dictionary to hold correlated variables
above_threshold_vars = {}

# For each column, record the variables that are above the threshold
for col in corr_matrix:
    above_threshold_vars[col] = list(corr_matrix.index[corr_matrix[col] > threshold])


# In[ ]:


# Track columns to remove and columns already examined
cols_to_remove = []
cols_seen = []
cols_to_remove_pair = []

# Iterate through columns and correlated columns
for key, value in above_threshold_vars.items():
    # Keep track of columns already examined
    cols_seen.append(key)
    for x in value:
        if x == key:
            next
        else:
            # Only want to remove one in a pair
            if x not in cols_seen:
                cols_to_remove.append(x)
                cols_to_remove_pair.append(key)
            
cols_to_remove = list(set(cols_to_remove))
print('Number of columns to remove: ', len(cols_to_remove))


# In[ ]:


to_drop = cols_to_remove


# In[ ]:


train = train.drop(columns = to_drop)


# In[ ]:


#free up memory
del corr_matrix
del to_drop
gc.collect()


# In[ ]:


response = train_data_v1['TARGET']
feature_name = train.columns.tolist()


# In[ ]:



#del numeric_train
#del categorical_train
#gc.collect()


# In[ ]:


#train.fillna(train.median()).head()
train = train.fillna(train.median())


# In[ ]:



from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=4, max_features=0.5, random_state=2018)
random_forest_model.fit(train, response)
features = train.columns.values


# In[ ]:


rF = SelectFromModel(random_forest_model, threshold=0.005)

# Train the selector
rF.fit(train,response)


# In[ ]:


features = train.columns.tolist()


# In[ ]:



model_features=[]
for f_index in rF.get_support(indices=True):
    model_features.append(features[f_index])


# In[ ]:



model_features.append( 'SK_ID_CURR')
model_features.append('TARGET')
model_features


# In[ ]:



train_data_final  = train_data_v1[model_features]
#del train_data_final


# In[ ]:


train_response = train_data_final.TARGET
train_predictor =  train_data_final.drop(columns=['TARGET','SK_ID_CURR'],axis=1)


# In[ ]:



train_predictor = train_predictor.fillna(train_predictor.median())
train_v2 = train_predictor


# In[ ]:


#normalize/scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit only to the training data
train_F_scaled = scaler.fit_transform(train_v2)
#train_F_scaled = scaler.transform(train_v2)


# In[ ]:


del train_data_final 


# In[ ]:


#Load libraries for cross validation
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold


# In[ ]:


from sklearn.neural_network import MLPClassifier


mlp = MLPClassifier(hidden_layer_sizes=(30,30,30),max_iter =500)


# In[ ]:


X_train, X_test,y_train, y_test = train_test_split(train_F_scaled , train_response,test_size =0.4, random_state=42)
mlp.fit(X_train, y_train)
y_pred_prob =mlp.predict_proba( X_test)[:,1]
roc_auc_score(y_test, y_pred_prob)


# In[ ]:



from sklearn.cross_validation import cross_val_score
import numpy as np


# 10-Fold Cross validation
print( np.mean(cross_val_score(mlp, train_F_scaled,train_response, scoring = 'roc_auc', cv=5)))


# In[ ]:


#fit random forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc


                           
randomForestModel = RandomForestClassifier(max_depth=5,random_state=0)


# In[ ]:


X_train, X_test,y_train, y_test = train_test_split(train_F_scaled , train_response,test_size =0.4, random_state=42)
randomForestModel.fit(X_train, y_train)
y_pred_prob = randomForestModel.predict_proba( X_test)[:,1]
roc_auc_score(y_test, y_pred_prob)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import numpy as np

# 10-Fold Cross validation
print( np.mean(cross_val_score(randomForestModel, train_F_scaled,train_response, scoring = 'roc_auc', cv=10)))


# In[ ]:


#fit SVM

from sklearn.svm import SVC
#
SVM_model = SVC(kernel= 'linear',probability= True) 
#SVM_model.fit(train_F_scaled , train_response)


# In[ ]:


X_train, X_test,y_train, y_test = train_test_split(train_F_scaled , train_response,test_size =0.4, random_state=42)
SVM_model.fit(X_train, y_train)
y_pred_prob = SVM_model.predict_proba( X_test)[:,1]
roc_auc_score(y_test, y_pred_prob)


# In[ ]:



from sklearn.cross_validation import cross_val_score

# 10-Fold Cross validation
print( np.mean(cross_val_score(SVM_model, train_F_scaled,train_response, scoring = 'roc_auc', cv=5)))
                                                                                                   


# In[ ]:


from sklearn.model_selection import GridSearchCV

params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
          'gamma': [0.0001, 0.001, 0.01, 0.1],
          'kernel':['linear','rbf'] }

#Create the GridSearchCV object
clf = GridSearchCV(SVM_model,params_grid)

#Fit the data with the best possible parameters
grid_clf = clf.fit(train_F_scaled, train_response)

#Print the best estimator with it's parameters
print (grid_clf.best_estimators)


# In[ ]:


# fit logistic regression 
from sklearn.linear_model import LogisticRegression

# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

# Train on the training data


# In[ ]:


X_train, X_test,y_train, y_test = train_test_split(train_F_scaled , train_response,test_size =0.4, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_prob = log_reg.predict_proba( X_test)[:,1]
roc_auc_score(y_test, y_pred_prob)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import numpy as np


print( np.mean(cross_val_score(log_reg, train_F_scaled,train_response, scoring = 'roc_auc', cv=10)))


# In[ ]:


import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

logistic = linear_model.LogisticRegression()
# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)
# Create grid search using 5-fold cross validation
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

# Fit grid search
best_model = clf.fit(X, y)
# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])


# In[ ]:


#KNN classifier 
from sklearn.neighbors import KNeighborsClassifier

KNN_model = KNeighborsClassifier(n_neighbors=5)

# Train the model usinGfit(X_train, y_train)g the training sets
#KNN_model.fit(train_F_scaled,train_response)


# In[ ]:


X_train, X_test,y_train, y_test = train_test_split(train_F_scaled , train_response,test_size =0.4, random_state=42)
KNN_model.fit(X_train, y_train)
y_pred_prob =  KNN_model.predict_proba( X_test)[:,1]
roc_auc_score(y_test, y_pred_prob)
#0.571898385215427


# In[ ]:


from sklearn.cross_validation import cross_val_score

print( np.mean(cross_val_score(KNN_model , train_F_scaled,train_response, scoring = 'roc_auc', cv=5)))
#0.5487848515932849


# In[ ]:


#naive bayes

from sklearn.naive_bayes import GaussianNB
naive_bayes_model = GaussianNB()


# In[ ]:


X_train, X_test,y_train, y_test = train_test_split(train_F_scaled , train_response,test_size =0.4, random_state=42)
naive_bayes_model.fit(X_train, y_train)
y_pred_prob =  naive_bayes_model.predict_proba( X_test)[:,1]
roc_auc_score(y_test, y_pred_prob)


# In[ ]:



from sklearn.cross_validation import cross_val_score
import numpy as np

print( np.mean(cross_val_score(naive_bayes_model ,train_F_scaled,train_response, scoring = 'roc_auc', cv=5)))


# In[ ]:


from lightgbm import LGBMClassifier
lgbm_model = LGBMClassifier(
        # n_estimators=1000,
        # num_leaves=20,
        # colsample_bytree=.8,
        # subsample=.8,
        # max_depth=7,
        # reg_alpha=.1,
        # reg_lambda=.1,
        # min_split_gain=.01
        n_estimators=10000,
        learning_rate=0.03,
        num_leaves = 22,
        colsample_bytree=0.8,
        subsample=0.8,
        max_depth=6,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_split_gain=0.01,
        min_child_weight=100,
        silent=-1,
        verbose=-1)


X_train, X_test,y_train, y_test = train_test_split(train_F_scaled , train_response,test_size =0.4, random_state=42)
lgbm_model.fit(X_train, y_train)
y_pred_prob =  lgbm_model.predict_proba( X_test)[:,1]
roc_auc_score(y_test, y_pred_prob)


# In[ ]:


from sklearn.cross_validation import cross_val_score

print( np.mean(cross_val_score(lgbm_model , train_F_scaled,train_response, scoring = 'roc_auc', cv=5)))


# In[ ]:


#this formulae is equivalent to cross validation above 
X =train_F_scaled
y = train_response

n=5
kf = StratifiedKFold(n_splits=n, random_state=None) 
test_lgbm=0
sum_lgbm =0
for train_index, test_index in kf.split(X,y):
     
       # print("Train:", train_index, "Validation:",test_index)
      X_train, X_test = X[train_index], X[test_index] 
      y_train, y_test = y[train_index], y[test_index]
      lgbm_model = LGBMClassifier(
        # n_estimators=1000,
        # num_leaves=20,
        # colsample_bytree=.8,
        # subsample=.8,
        # max_depth=7,
        # reg_alpha=.1,
        # reg_lambda=.1,
        # min_split_gain=.01
        n_estimators=10000,
        learning_rate=0.03,
        num_leaves = 22,
        colsample_bytree=0.8,
        subsample=0.8,
        max_depth=6,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_split_gain=0.01,
        min_child_weight=100,
        silent=-1,
        verbose=-1)
      lgbm_model.fit(X_train,y_train)  
      y_pred = lgbm_model .predict_proba( X_test)[:,1]
      y_class =lgbm_model .predict( X_test)
      test_lgbm += roc_auc_score(y_test, y_pred)
      sum_lgbm += metrics.accuracy_score( y_class,  y_test)
    
accuracy_ratio_lgbm =sum_lgbm

test_lgbm/n


# In[ ]:


#Plot RO's for all models 
plt.plot(fpr_log, tpr_log,label= 'Logistic regression' +str(auc_score_mean_log))
plt.plot(fpr_NN, tpr_NN, label = 'Nueral Network' + str(auc_score_mean_NN))
plt.plot(fpr_KNN, tpr_KNN, label = 'K Nearest Neighbour' +str(auc_score_mean_KNN)) 
plt.plot(fpr_NB, tpr_NB, label = 'Naive Bayes' + str(auc_score_mean_NB))
plt.plot(fpr_SVD, tpr_SVD, label = 'Support_Vector Machine' + str(auc_score_mean_SVM))
plt.plot([0, 1], [0, 1],'-', label = 'Poor model')
plt.title("ROC Curves for Logistic regression, Support Vector Machine, Neural Network, K Nearest Neighbours and Naive Bayes")
plt.show()

