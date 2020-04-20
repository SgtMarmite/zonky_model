# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
from dateutil.relativedelta import *
from datetime import date
from sklearn.compose import ColumnTransformer
import pickle
from sklearn.pipeline import Pipeline
import joblib


filename = 'archiv_pujcek_31_03_2020.csv'

def add_months_to_date(date,months_to_add):
    
    new_date = date + relativedelta(months =+ months_to_add)
    return new_date

def compare_dates(date_to_compare):
    
    if date.today() < date_to_compare.date():
        set_delete = 1
    else:
        set_delete = 0
    return set_delete    

def balance(data):
    count_class_0, count_class_1 = data.target.value_counts()
    
    df_class_0 = data[data['target'] == 0]
    df_class_1 = data[data['target'] == 1]
    
    df_class_0_under = df_class_0.sample(count_class_1, random_state=7)
    data = pd.concat([df_class_0_under, df_class_1], axis=0)
    
    return data


def dataload():
    #This function loads the data
    df = pd.read_csv(filename, sep=';', decimal=',')
    df = df.drop(['ID', 'Měsíc reportu', 'Měsíc poslední delikvence', 'Max. dnů po splatnosti', 'Pozdě zaplacené splátky', 'Max. částka po splatnosti', 'Zaplaceno na pokutách'], axis=1)    
    df = df.replace(['Ano', 'Ne'], [1, 0])
    df.loc[df.Zesplatnění == 1, 'target'] = 1
    df.loc[df.Stav == 'Zesplatněno', 'target'] = 1
    df.loc[df['Aktuálně dnů po splatnosti'] > (365/2), 'target'] = 1
    df.fillna(value=0, inplace=True)
    df.loc[df['Půjčovač investorem'] != 0, 'Půjčovač investorem'] = 1
    df.Poskytnuto = pd.to_datetime(df.Poskytnuto)
    df['assumed_repaid_date'] = df.apply(lambda x: add_months_to_date(x['Poskytnuto'],x['Původní počet splátek']),axis=1)
    df['set_delete'] = df.apply(lambda x: compare_dates(x['assumed_repaid_date']),axis=1)
    df['loan_start_month'] = df['Poskytnuto'].dt.month
    df = df[~((df.set_delete == 1) & (df.Stav == 'Aktivní'))]
    df = df.drop(['set_delete'], axis=1)
    df['Objem'] = pd.to_numeric(df['Objem'])
    df['Původní počet splátek'] = pd.to_numeric(df['Původní počet splátek'])
    df['Výše splátky'] = df['Objem']/df['Původní počet splátek']
    dataset = df.drop(['Poskytnuto', 'Odložené splátky', 'Stav','Aktuálně dnů po splatnosti', 'Aktuální částka po splatnosti', 'Zesplatnění', 'Aktuální počet splátek', 'Počet zbývajících splátek', 'Ztraceno', 'Měsíc doplacení', 'Splaceno jistina', 'Splaceno úrok', 'Zbývá splatit jistina', 'Zbývá splatit úrok', 'Půjčovač investorem', 'assumed_repaid_date'], axis=1)
    dataset = dataset[
    ['Kraj',
     'Příjem',
     'Účel',
     'loan_start_month',
     'Úroková sazba',
     'Objem',
     'Pořadí půjčky',
     'Pojištěno',
     'Původní počet splátek',
     'Příběh',
     'Výše splátky',
     'target'
     ]]
    return dataset

def categorical_encoding(X):
    
    global onehot
    global X_categorical
    global X_continuous
    X_categorical = X[['Kraj','Příjem','Účel','loan_start_month']]
    X_continuous = X[['Úroková sazba', 'Objem', 'Pořadí půjčky','Pojištěno', 'Původní počet splátek', 'Příběh', 'Výše splátky']]
    onehot = OneHotEncoder(categories='auto', sparse = False)
    onehot = onehot.fit(X_categorical)
    
    with open('zonky_onehot_encoder.pkl', 'wb') as handle:
        pickle.dump(onehot, handle, protocol=4)
        
    onehot = pickle.load(open('zonky_onehot_encoder.pkl', 'rb'))
    
    X_categorical = onehot.transform(X_categorical)
    X_ = np.column_stack((X_categorical,X_continuous))
    
    return X_

dataset = dataload()
#dataset = balance(dataset)

X = dataset[['Kraj', 'Příjem', 'Účel', 'loan_start_month', 'Úroková sazba', 'Objem', 'Pořadí půjčky','Pojištěno', 'Původní počet splátek', 'Příběh', 'Výše splátky']]
y = dataset[['target']]
y = y.values.ravel()

X = categorical_encoding(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7)

# Train XGBoost
from xgboost import XGBClassifier
xgb_model = XGBClassifier(objective = 'binary:logistic')
from sklearn.model_selection import RandomizedSearchCV
parameters = {
        'learning_rate': sp.stats.uniform(0.001, 0.2),
        'min_child_weight': [4, 5, 6, 7, 8],
        'booster': ['gbtree'], #dart
        'subsample': sp.stats.uniform(0.5, 0.45),
        'colsample_bytree': sp.stats.uniform(0.5, 0.45),
        'max_depth': sp.stats.randint(5, 12),
        'gamma': sp.stats.uniform(0.001, 0.5),
        'n_estimators': sp.stats.randint(80, 160),
        'max_delta_step': sp.stats.randint(0, 6)
        }

# Random search of parameters, using 5 fold cross validation,
random = RandomizedSearchCV(estimator = xgb_model, param_distributions = parameters, n_iter = 400, cv = 5, verbose=True, random_state=7, return_train_score=True, scoring='roc_auc',
                               n_jobs = 11)
# Fit the random search model
random.fit(X_train, y_train)
#Display best parameters
best_scores_array = random.cv_results_
best_params = random.best_params_
print("--------------------------------------------")
print("Best parameters set found by random search:")
print(best_params)
print("--------------------------------------------")
best_estimator = random.best_estimator_
print(best_estimator)


result_list = pd.DataFrame(random.cv_results_)[['params', 'mean_train_score', 'mean_test_score']] 
# Predict for test_set  
y_pred = best_estimator.predict(X_test)
y_pred = y_pred.reshape(-1, 1)
 
from sklearn.metrics import confusion_matrix, f1_score
f1score = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred) 
fpr, tpr, threshold = roc_curve(y_test, random.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

# image drawing
plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label = 'MLP AUC = %0.4f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

y_pred = best_estimator.predict_proba(X_test)
compare_results = np.concatenate((y_test.reshape(-1,1), y_pred[:,1].reshape(-1,1)), axis = 1)

all_data_best_estimator = best_estimator.fit(X,y)

with open('zonky_xgboost.pkl', 'wb') as handle:
    pickle.dump(all_data_best_estimator, handle, protocol=4)