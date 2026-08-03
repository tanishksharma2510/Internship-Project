import pickle
import pandas as pd
import numpy as np
import pycountry
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# Data Acquisition:
hotel_data=pd.read_csv("hotel_bookings_updated_2024.csv")

# Data Processing:
hotel_data['children']=hotel_data['children'].fillna(value=0)
hotel_data['Total Children']=hotel_data['children']+hotel_data['babies']
hotel_data['meal']=hotel_data['meal'].replace('Undefined',np.nan)
hotel_data['market_segment']=hotel_data['market_segment'].replace('Undefined',np.nan)
hotel_data['reserved_room_type']=hotel_data['reserved_room_type'].replace(['L','P'],[np.nan,np.nan])
hotel_data.dropna(subset=['meal','market_segment','reserved_room_type'],inplace=True)
hotel_data.drop(columns=['arrival_date_year','arrival_date_day_of_month','children','babies','distribution_channel','is_repeated_guest',
'booking_changes','agent','company','days_in_waiting_list','required_car_parking_spaces','total_of_special_requests','reservation_status',
'reservation_status_date','assigned_room_type'],
inplace=True)
hotel_data.drop_duplicates(inplace=True)
hotel_data=hotel_data.rename(columns={
    'hotel':'Hotel Category',
    'is_canceled':'Cancellation Status',
    'lead_time':'Lead Time (in days)', # Lead Time: Number of days between the booking date and the customer's arrival (check-in) date.
    'arrival_date_month':'Month',
    'arrival_date_week_number':'No. of Weeks',
    'stays_in_weekend_nights':'Total Weekend Nights Stay',
    'stays_in_week_nights':'Total Week Nights Stay',
    'adults':'Total Adults',
    'meal':'Meal Plan',
    'country':'Country',
    'market_segment':'Booking Source',
    'previous_cancellations':'Total Past Cancellations',
    'previous_bookings_not_canceled':'Total Past Non-cancellations',
    'reserved_room_type':'Room Category',
    'deposit_type':'Deposit Category',
    'customer_type':'Customer Category',
    'adr':'Average Daily Rate', # ADR: The average room price charged per occupied room per day.
    'city':'City',
})
country_dict={country.alpha_3: country.name for country in pycountry.countries}
hotel_data['Country']=hotel_data['Country'].map(country_dict).ffill()
hotel_data['Room Category']=hotel_data['Room Category'].replace(['A','B','C','D','E','F','G','H'],['Standard','Economy','Family','Deluxe',
'Superior','Executive','Suite','Presidential Suite'])
hotel_data['Meal Plan']=hotel_data['Meal Plan'].replace(['BB','HB','FB','SC'],['Bed & Breakfast','Half Board','Full Board','Self Catering'])
hotel_data['Booking Source']=hotel_data['Booking Source'].replace(['Online TA','Offline TA/TO','Complementary'],
['Online Travel Agent','Offline Travel Agent/Tour Operator','Free Booking'])
hotel_data['Customer Category']=hotel_data['Customer Category'].replace(['Transient','Transient-Party'],
['Individual Booking','Individual Booking (for Group)'])
order=['January','February','March','April','May','June','July','August','September','October','November','December']
hotel_data['Month']=pd.Categorical(values=hotel_data['Month'],categories=order,ordered=True)
hotel_data.sort_values(by='Month',ascending=True,inplace=True)
hotel_data['Hotel Category']=hotel_data['Hotel Category'].str.split(' - ').str[0]
hotel_data['Total Children']=hotel_data['Total Children'].astype(int)

# Data Exploration (EDA):
# print(hotel_data.info(),"\n\nHotel Bookings Demand Dataset Description:\n",hotel_data.describe(),"\n\nHotel Bookings Demand Dataset:\n",
# hotel_data.head(10),"\n\nShape of the Dataset:\n",hotel_data.shape,"\n\nColumn Labels:\n",hotel_data.columns,
# f"\n\nTotal Null Values={hotel_data.isnull().sum().sum()}",f"\n\nTotal Duplicated Values={hotel_data.duplicated().sum()}",
# f"\n\nTotal Unique Values=\n{hotel_data.nunique()}")

# Modelling:
# 1st Part: Finding the Best Model and Features
x=hotel_data.drop(columns=['Cancellation Status'])
y=hotel_data['Cancellation Status']
col=x.select_dtypes(include=['str','object','category']).columns.tolist()
enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1,encoded_missing_value=-1,dtype=int)
x[col]=enc.fit_transform(x[col])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
best_model={
    'HistGradientBoostingClassifier':{
        'model':HistGradientBoostingClassifier(random_state=42),
        'params':{
            'learning_rate':[0.05,0.1,0.15],
            'max_iter':[100,200,300],
            'max_leaf_nodes':[30,45,60],
            'min_samples_leaf':[50,70,80],
            'l2_regularization':[5.0,7.0,9.0]
        }
    },
    'XGBClassifier':{
        'model':XGBClassifier(random_state=42,learning_rate=0.1,n_estimators=100),
        'params':{
            'max_depth':[5,8,10],
            'min_child_weight':[6,8,10],
            'gamma':[2,3,4],
            'subsample':[0.7,0.8,0.9],
            'colsample_bytree':[0.7,0.8,0.9]
        }
    },
    'Logistic Regression':{
        'model':LogisticRegression(),
        'params':{
            'C':[0.001,0.01,0.1],
            'solver':['lbfgs','liblinear','saga']
        }
    },
    'Decision Trees':{
        'model':DecisionTreeClassifier(),
        'params':{
            'max_depth':[5,10,15],
            'min_samples_split':[6,12,18],
            'min_samples_leaf':[2,4,6],
            'max_features':['sqrt','log2'],
            'criterion':['gini','entropy','log_loss']
        }
    }
}
scores=[]
for mname,mp in best_model.items():
    clf=RandomizedSearchCV(estimator=mp['model'],param_distributions=mp['params'],n_iter=10,cv=5,scoring='f1_macro',n_jobs=-1,
    random_state=42)
    clf.fit(x_train,y_train)
    scores.append({
        'Model':mname,
        'Best Params':clf.best_params_,
        'Best Scores':clf.best_score_*100
    })
score_df=pd.DataFrame(scores)
print(score_df)

# 2nd Part: ML Pipeline
X=hotel_data.drop(columns=['Cancellation Status'])
Y=hotel_data['Cancellation Status']
cats_col=X.select_dtypes(include=['str','object','category']).columns.tolist()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42,stratify=Y)
best_params=score_df['Best Params'].iloc[0]
preprocessor=ColumnTransformer(transformers=[
        ('encoder',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1,encoded_missing_value=-1,dtype=int),cats_col),
    ],
    remainder='passthrough'
)
pipeline=Pipeline([
    ['preprocessor',preprocessor],
    ['imputer',IterativeImputer(max_iter=10,random_state=42)],
    ['model',HistGradientBoostingClassifier(**best_params,random_state=42)]
])
pipeline.fit(X_train,Y_train)
Y_pred=pipeline.predict(X_test)
# # Calculating Permutation Importance which tells us the importance of each feature in the model by showing the drop of a 
# # particular model's accuracy when a specific feature is randomly shuffled.
# result=permutation_importance(pipeline,X_test,Y_test,n_repeats=10,random_state=42,n_jobs=-1)
# feature_importances=pd.DataFrame({
#     'Feature':X.columns,
#     'Importance_Mean':result.importances_mean,
#     'Importance_Std':result.importances_std
# }).sort_values(by='Importance_Mean',ascending=False)
# print(feature_importances)
cm = confusion_matrix(Y_test, Y_pred)

artifacts = {
    "pipeline": pipeline,
    "hotel_data": hotel_data,
    "score_df": score_df,
    "X_columns": list(X.columns),
    "cats_col": cats_col,
    "Y_test": Y_test,
    "Y_pred": Y_pred,
    "cm": cm,
}

with open("artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)
print("\nSaved artifacts.pkl — copy this file (and NOT the CSV or training deps) into your deploy folder.")