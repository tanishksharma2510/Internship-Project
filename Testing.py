import numpy as np
import pandas as pd
import pycountry
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings("ignore",category=UserWarning,module="sklearn")

hotel_data=pd.read_csv("hotel_bookings_updated_2024.csv")

hotel_data['children']=hotel_data['children'].fillna(value=0)
hotel_data['Total Children']=hotel_data['children']+hotel_data['babies']
hotel_data['meal']=hotel_data['meal'].replace('Undefined',np.nan)
hotel_data['market_segment']=hotel_data['market_segment'].replace('Undefined',np.nan)
hotel_data['reserved_room_type']=hotel_data['reserved_room_type'].replace(['L','P'],[np.nan,np.nan])
hotel_data.dropna(subset=['meal','market_segment','reserved_room_type'],inplace=True)
hotel_data.drop(columns=['arrival_date_year','arrival_date_day_of_month','children','babies','distribution_channel','is_repeated_guest','booking_changes','agent',
'company','days_in_waiting_list','required_car_parking_spaces','total_of_special_requests','reservation_status','reservation_status_date','assigned_room_type'],
inplace=True)
hotel_data.drop_duplicates(inplace=True)
hotel_data=hotel_data.rename(columns={
    'hotel':'Hotel Category',
    'is_canceled':'Cancellation Status',
    'lead_time':'Lead Time (in days)',
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
    'adr':'Average Daily Rate',
    'city':'City',
})
country_dict={country.alpha_3: country.name for country in pycountry.countries}
hotel_data['Country']=hotel_data['Country'].map(country_dict).ffill()
hotel_data['Room Category']=hotel_data['Room Category'].replace(['A','B','C','D','E','F','G','H'],['Standard','Economy','Family','Deluxe','Superior','Executive','Suite','Presidential Suite'])
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
    clf=RandomizedSearchCV(estimator=mp['model'],param_distributions=mp['params'],n_iter=10,cv=5,scoring='accuracy',n_jobs=-1,
    random_state=42)
    clf.fit(x_train,y_train)
    scores.append({
        'Model':mname,
        'Best Params':clf.best_params_,
        'Best Scores':clf.best_score_*100
    })
score_df=pd.DataFrame(scores)
params1=score_df['Best Params'].iloc[0]
params2=score_df['Best Params'].iloc[1]
params3=score_df['Best Params'].iloc[2]
params4=score_df['Best Params'].iloc[3]
model1=HistGradientBoostingClassifier(**params1,random_state=42)
model2=XGBClassifier(**params2,random_state=42)
model3=LogisticRegression(**params3)
model4=DecisionTreeClassifier(**params4)
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)
df=pd.DataFrame({
    'Models':['HistGradientBoostingClassifier','XGBClassifier','Logistic Regression','Decision Trees'],
    'Accuracy':[
        accuracy_score(y_test,model1.predict(x_test))*100,
        accuracy_score(y_test,model2.predict(x_test))*100,
        accuracy_score(y_test,model3.predict(x_test))*100,
        accuracy_score(y_test,model4.predict(x_test))*100
    ],
    'Precision':[
        precision_score(y_test,model1.predict(x_test))*100,
        precision_score(y_test,model2.predict(x_test))*100,
        precision_score(y_test,model3.predict(x_test))*100,
        precision_score(y_test,model4.predict(x_test))*100
    ],
    'Recall':[
        recall_score(y_test,model1.predict(x_test))*100,
        recall_score(y_test,model2.predict(x_test))*100,
        recall_score(y_test,model3.predict(x_test))*100,
        recall_score(y_test,model4.predict(x_test))*100
    ],
    'F1 Score':[
        f1_score(y_test,model1.predict(x_test))*100,
        f1_score(y_test,model2.predict(x_test))*100,
        f1_score(y_test,model3.predict(x_test))*100,
        f1_score(y_test,model4.predict(x_test))*100
    ]
})
df.set_index('Models',inplace=True)
fig=df.plot(kind='bar',color=['blue','darkorange','green','red'],edgecolor='black')
plt.title('Performance Comparison of Classification Models',fontsize=20)
plt.xlabel('Classification Models',labelpad=20,fontsize=15)
plt.ylabel('Metrics Score (in %)',labelpad=20,fontsize=15)
plt.xticks(rotation=90)
plt.legend(loc='best',fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()