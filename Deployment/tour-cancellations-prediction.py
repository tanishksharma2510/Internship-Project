import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,confusion_matrix
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

with open("artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)
pipeline = artifacts["pipeline"]
hotel_data = artifacts["hotel_data"]
score_df = artifacts["score_df"]
X_columns = artifacts["X_columns"]
cats_col = artifacts["cats_col"]
Y_test = artifacts["Y_test"]
Y_pred = artifacts["Y_pred"]

# Deployment:
# 1st Part: App Layout
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout=html.Div(children=[
    html.H1("Tour Cancellations Prediction App",style={'textAlign':'center','color':'black','font-size':40,
    'font-family':'Times New Roman'}),
    dcc.Tabs(id='tabs-input',value='tab-1',children=[
        dcc.Tab(label='About the Model',value='tab-1'),
        dcc.Tab(label='Prediction',value='tab-2'),
        dcc.Tab(label='Dashboard',value='tab-3'),
    ]),
    dcc.Loading(
        id="loading-app",
        type="circle",
        color="red",
        target_components={"tabs-output": "children"},
        children=[html.Div(id="tabs-output")],
    ),
])

# Tabs:
@app.callback(Output(component_id='tabs-output',component_property='children'),
Input(component_id='tabs-input',component_property='value'))

def get_tab(tab_val):
    # Prediction Tab:
    if (tab_val=='tab-2'):
        return html.Div(children=[
            html.H2("Prediction of Tour Cancellations",style={'textAlign':'center','color':'black','font-size':30,
            'font-family':'Times New Roman'}),
            html.Hr(style={'background-color':'black','height':3}),
            html.Br(),
            html.Div(children=[
                html.P(children=["Please enter the following details of ",html.Strong("YOUR",style={'font-weight':900}),
                " tour booking: -"]),
                html.Br(),
                html.Div(children=[
                    html.Label("Type of Deposit:"),
                    dcc.Dropdown(
                        id='dropdown-deposit',
                        options=[{'label':i,'value':i} for i in sorted(hotel_data['Deposit Category'].unique())],
                        placeholder='Select from the given options',
                        searchable=False,
                        clearable=True,
                    ),
                    html.Br(),
                    html.Label("Country of Origin:"),
                    dcc.Dropdown(
                        id='dropdown-country',
                        options=[{'label':i,'value':i} for i in sorted(hotel_data['Country'].unique())],
                        placeholder='Select from the given options',
                        searchable=True,
                        clearable=True,
                    ),
                    html.Br(),
                    html.Label("Source of Booking:"),
                    dcc.Dropdown(
                        id='dropdown-booking-source',
                        options=[{'label':i,'value':i} for i in sorted(hotel_data['Booking Source'].unique())],
                        placeholder='Select from the given options',
                        searchable=False,
                        clearable=True,
                    ),
                ]),
                html.Br(),
                html.Br(),
                html.Div(children=[
                    html.Label("Enter your Lead Time (in days):"),
                    dcc.Input(
                        id='input-lead-time',
                        type='number',
                        placeholder='Enter here',
                        spellCheck=True,
                        required=True
                    ),
                    html.Br(),
                    html.Br(),
                    html.Label("Enter any Past Cancellations made by you:"),
                    dcc.Input(
                        id='input-past-cancellations',
                        type='number',
                        placeholder='Enter here',
                        spellCheck=True,
                        required=True,
                    )
                ])
            ],style={'font-size':15,'font-family':'Arial','color':'black'}),
            html.Br(),
            html.Div(children=[
                html.Button('Submit',id='button',style={'font-size':20,'background-color':'blue','color':'white'}),
                html.Br(),
                html.Br(),
                html.H3("Prediction of Tour:",style={'color':'black','font-size':20,'font-family':'Times New Roman'}),
                dcc.Loading(
                    id='loading-output',
                    type='cube',
                    color='seagreen',
                    children=[
                        html.Div(id='prediction'),
                        html.Div(id='chances'),
                    ]
                )
            ],style={'font-family':'Arial'})
        ])

    # Dashboard Tab:
    elif (tab_val=='tab-3'):
        return html.Div(children=[
            html.H2("Hotel Bookings Dashboard",style={'textAlign':'center','color':'black','font-size':30,
            'font-family':'Times New Roman'}),
            html.Hr(style={'background-color':'black','height':3}),
            html.Br(),
            html.Div(children=[
                html.Label("Select a City:"),
                dcc.Dropdown(
                    id='dropdown-1',
                    options=[{'label':i,'value':i} for i in sorted(hotel_data['City'].unique())],
                    placeholder='Select from the given options',
                    searchable=False,
                    clearable=True,
                ),
                html.Br(),
                html.Label("Select a Hotel Type:"),
                dcc.Dropdown(
                    id='dropdown-2',
                    options=[{'label':i,'value':i} for i in sorted(hotel_data['Hotel Category'].unique())],
                    placeholder='Select from the given options',
                    clearable=True,
                    searchable=False,
                ),
            ],style={'font-size':20,'font-family':'Arial','color':'black'}),
            html.Br(),
            html.Div(children=[
                html.Div(children=[
                    html.Label("Type Of Customer:", style={'fontWeight':'bold','fontSize':15,'font-family':'Arial'}),
                    dcc.Checklist(
                        id='checklist-1',
                        options=[{'label':i,'value':i} for i in sorted(hotel_data['Customer Category'].unique())],
                        inline=False,
                        labelStyle={'font-size':15,'color':'black','font-family':'Arial','marginBottom':'8px'}
                    ),
                    html.Br(),
                    html.Label("Deposit Filter:", style={'fontWeight':'bold','fontSize':15,'font-family':'Arial'}),
                    dcc.Checklist(
                        id='checklist-2',
                        options=[{'label':i,'value':i} for i in sorted(hotel_data['Deposit Category'].unique())],
                        inline=True,
                        labelStyle={'font-size':15,'color':'black','font-family':'Arial','marginBottom':'8px'}
                    )
                ],style={'width':'20%','minWidth':'200px','padding':'15px','backgroundColor':'beige','borderRadius':'5px',
                'marginRight':'20px'}),
                html.Div(children=[
                    dcc.Loading(
                        id='loading-1',
                        type='graph',
                        children=dcc.Graph(id='line-plot'),
                    ),
                    dcc.Loading(
                        id='loading-2',
                        type='graph',
                        children=dcc.Graph(id='area-plot'),
                    ),
                    dcc.Loading(
                        id='loading-3',
                        type='graph',
                        children=dcc.Graph(id='bar-plot'),
                    ),
                    dcc.Loading(
                        id='loading-4',
                        type='graph',
                        children=dcc.Graph(id='hist-plot'),
                    ),
                    dcc.Loading(
                        id='loading-5',
                        type='graph',
                        children=dcc.Graph(id='pie-plot'),
                    ),
                    dcc.Loading(
                        id='loading-6',
                        type='graph',
                        children=dcc.Graph(id='sun-plot'),
                    )
                ],style={'display':'block','width':'80%'}),
            ],style={'display':'flex','flexDirection':'row','padding':'10px','maxWidth':'100%'})
        ])
        
    # About the Model Tab:
    else:
        return html.Div(children=[
            html.Div(children=[
                html.H3("Overview",style={'font-size':20,'color':'black','font-family':'Times New Roman'}),
                html.Div(children=[
                    html.P("Tour Cancellations Prediction App is a machine learning-based web application designed to predict whether a hotel booking is likely to be cancelled. The system analyzes user-provided booking information and generates a prediction along with the confidence level of the result. It also provides an interactive dashboard for exploring booking trends and customer behavior using historical hotel booking data."),
                    html.P(["Some key features of this app include: -",html.Ul(children=[
                        html.Li("Predicts booking cancellation."),
                        html.Li("Displays prediction confidence."),
                        html.Li("Provides interactive dashboard for data insights."),
                        html.Li("Has user-friendly (easy to use) interface.")
                    ])])
                ],style={'font-size':15,'color':'black','font-family':'Arial'}),
            ]),
            html.Hr(style={'backgroundColor':'black','height':1}),
            html.Div(children=[
                html.H3("Purpose & Goal",style={'font-size':20,'color':'black','font-family':'Times New Roman'}),
                html.P("The primary purpose of this application is to predict the likelihood of hotel booking cancellations using machine learning techniques. By identifying bookings with a high probability of cancellation, the application helps support organizations with better decision-making, minimize potential revenue loss, and improve booking management. It also provides an interactive dashboard for analyzing booking trends and customer behavior through visualizations."),
                html.P(["The goal of this application is to: -",html.Ol(children=[
                    html.Li("Develop an accurate, user-friendly, and interactive prediction system."),
                    html.Li("Assist hotels and travel businesses in making data-driven decisions."),
                    html.Li("To enhance operational efficiency, optimize resource planning, and contribute to effective revenue protection strategies.")
                ])])
            ],style={'font-size':15,'color':'black','font-family':'Arial'}),
            html.Hr(style={'backgroundColor':'black','height':1}),
            html.Div(children=[
                html.H3("Model Information",style={'font-size':20,'color':'black','font-family':'Times New Roman'}),
                html.Div(children=[
                    html.P("This application uses the Histogram-Based Gradient Boosting Classifier to predict whether a hotel booking is likely to be cancelled. The model was trained using preprocessed historical booking data after handling missing values, encoding categorical features, and selecting relevant variables."),
                    html.P("This model was chosen because it: -"),
                    html.Div(children=[
                        html.Ul(children=[
                            html.Li("Provides high prediction F1 Score on structured tabular data."),
                            html.Li("Trains faster by grouping continuous values into histograms."),
                            html.Li("Handles large datasets efficiently with lower memory usage."),
                            html.Li("Captures complex non-linear relationships between booking features."),
                            html.Li("Offers good generalization, reducing the risk of overfitting.")
                        ],style={'width':'35%','minWidth':'300px','padding':'15px','marginRight':'20px'}),
                        html.Div(children=[dcc.Graph(figure=comp_fig,config={'responsive':True},style={'width':'100%','height':'300px'})],
                        style={'width':'60%'})
                    ],style={'display':'flex','flexDirection':'row','alignItems':'center','justifyContent':'space-between'})
                ],style={'font-size':15,'color':'black','font-family':'Arial'})
            ]),
            html.Hr(style={'backgroundColor':'black','height':1}),
            html.Div(children=[
                html.H3("Dataset and ETL Information",style={'font-size':20,'color':'black','font-family':'Times New Roman'}),
                html.Div(children=[
                    html.P("The application is built using the Hotel Booking Demand Dataset, which contains historical booking information collected from hotels. The dataset provides various booking-related attributes that are used to train the machine learning model and analyze customer booking patterns."),
                    html.Ol(children=[
                        html.Li([html.Strong("Dataset: ",style={'font-weight':900}),"Hotel Booking Demand Dataset"]),
                        html.Li([html.Strong("Source: ",style={'font-weight':900}),"Kaggle"]),
                        html.Li([html.Strong("Records: ",style={'font-weight':900}),"119,000+ hotel booking records"]),
                        html.Li([html.Strong("Features: ",style={'font-weight':900}),"33 columns (including target variable)"]),
                        html.Li([html.Strong("Target Variable: ",style={'font-weight':900}),"is_canceled"]),
                        html.Li([html.Strong("Data Type: ",style={'font-weight':900}),"Structured tabular data"]),
                        html.Li([html.Strong("Objective: ",style={'font-weight':900}),"Predict whether a hotel booking will be cancelled."])
                    ])
                ]),
                html.Br(),
                html.Div(children=[
                    html.P("The application follows the ETL (Extract, Transform, Load) process to prepare the dataset for analysis and prediction. This process ensures that the data is clean, consistent, and suitable for training the machine learning model."),
                    html.Ol(children=[
                        html.Li([html.Strong("Extract: ",style={'font-weight':900}),"Loaded the dataset from a CSV file."]),
                        html.Li([html.Strong("Transform: ",style={'font-weight':900}),html.Ul(children=[
                            html.Li("Removed unnecessary columns."),
                            html.Li("Handled missing values."),
                            html.Li("Encoded categorical variables."),
                            html.Li("Selected relevant features."),
                            html.Li("Split the data into training and testing sets."),
                        ])]),
                        html.Li([html.Strong("Load: ",style={'font-weight':900}),"Loaded the processed data into the machine learning pipeline for model training and prediction."])
                    ])
                ])
            ],style={'font-size':15,'color':'black','font-family':'Arial'}),
            html.Hr(style={'backgroundColor':'black','height':1}),
            html.Div(children=[
                html.H3("Performance Metrics",style={'font-size':20,'color':'black','font-family':'Times New Roman'}),
                html.Div(children=[
                    html.Div(children=[
                        html.Table(children=[
                            html.Thead(html.Tr([
                                html.Th("METRIC",style={'border':'1px solid black','padding':'8px'}),
                                html.Th("VALUE",style={'border':'1px solid black','padding':'8px'})
                            ])),
                            html.Tbody([
                                html.Tr([
                                    html.Td("Accuracy",style={'border':'1px solid black','padding':'8px'}),
                                    html.Td(f"{accuracy_score(Y_test,Y_pred)*100:.2f}%",style={'border':'1px solid black','padding':'8px'})
                                ]),
                                html.Tr([
                                    html.Td('Precision',style={'border':'1px solid black','padding':'8px'}),
                                    html.Td(f"{precision_score(Y_test,Y_pred,average='macro')*100:.2f}%",style={'border':'1px solid black','padding':'8px'})
                                ]),
                                html.Tr([
                                    html.Td('Recall',style={'border':'1px solid black','padding':'8px'}),
                                    html.Td(f"{recall_score(Y_test,Y_pred,average='macro')*100:.2f}%",style={'border':'1px solid black','padding':'8px'})
                                ]),
                                html.Tr([
                                    html.Td('F1 Score',style={'border':'1px solid black','padding':'8px'}),
                                    html.Td(f"{f1_score(Y_test,Y_pred,average='macro')*100:.2f}%",style={'border':'1px solid black','padding':'8px'})
                                ])
                            ],style={'textAlign':'center'})
                        ],style={'width':'100%','borderBottom':'1px solid #ddd','borderCollapse':'collapse'})
                    ],style={'width':'35%','minWidth':'300px','padding':'15px','marginRight':'20px'}),
                    html.Div(children=[dcc.Graph(figure=conmat,config={'responsive':True},style={'width':'100%','height':'300px'})],
                        style={'width':'60%'})
                ],style={'display':'flex','flexDirection':'row','alignItems':'center','justifyContent':'space-between','marginTop':'40px'}),
                html.P(["The performance of the model is evaluated using standard classification metrics, i.e., ",html.Em("Accuracy"),", ",html.Em("Precision"),", ",html.Em("Recall"),", and ",html.Em("F1 Score"),", and the ",html.Em("Confusion Matrix"),r". The model achieves an Accuracy of 82.27%, correctly predicting most booking outcomes. A Precision of 81.97% indicates that the majority of bookings predicted as cancelled were actually cancelled, while a Recall of 79.29% reflects the model's ability to identify a large proportion of actual cancellations. The F1-Score of 80.24% demonstrates a balanced performance between precision and recall."]),
                html.Br(),
                html.P("The confusion matrix visually summarizes the model's predictions by showing the number of correctly and incorrectly classified bookings. Around 30,000 bookings were correctly predicted as cancelled and not cancelled showing a strong sign of how our model generalizes to an unseen dataset."),
                html.Br(),
                html.P("Overall, these results suggest that the model demonstrates strong predictive performance and serves as an effective decision-support tool for identifying potential booking cancellations. By enabling early risk detection, it helps the hospitality industry to implement proactive strategies to minimize revenue loss and optimize booking management."),
            ],style={'font-size':15,'color':'black','font-family':'Arial'}),
            html.Hr(style={'backgroundColor':'black','height':1}),
            html.Div(children=[
                html.H3("Technologies Used",style={'font-size':20,'color':'black','font-family':'Times New Roman'}),
                html.Ol(children=[
                    html.Li("Python"),
                    html.Li("NumPy"),
                    html.Li("Pandas"),
                    html.Li("Plotly Express and Graph Objects"),
                    html.Li("Plotly Dash"),
                    html.Li("Scikit-Learn")
                ])
            ],style={'font-size':15,'color':'black','font-family':'Arial'}),
            html.Hr(style={'backgroundColor':'black','height':3}),
            html.Div(children=[
                html.H2("About the Developer",style={'font-size':30,'color':'black','font-family':'Times New Roman'}),
                html.P(["I am Tanishk Sharma, a B.Tech student from Amity University, Noida, specializing in ",html.Strong("Data Science",style={'font-weight':900})," with a strong interest in Machine Learning, Data Analytics, and Business Intelligence. My journey in data science began by completing the ",html.Strong("IBM Data Science Professional Certificate",style={'font-weight':900})," on ",html.Strong("Coursera",style={'font-weight':900}),", where I built a solid foundation in Python, data analysis, visualization, and machine learning."]),
                html.P(["To gain practical industry experience, I completed a ",html.Strong("Data Scientist Internship")," at ",html.Strong("TUMLARE SOFTWARE SERVICES (P) LTD.",style={'font-weight':900})," where I worked on data science tasks involving data preprocessing, exploratory data analysis, machine learning, and predictive analytics. This internship strengthened my technical skills and provided valuable exposure to real-world data-driven projects."]),
                html.P(["The knowledge and experience gained through my coursework and internship inspired me to develop this application, ",html.Strong("Predicting Tour Cancellations to Protect Revenue",style={'font-weight':900}),", which combines machine learning and interactive data visualization to support intelligent decision-making in the tourism and hospitality industry."])
            ],style={'font-size':20,'color':'black','font-family':'Arial'}),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Footer("© 2026 Tanishk Sharma. All rights reserved.",style={'textAlign':'center','fontSize':10,'color':'black','fontFamily':'Arial'})
        ])

# 2nd Part: Callback Function
# Prediction:
def compute_info2(deposit2,country,booking,lead_time,cancellations):
    input_dict={col:np.nan for col in X_columns}
    input_dict["Deposit Category"]=deposit2
    input_dict["Country"]=country
    input_dict["Booking Source"]=booking
    input_dict["Lead Time (in days)"]=lead_time
    input_dict["Total Past Cancellations"]=cancellations
    df2=pd.DataFrame([input_dict])[X_columns]
    df2[cats_col]=df2[cats_col].astype(str)
    return df2

@app.callback([Output(component_id='prediction',component_property='children'),
Output(component_id='chances',component_property='children')],
[State(component_id='dropdown-deposit',component_property='value'),
State(component_id='dropdown-country',component_property='value'),
State(component_id='dropdown-booking-source',component_property='value'),
State(component_id='input-lead-time',component_property='value'),
State(component_id='input-past-cancellations',component_property='value'),
Input(component_id='button',component_property='n_clicks')])

def get_pred(deposit2,country,booking,lead_time,cancellations,button=None):
    if (button is None):
        return "",""
    if (deposit2==None or country==None or booking==None or lead_time==None or cancellations==None):
        return html.P("*Please enter the required details",style={'font-size':10,'color':'red'}),""
    elif ((cancellations<0 or cancellations>30) and (lead_time<0 or lead_time>365)):
        return (html.P("*Lead Time detail must be between 0 and 365",style={'font-size':10,'color':'red'}),
        html.P("*Past cancellations detail must be more than 0",style={'font-size':10,'color':'red'})),""
    elif (lead_time<0 or lead_time>365):
        return html.P("*Lead Time detail must be between 0 and 365",style={'font-size':10,'color':'red'}),""
    elif (cancellations<0 or cancellations>30):
        return html.P("*Past cancellations detail must be between 0 and 30",style={'font-size':10,'color':'red'}),""
    else:
        data=compute_info2(deposit2,country,booking,lead_time,cancellations)
        pred=pipeline.predict(data)
        prob=pipeline.predict_proba(data)
        if (pred==0):
            return (html.P(["Guess = ",html.Span("Will Not Cancel !!!",style={'color':'green'})],style={'font-size':15,'color':'black'}),
            html.P(["Chances of Guess = ",html.Span(f"{prob[0][0]*100:.2f}%",style={'color':'green'})],
            style={'font-size':15,'color':'black'}))
        else:
            return (html.P(["Guess = ",html.Span("Will Cancel !!!",style={'color':'red'})],style={'font-size':15,'color':'black'}),
            html.P(["Chances of Guess = ",html.Span(f"{prob[0][1]*100:.2f}%",style={'color':'red'})],
            style={'font-size':15,'color':'black'}))

# Dashboard:
def compute_info1(hotel_data,city,hotel,customer,deposit):
    df=hotel_data[(hotel_data['City']==city) & (hotel_data['Hotel Category']==hotel)]
    if customer is not None and len(customer)>0:
        df=df[df['Customer Category'].isin(customer)]
    if deposit is not None and len(deposit)>0:
        df=df[df['Deposit Category'].isin(deposit)]
    df['Total Nights Stays']=df['Total Weekend Nights Stay']+df['Total Week Nights Stay']
    df['Total Cancellations']=df['Cancellation Status']+df['Total Past Cancellations']
    df['Total Non-Cancellations']=df['Total Past Non-cancellations']+(1-df['Cancellation Status'])
    # Line Data:
    line=df.groupby('No. of Weeks')['Total Non-Cancellations'].sum().reset_index()
    # Area Data:
    area=df.groupby(['Total Nights Stays','Month'])['Average Daily Rate'].mean().reset_index().sort_values(by='Month',ascending=True)
    # Bar Data:
    bar=df.groupby(['Room Category','Meal Plan'])['Total Cancellations'].sum().reset_index()
    # Histogram Data:
    hist=df[['Total Non-Cancellations','Lead Time (in days)']]
    # Pie Data:
    pie=df.groupby('Booking Source')['Total Cancellations'].sum().reset_index()
    # Sunburst Data:
    sunburst=df.groupby('Country')[['Total Adults','Total Children','Total Cancellations']].sum().reset_index()
    return line,area,bar,hist,pie,sunburst

@app.callback([Output(component_id='line-plot',component_property='figure'),
Output(component_id='area-plot',component_property='figure'),
Output(component_id='bar-plot',component_property='figure'),
Output(component_id='hist-plot',component_property='figure'),
Output(component_id='pie-plot',component_property='figure'),
Output(component_id='sun-plot',component_property='figure'),],
[Input(component_id='dropdown-1',component_property='value'),
Input(component_id='dropdown-2',component_property='value'),
Input(component_id='checklist-1',component_property='value'),
Input(component_id='checklist-2',component_property='value'),])

def get_graphs(city,hotel,customer=None,deposit=None):
    line_fig,area_fig,bar_fig,hist_fig,pie_fig,sun_fig={},{},{},{},{},{}
    if (city==None or hotel==None):
        return line_fig,area_fig,bar_fig,hist_fig,pie_fig,sun_fig
    else:
        line,area,bar,hist,pie,sun=compute_info1(hotel_data,city,hotel,customer,deposit)        
        # Line Graph:
        line_fig=px.line(line,x='No. of Weeks',y='Total Non-Cancellations',
        title='Trend of Total Non-Cancellations vs No. of Weeks spent by Customers',markers=True,color_discrete_sequence=['brown'])
        # Area Graph:
        area_fig=px.area(area,x='Total Nights Stays',y='Average Daily Rate',color='Month',
        title='Distribution of each Month in Nights Stay vs Average Daily Rate (ADR)',
        labels={'Total Nights Stays':'Total Nights Stays (in days)','Average Daily Rate (in Rs)':'Average Daily Rate'})
        # Bar Graph:
        bar_fig=px.bar(bar,x='Room Category',y='Total Cancellations',color='Meal Plan',barmode='group',
        color_discrete_sequence=['darkorange','yellow','navy','darkgreen'],
        title='Room-wise Comparison of Total Cancellations with Meal Plan',
        labels={'Room Category':'Type of Room Booked','Meal Plan':'Meal Plan Options'})
        # Histogram Plot:
        hist_fig=px.histogram(hist,x='Lead Time (in days)',y='Total Non-Cancellations',nbins=10,color_discrete_sequence=['gold'],
        title='Distribution of Total Non-Cancellations based on Lead Time (in days)')
        hist_fig.update_traces(marker=dict(line=dict(color='black',width=1.5)))
        hist_fig.update_yaxes(title_text='Total Non-Cancellations')
        # Pie Chart:
        pie_fig=px.pie(pie,values='Total Cancellations',names='Booking Source',title='Booking Sources Contribution in Total Cancellations')
        # SunBurst Chart:
        sun_fig=px.sunburst(sun,path=['Country','Total Adults','Total Children'],values='Total Cancellations',
        custom_data=['Country','Total Adults','Total Children'],title='Role of Country Behaviour in Total Cancellations',
        hover_data=['Country','Total Adults','Total Children'],
        labels={'Total Adults':'No. of Adults','Total Children':'No. of Children','Total Cancellations':'No. of Cancellations'})
        return line_fig,area_fig,bar_fig,hist_fig,pie_fig,sun_fig

# About the Model:
comp_fig=px.bar(score_df,x='Best Scores',y='Model',color_discrete_sequence=['coral'],
title="Model's F1 Score Comparison",labels={'Model':'Machine Learning Models','Best Scores':'F1 Score (in %)'},orientation='h')
cm=confusion_matrix(Y_test,Y_pred)
conmat=px.imshow(cm,x=['Cancelled','Not Cancelled'],y=['Cancelled','Not Cancelled'],color_continuous_scale='Reds',text_auto=True,
labels=dict(x='Predicted Cancellations',y='Actual Cancellations',color='Count'),title='HistGBC Model Performance')

# 3rd Part: Running the App
if __name__=='__main__':
    app.run(host="0.0.0.0",debug=True,use_reloader=False)