import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import Energy_Dashboard_helpers as helpers
import json
import plotly.graph_objs as go

# Load the training data
training_data = helpers.load_data()

# External stylesheets for the Dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Reading data from a CSV file into a Pandas DataFrame
df = pd.read_csv(r'testData_2019_Civil - testData_2019_Civil.csv')

# Prepare 2019 data
df_test = helpers.create_test_data(df)

# Creating a Dash application instance
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

# Function to generate an HTML table from a DataFrame
def generate_table(dataframe, start_row=0, max_rows=10):
    return html.Div([
        html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in dataframe.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
                ]) for i in range(start_row, min(start_row + max_rows, len(dataframe)))
            ])
        ])
    ])

def generate_slider(dataframe):
    return dcc.Slider(
        id='row-slider',
        min=0,
        max=len(dataframe),
        step=1,
        value=0,
        marks={i: dataframe.iloc[i]['Date'] for i in range(0, len(dataframe)+1, len(dataframe) // 10)}
    )

# Defining the layout of the Dash app
app.layout = html.Div([
    html.H2('Power Consumption Forecast Dashboard'),  # Main title
    html.P('Learn how to forecast hourly power consumption using machine learning.'),  # Subtitle
    dcc.Tabs(id='main-tabs', value='data-exploration', children=[  # Tabs for different sections
        dcc.Tab(label='Data Exploration', value='data-exploration'),  # Data Exploration tab
        # dcc.Tab(label='Feature Exploration', value='feature-selection'),  # Feature Exploration tab
        dcc.Tab(label='Forecasting', value='forecasting')  # Forecasting tab
    ],
    style={'fontSize': '20px'}),  # Set the font size of tab labels
    html.Div(id='main-tabs-content')  # Container for tab content
])

# Data Exploration Tab
data_exploration_tab = html.Div([
    html.H5('Explore the data using tabular data or graphs'),  # Subtitle for the Data Exploration tab
    dcc.Tabs(id='data-exploration-tabs', value='tab-1', children=[  # Tabs for different visualization types
        dcc.Tab(label='Table', value='tab-1', id='data-table'),  # Table tab
        dcc.Tab(label='Graph', value='tab-2'),  # Graph tab
    ]),
    html.Div(id='data-exploration-tabs-content') # Container for tab content
])

# Callback to render content based on the selected tab in Data Exploration section
@app.callback(Output('data-exploration-tabs-content', 'children'), [Input('data-exploration-tabs', 'value')])
def render_data_exploration_content(tab):
    if tab == 'tab-1':  # If Table tab is selected
        return html.Div([
            generate_slider(df),  # Generating slider
            html.Div(id='data-table')  # Container for the table
        ])
    elif tab == 'tab-2':  # If Graph tab is selected
        return html.Div([
            dcc.Dropdown(
                id='plot-type-dropdown',
                options=[
                    {'label': 'Bar Chart', 'value': 'bar'},
                    {'label': 'Line Chart', 'value': 'line'}
                ],
                value='bar',  # Default value for dropdown
                clearable=False
            ),
            dcc.Dropdown(
                id='data-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns],
                value=[df.columns[1]],  # Default value for dropdown
                multi=True  # Allow multiple selections
            ),
            dcc.Graph(id='yearly-data')  # Graph component for displaying data
        ])
    
# Updating the callback to generate the table based on slider value
@app.callback(
    Output('data-table', 'children'),
    [Input('row-slider', 'value')]
)
def update_table(start_row):
    return generate_table(df, start_row=start_row)

# Callback to render content based on the selected main tab
@app.callback(Output('main-tabs-content', 'children'), [Input('main-tabs', 'value')])
def render_main_tabs_content(tab):
    if tab == 'data-exploration':
        return data_exploration_tab  #Render Data Exploration tab content
    # elif tab == 'feature-selection':
    #     return feature_selection_tab
    elif tab == 'forecasting':
        return forecasting_tab  # Render Forecasting tab content

# Callback to update the graph based on selected plot type and data columns
@app.callback(Output('yearly-data', 'figure'),
              [Input('plot-type-dropdown', 'value'),
               Input('data-dropdown', 'value')])
def update_graph(plot_type, selected_data):
    if not selected_data:
        return {}  # Return empty graph if no data selected
    
    data = []
    for col in selected_data:
        data.append({'x': df['Date'], 'y': df[col], 'type': plot_type, 'name': f'{col} (kWh)'})
    
    return {
        'data': data,
        'layout': {
            'title': 'IST Civil Building Weather and Power Consumption Data',
        }
    }


# Feature Exploration
# --------------------------------------------------------------------------------------------------------------------------------
# # get the random Forest metrics
# random_forest_results = helpers.calculate_feature_importance_RF(training_data)

# # Layout with the Feature Exploration Tab
# feature_selection_tab = html.Div([
#     html.H5('Learn about the forecasting relevance of the features.'),  # Subtitle for the Feature Exploration tab
#     html.P('Some features have been modified and some have been added. Analyse them by selecting the feature selection method.'),  # Explanation text
#     dcc.Dropdown(
#         id='feature-selection-method-dropdown',
#         options=[
#             {'label': 'Correlation Matrix', 'value': 'correlation'},
#             {'label': 'Random Forest', 'value': 'random_forest'}
#         ],
#         value='correlation',  # Default value
#         clearable=False
#     ),
#     html.Div(id='feature-selection-output')
# ])

# # Callback for performing feature selection and displaying results
# @app.callback(
#     Output('feature-selection-output', 'children'),
#     Input('feature-selection-method-dropdown', 'value')
# )
# def perform_feature_selection(method):
#     if method == 'correlation':
#         return html.Div([
#         html.Img(src='correlation_matrix.png',
#         style={'max-width': '100%', 'height': 'auto'})
#     ])
#     elif method == 'random_forest':
#         return html.Div([
#             html.H4('Random Forest Feature Selection Results:'),
#             html.Table([
#                 html.Thead(
#                     html.Tr([html.Th(col) for col in random_forest_results.columns])
#                 ),
#                 html.Tbody([
#                     html.Tr([
#                         html.Td(random_forest_results.iloc[i][col]) for col in random_forest_results.columns
#                     ]) for i in range(min(len(random_forest_results), 10))  # Display first 10 rows
#                 ])
#             ])
#         ])



# Forecasting
# --------------------------------------------------------------------------------------------------------------------------------
# Define forecasting methods
methods = {
    'Linear Regression',
    'Support Vector Machine (RBF kernel)',
    'Multi-layer Perceptron',
    'Decision Tree',
    'Random Forest',
    'Gradient Boosting'
}

# Define available metrics
available_metrics = ['Mean Absolute Error (MAE)', 'Mean Bias Error (MBE)', 'Mean Squared Error (MSE)',
                     'Root Mean Squared Error (RMSE)', 'Coefficient of Variation RMSE (cvRMSE)', 'Normalized Mean Bias Error (NMBE)']

# Forecasting Tab
forecasting_tab = html.Div([
    html.H5('Let´s predict the power consumption for the next hour!'),  # Subtitle for the Forecasting tab
    html.P('Train a model on 2017 and 2018 data and predict the 2019 power consumption based on the previous hour. Be patient ;)'),  # Explanation text
    dcc.Dropdown(  # Dropdown for selecting forecasting method
        id='forecasting-method-dropdown',
        options=[{'label': method, 'value': method} for method in methods],
        value=None,
        placeholder='Select model for forecasting',
        clearable=False
    ),
    dcc.Dropdown(
        id='forecast-variable-dropdown',
        options=[{'label': col, 'value': col} for col in df_test.columns],
        value=None,  # Default value
        placeholder='Select features for forecasting',
        multi=True  # Allow multiple selections
    ),
    html.Button('Forecast', id='train-model-button', n_clicks=0), # Button to train the model
    html.Div(id='model-training-output'),  # Placeholder for displaying model training results
    html.Div(id='metrics-output')  # Container for displaying forecasting metrics
])

# Callback to display selected forecasting method
@app.callback(Output('forecasting-results', 'children'),
              Input('forecasting-method-dropdown', 'value'))
def display_forecasting_results(method):
    return html.Div(f'Selected Forecasting Method: {method}')

# Callback for training the model and displaying results
@app.callback(
    Output('model-training-output', 'children'),
    Input('train-model-button', 'n_clicks'),
    State('forecasting-method-dropdown', 'value'),
    State('forecast-variable-dropdown', 'value')
)
def train_and_display_model(n_clicks, model_name, features):
    if n_clicks > 0:
        if features is None or len(features) == 0 or model_name is None or len(model_name) == 0:
            return html.Div('Please select at least one feature and a model for training the model.')
        else:
            model_info, trained_model = helpers.train_model(training_data, model_name, features)
            predictions = helpers.get_predictions(df_test, trained_model, features)
            predictions_dict = {'model_info': model_info, 'predictions': predictions.tolist()}  # Convert predictions to a list
            predictions_json = json.dumps(predictions_dict, indent=2)

            # Plotting
            trace_actual = go.Scatter(x=df_test.index, y=df['Civil (kWh)'], mode='lines', name='Actual Data')
            trace_predicted = go.Scatter(x=df_test.index, y=predictions, mode='lines', name='Predicted Data')
            layout = go.Layout(title='Actual vs Predicted Data', xaxis=dict(title='Index'), yaxis=dict(title='Civil (kWh)'))
            fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)

            return html.Div([
                html.H5(f'{model_name} model using {features} as features trained successfully'),
                html.Div('Let´s see the predictions!'),
                dcc.Graph(id='prediction-plot', figure=fig),  # Display the plot
                html.Div('Let´s evaluate the results!'),
                html.Div([  # Container for metric dropdown menu
                    dcc.Dropdown(
                        id='metric-dropdown',
                        options=[{'label': metric, 'value': metric} for metric in available_metrics],
                        value=None,  # Default value
                        placeholder='Select an evaluation metric',
                        clearable=False
                    )
                ])
            ])


# Forecasting metrics
# Define a function to calculate metrics
def calculate_metrics(y_true, y_pred):
    MAE = metrics.mean_absolute_error(y_true, y_pred)
    MBE = np.mean(y_true.values - y_pred)
    MSE = metrics.mean_squared_error(y_true, y_pred)
    RMSE = np.sqrt(MSE)
    cvRMSE = RMSE / np.mean(y_true)
    NMBE = MBE / np.mean(y_true)
    
    # Define the metrics dictionary with both abbreviations and long words
    metrics_dict = {
        'Mean Absolute Error (MAE)': MAE,
        'MAE': MAE,
        'Mean Bias Error (MBE)': MBE,
        'MBE': MBE,
        'Mean Squared Error (MSE)': MSE,
        'MSE': MSE,
        'Root Mean Squared Error (RMSE)': RMSE,
        'RMSE': RMSE,
        'Coefficient of Variation RMSE (cvRMSE)': cvRMSE,
        'cvRMSE': cvRMSE,
        'Normalized Mean Bias Error (NMBE)': NMBE,
        'NMBE': NMBE
    }
    
    return metrics_dict

# Callback to update the selected metric
@app.callback(
    Output('metrics-output', 'children'),
    [Input('metric-dropdown', 'value'),
     Input('prediction-plot', 'figure')]
)
def update_metric(metric, prediction_plot):
    if not prediction_plot or 'y' not in prediction_plot['data'][1] or metric is None:  # Check if prediction plot is available
        return ''  # Return empty string if prediction plot is not available

    y_true = df['Civil (kWh)'][1:]  # Get actual values from the DataFrame
    y_pred = prediction_plot['data'][1]['y']  # Get predicted values from the plot

    calculated_metrics = calculate_metrics(y_true, y_pred)  # Calculate metrics
    selected_metric_value = calculated_metrics[metric]  # Get the value of the selected metric

    return html.Div(f'{metric}: {selected_metric_value}')  # Display selected metric value


# Running the Dash app
if __name__ == '__main__':
    app.run_server()
