import numpy as np
import pandas as pd

import dash as dash
import dash_core_components as dash_component
import dash_html_components as dash_html
import dash_table as dash_table

from datetime import datetime as dt
from dash.dependencies import Input, Output
from joblib import dump, load

df = pd.read_csv('data/engineered_factset_campaign.csv')
df_pricing = pd.read_csv('data/clean_factset_pricing.csv', parse_dates=['date'])

df_campaign_return_data = pd.read_csv('results/campaign_return_model_data.csv')
campaign_return_model = load('results/campaign_return_model.joblib') 

app = dash.Dash(
    __name__,
    external_stylesheets=[
        'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css',
        'https://codepen.io/chriddyp/pen/bWLwgP.css'
    ]
)
app.title = 'Moelis Capstone Dashboard'
app.config['suppress_callback_exceptions'] = True

app.layout = dash_html.Div(
    children=[
        dash_html.H1(children='Moelis Capstone Project Dashboard'),
        dash_html.Div(
            children=[
                dash_html.H2(
                    '''
                    Introduction
                    '''
                ),
                dash_html.P(
                    '''
                    This project is a collaborative work between Moelis & Company and Data Science Institute at Columbia University
                    aimed at advising companies that are approached by activists. Specifically, the project’s objective is to help
                    the Moelis Activism Advisory team better advise their client companies that are the target of activist
                    campaigns by predicting both the objective and the likelihood of success of an activist campaign
                    once a client company has been approached.
                    '''
                )
            ],
            className="jumbotron"
        ),
        dash_html.H2(children='Inputs'),
        dash_html.H3('Selected Campaign'),
        dash_html.P(children='Select a specific campaign for analysis.'),
        dash_component.Dropdown(
            id='selected-campaign-id',
            options=[{'label': f"{row.campaign_title} ({row.campaign_id})", 'value': row.campaign_id} for index, row in df.iterrows()],
            value='1023510334C'
        ),
        dash_html.H3('Selected Filtration Date'),
        dash_html.P(children='Select a filtration date. All analysis will be restricted to information available only up until this date.'),
        dash_component.DatePickerSingle(
            id='selected-filtration-date',
            min_date_allowed=df.campaign_announcement_date.min(),
            max_date_allowed=df.campaign_announcement_date.max(),
            initial_visible_month=dt(2017, 12, 15),
            date=str(dt(2017, 12, 31, 23, 59, 59))
        ),

        dash_html.H2(children='Overrides'),
        dash_html.P(children='Under construction...'),

        dash_html.H2(children='Data'),
        dash_html.H3(children='Campaign Data'),
        dash_html.Div(id='campaign-table-container'),
        dash_html.H3(children='Activist Data'),
        dash_html.Div(id='activist-table-container'),
        dash_html.H3(children='Target Company Data'),
        dash_html.Div(id='target-graph-container'),

        dash_html.H2(children='Models'),
        # dash_html.H3(children='Campaign Objective Model'),
        # dash_html.P(children='Under construction...'),
        # dash_html.H3(children='Campaign Outcome Model'),
        # dash_html.P(children='Under construction...'),
        dash_html.H3(children='Campaign Return Model'),
        dash_html.Div(id='return-prediction-container'),

    ],
    className='pretty_container',
    style={
        'width': '60%',
        'margin': 'auto',
        'justify-content': 'center',
        'padding': '50px'
    }
)

@app.callback(
    Output('campaign-table-container', 'children'),
    [Input('selected-campaign-id', 'value')]
)
def update_campaign_table(selected_campaign_id):
    df_display = df.loc[
        lambda df: df.campaign_id == selected_campaign_id,
        [
            'campaign_id', 
            'activist_id',
            'activist_name',
            'company_id',
            'company_name',
            'campaign_announcement_date',
            'campaign_title',
            'campaign_objective_primary',
            'value_demand',
            'governance_demand'
        ]
    ]
    df_display = df_display.set_index('campaign_id').transpose().reset_index()
    return dash_html.Div(children=[display_table(df_display)])

@app.callback(
    Output('activist-table-container', 'children'),
    [Input('selected-campaign-id', 'value')]
)
def update_activist_table(selected_campaign_id):
    selected_activist_id = df[lambda df: df.campaign_id == selected_campaign_id].activist_id.iloc[0]
    df_display = df.loc[
        lambda df: df.activist_id == selected_activist_id,
        [
            'campaign_id', 
            'campaign_announcement_date',
            'campaign_title',
            'campaign_objective_primary',
            'activist_campaign_tactic',
            'ownership_pecent_on_announcement'
        ]
    ]
    return dash_html.Div(children=[display_table(df_display)])

@app.callback(
    Output('target-graph-container', 'children'),
    [Input('selected-campaign-id', 'value')]
)
def update_graph(selected_campaign_id):
    selected_company_id = df[lambda df: df.campaign_id == selected_campaign_id].company_id.iloc[0]
    df_display = df_pricing[lambda df: df.company_id == selected_company_id]
    return dash_component.Graph(figure={
        'data': [
            {
                'x': df_display.date,
                'y': df_display.price,
                'type': 'line',
                'name': 'cumsum'
            }
        ],
        'layout': {
            'title': f'Stock Price of Target Company ({selected_company_id})'
        }
    })

@app.callback(
    Output('return-prediction-container', 'children'),
    [Input('selected-campaign-id', 'value')]
)
def update_model_prediction(selected_campaign_id):
    df_x = df_campaign_return_data[lambda df: df.campaign_id == selected_campaign_id].iloc[:, 1::]
    y_predicted = campaign_return_model.predict(df_x)
    y_predicted_label = 'POSITIVE' if y_predicted == 1 else 'NEGATIVE'
    return dash_html.Div(children=f"The predicted campaign return is: {y_predicted_label}")

def display_table(df):
    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': str(i), 'id': str(i)} for i in df.columns],
        style_table={'overflowX': 'scroll'},
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'border': 'thin lightgrey solid'
        }
    )


if __name__ == '__main__':
    app.run_server(debug=True)