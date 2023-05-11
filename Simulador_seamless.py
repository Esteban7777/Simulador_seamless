# -*- coding: utf-8 -*-
"""
Created on Sun May  7 09:36:08 2023

@author: jelondos
"""



import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

data = pd.read_csv('data.csv')

X = data[['LARGO DE CUERPO', 'AREA','PESO PRENDA']]
y = data['Teorica']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

app = dash.Dash('mi_app')

app.layout = html.Div(
    style={'backgroundColor': '#f7f7f7', 'padding': '20px'},
    children=[
        html.H1(
            children='Simulador Seamless',
            style={'textAlign': 'center', 'color': '#333333'}
        ),
        html.Div(
            children='''
            Predicción de teórica basada en peso prenda, largo cuerpo y área.
            ''',
            style={'textAlign': 'center', 'color': '#555555', 'marginBottom': '20px'}
        ),

        html.Div(
            style={'display': 'flex', 'justifyContent': 'center'},
            children=[
                html.Div(
                    style={'marginRight': '20px'},
                    children=[
                        html.Label('Largo cuerpo'),
                        dcc.Input(id='input-largo', type='number', value=0),
                    ]
                ),
                html.Div(
                    style={'marginRight': '20px'},
                    children=[
                        html.Label('Área'),
                        dcc.Input(id='input-area', type='number', value=0),
                    ]
                ),
                html.Div(
                    children=[
                        html.Button('Predecir', id='button', style={'backgroundColor': '#4CAF50', 'color': 'white'}),
                    ]
                ),
            ]
        ),

        html.Div(
            style={'textAlign': 'center', 'marginTop': '20px'},
            children=[
                html.Label('Predicción de teórica:', style={'marginRight': '10px'}),
                html.Div(id='output-teorica', style={'display': 'inline-block', 'fontWeight': 'bold'}),
            ]
        ),
    ]
)


@app.callback(
    dash.dependencies.Output('output-teorica', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-largo', 'value'),
     dash.dependencies.State('input-area', 'value')])
def update_output(n_clicks, input_largo, input_area):
    peso_calculado = 1.40 * input_largo  # Calcula el peso basado en el largo del cuerpo
    input_data = np.array([[peso_calculado, input_largo, input_area]])
    prediction = model.predict(input_data)
    return prediction[0]


if __name__ == '__main__':
    app.run_server()