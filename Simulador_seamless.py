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

X = data[['LARGO DE CUERPO', 'AREA','PESO PRENDA','DE_MODELO']]
y = data['Teorica']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

app = dash.Dash('mi_app')


## Estas es la parte del front
app.layout = html.Div(
    style={'backgroundColor': '#f7f7f7', 'padding': '20px'},
    children=[
        html.H1(
            children='Simulador Seamless',
            style={'textAlign': 'center', 'color': '#333333'}
        ),
        html.Div(
            children='Predicción de teórica basada en peso prenda, largo cuerpo y área.',
            style={'textAlign': 'center', 'color': '#555555', 'marginBottom': '20px'}
        ),

        html.Div(
            style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'},
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
                    style={'marginRight': '20px'},
                    children=[
                        html.Label('Máquina'),
                        dcc.Dropdown(
                            id='input-maquina',
                            options=[
                                {'label': 'SM8 TOP-2S', 'value': 'SM8 TOP-2S'},
                                {'label': 'SM8 TOP', 'value': 'SM8 TOP'},
                                {'label': 'TOP2 V', 'value': 'TOP2 V'},
                                {'label': 'SM8 TR1', 'value': 'SM8 TR1'}
                            ],
                            value='SM8 TOP-2S',   # Valor predeterminado
                            style={'width': '100%'}  # Ancho del dropdown ajustable
                        )
                    ]
                ),
            ]
        ),
        
        html.Div(
            style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '20px'},
            children=[
                html.Button('Predecir', id='button', style={'backgroundColor': '#4CAF50', 'color': 'white'}),
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
     dash.dependencies.State('input-area', 'value'),
     dash.dependencies.State('input-maquina', 'value')])

def update_output(n_clicks, input_largo, input_area, input_maquina):
    peso_calculado = 1.40 * input_largo  # Calcula el peso basado en el largo del cuerpo
    input_data = np.array([[peso_calculado, input_largo, input_area]])
    # Crear DataFrame temporal con las columnas necesarias para predecir
    temp_df = pd.DataFrame({
        'LARGO DE CUERPO': [input_largo],
        'AREA': [input_area],
        'PESO PRENDA': [peso_calculado],
        'DE_MODELO': [input_maquina]
    })
    # Obtener las columnas utilizadas durante el entrenamiento del modelo
    feature_names = X_train.columns.tolist()
    # Realizar one-hot encoding en el DataFrame temporal
    temp_df_encoded = pd.get_dummies(temp_df)
    # Alinear las columnas del DataFrame temporal con las características utilizadas en el modelo
    input_data_encoded = temp_df_encoded.reindex(columns=feature_names, fill_value=0)
    # Realizar la predicción
    prediction = model.predict(input_data_encoded)
    return prediction[0]


if __name__ == '__main__':
    app.run_server()