# -*- coding: utf-8 -*-
"""
Created on Sun May  7 09:36:08 2023

@author: jelondos, ejruizca
"""



import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np  
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import xgboost as xgb

data = pd.read_csv('data.csv')

X = data[['LARGO DE CUERPO', 'AREA','PESO PRENDA']]
y = data['Teorica']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

external_stylesheets = ['./assets/style.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    className="contenedor",
    children=[
        html.Div(
            children=[
                html.H2("simulador seamleass"),
                html.H5("Predicción de teórica basada en peso prenda, largo cuerpo y área."),
            ]
        ),
        html.Div(
            className="contenedorDivsInputs",
            children=[
                html.Div( 
                className="contenedorInputs",
                children=[
                    html.Div(
                        className="contenedorInput",
                        children=[
                            html.Label("largo cuerpo:"),
                            dcc.Input(type="number", id='input-largo', value=0),
                        ]
                    ),
                    html.Div(
                        className="contenedorInput",
                        children=[
                            html.Label("área:"),
                            dcc.Input(type="number", id='input-area',  value=0),
                        ]
                    ), 
                ]),
               
                html.Div(
                className='contenedorInput',
                    children=[
                        html.Label('maquina:'),
                        html.Select(
                            id='input-maquina',
                            children=[
                                html.Option('Seleccione una opción', **{'data-value': 'SM8 TOP-2S'}),
                                html.Option('SM8 TOP-2S', **{'data-value': 'SM8 TOP-2S'}),
                                html.Option('SM8 TOP', **{'data-value': 'SM8 TOP'}),
                                html.Option('SM8 TR1', **{'data-value': 'SM8 TR1'}),
                                html.Option('TOP2 V', **{'data-value': 'TOP2 V'})
                            ]
                        )
                    ]
                )
            ]
        ),
        html.Div(
            children=[
                html.Div(className="contendorButton", children=[html.Button("predecir", id='button')])
            ]
        ),
        html.Div(
            children=[
                html.Div(
                    className='contendorTeorica',
                    children=[
                        html.P('predicción de teórica: '),
                        html.Span(id='output-teorica')
                    ]
                ),
                html.Div(
                    className='contendorTeorica',
                    children=[
                        html.P('Peso: '),
                        html.Span(id='output-peso')
                    ]
                ),
        ]),
    ]
)

@app.callback(
    dash.dependencies.Output('output-peso', 'children'),
    dash.dependencies.Output('output-teorica', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-largo', 'value'),
     dash.dependencies.State('input-area', 'value'),
     dash.dependencies.State('input-maquina', 'data-value')])
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
    return peso_calculado, prediction[0]

if __name__ == '__main__':
    app.run_server(debug=True)