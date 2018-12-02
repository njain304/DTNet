import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

digits_display_images = [{'label': 'New York City', 'value': 'NYC'},
                        {'label': 'Montreal', 'value': 'MTL'},
                        {'label': 'San Francisco', 'value': 'SF'}]
app.layout = html.Div([
    html.H1(
            children = 'Study on Domain Trasnfer Networks',
            style = {
                    'textAlign': 'center'
            }
    ),
    html.H6(
            children = 'Rishabh Bhardwaj, Neha Jain, Prem Sagar Gali',
            style = {
                    'textAlign': 'center'
            }
    ),
    html.Div([
        dcc.Tabs(id="tabs", children=[
            dcc.Tab(label='Tab one', children=[
                html.Div(children=[
                        html.Div(children=[
                                html.H3('Tab content 1'),
                                dcc.Upload(
                                    id='upload-image',
                                    children=[
                                        'Drag and Drop or ',
                                        html.A('Select an Image')
                                    ],
                                    style={
                                        'width': '100%',
                                        'height': '50px',
                                        'lineHeight': '50px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center'
                                    },
                                    accept='image/*'
                                ),
                                html.Div(id ='result-tab-1'),
                                dcc.Graph(
                                    id='graph-1-tabs',
                                    figure={
                                        'data': [{
                                            'x': [1, 2, 3],
                                            'y': [3, 1, 2],
                                            'type': 'bar'
                                        }]
                                    }
                                )
                        ])
                ]),
            ]),
            dcc.Tab(label='Digits Transfer', children=[
                    html.H4('SVHN (Source) -> MNIST (Target)',
                            style = {
                                'textAlign': 'center'
                            }
                        ),
                    html.P('For Digit Transfer, we use Street View House Numbers(SVHN) and MNIST databse of handwritten digits. SVHN training set consists of 73257 images, and MNIST training set size is 60000. All images are resized to (32,32) and SVHN images are normalize to [-1,1].'),
                    html.P('We take SVHN as `Source` and MNIST as `Target`. Features (F-Model) for the SVHN images are extracted using four blocks of convolution layers with ReLU nonlinearity.To encode the features, we have taken first 7 layers of F model as `f` block in the digit model, so that it encodes the features from the images.'),
                    dcc.Dropdown(
                        id='digits-dropdown',
                        options=digits_display_images,
                        value='NYC'
                    ),
                    html.Div(id='digits-result')
            ]),
            dcc.Tab(label='Tab three', children=[
                    dcc.Graph(
                        id='example-graph-2',
                        figure={
                            'data': [
                                {'x': [1, 2, 3], 'y': [2, 4, 3],
                                    'type': 'bar', 'name': 'SF'},
                                {'x': [1, 2, 3], 'y': [5, 4, 3],
                                 'type': 'bar', 'name': u'Montral'},
                            ]
                        }
                    )
            ]),
        ])
    ])
])
@app.callback(
    Output(component_id='result-tab-1', component_property='children'),
    [Input(component_id = 'upload-image',component_property = 'contents')]
)
def update_digits_pred(contents):
    return html.Img(src=contents)

@app.callback(
    dash.dependencies.Output('digits-result', 'children'),
    [dash.dependencies.Input('digits-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)

if __name__ == '__main__':
    app.run_server(debug=True)
