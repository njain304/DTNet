import base64

import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import plotly.plotly as py
import plotly.tools as tls
import matplotlib.pyplot as plt

import numpy as np
import time
from io import BytesIO as _BytesIO
from PIL import Image

import digits_server

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

HTML_IMG_SRC_PARAMETERS = 'data:image/png;base64, '

def pil_to_b64(im, enc_format='png', verbose=False, **kwargs):
    """
    Converts a PIL Image into base64 string for HTML displaying
    :param im: PIL Image object
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :return: base64 encoding
    """
    t_start = time.time()

    buff = _BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

    t_end = time.time()
    return encoded


def numpy_to_b64(np_array, enc_format='png', scalar=True, **kwargs):
    """
    Converts a numpy image into base 64 string for HTML displaying
    :param np_array:
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :param scalar:
    :return:
    """
    # Convert from 0-1 to 0-255
    if scalar:
        np_array = np.uint8(255 * np_array)
    else:
        np_array = np.uint8(np_array)

    im_pil = Image.fromarray(np_array)

    return pil_to_b64(im_pil, enc_format, **kwargs)

# Sample Data Collection Start
digits_display_images = [{'label': str(i)+'.png', 'value': str(i)} for i in xrange(1, 50)]

sample_digit_img_path = './results/digits/outputs/1.png'
sample_digit_img = Image.open(sample_digit_img_path)
sample_digit_encoded_img = pil_to_b64(sample_digit_img)

# Sample Data Collection End

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
                    # html.Section(id="slideshow", children=[
                    #     html.Div(id="slideshow-container", children=[
                    #         html.Div(id="image"),
                    #         dcc.Interval(id='interval', interval=3000)
                    #     ])
                    # ]),
                    html.H6('Sample Generated Images (SVHN generated in MNIST domain)'),
                    html.Div(id='digits-sample-container', children=[
                            html.Img(
                                src=HTML_IMG_SRC_PARAMETERS + sample_digit_encoded_img,
                                width='200px'
                            ),
                            html.Img(
                                src=HTML_IMG_SRC_PARAMETERS + sample_digit_encoded_img,
                                width='200px'
                            )
                    ]),
                    html.H6('Try out on SVHN Test Dataset...'),
                    html.P('Select test image from dropdown:-'),
                    dcc.Dropdown(
                        id='digits-dropdown',
                        options=digits_display_images,
                        value='1'
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
    print(value)

    orig_img = digits_server.get_svhn_image(int(value))
    out_img = digits_server.digits_predict(orig_img)
    #print(type(out_img))
    #print(type(orig_img))
    encoded_orig = pil_to_b64(orig_img)
    encoded_out = numpy_to_b64(out_img)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1,2,1)
    # ax1.imshow(orig_img)
    # ax2 = fig.add_subplot(1,2,2)
    # ax2.imshow(out_img)
    # plotly_fig = tls.mpl_to_plotly(fig)
    #plt.savefig('./results/digits/'+str(i)+'.png')
    return html.Div(children=[
                html.Img(
                    id='img-'+str(value),
                    src=HTML_IMG_SRC_PARAMETERS + encoded_orig,
                    width='200px'
                ),
                html.Img(
                    id='img-'+str(value),
                    src=HTML_IMG_SRC_PARAMETERS + encoded_out,
                    width='200px'
                )
            ])

if __name__ == '__main__':
    app.run_server(debug=True)
