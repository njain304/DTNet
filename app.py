import base64
import time
from io import BytesIO as _BytesIO

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import digits_server
from predict_all import *

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


def b64_to_pil(string):
    string += "=" * ((4 - len(string) % 4) % 4)
    decoded = base64.b64decode(string)
    buffer = _BytesIO(decoded)
    buffer.seek(0)
    im = Image.open(buffer)

    return im


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
digits_display_images = [{'label': str(i) + '.png', 'value': str(i)} for i in xrange(1, 50)]

sample_digits_display = [21, 333, 467, 20, 70, 557, 52, 324, 677, 101, 320, 98, 302, 259]
sample_digits_imgs = [Image.open('./results/digits/outputs/' + str(i) + '.png') for i in sample_digits_display]
sample_digits_encoded = [pil_to_b64(i) for i in sample_digits_imgs]

# Sample Data Collection End

app.layout = html.Div([
    html.H1(
        children='Unsupervised Cross-Domain Image Generation using GANs',
        style={
            'textAlign': 'center'
        }
    ),
    html.H6(
        children='Rishabh Bhardwaj, Neha Jain, Prem Sagar Gali',
        style={
            'textAlign': 'center'
        }
    ),
    html.Div([
        dcc.Tabs(id="tabs", children=[
            dcc.Tab(label='Digits Transfer', children=[
                html.H4('SVHN (Source) -> MNIST (Target)',
                        style={
                            'textAlign': 'center'
                        }
                        ),
                html.P(
                    'For Digit Transfer, we use Street View House Numbers(SVHN) and MNIST databse of handwritten digits. SVHN training set consists of 73257 images, and MNIST training set size is 60000. All images are resized to (32,32) and SVHN images are normalize to [-1,1].'),
                html.P(
                    'We take SVHN as `Source` and MNIST as `Target`. Features (F-Model) for the SVHN images are extracted using four blocks of convolution layers with ReLU nonlinearity.To encode the features, we have taken first 7 layers of F model as `f` block in the digit model, so that it encodes the features from the images.'),
                # html.Section(id="slideshow", children=[
                #     html.Div(id="slideshow-container", children=[
                #         html.Div(id="image"),
                #         dcc.Interval(id='interval', interval=3000)
                #     ])
                # ]),
                html.H6('Sample Generated Images (SVHN generated in MNIST domain)'),
                html.Div(id='digits-sample-container', children=[
                    html.Img(
                        src=HTML_IMG_SRC_PARAMETERS + sample_digits_encoded[0],
                        width='200px'
                    ),
                    html.Img(
                        src=HTML_IMG_SRC_PARAMETERS + sample_digits_encoded[1],
                        width='200px'
                    ),
                    html.Img(
                        src=HTML_IMG_SRC_PARAMETERS + sample_digits_encoded[2],
                        width='200px'
                    ),
                    html.Img(
                        src=HTML_IMG_SRC_PARAMETERS + sample_digits_encoded[3],
                        width='200px'
                    ),
                    html.Img(
                        src=HTML_IMG_SRC_PARAMETERS + sample_digits_encoded[4],
                        width='200px'
                    ),
                    html.Img(
                        src=HTML_IMG_SRC_PARAMETERS + sample_digits_encoded[5],
                        width='200px'
                    ),
                    html.Img(
                        src=HTML_IMG_SRC_PARAMETERS + sample_digits_encoded[6],
                        width='200px'
                    ),
                    html.Img(
                        src=HTML_IMG_SRC_PARAMETERS + sample_digits_encoded[7],
                        width='200px'
                    ),
                    html.Img(
                        src=HTML_IMG_SRC_PARAMETERS + sample_digits_encoded[8],
                        width='200px'
                    ),
                    html.Img(
                        src=HTML_IMG_SRC_PARAMETERS + sample_digits_encoded[9],
                        width='200px'
                    ),
                    html.Img(
                        src=HTML_IMG_SRC_PARAMETERS + sample_digits_encoded[10],
                        width='200px'
                    ),
                    html.Img(
                        src=HTML_IMG_SRC_PARAMETERS + sample_digits_encoded[11],
                        width='200px'
                    ),
                    html.Img(
                        src=HTML_IMG_SRC_PARAMETERS + sample_digits_encoded[12],
                        width='200px'
                    ),
                    html.Img(
                        src=HTML_IMG_SRC_PARAMETERS + sample_digits_encoded[13],
                        width='200px'
                    )
                ]),
                html.H6('Try out on SVHN Test Dataset...',
                        style={
                            'textAlign': 'center'
                        }
                        ),
                html.P('Select test image from dropdown:-',
                       style={
                           'textAlign': 'center'
                       }
                       ),
                dcc.Dropdown(
                    id='digits-dropdown',
                    options=digits_display_images,
                    value='20',
                    style={
                        'textAlign': 'center',
                        'width': '45%',
                        'margin-left': '28%'
                    }
                ),
                html.Hr(),
                html.P('Gnereated image in MNIST domain for selected SVHN image..',
                       style={
                           'textAlign': 'center'
                       }
                       ),
                html.Div(id='digits-result',
                         style={
                             'textAlign': 'center',
                         }
                         ),
                html.Hr()
            ]),
            dcc.Tab(label='Face To Emoji', children=[
                html.Div(children=[
                    html.Div(children=[
                        html.H4('CelebA (Source) -> Emoji (Target)',
                                style={
                                    'textAlign': 'center'
                                }
                                ),
                        html.P(
                            'For Emoji generation of Faces, we have used celebA dataset (200k images) and generated 100k Emojis using BitMoji API.'),
                        html.P(
                            'We take celebA as `Source` and Emoji as `Target`. We have used Openface model as `f` block for feature exctraction.'),
                        html.H6('Sample Generated Images (Faces generated in Emoji domain)',
                                style={
                                    'textAlign': 'center'
                                }
                                ),
                        dcc.Upload(
                            id='upload-image',
                            children=[
                                'Drag and Drop or ',
                                html.A('Select an Image')
                            ],
                            style={
                                'width': '28%',
                                'height': '50px',
                                'lineHeight': '50px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin-left': '35%'
                            },
                            accept='image/*'
                        ),
                        html.Div(id='result-tab-emoji')
                    ])
                ]),
            ]),
            dcc.Tab(label='Face To Cartoons', children=[
                html.Div(children=[
                    html.Div(children=[
                        html.H4('CelebA (Source) -> CartoonSet (Target)',
                                style={
                                    'textAlign': 'center'
                                }
                                ),
                        html.P(
                            'For Emoji generation of Faces, we have used celebA dataset (200k images) and generated 100k Emojis using BitMoji API.'),
                        html.P(
                            'We take celebA as `Source` and Emoji as `Target`. We have used Openface model as `f` block for feature exctraction.'),
                        html.H6('Sample Generated Images (Faces generated in Emoji domain)',
                                style={
                                    'textAlign': 'center'
                                }
                                ),
                        dcc.Upload(
                            id='upload-image-cartoon',
                            children=[
                                'Drag and Drop or ',
                                html.A('Select an Image')
                            ],
                            style={
                                'width': '28%',
                                'height': '50px',
                                'lineHeight': '50px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin-left': '35%'
                            },
                            accept='image/*'
                        ),
                        html.Div(id='result-tab-cartoon')
                    ])
                ]),
            ]),
            dcc.Tab(label='Face To Simpson', children=[
                html.Div(children=[
                    html.Div(children=[
                        html.H4('CelebA (Source) -> Simpson Dataset (Target)',
                                style={
                                    'textAlign': 'center'
                                }
                                ),
                        html.P(
                            'For Emoji generation of Faces, we have used celebA dataset (200k images) and generated 100k Emojis using BitMoji API.'),
                        html.P(
                            'We take celebA as `Source` and Emoji as `Target`. We have used Openface model as `f` block for feature exctraction.'),
                        html.H6('Sample Generated Images (Faces generated in Emoji domain)',
                                style={
                                    'textAlign': 'center'
                                }
                                ),
                        dcc.Upload(
                            id='upload-image-simpson',
                            children=[
                                'Drag and Drop or ',
                                html.A('Select an Image')
                            ],
                            style={
                                'width': '28%',
                                'height': '50px',
                                'lineHeight': '50px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin-left': '35%'
                            },
                            accept='image/*'
                        ),
                        html.Div(id='result-tab-simpson')
                    ])
                ]),
            ]),
        ])
    ])
])


@app.callback(
    Output(component_id='result-tab-emoji', component_property='children'),
    [Input(component_id='upload-image', component_property='contents')]
)
def update_emoji(contents):

    print(contents)
    if contents:
        data = str(contents)[23:]
        img = b64_to_pil(data)
        tgt_img = predict_emoji(img)
        #tgt_img = Image.fromarray(tgt_img.astype('uint8'), 'RGB')
        out = numpy_to_b64(tgt_img)
    else:
        out = contents
    return html.Div(children=[
        html.Img(
            src=contents,
            width='200px'
        ),
        html.Img(

            src=HTML_IMG_SRC_PARAMETERS + out,
            width='200px'
        )
    ], style={
        'textAlign': 'center'
    })


@app.callback(
    Output(component_id='result-tab-cartoon', component_property='children'),
    [Input(component_id='upload-image-cartoon', component_property='contents')]
)
def update_cartoon(contents):

    if contents:
        data = str(contents)[23:]

        img = b64_to_pil(data)
        tgt_img = predict_cartoon(img)
        #tgt_img = Image.fromarray(tgt_img.astype('uint8'), 'RGB')
        out = numpy_to_b64(tgt_img)
    else:
        out = contents
    return html.Div(children=[
        html.Img(
            src=contents,
            width='200px'
        ),
        html.Img(

            src=HTML_IMG_SRC_PARAMETERS + out,
            width='200px'
        )
    ], style={
        'textAlign': 'center'
    })


@app.callback(
    Output(component_id='result-tab-simpson', component_property='children'),
    [Input(component_id='upload-image-simpson', component_property='contents')]
)
def update_simpson(contents):

    if contents:
        data = str(contents)[23:]

        img = b64_to_pil(data)
        tgt_img = predict_simpsons(img)
        #tgt_img = Image.fromarray(tgt_img.astype('uint8'), 'RGB')
        out = numpy_to_b64(tgt_img)
    else:
        out = contents
    return html.Div(children=[
        html.Img(
            src=contents,
            width='200px'
        ),
        html.Img(

            src=HTML_IMG_SRC_PARAMETERS + out,
            width='200px'
        )
    ], style={
        'textAlign': 'center'
    })


@app.callback(
    dash.dependencies.Output('digits-result', 'children'),
    [dash.dependencies.Input('digits-dropdown', 'value')])
def update_output(value):
    print(value)

    orig_img = digits_server.get_svhn_image(int(value))
    out_img = digits_server.digits_predict(orig_img)

    encoded_orig = pil_to_b64(orig_img)
    encoded_out = numpy_to_b64(out_img)

    return html.Div(children=[
        html.Img(
            id='img-' + str(value),
            src=HTML_IMG_SRC_PARAMETERS + encoded_orig,
            width='200px'
        ),
        html.Img(
            id='img-' + str(value),
            src=HTML_IMG_SRC_PARAMETERS + encoded_out,
            width='200px'
        )
    ], style={
        'textAlign': 'center'
    })


if __name__ == '__main__':
    app.run_server(debug=True)
