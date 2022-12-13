import io
import base64
import cv2

import numpy as np
import plotly.graph_objs as go
from PIL import Image
from plotly.graph_objects import Figure
from glass.evaluation.text_evaluator import get_instances_text


def visualize(preds, image, text_encoder, title=None, vis_width: int = 720,
              vis_text: bool = True, save_output_image_path: str = None):
    image_height, image_width = image.shape[:2]
    scale = vis_width / image_width
    dim = (int(np.round(image_width * scale)), int(np.round(image_height * scale)))
    resized_image = cv2.resize(image, dim)
    image_height, image_width = resized_image.shape[:2]

    trace_list, annotation_list = _make_trace_words_preds(
        image_width=image_width,
        image_height=image_height,
        scale=scale,
        preds=preds,
        text_encoder=text_encoder,
        vis_text=vis_text,
    )

    layout = _make_layout(image_width=image_width,
                          image_height=image_height,
                          image=resized_image,
                          title=title,
                          vis_text=vis_text)

    # Adding the visualized text transcription annotations
    layout.update({'annotations': annotation_list})

    figure = Figure({'data': trace_list or {}, 'layout': layout})

    if save_output_image_path is not None:
        figure.write_image(save_output_image_path)

    return figure


def _load_image_to_plotly_src(image_path):
    im = Image.open(image_path)
    im = _fix_rgba_pil_images(pil_image=im)
    # convert image to bytes
    img_bytes_arr = io.BytesIO()
    im.save(img_bytes_arr, format='PNG')
    img_bytes_arr = img_bytes_arr.getvalue()
    encoded_image = base64.b64encode(img_bytes_arr)
    new_src = 'data:image/png;base64,{}'.format(encoded_image.decode("utf-8"))
    return new_src


def _make_trace_words_preds(preds, text_encoder, image_width, image_height, scale, vis_text):
    polygons = preds.pred_polygons.cpu().numpy() * scale
    angles = preds.pred_boxes.tensor.cpu().numpy()[:, 4]  # We don't scale rotated boxes, there's no need

    scores = preds.scores.cpu().numpy()

    texts, text_scores, _ = get_instances_text(preds.pred_text_prob, text_encoder)

    trace_list = list()
    annotation_list = list()

    for poly, score, angle, text, text_score in zip(polygons, scores, angles, texts, text_scores):
        draw_x = poly[:, 0]
        draw_x = np.append(draw_x, draw_x[0])
        draw_y = image_height - poly[:, 1]
        draw_y = np.append(draw_y, draw_y[0])
        left, right = np.min(draw_x), np.max(draw_x)
        top, bottom = np.min(draw_y), np.max(draw_y)

        draw_vertices = np.stack((draw_x, draw_y)).T
        color = 'blue'
        dash = 'solid'
        line_width = 3

        trace_list.append(
            go.Scatter(
                x=draw_x,
                y=draw_y,
                hoveron='fills',
                name='',
                opacity=0.66,
                text=f'<b>{text}</b><br>Detect score: {score * 100:.1f}<br>Text score: {text_score*100:.1f}',
                mode='lines',
                line=dict(width=line_width, color=color, dash=dash),
                showlegend=False
            )
        )

        # Additional attributes
        font_face = 'Arial'

        if vis_text:
            text_x = [image_width + (right + left) / 2]
            text_y = [(top + bottom) / 2]

            # The polygon / bounding box vertices from top left going clockwise
            fontsize = _get_font_size_from_box_size(
                width=np.linalg.norm(draw_vertices[1] - draw_vertices[0]),
                height=np.linalg.norm(draw_vertices[2] - draw_vertices[1]),
                text=text)

            norm_angle = ((angle + 180) % 360) - 180
            text_angle = norm_angle - 180 if np.abs(norm_angle) > 90 else norm_angle
            annotation_list.append(
                go.layout.Annotation(
                    x=text_x[0],
                    y=text_y[0],
                    text=text,
                    align='center',
                    showarrow=False,
                    yanchor='middle',
                    font={'family': font_face,
                          'size': max(7, fontsize) - 2,
                          'color': color},
                    textangle=-text_angle,
                )
            )
    return trace_list, annotation_list


def _make_layout(image_width=50, image_height=100, image_path=None, image=None, title=None,
                 vis_text=False):
    if image is not None:
        source = Image.fromarray(image)
    elif image_path is not None:
        source = _load_image_to_plotly_src(image_path=image_path)
    else:
        raise ValueError
    layout = go.Layout(
        plot_bgcolor='#f0f2f6',  # streamlit native gray background
        hovermode='x',
        title=title,
        xaxis=go.layout.XAxis(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0, image_width * 2 if vis_text else image_width]
        ),
        yaxis=go.layout.YAxis(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-5, image_height + 5],
            # range=[0, image_height],
            scaleanchor='x'
        ),
        autosize=False,
        height=image_height,
        width=image_width * 2 if vis_text else image_width,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        images=[dict(
            source=source,
            x=0,
            sizex=image_width,
            y=image_height,
            sizey=image_height,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch"
        )]
    )
    return layout


def _get_font_size_from_box_size(width, height, text):
    min_font_size = 5
    max_font_size = 192
    scale_coeff = 1.1
    if height < 5 or width < 4 or not text:
        return min_font_size
    width_per_character = width / len(text)
    dimension_to_consider = min(width_per_character * 1.5, height)
    return min(int(np.round(dimension_to_consider * scale_coeff / 2) * 2), max_font_size)


def _fix_rgba_pil_images(pil_image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    if pil_image.mode != 'RGBA':
        return pil_image.convert('RGB')

    pil_image.load()  # needed for split()
    background = Image.new('RGB', pil_image.size, color)
    background.paste(pil_image, mask=pil_image.split()[3])  # 3 is the alpha channel
    return background
