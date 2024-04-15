import math

import cv2
from PIL.Image import Image, fromarray
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from PIL import ImageColor
import numpy as np
from xml.dom.minidom import parse
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def kp_sort(keypoints_perInstance):
    keypoints_perInstance = keypoints_perInstance[:, :-1].tolist()
    newList = [keypoints_perInstance[0]]
    keypoints_perInstance.pop(0)
    while len(keypoints_perInstance) > 1:
        min = 752 * 752
        nearestIndex = 0
        for i in range(0, len(keypoints_perInstance)):
            dis = math.sqrt((keypoints_perInstance[i][0] - newList[-1][0])**2 +
                            (keypoints_perInstance[i][1] - newList[-1][1])**2)
            if dis < min:
                min = dis
                nearestIndex = i
        newList.append(keypoints_perInstance[nearestIndex])
        keypoints_perInstance.pop(nearestIndex)
    newList.append(keypoints_perInstance[-1])
    return newList
def draw_text(draw,
              box: list,
              cls: int,
              score: float,
              category_index: dict,
              color: str,
              font: str = 'arial.ttf',
              font_size: int = 24):

    try:
        font = ImageFont.truetype(r'/public/home/liubingwen/paper/point_rcnn_best/fonts/ARIAL.TTF', font_size)
    except IOError:
        raise Exception("eee")
        font = ImageFont.load_default()

    left, top, right, bottom = box

    left *= 4
    top *= 4
    right *= 4
    bottom *= 4

    display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
    display_str_heights = [font.getsize(ds)[1] for ds in display_str]
    display_str_height = (1 + 2 * 0.05) * max(display_str_heights) * 5

    if top > display_str_height:
        text_top = top - display_str_height
        text_bottom = top
    else:
        text_top = bottom
        text_bottom = bottom + display_str_height

    for ds in display_str:
        text_width, text_height = font.getsize(ds)
        text_width *= 5
        text_height *= 5
        margin = np.ceil(0.05 * text_width)
        draw.rectangle([(left, text_top),
                        (left + text_width + 2 * margin, text_bottom)], fill=color)
        draw.text((left + margin, text_top),
                  ds,
                  fill='black',
                  font=font)
        left += text_width


def draw_masks(image, masks, colors, thresh: float = 0.7, alpha: float = 0.5):
    np_image = np.array(image)
    masks = np.where(masks > thresh, True, False)

    img_to_draw = np.copy(np_image)
    for mask, color in zip(masks, colors):
        img_to_draw[mask] = color

    out = np_image * (1 - alpha) + img_to_draw * alpha
    return fromarray(out.astype(np.uint8))



    return draw
def drawGtAndPredict(
        kpRes,
        boxes: np.ndarray = None,
        classes: np.ndarray = None,
        scores: np.ndarray = None,
        keypoints: np.ndarray = None,
        box_thresh: float = 0.5,
        xml_path: str = None
):

    idxs = np.greater(scores, box_thresh)
    boxes = boxes[idxs]
    classes = classes[idxs]
    scores = scores[idxs]
    keypoints = keypoints[idxs]

    if len(boxes) == 0:
        return kpRes

    # colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in classes]

    if keypoints is not None:
        color_predict = (0, 0, 255)
        for i, (cls, keypoints_perInstance) in enumerate(zip(classes, keypoints)):
            # color = (0, 0, 255) if cls == 1 else colors_kp[i]
            if cls == 2:
                for keypoint in keypoints_perInstance:
                    cv2.circle(kpRes,
                               (int(keypoint[0]), int(keypoint[1])),
                               radius=1,
                               color=color_predict,
                               thickness=1,
                               lineType=cv2.LINE_AA)
    color_gt = (255, 0, 0)
    drawGtPoints(kpRes, xml_path, color_gt)
    drawExplanation(kpRes)

    return cv2

def drawExplanation(imageDraw):
    cv2.rectangle(imageDraw, (650, 450),  (660, 460),  (0, 0, 255), thickness=-1, lineType=cv2.LINE_8)
    cv2.putText(imageDraw, text='predict', org=(680, 460),
                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)
    cv2.rectangle(imageDraw, (650, 465),  (660, 475), (255, 0, 0), thickness=-1, lineType=cv2.LINE_8)
    cv2.putText(imageDraw, text='label', org=(680, 475),
                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=1)
def drawGtPoints(imageDraw, xml_path, color):
    dom = parse(xml_path)
    objects = dom.getElementsByTagName("object")
    for index, object in enumerate(objects):
        if object.getElementsByTagName('name')[0].childNodes[0].data == 'leaf':
            keypoints = object.getElementsByTagName('keypoints')[0].childNodes[0].data
            keypoints = keypoints[1:-1].split(', ')
            for i in range(0, len(keypoints), 2):
                cv2.circle(imageDraw,
                           (int(keypoints[i]), int(keypoints[i+1])),
                           radius=1,
                           color=color,
                           thickness=1,
                           lineType=cv2.LINE_AA)

def draw_objs(image: Image,
              kpRes,
              boxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              keypoints: np.ndarray = None,
              category_index: dict = None,
              box_thresh: float = 0.1,
              line_thickness: int = 8,
              font: str = 'arial.ttf',
              font_size: int = 24,
              draw_boxes_on_image: bool = True,
              draw_keypoints_on_image: bool = True):


    idxs = np.greater(scores, box_thresh)
    boxes = boxes[idxs]
    classes = classes[idxs]
    scores = scores[idxs]
    keypoints = keypoints[idxs]

    if len(boxes) == 0:
        return image

    colors_kp = [ImageColor.getrgb(STANDARD_COLORS[i % len(STANDARD_COLORS)]) for i in range(boxes.shape[0])]
    if draw_boxes_on_image:
        box_draw = ImageDraw.Draw(image)
        for box, cls, score in zip(boxes, classes, scores):
            left, top, right, bottom = box
            color = 'red' if cls == 1 else 'green'
            box_draw.line([(left*1, top*1), (left*1, bottom*1), (right*1, bottom*1),
                       (right*1, top*1), (left*1, top*1)], width=line_thickness*1, fill=color)
    if draw_keypoints_on_image and (keypoints is not None):

        # kpRes = np.zeros_like(kpRes)
        # kp_draw = ImageDraw.Draw(kpRes)

        for i, (cls, keypoints_perInstance)   in enumerate(zip(classes, keypoints)):
            # color = 'blue' if cls == 1 else 'red'
            color = (0, 0, 255) if cls == 1 else colors_kp[i]

            # keypoints_perInstance = kp_sort(keypoints_perInstance)
            # for index, keypoints in enumerate(keypoints_perInstance):
            #
            #     if index < len(keypoints_perInstance)-1:
            #         cv2.line(kpRes,
            #                  (int(keypoints[0]), int(keypoints[1])),
            #                  (int(keypoints_perInstance[index+1][0]), int(keypoints_perInstance[index+1][1])),
            #                  color=color,
            #                  thickness=1)

            for keypoint in keypoints_perInstance:
                cv2.circle(kpRes,
                           (int(keypoint[0])*1, int(keypoint[1])*1),
                           radius=3,
                           color=color,
                           thickness=-1,
                           lineType=cv2.LINE_AA)

                # kp_draw.cir(xy=(keypoint[0], keypoint[1]), fill=color)

           #  kpRes = draw_kp(kpRes, keypoints_perInstance, color)
            # color = 'white'
            # kpRes = draw_keypoints(kpRes, torch.tensor(keypoints_perInstance).unsqueeze(0), colors=color, radius=1)

        # unloader = transforms.ToPILImage()
        # kpRes = kpRes.cpu().clone()
        # kpRes = kpRes.squeeze(0)
        # kpRes = unloader(kpRes)


    return image, cv2
def draw_black(image: Image,
              kpRes,
              boxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              keypoints: np.ndarray = None,
              box_thresh: float = 0.1,
              ):

    idxs = np.greater(scores, box_thresh)
    boxes = boxes[idxs]
    classes = classes[idxs]
    keypoints = keypoints[idxs]

    if len(boxes) == 0:
        return image

    colors_kp = [ImageColor.getrgb(STANDARD_COLORS[i % len(STANDARD_COLORS)]) for i in range(boxes.shape[0])]


    for i, (cls, keypoints_perInstance)   in enumerate(zip(classes, keypoints)):
        color = (0, 0, 255) if cls == 1 else colors_kp[i]

        for keypoint in keypoints_perInstance:

            cv2.circle(kpRes,
                       (int(keypoint[0]) * 4, int(keypoint[1]) * 4),
                       radius=25,
                       color=color,
                       thickness=-1,
                       lineType=cv2.LINE_AA)

    return cv2