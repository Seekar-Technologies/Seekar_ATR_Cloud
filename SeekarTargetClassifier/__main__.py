# __main__.py
# Seekar Technologies, LLC.
########################################################################################################################
# This file compiles the model, launches the recognition program, and sets up file processing.
# The file processes command-line arguments and performs some error checking on the I/O files input by the user.
########################################################################################################################
from SeekarTargetClassifier import YOLO, error_checking
downloadedModel = YOLO
import cv2
import json
import os
from PIL import Image, ImageDraw
from timeit import default_timer as timer
from SeekarTargetClassifier.classifier_utils import class_labels, classify_object
import pandas as pd
import numpy as np
import argparse
########################################################################################################################
########################################################################################################################
DEMO_MODE = False
SHOULD_REPEAT = False
DETAIL_REPORT = False
########################################################################################################################
########################################################################################################################
## for AWS Lambda script, seems to resolve some issues with file routing
# DETECTION_RESULTS_FILE = 'DETECTION_DATA/DETECTION_RESULTS/DETECTION_RESULTS.txt'
# DETECTION_RESULTS_IMAGE = 'DETECTION_DATA/DETECTION_RESULTS/DETECTION_RESULTS.png'
# MODEL_WEIGHTS = 'Model/trained_weights_final.h5'
# MODEL_CLASSES = 'Model/data_classes.txt'
# ANCHORS_PATH = 'src/keras_yolo3/model_data/yolo-tiny_anchors.txt'
# INPUT_IMAGE_PATH_PREFIX = 'DETECTION_DATA/TEST_DETECTION_IMAGE/'

## for normal console operation
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
DETECTION_RESULTS_FILE = 'SeekarTargetClassifier/DETECTION_DATA/DETECTION_RESULTS/DETECTION_RESULTS.txt'
DETECTION_RESULTS_IMAGE = 'SeekarTargetClassifier/DETECTION_DATA/DETECTION_RESULTS/DETECTION_RESULTS.png'
MODEL_WEIGHTS = 'SeekarTargetClassifier/Model/trained_weights_final.h5'
MODEL_CLASSES = 'SeekarTargetClassifier/Model/data_classes.txt'
ANCHORS_PATH = os.path.join('SeekarTargetClassifier', 'src', 'keras_yolo3', 'model_data', 'yolo-tiny_anchors.txt')
INPUT_IMAGE_PATH_PREFIX = 'SeekarTargetClassifier/DETECTION_DATA/TEST_DETECTION_IMAGE/'
MIN_CONFIDENCE_THRESHOLD = 0.2
########################################################################################################################
########################################################################################################################

rep_count = 0
yolo = YOLO(
    **{
        "model_path": MODEL_WEIGHTS,
        "anchors_path": ANCHORS_PATH,
        "classes_path": MODEL_CLASSES,
        "score": MIN_CONFIDENCE_THRESHOLD,
        "gpu_num": 1,
        "model_image_size": (416, 416),
    }
)


def classify_targets_in_image(yolo, INPUT_IMAGE_PATH):
    out_df = pd.DataFrame(
        columns=[
            "image",
            "image_path",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "label",
            "confidence",
            "x_size",
            "y_size",
        ]
    )

    class_file = open(MODEL_CLASSES, "r")
    if INPUT_IMAGE_PATH:
        start = timer()
        text_out = ""
        pre_image = Image.open(INPUT_IMAGE_PATH)
        prediction, img_array = classify_object(
            yolo,
            pre_image
        )
        y_size, x_size, _ = np.array(img_array).shape
        for single_prediction in prediction:
            out_df = out_df.append(
                pd.DataFrame(
                    [
                        [
                            os.path.basename(INPUT_IMAGE_PATH.rstrip("\n")),
                            INPUT_IMAGE_PATH.rstrip("\n"),
                        ]
                        + single_prediction
                        + [x_size, y_size]
                    ],
                    columns=[
                        "image",
                        "image_path",
                        "xmin",
                        "ymin",
                        "xmax",
                        "ymax",
                        "label",
                        "confidence",
                        "x_size",
                        "y_size",
                    ],
                )
            )

    # OPTION 1: uncomment if we only want to return the label of what was classified
    if not DETAIL_REPORT:
        if len(out_df) > 0:
            # print('target detected')
            idx = out_df.values[0]
            result = f'{class_labels[idx[6]]}'
            print(result)
            txtFile = open(str(DETECTION_RESULTS_FILE), 'a')
            txtFile.truncate(0)
            txtFile.write(f'{result}')
            txtFile.close()
        else:
            print('NONE')
    else:
    # OPTION 2: uncomment if we want to return the bounding box, confidence, label, and image dimensions
        if len(out_df) > 0:
            idx = out_df.values[0]
            bbox = [int(idx[2]), int(idx[3]), int(idx[4]), int(idx[5])]
            label = class_labels[idx[6]]
            conf = float(idx[7])
            im_width = float(idx[8])
            im_height = float(idx[9])

            post_image = Image.open(INPUT_IMAGE_PATH)
            ann_image = ImageDraw.Draw(post_image)
            ann_image.rectangle([(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))], outline='red', width=5)

            if DEMO_MODE:
                post_image.show()

            post_image.save(DETECTION_RESULTS_IMAGE)
            results = {'bbox':bbox, 'label':label, 'conf':conf, 'im_width':im_width, 'im_height':im_height}
            txtFile = open(DETECTION_RESULTS_FILE, "a")
            txtFile.truncate(0)
            txtFile.write('bbox: ' + str(bbox)
                          + '\nlabel: ' + str(label)
                          + '\nconfidence: ' + str(conf)
                          + '\nim_width: ' + str(im_width)
                          + '\nim_height: ' + str(im_height))
            txtFile.close()

            # Uncomment for a pure string response
            # print('\n\n\nbbox: ' + str(bbox)
            #       + '\nlabel: ' + str(label)
            #       + '\nconfidence: ' + str(conf)
            #       + '\nim_width: ' + str(im_width)
            #       + '\nim_height: ' + str(im_height))

            # Uncomment for a pure json response
            json_resp = {"bbox": str(bbox),
                         "label": str(label),
                         "confidence": str(conf),
                         "im_width": str(im_width),
                         "im_height": str(im_height)
                         }
            print(json.dumps(json_resp))

        else:
            # Uncomment for a pure string response
            print('none')
            # Uncomment for a pure json response
            json_resp = {"bbox": "none",
                         "label": "none",
                         "confidence": "none",
                         "im_width": "none",
                         "im_height": "none"
                         }
            print(json.dumps(json_resp))



def check_image_specs(image_name) -> bool:
    pre_image = Image.open(image_name)
    w, h = pre_image.size
    if w != 1200 and h != 1200:
        return False
    else:
        return True


def reformat_image_for_detection(image_name):
    const_w = 1600
    const_h = 1200
    pre_image = cv2.imread(image_name, cv2.IMREAD_COLOR)
    h, w = pre_image.shape[:2]
    cx = w / 2
    cy = h / 2
    #center crop
    if w > const_w and h > const_h:
        x_min = int(cx - (0.5 * const_w))
        x_max = int(cx + (0.5 * const_w))
        y_min = int(cy - (0.5 * const_h))
        y_max = int(cy + (0.5 * const_h))
        pre_imageX = cv2.imread(image_name, cv2.IMREAD_COLOR)
        crop = pre_imageX[y_min:y_max, x_min:x_max]
        cv2.imwrite('X.jpeg', crop)
        return 'X.jpeg'
    else:
        new_size = (const_w, const_h)
        pre_imageX = cv2.imread(image_name, cv2.IMREAD_COLOR)
        pre_imageX.resize(new_size)
        cv2.imwrite('X.jpeg', pre_imageX)
    return image_name


# construct a parser to accept and parse command-line arguments
parser = argparse.ArgumentParser()
# add the argument for the input file
parser.add_argument('image_name', type=str, default='wolf.jpeg',
                    help='name of image or image path with a .jpeg, .jpg, or .png extension')
# declare the arguments as separate variables
parse_args = vars(parser.parse_args())
# initialize input and output files for error checking
image_name = parse_args['image_name']

#error handling for input file
if SHOULD_REPEAT:
    while SHOULD_REPEAT:
        rep_count += 1
        # image_name = input()

        if len(image_name) > 1:
            #verify that the input file the user specified is in 'io_files' directory.
            if error_checking.check_project_for_input_file_in_correct_directory(image_name) == True:
                if check_image_specs(image_name):
                    classify_targets_in_image(yolo, image_name)
                else:
                    classify_targets_in_image(yolo, reformat_image_for_detection(image_name))

            #if specified input file exists, but in wrong directory, move it to correct directory
            elif error_checking.check_project_for_input_file(image_name) == True:
                input_file = error_checking.find_input_file(image_name)
                print(f'Your input file was placed in the incorrect spot and it was found at the following path: {input_file}')
                print(f'The program relocated your file to the most parent directory.')
                if check_image_specs(image_name):
                    classify_targets_in_image(yolo, image_name)
                else:
                    classify_targets_in_image(yolo, reformat_image_for_detection(image_name))
            else:
                print(
                    f'The {image_name} file you specified was not found. Move file to most parent directory.')
            continue
        else:
            SHOULD_REPEAT = False
else:
    # image_name = input()
    if len(image_name) > 1:
        #verify that the input file the user specified is in 'io_files' directory.
        if error_checking.check_project_for_input_file_in_correct_directory(image_name) == True:
            if check_image_specs(image_name):
                classify_targets_in_image(yolo, image_name)
            else:
                classify_targets_in_image(yolo, reformat_image_for_detection(image_name))

        #if specified input file exists, but in wrong directory, move it to correct directory
        elif error_checking.check_project_for_input_file(image_name) == True:
            input_file = error_checking.find_input_file(image_name)
            print(f'Your input file was placed in the incorrect spot and it was found at the following path: {input_file}')
            print(f'The program relocated your file to the most parent directory.')
            if check_image_specs(image_name):
                classify_targets_in_image(yolo, image_name)
            else:
                classify_targets_in_image(yolo, reformat_image_for_detection(image_name))
        else:
            print(
                f'The {image_name} file you specified was not found. Move file to most parent directory.')


########################################################################################################################
########################################################################################################################
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, save_model

model = tf.keras.models.load_model('Model/1/trained_weights_final','h5')
tf.saved_model.save(model, 'C:/Users/kkfra/OneDrive/Documents/Seekar/ATR/SeekarATR/SeekarTargetClassifierBareBones_TFX/SeekarTargetClassifier/Model/2/trained_weights_final')





########################################################################################################################
# Seekar Technologies, LLC.
########################################################################################################################