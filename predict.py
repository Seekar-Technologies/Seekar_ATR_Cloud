#
# docker run -p 8501:8501 --name seekar_atr --mount type=bind,source=C:/Users/kkfra/OneDrive/Documents/Seekar/ATR/SeekarATR/SeekarTargetClassifierBareBones_TFX/SeekarTargetClassifier/Model/1,target=/models/Model/1 -e MODEL_NAME=Model -t tensorflow/serving
#

from keras.layers import Input
import matplotlib.pyplot as plt
import requests
import base64
import json
import numpy as np
from tensorflow.keras.datasets.mnist import load_data
from PIL import Image
import pprint



def gray_to_rgb(img):
   x=np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
   mychannel=np.repeat(x[:, :, np.newaxis], 3, axis=2)
   return mychannel


image = Image.open('wolf.jpeg')
data = np.asarray(image)
data2 = Image.fromarray(data)

# data_test = gray_to_rgb(data)
data_test = data.reshape(data.shape[1], data.shape[0], data.shape[2])
# data_test = data.reshape(data.shape[0], data.shape[1], data.shape[2])
# data_test = data_test.astype('float32') / 255.0
# print(data_test.shape)
print(f'\tdata_test:\n\t{data_test.shape}')

d = Input(shape=(data.shape[1], data.shape[0], 3), dtype="uint8", name="input")
print(f'\td:\n\t{d}')

x = {
   'input_1': data_test,
   'input_2': 3,
   'input_3': 1
}

# x = {
#    'input_1': d.shape,
#    'input_2': 1,
#    'input_3': 1
# }

# #load MNIST dataset
# (_, _), (x_test, y_test) = load_data()
# # reshape data to have a single channel
# x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
# # normalize pixel values
# x_test = x_test.astype('float32') / 255.0
# print(x_test[0:4])

#server URL
"""
http://{HOST}:{PORT}/v1/models/{MODEL_NAME}:{VERB}
HOST: The domain name or IP address of your model server
PORT: The server port for your URL. By default, TF Serving uses 8501 for REST Endpoint.
MODEL_NAME: The name of the model you’re serving.
VERB: The verb has to do with your model signature. You can specify one of predict, classify or regress.
"""
url = 'http://localhost:8501/v1/models/Model:predict'


class NumpyEncoder(json.JSONEncoder):
   def default(self, obj):
      if isinstance(obj, np.ndarray):
         return obj.tolist()
      return json.JSONEncoder.default(self, obj)



def make_prediction(instances):
   data = json.dumps({"signature_name": "serving_default", "instances": [instances]}, cls=NumpyEncoder)
   # print(data)
   headers = {"content-type": "application/json"}
   json_response = requests.post(url, data=data, headers=headers)
   print(f'response:\n\t{json.loads(json_response.text)}')
   # resp_dict = dict(json.loads(json_response.text))
   # for key in resp_dict.keys():
   #    print(key)
   predictions = json.loads(json_response.text)['predictions']
   return predictions


predictions = make_prediction(x)

for i, pred in enumerate(predictions):
   print(f"True Value: {y_test[i]}, Predicted Value: {np.argmax(pred)}")






# #load MNIST dataset
# (_, _), (x_test, y_test) = load_data()
# # reshape data to have a single channel
# x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
# # normalize pixel values
# x_test = x_test.astype('float32') / 255.0
#
# #server URL
# """
# http://{HOST}:{PORT}/v1/models/{MODEL_NAME}:{VERB}
# HOST: The domain name or IP address of your model server
# PORT: The server port for your URL. By default, TF Serving uses 8501 for REST Endpoint.
# MODEL_NAME: The name of the model you’re serving.
# VERB: The verb has to do with your model signature. You can specify one of predict, classify or regress.
# """
# url = 'http://localhost:8501/v1/models/Model:predict'
# metadata_url = 'http://localhost:8501/v1/models/Model:predict'
#
#
#
# class NumpyEncoder(json.JSONEncoder):
#    def default(self, obj):
#       if isinstance(obj, np.ndarray):
#          return obj.tolist()
#       return json.JSONEncoder.default(self, obj)
#
#
# def get_metadata():
#    data = json.dumps({"signature_name": "serving_default", "inputs": []})
#    headers = {"content-type": "application/json"}
#    json_response = requests.post(metadata_url, data=data, headers=headers)
#    print(json.loads(json_response.text))
#
#
# def make_prediction(instances):
#    # data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
#    data = json.dumps({"signature_name": "serving_default", "instances": [instances]}, cls=NumpyEncoder)
#    print(data)
#    headers = {"content-type": "application/json"}
#    json_response = requests.post(url, data=data, headers=headers)
#    print(f'response:\n\t{json.loads(json_response.text)}')
#    predictions = json.loads(json_response.text)['predictions']
#    return predictions
#
# def gray_to_rgb(img):
#    x=np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
#    mychannel=np.repeat(x[:, :, np.newaxis], 3, axis=2)
#    return mychannel
#
# # x = [ [[x_test.shape[0], x_test.shape[1]], [x_test.shape[0], x_test.shape[1]]]]
# # x = [
# #    {
# #       "width": x_test.shape[1],
# #       "height": x_test.shape[2]
# #    },
# #    {
# #       "width": x_test.shape[1],
# #       "height": x_test.shape[2]
# #    }
# # ]
#
# image_data = np.array(x_test[0], dtype="float32")
# image_data /= 255.0
# image_data = gray_to_rgb(image_data)
# # image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
# image_tensor = tf.convert_to_tensor(image_data[:,:,:1])
# print(f'\n\nIMAGE TENSOR:\n\t{image_tensor}')
#
# x = {
#    'input_1': image_data,
#    'input_2': 1,
#    'input_3': 3
# }
# # y = {k:x_test[0:4].tolist() for k,v in input.iteritems()}
#
# # predictions = make_prediction(x_test[0:4])
# predictions = make_prediction(x)
#
# # get_metadata()
#
# for i, pred in enumerate(predictions):
#    print(f"True Value: {y_test[i]}, Predicted Value: {np.argmax(pred)}")

