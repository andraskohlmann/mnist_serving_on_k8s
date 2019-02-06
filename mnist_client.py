import json
import random
import sys

from tensorflow import keras
import requests
import numpy as np

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

train_images = train_images / 255.0
test_images = test_images / 255.0

server_uri = 'localhost:8501'
if len(sys.argv) > 1:
    server_uri = sys.argv[1]

rando = random.randint(0, len(test_images) - 1)
data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data) - 52:]))

headers = {"content-type": "application/json"}
json_response = requests.post('http://{}/v1/models/mnist/1:predict'.format(server_uri), data=data, headers=headers)
print(json_response)
predictions = json.loads(json_response.text)['predictions']

print('The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
    class_names[np.argmax(predictions[0])], test_labels[0], class_names[np.argmax(predictions[0])], test_labels[0]))
