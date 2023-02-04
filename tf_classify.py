from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2


np.set_printoptions(suppress=True)

model = load_model("model/keras_Model.h5", compile=False)

class_names = open("model/labels.txt", "r").readlines()

model2 = load_model("model/keras_Model.h5", compile=False)

class_names2 = open("model/labels.txt", "r").readlines()


def classify(cvimg):
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(cvimg, (224, 224), interpolation=cv2.INTER_AREA)

    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name[2:]

def coneify(cvimg):
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(cvimg, (224, 224), interpolation=cv2.INTER_AREA)

    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    prediction = model2.predict(image)
    index = np.argmax(prediction)
    class_name = class_names2[index]
    confidence_score = prediction[0][index]

    return class_name[2:]