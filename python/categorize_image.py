#!/opt/anaconda3/bin/python

import getopt
import sys
import logging
import numpy as np

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json

logger = logging.getLogger("root")
logger.setLevel(logging.DEBUG)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


fashion_categories = ["T-shirt/top", "Trouser",
                      "Pullover", "Dress", "Coat",
                      "Sandal", "Shirt", "Sneaker",
                      "Bag", "Ankle boot"]


def load_image(filename):
    logger.info("loading image: " + filename)
    img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img


def run_example(img):
    model_json_file = open('../model/mnist_fashion_cnn.json', 'r')
    loaded_model_json = model_json_file.read()
    model_json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("../model/mnist_fashion_cnn.h5")
    logger.info("Loaded model from disk")
    result = model.predict(img)
    logger.info(result)
    category = np.argmax(result,axis=1)[0]
    logger.info(category)
    logger.info(fashion_categories[category])


def usage():
    logger.info("./categorize_image.py --image ../test/test.png")


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi", ["help", "image="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    input_file = None
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-i", "--image"):
            input_file = a
        else:
            assert False, "unhandled option"
    img = load_image(input_file)
    logger.info("image loaded")
    run_example(img)


if __name__ == "__main__":
    main()
