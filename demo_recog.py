import tensorflow as tf
import string
import numpy as np
import random
import os
import cv2
from argparse import ArgumentParser

from tensorflow.python.keras.activations import softmax
# import statistics as stat
# Custom metris / losses
from custom import cat_acc, cce, plate_acc, top_3_k

# For measuring inference time
from timeit import default_timer as timer

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


custom_objects = {
    'cce': cce,
    'cat_acc': cat_acc,
    'plate_acc': plate_acc,
    'top_3_k': top_3_k,
    'softmax': softmax
}


class OCR:
    def __init__(self):
        #self.alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
        self.alphabet = string.digits + string.ascii_uppercase + '_'
        self.model = tf.keras.models.load_model(
            './m1_93_vpa_2.0M-i2.h5', custom_objects=custom_objects)

    def check_low_conf(self, probs, thresh=0.3):
        '''
        Add position of chars. that are < thresh
        '''
        return [i for i, prob in enumerate(probs) if prob < thresh]

    @tf.function
    def predict_from_array(self, img):
        pred = self.model(img, training=False)
        return pred

    def probs_to_plate(self, prediction):
        # 37 * 7 = 259
        # A, B, C .. Z y 0 .. 9, _
        '''
        [
            [0.05, 0.02, ...],
            [],
            ...
            []
        ]
        '''

        '''
               [  0,   1,   2]
        Argmax [0.5, 0.4, 0.1] -> [ 0 -> A ] 
        Max [0.5, 0.4, 0.1] -> 0.5
        '''
        prediction = prediction.reshape((7, 37))
        probs = np.max(prediction, axis=-1)
        prediction = np.argmax(prediction, axis=-1)
        '''
        #self.alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
        
        prediction = [0, 4, 36, 1, 4, 2, 8]

        [0, 4, _, ... ]
        '''
        plate = list(map(lambda x: self.alphabet[x], prediction))
        # plate = ['G', 'T', 'A', '4', '9', '8', '_']
        # probs = [0.3, 0.4, ...]
        return plate, probs

    def ocr(self, sub_frame):
        im = cv2.cvtColor(sub_frame, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(im, dsize=(140, 70), interpolation=cv2.INTER_LINEAR)
        img = img[np.newaxis, ..., np.newaxis] / 255.
        img = tf.constant(img, dtype=tf.float32)
        prediction = self.predict_from_array(img).numpy()
        plate, probs = self.probs_to_plate(prediction)
        # plate = ['G', 'T', 'A', '4', '9', '8', '_']
        plate_str = ''.join(plate)
        print(f'License Plate #: {plate_str}', flush=True)
        print(f'Confidence: {probs}', flush=True)
        return plate_str
        # plate = GTA498_
        # low_conf_chars = 'Low conf. on: ' + \
        #     ' '.join([plate[i] for i in check_low_conf(probs, thresh=.15)])


if __name__ == "__main__":
    # imread
    gta = cv2.imread('.\data\images\GTA.jpg')
    # nom_var = NOM_CLASE()
    gta_string = OCR()
    # nomb_objeto.nombre_funcion(param1, ..., param_n)
    print(gta_string.ocr(gta))
