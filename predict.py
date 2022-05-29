import argparse
import os
import pickle
import time

import cv2
import numpy as np
import imutils
import hinge_and_cold_feature_extraction.src.hinge_feature_extraction as hcf

parser = argparse.ArgumentParser(
    description='take full paths for input and output files')

parser.add_argument('--test', type=str,
                    help='full path to input images')
parser.add_argument('--out', type=str, help='full path to output directory')

args = parser.parse_args()
test_path = os.path.join(args.test, '')
out_path = os.path.join(args.out, '')
# print(out_path)
if not os.path.exists(test_path):
    print('Error: Input path does not exist')
    exit()


filename = 'hinge_rfc_checkpoint.ckpt'
if os.path.exists('hinge_rfc_checkpoint.ckpt'):
    model = pickle.load(open(filename, 'rb'))

    if os.path.exists(os.path.join(
            out_path, 'results.txt')):
        os.remove(os.path.join(out_path, 'results.txt'))
    if os.path.exists(os.path.join(out_path, 'times.txt')):
        os.remove(os.path.join(out_path, 'times.txt'))
    test_img_paths = []
    try:
        prediction_file = open(os.path.join(
            out_path, 'results.txt'), 'a', buffering=1)
        timing_file = open(os.path.join(
            out_path, 'times.txt'), 'a', buffering=1)
    except Exception as e:
        print('Error: Output path does not exist')
        exit()
    for img in os.listdir(test_path):
        test_img_paths.append(os.path.join(test_path, img))
    for img_path in test_img_paths:
        test_img = cv2.imread(img_path, 0)
        hinge_classifier = hcf.Hinge((10, 3, False, True))
        start = time.time()
        img_feautres = hinge_classifier.get_hinge_features(test_img)
        img_feautres = np.reshape(img_feautres, (1, -1))
        try:
            prediction = 1 - model.predict(img_feautres)[0]
        except Exception as e:
            prediction = -1
        duration = time.time() - start
        if duration == 0:
            duration = 0.001
        prediction_file.write(str(prediction) + '\n')
        timing_file.write(str(round(duration, 2)) + '\n')
    prediction_file.close()
    timing_file.close()
else:
    print('Error: Checkpoint does not exist')
    exit()
