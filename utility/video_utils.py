import cv2
import numpy as np
import os
import sys
sys.path.insert(1, os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import tensorflow as tf
from face_detector import FaceDetector
from face_det.FaceBoxes import FaceBoxes
from face_det.TDDFA import TDDFA
import yaml
from focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class VideoUtils(object):

    def load_facedetector(root_path):

        cfg = yaml.load(open('%s/configs/mb1_120x120.yml' % root_path), Loader=yaml.SafeLoader)
        cfg['bfm_fp'] = os.path.join(root_path, cfg['bfm_fp'])
        cfg['checkpoint_fp'] = os.path.join(root_path, cfg['checkpoint_fp'])
        face_boxes = FaceBoxes(gpu_mode=True)
        tddfa = TDDFA(gpu_mode=True, **cfg)
        return face_boxes, tddfa

    def detect_faces(img):

        root_path = "face_det"
        cfg = yaml.load(open('%s/configs/mb1_120x120.yml' % root_path), Loader=yaml.SafeLoader)
        cfg['bfm_fp'] = os.path.join(root_path, cfg['bfm_fp'])
        cfg['checkpoint_fp'] = os.path.join(root_path, cfg['checkpoint_fp'])

        face_boxes = FaceBoxes(gpu_mode=True)
        faces = face_boxes(img)
        faces = np.array([[top, right, bottom, left] for left, top, right, bottom, _ in faces]).astype(int).tolist()
        return faces

    def cut_face_locations(frame, boxes):
        face_location = [[left, top, right, bottom] for top, right, bottom, left in boxes]
        try:
            xmin, ymin, xmax, ymax = max(face_location, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))
        except ValueError as e:
            return None

        return frame[max(ymin, 0):ymax, max(xmin, 0):xmax].copy()


    def get_liveness_score(frame, boxes):

        face_location = [[left, top, right, bottom] for top, right, bottom, left in boxes]
        try:
            xmin, ymin, xmax, ymax = max(face_location, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))
        except ValueError as e:
            return None

        def get_normal_face(img):
            return img/255.

        face_img = frame[max(ymin, 0):ymax, max(xmin, 0):xmax].copy()
        standard_face = np.array(get_normal_face(cv2.resize(face_img, (112, 112))))
        img_array_expanded = np.expand_dims(standard_face, axis=0)
        return img_array_expanded, standard_face

    def load_keras_model(model_path):
        try:
            model = tf.keras.models.load_model(model_path, custom_objects={"lr": lambda x: x, "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss})
            model.status = 0
        except Exception as e:
            class empty_model(object):
                def __init__(self):
                    self.status = 600
            model = empty_model()
            print(e)
            print('[ERROR] Model failed to load. Check file path.')
            sys.exit()
        return model

    def get_normal_face(img):
        return img/255.

    def get_max_face(face_locations):
        return max(face_locations, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))

    def cv_draw_landmark(img_ori, pts, box=None, size=1):
        GREEN = (0, 255, 0)
        img = img_ori.copy()
        n = pts.shape[1]
        if n <= 106:
            for i in range(n):
                cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, GREEN, -1)
        else:
            sep = 1
            for i in range(0, n, sep):
                cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, GREEN, 1)
        if box is not None:
            left, top, right, bottom = np.round(box).astype(np.int32)
            left_top = (left, top)
            right_top = (right, top)
            right_bottom = (right, bottom)
            left_bottom = (left, bottom)
            cv2.line(img, left_top, right_top, GREEN, 1, cv2.LINE_AA)
            cv2.line(img, right_top, right_bottom, GREEN, 1, cv2.LINE_AA)
            cv2.line(img, right_bottom, left_bottom, GREEN, 1, cv2.LINE_AA)
            cv2.line(img, left_bottom, left_top, GREEN, 1, cv2.LINE_AA)

        return img


    def estimate_blur(image: np.array):

        # Input an image array
        if type(image).__module__ == 'numpy':
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return np.var(cv2.Laplacian(image, cv2.CV_64F))
        else:
            image = cv2.imread(image)
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return np.var( cv2.Laplacian(image, cv2.CV_64F))

    def rgb2hsv(img_path):

        # Input an image array
        if type(img_path).__module__ == 'numpy':
            hsv = cv2.cvtColor(img_path, cv2.COLOR_BGR2HSV)
            return hsv[...,2].mean()
        else:
            img = cv2.imread(img_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            return hsv[...,2].mean()
