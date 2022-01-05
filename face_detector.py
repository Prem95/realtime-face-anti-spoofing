import os
import sys
root_path = "face_det"
sys.path.append(root_path)
import numpy as np
import yaml
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA

cfg = yaml.load(open('%s/configs/mb1_120x120.yml' % root_path), Loader=yaml.SafeLoader)
cfg['bfm_fp'] = os.path.join(root_path, cfg['bfm_fp'])
cfg['checkpoint_fp'] = os.path.join(root_path, cfg['checkpoint_fp'])

use_gpu_flag = 1

onnx_flag = False
if onnx_flag:
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    face_boxes = FaceBoxes_ONNX()
else:
    face_boxes = FaceBoxes(gpu_mode=False if use_gpu_flag == 0 else True)
    tddfa = TDDFA(gpu_mode=False if use_gpu_flag == 0 else True, **cfg)

class FaceDetector(object):

    def __call__(self, img, dense_flag=False):

        faces = face_boxes(img)
        param_lst, roi_box_lst = tddfa(img, faces)
        landmarks = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        faces = np.array([[top, right, bottom, left] for left, top, right, bottom, _ in faces]).astype(int).tolist()
        return faces, landmarks
