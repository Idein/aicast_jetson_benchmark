
import argparse
import time

import cv2
import numpy as np

from model import YoloX_S
from util.pre_post import (
    COCO_CLASSES,
    multiclass_nms_class_agnostic_aicast,
    preproc,
    vis,
)
from util.yolox_layer import YoloXLayer


class YoloXDection:
    def __init__(self):
        self.yolox_layer = YoloXLayer(80, 0.1, in_size=640, output_sizes=[[80, 80], [40, 40], [20, 20]])
        self.model = YoloX_S()

    def forward(self, origin_img, input_shape, thresh: float, num_iteration: int, print_summary=True, visualize=True):
        execution_times = {
            "preprocess": [],
            "infer": [],
            "postprocess": [],
            "all": []
        }
        input_shape = tuple(map(int, input_shape.split(',')))
        for i in range(num_iteration):
            t1 = time.time()
            input, ratio = preproc(origin_img, input_shape, swap=(0, 1, 2))
            t2 = time.time()
            val0, val1, val2 = self.model.infer(input)
            t3 = time.time()
            bboxes, scores, classes = self.yolox_layer.run([val0, val1, val2])
            bboxes /= ratio
            dets = multiclass_nms_class_agnostic_aicast(bboxes, scores, classes, nms_thr=0.45)
            t4 = time.time()
            execution_times["preprocess"].append(t2 - t1)
            execution_times["infer"].append(t3 - t2)
            execution_times["postprocess"].append(t4 - t3)
            execution_times["all"].append(t4 - t1)
        if dets is not None and visualize is True:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=thresh, class_names=COCO_CLASSES)
        if print_summary:
            for k, v in execution_times.items():
                print(k, np.mean(v))
            print("FPS", 1 / np.mean(execution_times["all"]))
        return origin_img, dets


if __name__ == '__main__':
    parser = argparse.ArgumentParser("aicast Demo")
    parser.add_argument('--input_shape', type=str, default="640,640", help='input shape')
    parser.add_argument('--image_path', type=str, default=None, help='image path')
    parser.add_argument('--output_path', type=str, default=None, help='output path')
    parser.add_argument('--thresh', type=float, default=0.4)
    parser.add_argument('--num_iteration', type=int, default=30, help='measure average execution time for this iteration')
    args = parser.parse_args()

    detection = YoloXDection()
    origin_img = cv2.imread(args.image_path)
    output_img, _ = detection.forward(origin_img, args.input_shape, args.thresh, args.num_iteration)
    cv2.imwrite(args.output_path, output_img)
