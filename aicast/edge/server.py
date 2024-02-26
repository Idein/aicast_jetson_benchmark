import cv2
import numpy as np
from flask import Flask, jsonify, request

from model import YoloX_S
from util.pre_post import multiclass_nms_class_agnostic_aicast, preproc
from util.yolox_layer import YoloXLayer

app = Flask(__name__)

model = YoloX_S()
conf = 0.1
nms_thre = 0.45
yolox_layer = YoloXLayer(80, conf, in_size=640, output_sizes=[[80, 80], [40, 40], [20, 20]])


def process_image(file_stream):
    img = cv2.imdecode(np.fromstring(file_stream.read(), np.uint8), cv2.IMREAD_COLOR)
    resized_img = cv2.resize(img, (640, 640))
    input_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    return input_img, img.shape


coco80_to_91class_map = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]


def get_coco91_from_coco80(idx):
    return coco80_to_91class_map[idx]


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400
    if 'image_id' in request.form:
        image_id = int(request.form['image_id'])
    else:
        return jsonify({"error": "No image_id provided"}), 400
    file_stream = request.files['image']
    origin_img = cv2.imdecode(np.fromstring(file_stream.read(), np.uint8), cv2.IMREAD_COLOR)
    input, ratio = preproc(origin_img, (640, 640))
    val0, val1, val2 = model.infer(input)
    bboxes, scores, classes = yolox_layer.run([val0, val1, val2])
    bboxes /= ratio

    dets = multiclass_nms_class_agnostic_aicast(bboxes, scores, classes, nms_thr=nms_thre)
    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    results = []
    for bbox, score, cls in zip(final_boxes, final_scores, final_cls_inds):
        x1, y1, x2, y2 = bbox
        result = {
            "image_id": image_id,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score),
            "category_id": get_coco91_from_coco80(int(cls))
        }
        results.append(result)

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', use_reloader=False)
