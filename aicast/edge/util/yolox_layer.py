import numpy as np


class YoloXLayer():
    """
    anchor free yolo layer for yoloX.
    """
    def __init__(
        self, 
        n_class,
        conf_thresh,
        in_size=640, 
        output_sizes=[[80, 80], [40, 40], [20, 20]],
        n_max_output_bbox=30000
    ):
        self.n_class = n_class
        self.conf_thresh = conf_thresh
        self.n_output = len(output_sizes)
        self.stride = [in_size // o[0] for o in output_sizes]
        self.grid = [
            self.make_grid(nx, ny)
            for nx, ny in output_sizes
        ]
        self.n_max_output_bbox = n_max_output_bbox

    def make_grid(self, nx, ny):
        x = np.arange(nx, dtype=np.float32)
        y = np.arange(ny, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        return np.stack((xv, yv), 2)

    def run(self, preds):
        """
        preds: list of model output np.ndarray
        len(preds) == len(self.output_sizes)
        By default, len(preds) == 3 and
        preds[0].shape = [80, 80, (5 + self.n_class)]
        preds[1].shape = [40, 40, (5 + self.n_class)]
        preds[2].shape = [20, 20, (5 + self.n_class)]
        """
        boxes = []
        confs = []
        classes = []
        for i in range(self.n_output):
            # filter by object threshold
            iy, ix = np.where(preds[i][..., 4] > self.conf_thresh)
            if len(iy) == 0:
                continue
            x = preds[i][iy, ix, :]
            #compute bbox
            grid = self.grid[i][iy, ix]
            xy = (x[:, :2] + grid) * self.stride[i]
            wh = np.exp(x[:, 2:4]) * self.stride[i]
            cls = x[:, 5:5+self.n_class].argmax(1)
            class_conf = np.take_along_axis(x[:, 5:5+self.n_class], cls.reshape(-1, 1), axis=1).flatten()
            obj_conf = x[:, 4]
            conf = class_conf * obj_conf
            boxes.append(np.hstack([xy-wh/2, xy+wh/2]))
            classes.append(cls)
            confs.append(conf)
        
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32), np.zeros((0, ), dtype=np.float32)

        boxes = np.vstack(boxes)
        classes = np.hstack(classes)
        confs = np.hstack(confs)

        # filter by conf
        sel = np.where(confs > self.conf_thresh)
        boxes = boxes[sel]
        classes = classes[sel]
        confs = confs[sel]
        # filter by bbox num
        sel = np.argsort(-confs, axis=0)[:self.n_max_output_bbox]
        boxes = boxes[sel]
        classes = classes[sel]
        confs = confs[sel]

        return boxes, confs, classes