import glob

import numpy as np
from PIL import Image

images = []
for i,f in enumerate(glob.glob('/data/dataset/coco/test2017/*.jpg', recursive=True)):
    if i==4000:
        break
    image = Image.open(f).convert('RGB').resize((640, 640))
    npimg = np.asarray(image)
    print(npimg.shape)
    images.append(npimg)
    
np.save('calib_set.npy', np.stack(images, axis=0))