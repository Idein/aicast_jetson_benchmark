import argparse
import glob
import json
import os

import requests
from tqdm import tqdm


def glob_images(coco_dir):
    images = glob.glob(f"{coco_dir}/val2017/*.jpg")
    print(len(images))
    return images


def run(host, image_id, image_path):
    url = f'http://{host}:5000/predict'
    files = {'image': open(image_path, 'rb')}
    response = requests.post(url, files=files, data={"image_id": image_id})
    if response.status_code == 200:
        res = response.json()
        return res
    else:
        print("error")
        return []


def main(args):
    image_paths = glob_images(args.coco_dir)
    all_results = []
    for image_path in tqdm(image_paths):
        image_id = int(os.path.basename(image_path).replace(".jpg", ""))
        results = run(args.host, image_id, image_path)
        all_results.extend(results)
    open(args.output, "w").write(json.dumps(all_results))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.0.20")
    parser.add_argument("--output", default="result.json")
    parser.add_argument("--coco-dir", default="/dataset/coco")
    args = parser.parse_args()
    main(args)
