import argparse

from metric import CocoMetric


def main(args):
    metric = CocoMetric(f"{args.coco_dir}/annotations/instances_val2017.json")
    metric.load_results_from_json(args.result_json)
    metric.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_json", default="result.json")
    parser.add_argument("--coco-dir", default="/dataset/coco")
    args = parser.parse_args()
    main(args)