# aicast_jetson_benchmark

Benchmark YOLOX-S-Leaky on Jetson Nano, Jetson Orin Nano, and ai cast.

Blog Post (Japanese): [ai castの性能評価 (vs Jetson Orin Nano)](https://note.com/idein/n/n63728d3c107e)

## Environment
- Jetson Orin Nano
    * JetPack 5.1.2 TensorRT 8.5.2 CUDA 11.4
- Jetson Nano
    * JetPack 4.6.1 TensorRT 8.2.1 CUDA 10.2
- [ai cast](https://www.idein.jp/ja/blog/20230221-aicast)
    * Raspberry Pi Compute Module 4 + Hailo-8
    * Hailo Runtime v4.10.0

For more information on benchmarking instructions for each of these devices, please refer [jetson/README.md](./jetson/README.md) and [aicast/README.md](./aicast/README.md)

## Performance

| device            | runtime/precision   | FPS   | infer latency(ms) | all latency(ms) |
|---------------------|-------------------|------:|------------------:|----------------:|
| Jetson Orin Nano    | trtexec/FP32      | 49.0  | 20.9              | 44.5            |
| Jetson Orin Nano    | trtexec/FP16      | 97.8  | 10.7              | 36.9            |
| Jetson Orin Nano    | trtexec/INT8      | 126.3 | 8.6               | 36.5            |
| Jetson Nano         | trtexec/FP32      | 6.5   | 153.3             | 224.8           |
| Jetson Nano         | trtexec/FP16      | 10.6  | 93.7              | 154.5           |
| ai cast             | Hailo8/INT8       | 250.5 | 8.6               | 35.6            |

- `FPS` and `infer latency` are measured using benchmark tools. (jetson: `trtexec`, ai cast: `hailortcli`)
- `all latency` are measured by a python script. The latency includes the time for pre/post processing.

The highest FPS was achieved by ai cast due to the Hailo-8's dataflow architecture.   
The inference latency of ai cast and Jetson Orin Nano (INT8) was nearly identical.   
In these cases, the time taken for pre-processing was almost equal to the inference process itself, suggesting that there is potential for further speed improvements. This could be achieved by transitioning the pre-processing from the Python implementation to a C implementation, or, in the case of Jetson, to a CUDA implementation.


## Accuracy

| device            | runtime/precision   | mAP@IoU0.50:0.95   | mAP@IoU0.50 | mAP@IoU0.75 |
|---------------------|-------------------|------:|------------------:|----------------:|
| Jetson Orin Nano    | trtexec/FP32      | 35.2  | 52.2              | 38.1            |
| Jetson Orin Nano    | trtexec/FP16      | 35.2  | 52.2              | 38.0            |
| Jetson Orin Nano    | trtexec/INT8      | 31.0 | 46.8             | 33.7            |
| ai cast             | Hailo8/INT8       | 34.3   | 51.8             | 37.1           |

- Accuracy is measured using `COCO2017` val dataset and `pycocotools`.

Although model quantization generally leads to a reduction in accuracy, ai cast demonstrates that the decrease in accuracy is minimal compared to that of FP32.

We believe the lower accuracy of TensorRT (INT8) compared to ai Cast (INT8) is attributable to issues with the calibration method. While our focus in this discussion is on ai Cast, we will not explore this issue in depth. However, we posit that accuracy could be enhanced by refining the calibration approach.

## Power Consumption

Connect a [watt monitor](https://www.sanwa.co.jp/product/syohin?code=TAP-TST8) to the AC adapter to easily measure and compare power consumption under the following three conditions:

1. Idle
1. During a running demo (`demo_onnx/trt.py`, `demo_aicast.py`): Executes inference on a single thread.
1. During a benchmark (`trtexec`, `hailortcli`): Exhibits the highest power consumption due to continuous data transmission.

| device            | Idle(W)   | Demo(W) | Benchmark(W) |
|---------------------|------:|------------------:|----------------:|
| Jetson Orin Nano (INT8)    | 6.8 | 8.8 | 12.7 |
| Jetson Nano (FP16)    | 1.5 | 5.1 | 8.8 |
| ai cast         | 3.7 | 4.5 | 7.8 |

It was observed that ai Cast maintains low power consumption even under heavy loads.