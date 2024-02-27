# Jetson

Due to licensing issues, only the differences are published.

```sh
git clone git@github.com:xuanandsix/Tensorrt-int8-quantization-pipline.git
cd Tensorrt-int8-quantization-pipline
git apply patch ../0001-evaluate-yolox_s_leaky-on-jetson.patch
```

Download model:

```
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/yolox_s_leaky/pretrained/2023-05-31/yolox_s_leaky.zip
unzip yolox_s_leaky.zip
```

## Performance

### FP32, FP16

```sh
export PATH=/usr/src/tensorrt/bin:$PATH
# FP32
trtexec --onnx=./yolox_s_leaky.onnx --saveEngine=./default.trt
# FP16
trtexec --onnx=./yolox_s_leaky.onnx --saveEngine=./fp16.trt --fp16
```

Log (Jeston Orin Nano FP16):
```
[02/26/2024-10:43:06] [I] === Performance summary ===
[02/26/2024-10:43:06] [I] Throughput: 97.8649 qps
[02/26/2024-10:43:06] [I] Latency: min = 10.642 ms, max = 10.9861 ms, mean = 10.7971 ms, median = 10.7972 ms, percentile(90%) = 10.9329 ms, percentile(95%) = 10.949 ms, percentile(99%) = 10.9744 ms
[02/26/2024-10:43:06] [I] Enqueue Time: min = 1.1098 ms, max = 2.30975 ms, mean = 1.36227 ms, median = 1.19873 ms, percentile(90%) = 1.9928 ms, percentile(95%) = 2.07275 ms, percentile(99%) = 2.15002 ms
[02/26/2024-10:43:06] [I] H2D Latency: min = 0.299072 ms, max = 0.373962 ms, mean = 0.316446 ms, median = 0.310486 ms, percentile(90%) = 0.332031 ms, percentile(95%) = 0.345398 ms, percentile(99%) = 0.354126 ms
[02/26/2024-10:43:06] [I] GPU Compute Time: min = 10.0343 ms, max = 10.3575 ms, mean = 10.1824 ms, median = 10.1851 ms, percentile(90%) = 10.3206 ms, percentile(95%) = 10.338 ms, percentile(99%) = 10.356 ms
[02/26/2024-10:43:06] [I] D2H Latency: min = 0.17041 ms, max = 0.312134 ms, mean = 0.29821 ms, median = 0.298584 ms, percentile(90%) = 0.302856 ms, percentile(95%) = 0.304077 ms, percentile(99%) = 0.310608 ms
[02/26/2024-10:43:06] [I] Total Host Walltime: 3.0348 s
[02/26/2024-10:43:06] [I] Total GPU Compute Time: 3.02418 s
[02/26/2024-10:43:06] [I] Explanations of the performance metrics are printed in the verbose logs.
[02/26/2024-10:43:06] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v8502] # trtexec --onnx=./yolox_s_leaky.onnx --saveEngine=./fp16.trt --fp16
```

### INT8
Jetson Orin Nano has Tensor Core, which supports INT8 precision inference.

We've made several modifications based on [Tensorrt-int8-quantization-pipline](https://github.com/xuanandsix/Tensorrt-int8-quantization-pipline).
- The preprocessing normalization has been eliminated.
- Fixes to match TensorRT version updates.
- The Calibrator has been replaced with `IInt8MinMaxCalibrator` approach.

Copy COCO test dataset to Jetson and run quantization.

```
python3 ./sample.py  --training_data_path ~/idein/coco/test2017/
python3 quantization.py
```

`modelInt8.engine` is saved as quantized engine. 

```sh
# INT8
trtexec --onnx=./yolox_s_leaky.onnx --loadEngine=./modelInt8.engine
```

## Demo

- FP32, FP16: use onnxruntime-gpu 1.16.0 for Jetson Orin Nano, 1.11.0 for Jetson Nano from [Jetson Zoo](https://elinux.org/Jetson_Zoo).
- INT8: use TensorRT python bindings.

```sh
# FP32
python3 demo_onnx.py --model_path ./yolox_s_leaky.onnx --label_name_path ./coco.label --image_path ./dog.jpg --output_path ./result.jpg --mode trt --warmup
preprocess 0.01185154914855957
infer 0.02301931381225586
postprocess 0.009637117385864258
all 0.04450798034667969
FPS 22.467880865652454
# FP16
python3 demo_onnx.py --model_path ./yolox_s_leaky.onnx --label_name_path ./coco.label --image_path ./dog.jpg --output_path ./result.jpg --mode trt_fp16 --warmup
preprocess 0.011473894119262695
infer 0.01324462890625
postprocess 0.012219429016113281
all 0.03693795204162598
FPS 27.07242672449961
# INT8
python3 demo_trt.py --model_path ./modelInt8.engine --label_name_path ./coco.label --image_path ./dog.jpg --output_path ./result_int8.jpg 
preprocess 0.012786388397216797
infer 0.014822721481323242
postprocess 0.008907318115234375
all 0.036516427993774414
FPS 27.384934807163702
```

## Accuracy

For accuracy evaluation, we will use Flask to turn edge devices into inference servers, allowing images to be sent from the client side and inference results to be received. The server-side code is intended purely for experimental purposes, assuming sequential execution by a single client.

```
python3 server.py --mode trt --model_path ./yolox_s_leaky.onnx
```

Execute the client-side code on the host PC, and the inference results will be saved in a JSON file. Finally, accuracy evaluation will be performed using pycocotools.

```sh
git clone https://github.com/Idein/aicast_jetson_benchmark.git
cd host
python3 client.py --host 192.168.0.20 --output ./jetson_orin_nano_trt.json 
python3 eval.py --result_json ./jetson_orin_nano_trt.json
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.321
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.476
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.346
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.150
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.353
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.452
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.274
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.395
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.403
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.192
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.439
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.559
```