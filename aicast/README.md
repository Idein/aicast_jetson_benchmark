# ai cast

ai cast cannot achieve the same speed performance as listed on [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo). This is because, while the Hailo-8 M.2 module connects via PCIe Gen3.0 2-lane, ai cast is limited to a PCIe Gen2.0 1-lane connection due to the constraints of the Raspberry Pi CM4. This results in slower data transfer speeds between the Raspi and Hailo.

## Compile Model

Install Hailo Dataflow Compiler v3.24 on host PC.

```sh
pip install hailo_dataflow_compiler-3.24.0-py3-none-linux_x86_64.whl
```

### Parse ONNX to HAR
```sh
hailo parser onnx yolox_s_leaky.onnx --start-node-names images --end-node-names Concat_201 Concat_217 Concat_233 --hw-arch hailo8
[info] Translation completed on ONNX model yolox_s_leaky
[info] Initialized runner for yolox_s_leaky
[info] Saved Hailo Archive file at yolox_s_leaky.har
```

### Quantize HAR
Create calibdation dataset.

```
python3 compile/make_calib_set.py
```

```sh
hailo optimize yolox_s_leaky.har --calib-set-path calib_set.npy --model-script model.alls --hw-arch hailo8
```

### Compile HAR to HEF

```sh
hailo compiler yolox_s_leaky_quantized.har --hw-arch hailo8
```

## Device Setup

Install [Hailo Runtime Driver](https://github.com/hailo-ai/hailort-drivers) and [Hailo Runtime](https://github.com/hailo-ai/hailort) on ai cast.

- Driver
```sh
sudo apt install git
sudo apt install raspberrypi-kernel-headers
git clone https://github.com/hailo-ai/hailort-drivers.git
cd hailort-drivers
git checkout v4.10.0
cd linux/pcie
sed -i "s/raspi/$(hostname)/g" Kbuild
make all
sudo make install
sudo modprobe hailo_pci
cd ../../
./download_firmware.sh
sudo mkdir /lib/firmware/hailo
sudo mv hailo8_fw.4.10.0.bin /lib/firmware/hailo/hailo8_fw.bin
sudo cp ./linux/pcie/51-hailo-udev.rules /etc/udev/rules.d/
```

- Runtime

```sh
sudo apt-get install cmake
git clone https://github.com/hailo-ai/hailort.git -b v4.10.0
cd hailort
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 3
sudo make install
```

## Performance

```
$ hailortcli benchmark --no-power true yolox_s_leaky.hef
Starting Measurements...
Measuring FPS in hw_only mode
Network yolox_s_leaky/yolox_s_leaky: 100% | 3762 | FPS: 250.78 | ETA: 00:00:00
Measuring FPS in streaming mode
Network yolox_s_leaky/yolox_s_leaky: 100% | 3539 | FPS: 235.89 | ETA: 00:00:00
Measuring HW Latency
Network yolox_s_leaky/yolox_s_leaky: 100% | 1092 | HW Latency: 8.61 ms | ETA: 00:00:00
```

```
$ hailortcli run yolox_s_leaky.hef  --measure-latency --measure-overall-latency
Running streaming inference (yolox_s_leaky.hef):
  Transform data: true
    Type:      auto
    Quantized: true
Network yolox_s_leaky/yolox_s_leaky: 100% | 456 | HW Latency: 8.61 ms | ETA: 00:00:00
> Inference result:
 Network group: yolox_s_leaky
    Frames count: 456
    HW Latency: 8.61 ms
    Overall Latency: 10.92 ms
```

## Demo

For the inference demo and accuracy evaluation, the pre/post processing used the same implementation as that used in Jetson.

The reason we are using C instead of Python to invoke Hailo Runtime is that the python binding provided by Hailo requires Python 3.8, but we can only use Python 3.7 due to our development environment.

```sh
git clone https://github.com/Idein/aicast_jetson_benchmark.git
cd aicast/edge
make #compile src/model.c
python3 demo_aicast.py --image_path dog.jpg --output_path result.jpg
preprocess 0.018979724248250326
infer 0.014112377166748047
postprocess 0.002598913510640462
all 0.035691014925638836
```

## Accuracy
As in the Jetson experiment, ai cast is used as an inference server, and images are sent from the host PC for accuracy evaluation.

```sh
python3 server.py
```

```
python3 client.py --host 192.168.0.18 --output ./aicast.json 
python3 eval.py --result_json ./aicast.json
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.343
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.518
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.170
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.378
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.460
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.275
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.407
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.213
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.458
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.547
```