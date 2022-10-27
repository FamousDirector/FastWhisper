# FastWhisper
This is an optimized implementation of [OpenAI's Whisper](https://github.com/openai/whisper) using a greedy decode
for multilingual transcription. It supports all sizes of the Whisper model (from `tiny` to `large`). 

This codebase exports the models into TorchScript, ONNX, and TensorRT formats.

## Getting Started
[Docker](https://www.docker.com/), `docker-compose` and `nvidia-container-toolkit` is required to be installed.

Simply run `bash run.sh`; then you can access a simple UI at [http://localhost:7860/](http://localhost:7860/).

Please note the initial setup can be quite slow and requires significant memory. 
Additionally, the TensorRT export will require an Nvidia GPU.

By default, the model selects `tiny` model to be exported to the optimized frameworks. 
This can be adjusted by changing th `MODEL_NAME` in `run.sh`. 
Please note the larger models *will take much longer and use more memory!* 
The `medium` size took 4 hours and 40GB+ of memory on my system!

## Model Performance
With my system with an AMD Ryzen Threadripper PRO 3975WX and an Nvidia RTX A6000, 
the following inference time on a ~5 second audio clip:

| Model Framework (Model)     | `tiny` | `medium`     |
| :---        |    :----:   |          ---: |
| PyTorch (Original)      | 52.9 ms       | 327 ms   |
| PyTorch (Modded)    | 41.6 ms        | 261 ms      |
| TorchScript (Modded)    | 32.7 ms        | 209 ms      |
| ONNX (Modded)    | 16.8 ms        | 142 ms      |
| TensorRT (Modded)    | 8.1 ms        | 60 ms      |

Note the PyTorch (Original) model is using a Beam Search 
while the PyTorch (Modded) model is using a Greedy Search for decoding.

Note, the first few inference times will be quite long while the model "warms-up".

## Disclaimer
The accelerated models should be validated for accuracy against the original model before being used. 
Limited testing has been done. Use at your own risk.

## Sources
Credit to https://github.com/evanarlian/whisper-torchscript/ for creating a first cut of a scriptable model.