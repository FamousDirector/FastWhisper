import gradio as gr
import whisper
import torch
import os
import time
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

from whisper.tokenizer import get_tokenizer

import scriptable_model

model_name = os.getenv("MODEL_NAME", "")
if model_name == "":
    model_name = "medium"
    print(f"Did not get a model_name from env variable. Defaulting to {model_name}...")

print(f"Using {model_name} Whisper model")

model = whisper.load_model(model_name)

whisper_path = Path(f"~/.cache/whisper/{model_name}.pt").expanduser()
with open(whisper_path, "rb") as f:
    checkpoint = torch.load(f)

modded = scriptable_model.Whisper(model.dims).eval().to(model.device)
modded.load_state_dict(checkpoint["model_state_dict"])
scripted_model = torch.jit.script(modded).eval().to(model.device)

# setup tokenizer
tokenizer = get_tokenizer(True).tokenizer  # use HF Transformer package tokenizer
# suppressed tokens, see SuppressBlank and SuppressTokens class
suppress_blanks = [220, 50257]
suppress_nonspeech = [1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92,
                      93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253,
                      3268, 3536,
                      3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562,
                      13793, 14157,
                      14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279,
                      29464, 31650,
                      32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50360, 50361, 50362]

# onnx setup
onnx_model = onnx.load(f'{model_name}.onnx')
onnx_input_names = [node.name for node in onnx_model.graph.input]
onnx_output_names = [node.name for node in onnx_model.graph.output]
print(f"ONNX Input names are: {onnx_input_names}")
print(f"ONNX Output names are: {onnx_output_names}")
del onnx_model  # cleanup

ort_sess = ort.InferenceSession(f'{model_name}.onnx', providers=['CUDAExecutionProvider'])

# this "builds" a TRT graph and takes significant time
# trt_ort_sess = ort.InferenceSession('tiny.onnx', providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])

# TRT
load_engine = EngineFromBytes(BytesFromPath(f"{model_name}_fp16.engine"))
trt_runner = TrtRunner(load_engine)
trt_runner.activate()


def detect_audio(audio_data, language):
    starting_tokens = tokenizer.encode(f"<|startoftranscript|><|{language}|><|transcribe|><|notimestamps|>",
                                       return_tensors="pt").to(model.device)

    print('--------------------')
    last_reported_time = time.time()

    # simple helper function to report time taken inbetween calls
    def report_time_taken(text_to_print="Time taken: "):
        nonlocal last_reported_time
        print(text_to_print + " {0:.4f} ms".format((time.time() - last_reported_time) * 1000))
        last_reported_time = time.time()

    # grab audio
    audio = whisper.load_audio(audio_data)
    report_time_taken("Initial audio import time: ")

    # pre processing audio signal
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(model.device)
    report_time_taken("Audio pre processing time: ")

    # original whisper inference
    options = whisper.DecodingOptions(language=language, without_timestamps=True)
    original_result = whisper.decode(model, mel, options)[0].text
    report_time_taken("Regular model time: ")

    # modded model PyTorch
    modded_transcribed = modded(starting_tokens, mel, suppress_blanks, suppress_nonspeech)
    modded_result = tokenizer.batch_decode(modded_transcribed, skip_special_tokens=True)[0]
    report_time_taken("Modded time: ")

    # modded model TorchScript
    scripted_transcribed = scripted_model(starting_tokens, mel, suppress_blanks, suppress_nonspeech)
    scripted_result = tokenizer.batch_decode(scripted_transcribed, skip_special_tokens=True)[0]
    report_time_taken("Scripted time: ")

    # modded model ONNX
    onnx_transcribed = ort_sess.run(None, {onnx_input_names[0]: starting_tokens.cpu().numpy(),
                                           onnx_input_names[1]: mel.cpu().numpy()})
    onnx_transcribed = torch.tensor(onnx_transcribed[0])
    onnx_result = tokenizer.batch_decode(onnx_transcribed, skip_special_tokens=True)[0]
    report_time_taken("ONNX time: ")

    # modded model TensorRT
    trt_transcribed = trt_runner.infer(feed_dict={onnx_input_names[0]: starting_tokens.cpu().numpy().astype(np.int32),
                                                  onnx_input_names[1]: mel.cpu().numpy()})
    trt_transcribed = torch.tensor(trt_transcribed[onnx_output_names[0]])
    trt_result = tokenizer.batch_decode(trt_transcribed, skip_special_tokens=True)[0]
    report_time_taken("TRT time: ")

    return original_result, scripted_result, onnx_result, trt_result


if __name__ == '__main__':
    iface = gr.Interface(fn=detect_audio,
                         inputs=[
                             gr.Audio(label="Input Audio", type="filepath"),
                             gr.Dropdown(choices=["en", "fr"], label="Transcription Language"),
                         ],
                         outputs=[
                             gr.Textbox(label="Original Output"),
                             gr.Textbox(label="Scripted Output"),
                             gr.Textbox(label="ONNX Output"),
                             gr.Textbox(label="TRT Output"),
                         ],
                         examples=[
                             ["AnotherTestRecording.wav", "en"]
                         ],
                         title="Audio App",
                         description="A simple app for Whisper Transcription"
                         )
    iface.launch(server_name="0.0.0.0")
