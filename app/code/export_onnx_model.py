import os
import whisper
import torch
from pathlib import Path

from whisper.tokenizer import get_tokenizer

import scriptable_model

model_name = os.getenv("MODEL_NAME", "")
if model_name == "":
    model_name = "tiny"
    print(f"Did not get a model_name from env variable. Defaulting to {model_name}...")
model = whisper.load_model(model_name)

whisper_path = Path(f"~/.cache/whisper/{model_name}.pt").expanduser()
with open(whisper_path, "rb") as f:
    checkpoint = torch.load(f)

modded = scriptable_model.Whisper(model.dims).eval()
modded.load_state_dict(checkpoint["model_state_dict"])

# cleanup
del model

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

starting_tokens = tokenizer.encode("<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
                                   return_tensors="pt")

audio = whisper.load_audio("AnotherTestRecording.wav")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).unsqueeze(0)
print(f"Exporting {model_name} model to ONNX.")
torch.onnx.export(modded, (starting_tokens, mel, suppress_blanks, suppress_nonspeech),
                  f"{model_name}.onnx",
                  opset_version=16, )  # higher opset can sometimes speed up export
