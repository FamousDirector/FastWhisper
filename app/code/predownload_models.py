import whisper

for n in whisper._MODELS:
    print("Downloading {}...".format(n))
    whisper.load_model(n)
