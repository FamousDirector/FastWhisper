FROM nvcr.io/nvidia/pytorch:22.09-py3

WORKDIR "/opt/project"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y ffmpeg

# install python depencies
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

RUN python3 -m pip install git+https://github.com/openai/whisper.git

# pre download models
RUN python3 -c "import whisper; [whisper.load_model(n) for n in whisper._MODELS]"

# setup main.py to run app
COPY code/ .
CMD ["python3", "-u", "main.py"]