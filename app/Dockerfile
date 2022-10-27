FROM nvcr.io/nvidia/pytorch:22.09-py3

WORKDIR "/opt/project"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y ffmpeg

# install python depencies
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

RUN python3 -m pip install git+https://github.com/openai/whisper.git

# setup main.py to run app
COPY code/ .
RUN python3 predownload_models.py
CMD ["python3","main.py"]