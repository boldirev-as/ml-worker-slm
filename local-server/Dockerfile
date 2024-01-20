FROM python:3.10

COPY requirements.txt requirements.txt

# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt
RUN apt install libcairo2

# FROM nvidia/cuda:12.3.1-base-ubuntu20.04

COPY .. .
CMD ["uvicorn", "printer_server:app", "--host 0.0.0.0", "--port 8090"]
# CMD ["nvidia-smi"]
