FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update  -y 
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install libgl1-mesa-glx libglib2.0-0 libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk libharfbuzz-dev libfribidi-dev libxcb1-dev git -y
RUN python -m pip install --upgrade pip

COPY requirements.txt requirements.txt
COPY src/app.py app.py 

RUN python -m pip install -r requirements.txt
RUN python -m pip install Pillow-SIMD

CMD ["python","app.py"]
