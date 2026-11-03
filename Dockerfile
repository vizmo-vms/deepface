# syntax=docker/dockerfile:1.7
# base image
# The slim variant has the same Python build without the full image's compilers, ImageMagick
# and version control tools. Every package below installs from a prebuilt wheel, apart from
# fire, which is pure Python.
FROM python:3.8.12-slim
LABEL org.opencontainers.image.source https://github.com/serengil/deepface

# -----------------------------------
# create required folder
RUN mkdir -p /app && chown -R 1001:0 /app
RUN mkdir /app/deepface

# -----------------------------------
# switch to application directory
WORKDIR /app

# -----------------------------------
# install dependencies
# requirements_docker.txt pins every package, so --no-deps keeps pip from pulling in the CUDA
# build of TensorFlow or the GUI build of OpenCV for packages that ask for them. The pip wheels
# bring their own FFmpeg (OpenCV) and HDF5 (h5py), so the image needs no system packages.
COPY ./requirements_docker.txt /app/requirements_docker.txt
RUN pip install --no-cache-dir --no-deps -r /app/requirements_docker.txt

# -----------------------------------
# Copy required files from repo into image
COPY ./deepface /app/deepface
# setup.py reads these files
COPY ./requirements.txt /app/requirements.txt
COPY ./package_info.json /app/
COPY ./setup.py /app/
COPY ./README.md /app/
COPY ./entrypoint.sh /app/deepface/api/src/entrypoint.sh

# install deepface from source code (always up-to-date)
RUN pip install --no-cache-dir --no-deps -e .

# -----------------------------------
# Pre-download weights, pinned by checksum

# Facenet512
ADD --checksum=sha256:3f76b5117a9ca574d536af8199e6720089eb4ad3dc7e93534496d88265de864f \
    https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5 \
    /root/.deepface/weights/facenet512_weights.h5

# Yunet
ADD --checksum=sha256:8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4 \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
    /root/.deepface/weights/face_detection_yunet_2023mar.onnx

# -----------------------------------
# environment variables
ENV PYTHONUNBUFFERED=1

# -----------------------------------
# run the app (re-configure port if necessary)
WORKDIR /app/deepface/api/src
EXPOSE 5000
# CMD ["gunicorn", "--workers=1", "--timeout=3600", "--bind=0.0.0.0:5000", "app:create_app()"]
ENTRYPOINT [ "sh", "entrypoint.sh" ]
