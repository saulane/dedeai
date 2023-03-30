FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# install utilities
RUN apt-get update && apt-get install --no-install-recommends -y curl ffmpeg libsm6 libxext6  -y
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/opt/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/miniconda \
    && rm ~/miniconda.sh \
    && sed -i "$ a PATH=/opt/miniconda/bin:\$PATH" /etc/environment

# Installing python dependencies
RUN python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 --version && \
    pip3 --version

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace
COPY . /workspace

RUN pip3 install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

RUN chown -R 42420:42420 /workspace
ENV HOME=/workspace