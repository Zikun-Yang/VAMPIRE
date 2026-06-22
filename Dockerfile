FROM ubuntu:22.04

ARG VAMPIRE_VERSION
ARG CONDA_MIRROR

ENV LANG=en_US.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

LABEL Author="Zikun-Yang"
LABEL Version="v0.4.1"

RUN set -e && \
    apt-get update && apt-get install -y \
        ca-certificates \
        wget \
        curl \
        bzip2 \
        libcurl4 \
        libfontconfig1 \
        libfreetype6 \
        libpng16-16 \
        libtiff5 \
        libjpeg-turbo8 \
        locales \
        && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 && \
    \
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    export PATH="/opt/conda/bin:$PATH" && \
    \
    if [ "${CONDA_MIRROR}" = "cn" ]; then \
        echo "==> Configuring Tsinghua (TUNA) mirrors for Chinese network" && \
        TUNA="https://mirrors.tuna.tsinghua.edu.cn/anaconda" && \
        /opt/conda/bin/conda config --remove-key channels || true && \
        /opt/conda/bin/conda config --add channels "${TUNA}/cloud/bioconda/" && \
        /opt/conda/bin/conda config --add channels "${TUNA}/cloud/conda-forge/" && \
        /opt/conda/bin/conda config --add channels "${TUNA}/pkgs/main/" && \
        /opt/conda/bin/conda config --set show_channel_urls yes && \
        /opt/conda/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
        /opt/conda/bin/pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn; \
    else \
        echo "==> Configuring official conda channels (international)" && \
        /opt/conda/bin/conda config --add channels conda-forge && \
        /opt/conda/bin/conda config --add channels bioconda && \
        /opt/conda/bin/conda config --set show_channel_urls yes; \
    fi && \
    \
    /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda clean -afy && \
    \
    /opt/conda/bin/conda install -y mafft && \
    \
    if [ -n "$VAMPIRE_VERSION" ]; then \
        /opt/conda/bin/pip install --root-user-action=ignore "vampire-tr==${VAMPIRE_VERSION}"; \
    else \
        /opt/conda/bin/pip install --root-user-action=ignore vampire-tr; \
    fi && \
    \
    /opt/conda/bin/conda clean -afy && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:${PATH}"

CMD ["vampire", "--help"]