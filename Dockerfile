FROM condaforge/miniforge3:latest

ARG VAMPIRE_VERSION
ARG CONDA_MIRROR

ENV LANG=en_US.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

LABEL Author="Zikun-Yang"
LABEL Version="v0.4.1"

USER root

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
    if [ "${CONDA_MIRROR}" = "cn" ]; then \
        echo "==> Configuring Tsinghua (TUNA) mirrors for Chinese network" && \
        TUNA="https://mirrors.tuna.tsinghua.edu.cn/anaconda" && \
        conda config --remove-key channels || true && \
        conda config --add channels "${TUNA}/cloud/bioconda/" && \
        conda config --add channels "${TUNA}/cloud/conda-forge/" && \
        conda config --add channels "${TUNA}/pkgs/main/" && \
        conda config --set show_channel_urls yes && \
        pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
        pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn; \
    else \
        echo "==> Configuring official conda channels (international)" && \
        conda config --add channels conda-forge && \
        conda config --add channels bioconda && \
        conda config --set show_channel_urls yes; \
    fi && \
    \
    conda init bash && \
    conda clean -afy && \
    \
    if [ -n "$VAMPIRE_VERSION" ]; then \
        pip install --root-user-action=ignore "vampire-tr==${VAMPIRE_VERSION}"; \
    else \
        pip install --root-user-action=ignore vampire-tr; \
    fi && \
    \
    conda clean -afy && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:${PATH}"

CMD ["vampire", "--help"]