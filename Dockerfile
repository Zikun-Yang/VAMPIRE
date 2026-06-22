FROM ubuntu:22.04

ARG VAMPIRE_VERSION
ARG CONDA_MIRROR=global
ARG TARGETARCH
ARG MINIFORGE_VERSION=25.3.1-0

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:${PATH}"

LABEL org.opencontainers.image.title="VAMPIRE"
LABEL org.opencontainers.image.authors="Zikun Yang"
LABEL org.opencontainers.image.description="VAMPIRE: Unified framework for de novo tandem repeat annotation and analysis"

RUN set -eux; \
    \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        bzip2 \
        locales \
        libcurl4 \
        libfontconfig1 \
        libfreetype6 \
        libpng16-16 \
        libtiff5 \
        libjpeg-turbo8; \
    \
    locale-gen en_US.UTF-8; \
    update-locale LANG=en_US.UTF-8; \
    \
    case "${TARGETARCH}" in \
        amd64) ARCH=x86_64 ;; \
        arm64) ARCH=aarch64 ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" && exit 1 ;; \
    esac; \
    \
    wget -q \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-${ARCH}.sh" \
      -O /tmp/miniforge.sh; \
    \
    bash /tmp/miniforge.sh -b -p /opt/conda; \
    rm -f /tmp/miniforge.sh; \
    \
    if [ "${CONDA_MIRROR}" = "cn" ]; then \
        echo "Using Tsinghua mirrors"; \
        TUNA=https://mirrors.tuna.tsinghua.edu.cn/anaconda; \
        \
        conda config --remove-key channels || true; \
        conda config --add channels "${TUNA}/cloud/bioconda"; \
        conda config --add channels "${TUNA}/cloud/conda-forge"; \
        conda config --add channels "${TUNA}/pkgs/main"; \
        conda config --set show_channel_urls yes; \
        \
        mkdir -p /root/.pip; \
        printf "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\ntrusted-host = pypi.tuna.tsinghua.edu.cn\n" > /root/.pip/pip.conf; \
    else \
        conda config --add channels conda-forge; \
        conda config --add channels bioconda; \
        conda config --set show_channel_urls yes; \
    fi; \
    \
    VERSION="${VAMPIRE_VERSION#v}"; \
    \
    if [ -n "${VERSION}" ]; then \
        /opt/conda/bin/pip install \
            --no-cache-dir \
            --root-user-action=ignore \
            "vampire-tr==${VERSION}"; \
    else \
        /opt/conda/bin/pip install \
            --no-cache-dir \
            --root-user-action=ignore \
            vampire-tr; \
    fi; \
    \
    conda clean -afy; \
    apt-get clean; \
    rm -rf \
        /var/lib/apt/lists/* \
        /root/.cache \
        /tmp/*

CMD ["vampire", "--help"]