# syntax=docker/dockerfile:1

ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
ARG MICROMAMBA_IMAGE=mambaorg/micromamba:2.0.5-ubuntu22.04

FROM ${MICROMAMBA_IMAGE} AS environment-builder

USER root

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY environment.ts3.yml /tmp/environment.ts3.yml

RUN micromamba create --yes --prefix /opt/conda/envs/ts3 --file /tmp/environment.ts3.yml \
    && micromamba clean --all --yes

COPY requ/third_parties.lock /tmp/third_parties.lock

RUN set -eux; \
    mkdir -p /tmp/third_parties; \
    while read -r name url commit extra; do \
        case "${name}" in ''|'#'*) continue ;; esac; \
        test -n "${name}"; \
        test -n "${url}"; \
        test -n "${commit}"; \
        test -z "${extra}"; \
        git clone "${url}" "/tmp/third_parties/${name}"; \
        git -C "/tmp/third_parties/${name}" checkout --detach "${commit}"; \
        micromamba run --prefix /opt/conda/envs/ts3 python -m pip install --no-deps "/tmp/third_parties/${name}"; \
    done < /tmp/third_parties.lock; \
    micromamba run --prefix /opt/conda/envs/ts3 python -m pip check; \
    rm -rf /tmp/third_parties /tmp/third_parties.lock

FROM ${CUDA_IMAGE} AS runtime

ENV CONDA_PREFIX=/opt/conda/envs/ts3
ENV PATH=${CONDA_PREFIX}/bin:${PATH}
ENV PYTHONPATH=/workspace
ENV HF_HOME=/workspace/.cache/huggingface

RUN apt-get update \
    && apt-get install --yes --no-install-recommends bash ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=environment-builder /opt/conda/envs/ts3 /opt/conda/envs/ts3

WORKDIR /workspace
COPY . /workspace

CMD ["bash"]
