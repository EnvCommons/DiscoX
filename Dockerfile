FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    python3 python3-pip curl git \
    && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN uv venv --python 3.11

# Install dependencies
COPY requirements.txt /app/
RUN uv pip install -r /app/requirements.txt

# Download DiscoX dataset (1.98 MB)
RUN curl -L -o discox.parquet \
    https://huggingface.co/datasets/ByteDance-Seed/DiscoX/resolve/main/data/train-00000-of-00001.parquet

# Copy code
COPY discox.py server.py /app/

EXPOSE 8080
CMD ["uv", "run", "python", "/app/server.py"]
