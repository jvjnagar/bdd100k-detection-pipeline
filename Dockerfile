FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir opencv-python-headless

ENV MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp

COPY src/ src/
COPY scripts/ scripts/
COPY train.py ./
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && mkdir -p data output third_party weights

# One image, every stage:
#   analyze (default) | dashboard | train | evaluate | test | bash
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["analyze"]
