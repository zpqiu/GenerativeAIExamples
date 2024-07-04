FROM downloads.unstructured.io/unstructured-io/unstructured:latest

# Install common dependencies for all examples
# RUN --mount=type=bind,source=requirements.txt,target=/opt/requirements.txt \
    # /usr/bin/pip3 install --no-cache-dir -r /opt/requirements.txt

# Copy required common modules for all examples
COPY . /opt/pdfparser

ENV PYTHONPATH=/app:/home/notebook-user/.local/lib/python3.11/site-packages

# RUN chown -R notebook-user:notebook-user /pdfs
USER root

WORKDIR /opt
# Expose the port
ENTRYPOINT ["uvicorn", "pdfparser.server:app"]

# build the image
# docker build -t pdfparser .