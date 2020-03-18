FROM nvidia/cuda

WORKDIR /nlp
COPY ./ ./
RUN python3 -m pip install -r requirements.txt

CMD ["python3", "/nlp/src/run.py"]
