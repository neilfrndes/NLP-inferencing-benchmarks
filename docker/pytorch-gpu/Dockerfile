FROM pytorch/manylinux-cuda102

WORKDIR /nlp
RUN pip install transformers

COPY src src/ 
COPY data data/


CMD ["python", "/nlp/src/run.py"]
