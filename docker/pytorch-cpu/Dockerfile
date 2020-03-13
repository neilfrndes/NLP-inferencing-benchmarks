FROM pytorch/pytorch

WORKDIR /nlp
RUN pip install transformers

COPY src src/ 
COPY data data/

ENTRYPOINT ["python", "/nlp/src/run.py"]
