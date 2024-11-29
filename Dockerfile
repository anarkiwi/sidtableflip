FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
COPY requirements.txt test-requirements.txt /root
ARG PIP_OPTS=""
ENV PIP_OPTS=$PIP_OPTS
RUN apt-get -yq update && apt-get install -yq python3-pip
RUN pip install $PIP_OPTS -r /root/requirements.txt -r /root/test-requirements.txt
WORKDIR /
COPY sidtableflip /sidtableflip
COPY tests /tests
RUN black --check sidtableflip tests && PYTHONPATH=. pytest -svvv /tests && PYTHONPATH=. pylint -E sidtableflip && PYTHONPATH=. pytype sidtableflip

# docker build -f Dockerfile . -t anarkiwi/sidtableflip
# docker run --gpus=all -v /scratch:/scratch -ti anarkiwi/sidtableflip /sidtableflip/train.py --batch-size 64
