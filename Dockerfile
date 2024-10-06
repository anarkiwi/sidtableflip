FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
COPY requirements.txt test-requirements.txt /root
RUN apt-get -yq update && apt-get install -yq python3-pip && pip install -r /root/requirements.txt -r /root/test-requirements.txt
COPY sidtableflip /sidtableflip
COPY tests /tests
WORKDIR /
RUN PYTHONPATH=. pytest /tests
RUN perl -pi -e "s/class TransformerDecoder\(nn.Module\):/from monkey import Monkey\nclass TransformerDecoder\(Monkey\):/" $(python3 -c "import torchtune.modules.transformer;print(torchtune.modules.transformer.__file__)")

# docker build -f Dockerfile . -t anarkiwi/sidtableflip
# docker run --gpus=all -v /scratch:/scratch -ti anarkiwi/sidtableflip /sidtableflip/train.py --batch-size 64
