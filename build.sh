docker build --build-arg PIP_OPTS="--index-url http://192.168.5.1:5001/index/ --trusted-host 192.168.5.1" -f Dockerfile . -t anarkiwi/sidtableflip
