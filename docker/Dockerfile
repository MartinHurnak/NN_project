FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

WORKDIR /NN_project

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . /NN_project

EXPOSE 8888

EXPOSE 6006

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"]
