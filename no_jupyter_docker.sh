sudo docker run --gpus all -p 8888:8888 -p 6006:6006 -v ${pwd}:/NN_project  -it --entrypoint /bin/bash mataur/nsiete_fiit