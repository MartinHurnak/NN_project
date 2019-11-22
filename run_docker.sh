sudo docker pull mataur/nsiete_fiit
sudo docker run -p 8888:8888 -p 6006:6006 -v $(pwd):/NN_project -it mataur/nsiete_fiit
