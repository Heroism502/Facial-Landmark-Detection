

export PYTHONPATH=/home/caffe/python:$PYTHONPATH

/home/caffe-master/build/tools/caffe train --solver=/home/train/solver_Head.prototxt #-weights=/home/train/mobilenet_v2_dwise.caffemodel --gpu=1 2>&1 |tee logResNetHead 



