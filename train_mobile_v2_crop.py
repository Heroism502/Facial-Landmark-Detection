# -*- coding: utf-8 -*-
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('./solver_Head.prototxt')

solver.solve()

