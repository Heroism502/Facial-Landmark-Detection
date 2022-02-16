# -*- coding: utf-8 -*-
"""

"""
import sys
import math
sys.path.append('/home/soft/caffe-master/python')

import caffe
import numpy as np

def poly_weaken(base_scale, iter, max_iter):
    power = 0.9
    temp = 1 - iter/max_iter
    output = base_scale * math.pow(temp, power)
    return output


class NormlizedMSE(caffe.Layer):
    """
    Compute the normlized MSE Loss 
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match

        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        #        self.diff = np.zeros(bottom[0].count,dtype='float32')
        # loss output is scalar
        top[0].reshape(1)

    #  print('NormlizedMSE bottom shape ',bottom[0].shape[0],bottom[0].shape[1])

    def forward(self, bottom, top):
        #Mean square error of landmark regression normalized w.r.t.
        base_scale = 0.2
        max_iter = 640000
        first = True
        loss = 0
        #该参数的设置标准: = 需要迭代的最大次数 * iter_size
        #例如:希望迭代80000次后base_scale衰减完,output为0.则该参数设置为80000*iter_size (iter_size在prototxt中为6）
        y_true = bottom[1].data  # 实际标注关键点值
        y_pred = bottom[0].data  # 预测关键点值
        print y_true.shape,y_pred.shape
        '''计算外眼角之间的距离作为归一化因子'''
        delX = y_true[:, 72] - y_true[:, 90]  # del X size 16
        delY = y_true[:, 73] - y_true[:, 91]  # del y size 16
        #print('delx shape: ', delX.shape)
        self.interOc = (1e-6 + (delX * delX + delY * delY) ** 0.5).T  # Euclidain distance
        print('interoc shape: ', self.interOc.shape)

        '''计算迭代次数加权值'''
        if first:
            num_iter = 0
            first = False
        else:
            num_iter = num_iter + 1
        print 'num_iter:', num_iter

        if num_iter > max_iter:
            output = 0
        else:
            output = poly_weaken(base_scale, num_iter, max_iter)
        print 'output:', output

        '''计算预测值和实际值均方差'''
        # Cannot multiply shape (16,10) by (16,1) so we transpose to (10,16) and (1,16)
        for i in range(0, 135):
            if i >= 0 and i <= 61:  #脸部轮廓区域
                diff = (y_pred[:, i] - y_true[:, i])      # Transpose so we can divide a (16,10) array by (16,1)
                print 'part1 diff: ', diff.shape
            elif i >= 62 and i <= 120:  #人脸五官区域
                diff = ((y_pred[:, i] - y_true[:, i]) * (1 + base_scale - output))
                print 'part2 diff: ', diff.shape
            else:                       #嘴唇内部区域
                diff = ((y_pred[:, i] - y_true[:, i]) * (1 - base_scale + output))
                print 'part3 diff: ', diff.shape
            i += 1

        diff = diff.T
        print 'diff.shape:　', diff.shape

        self.diff[...] = (diff / self.interOc).T  # We transpose back to (16,10)
        top[0].data[...] = np.sum(self.diff ** 2) / bottom[0].num / 2.  # Loss is scalar

        # loss = np.zeros((bottom[0].num), dtype=np.float32)
        # for k in xrange(0,68):
        #     delX = y_pred[:,k*2]-y_true[:,k*2] # del X size 16
        #     delY = y_pred[:,k*2+1]-y_true[:,k*2+1] # del y size 16
        #     loss = loss + (delX*delX + delY*delY)**0.5
        # loss = loss.T
        # top[0].data[...] = np.sum(loss/self.interOc) / bottom[0].num/68

    def backward(self, top, propagate_down, bottom):

        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num


#            print(bottom[i].diff[...])

##################################

class EuclideanLossLayer(caffe.Layer):
    # ORIGINAL EXAMPLE
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff ** 2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num


