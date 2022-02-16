# -*- coding: utf-8 -*-

import numpy as np
_A = np.array  # A shortcut to creating arrays in command line
import os
import cv2
import sys
from pickle import load, dump
from zipfile import ZipFile
from urllib import urlretrieve
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 显卡
reload(sys)
sys.setdefaultencoding('utf-8')
from DataRow_diffcrop import DataRow, ErrorAcum, Predictor, createDataRowsFromCSV, getValidWithBBox, writeHD5, createBoxRowsFromCSV

TRAIN_DATA_PATH = []



Root = '/home/Facelandmark/part4_300W_VW'
TRAIN_DATA_PATH.append([os.path.join(Root,'part_15/image'),# 图片路径-公司互联网
                        os.path.join(Root,'part_15/68.txt')])
TRAIN_DATA_PATH.append([os.path.join(Root,'part_16/image'),# 图片路径-公司互联网
                        os.path.join(Root,'part_16/68.txt')])

MEAN_TRAIN_SET = [127.5, 127.5, 127.5]
SCALE_TRAIN_SET = 0.0078125
NORM_IMAGE_SIZE = 96

STEPS = [#'downloadAFW', #no downland permission
         #'createAFW_TestSet',
         #'testAFW_TestSet',
         #'downloadAFLW',
         # 'testSetHD5',
         # 'testSetPickle',
         'trainSetHD5'
         #'calcTrainSetMean',
         #'createAFLW_TestSet',
         #'testAFLW_TestSet',
         #'testErrorMini']
         ]


if 'trainSetHD5' in STEPS:

    dataset_num = len(TRAIN_DATA_PATH)
    all_dataRowsTrain_CSV = []
    all_dataBboxTrain_CSV = []
    for DATA_PATH, CSV_TRAIN in TRAIN_DATA_PATH:
        dataRowsTrain_CSV = createDataRowsFromCSV(CSV_TRAIN, DataRow.DataRowFromNameBoxInterlaved, DATA_PATH)
        dataBboxTrain_CSV = createBoxRowsFromCSV(CSV_TRAIN, DATA_PATH)
        assert len(dataBboxTrain_CSV) == len(dataBboxTrain_CSV)
        print 'data path: ', DATA_PATH
        print "Finished reading %d rows from training data." % len(dataRowsTrain_CSV)
        all_dataRowsTrain_CSV.extend(dataRowsTrain_CSV)
        all_dataBboxTrain_CSV.extend(dataBboxTrain_CSV)

    dataRowsTrainValid, R = getValidWithBBox(all_dataRowsTrain_CSV, all_dataBboxTrain_CSV)
    print "Original train:", len(all_dataRowsTrain_CSV), "Valid Rows:", len(dataRowsTrainValid), " noFacesAtAll", R.noFacesAtAll, " outside:", R.outsideLandmarks, " couldNotMatch:", R.couldNotMatch
    dataRowsTrain_CSV=[]  # remove from memory
    dataBboxTrain_CSV=[]

    writer_root = '/home/project/landmark/landmark/data_deal/'
    writeHD5(dataRowsTrainValid, writer_root+'train_300vw_9.hd5',
             writer_root + 'train_300vw_9.txt', MEAN_TRAIN_SET,
             SCALE_TRAIN_SET, IMAGE_SIZE=NORM_IMAGE_SIZE, mirror=True)
    print "Finished writing train to caffeData/train.txt"

