# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
import sys
sys.path.insert(0, '/home//Danbin_work/caffe/python')
import caffe
DEBUG = True

def calculate_NME(true_pts, pred_pts, target_parts=None, normalization='centers', showResults=False, verbose=False):

    assert true_pts.shape[0] == 68
    assert true_pts.shape[1] == 2
    assert true_pts.shape == pred_pts.shape

    if normalization == 'centers':
        normDist = np.linalg.norm(np.mean(true_pts[[43, 44, 46, 47], :], axis=0) - np.mean(true_pts[[37, 38, 40, 41], :], axis=0))
    elif normalization == 'corners':
        normDist = np.linalg.norm(true_pts[36] - true_pts[45])
    elif normalization == 'diagonal':
        height, width = np.max(true_pts, axis=0) - np.min(true_pts, axis=0)
        normDist = np.sqrt(width ** 2 + height ** 2)

    if target_parts is None:
        # print('Calculate all points error.')
        error = np.mean(np.sqrt(np.sum((true_pts - pred_pts)**2, axis=1))) / normDist
        error = error * 100
        return error
    else:
        # print('Calculate points error of idx: ', target_parts)
        target_true_pts = true_pts[target_parts, :]
        target_pred_pts = pred_pts[target_parts, :]
        error = np.mean(np.sqrt(np.sum((target_true_pts - target_pred_pts)**2, axis=1))) / normDist
        error = error * 100
        return error

def load_model(proto_file, model_file, gpu_id=0):
    if gpu_id >= 0:
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(proto_file, model_file, caffe.TEST)
    transformer = caffe.io.Transformer({'X': net.blobs['X'].data.shape})
    transformer.set_transpose('X', (2, 0, 1))
    return net, transformer


def predict_landmarks(net, transformer, output_layer, cropped_face, cropped_offset, norm_size=64):

    resize_face = cv2.resize(cropped_face, (norm_size, norm_size))
    input_data = resize_face - 127.5
    input_data = input_data * 0.0078125
    net.blobs['X'].data[...] = transformer.preprocess('X', input_data)
    # print("data shape:", net.blobs['X'].data.shape)
    out = net.forward()
    output = net.blobs[output_layer].data
    # print(" output shape:", output.shape)
    output = output.flatten()
    # print(" output shape:", output.shape)
    assert output.shape[0] == 136

    pred_landmarks = np.array([[output[i * 2], output[i * 2 + 1]] for i in range(68)])
    crop_offset_x = cropped_offset[0]
    crop_offset_y = cropped_offset[1]
    # print 'offset_x ', crop_offset_x
    # print 'offset_y ', crop_offset_y

    ''' 将预测的结果转换回到图像坐标'''
    map_to_src_landmarks = np.zeros(pred_landmarks.shape)
    for i in range(68):
        map_to_src_landmarks[i][0] = (pred_landmarks[i][0] + 0.5) * cropped_face.shape[1] + crop_offset_x
        map_to_src_landmarks[i][1] = (pred_landmarks[i][1] + 0.5) * cropped_face.shape[0] + crop_offset_y
    return map_to_src_landmarks

def read_landmarktxt(filepath):
    landmark_dict = {}
    bbox_dict = {}

    with open(filepath, 'r') as f:
        lines = [p.strip() for p in f.readlines()]

    for k, row in enumerate(lines):
        parts = row.split(' ')
        # print 'row ', k
        # print 'parts len: ', len(parts)
        # print parts
        filename = parts[0]
        #print filename, parts[1],parts[2]
        #print parts
        bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
        pts = []
        for i in range(68):
            pts.append([float(parts[i*2+5]), float(parts[i*2+6])])
        assert len(pts) == 68
        landmark_dict[filename] = pts
        bbox_dict[filename] = bbox

    return landmark_dict, bbox_dict

def crop_face(img, det_box, gt_landmark):
    # 扩充det_box 上下左右各10％ s = 0.1
    # 扩充det_box 上下左右各15％ s = 0.15
    # 得到标注关键点最大最小值
    all_x = []
    all_y = []
    for i in range(68):
        all_x.append(gt_landmark[i][0])
        all_y.append(gt_landmark[i][1])
    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)

    #修改扩充框区域
    s = 0.1
    width = det_box[2]
    height = det_box[3]

    new_top = det_box[1]
    new_bottom = det_box[1] + det_box[3] + 0.5 * s * height
    new_left = det_box[0] - 0.5 * s * width
    new_right = det_box[0] + det_box[2] + 0.5 * s * width
    new_top = new_top + 0.8 * (height - width)

    new_top = np.maximum(0, new_top)
    new_left = np.maximum(0, new_left)
    new_bottom = np.minimum(img.shape[0], new_bottom)
    new_right = np.minimum(img.shape[1], new_right)

    # 原人脸框扩充0.1后对应的人脸框
    # 加判断--如果有关键点在人脸扩充框之外 -- 设置/强行设置最小外接矩形
    if new_left > x_min or new_top > y_min or new_right < x_max or new_bottom < y_max:
        # 有关键点在人脸扩充框之外
        if new_left > x_min:
            new_left = x_min - 0.07 * width  # 根据最小关键点坐ｘ标再往外扩充0.05
            if new_left < 0:  # 边界判断
                new_left = 0
        if new_top > y_min:
            new_top = y_min - 0.08 * height  # 防止top点正好在线上
            if new_top < 0:
                new_top = 0
        if new_right < x_max:
            new_right = x_max + 0.07 * width
            if new_right > img.shape[1]:
                new_right = img.shape[1]
        if new_bottom < y_max:
            new_bottom = y_max + 0.05 * height
            if new_bottom > img.shape[0]:
                new_bottom = img.shape[0]

    ''' 裁剪之后的人脸 '''
    crop_face = img[int(new_top):int(new_bottom), int(new_left):int(new_right), :]
    ''' 裁剪之后人脸的关键点坐标位置，以裁剪图像左上角为原点'''
    crop_landmark = [[p[0] - new_left, p[1] - new_top] for p in gt_landmark]
    ''' 裁剪之后图像的SSD检测框位置，以裁剪图像左上角为原点'''
    crop_bbox = [det_box[0]-new_left, det_box[1]-new_top, det_box[2], det_box[3]]
    ''' 裁剪图像相对于原图的偏移量'''
    crop_offset = [new_left, new_top]
    return crop_face, crop_landmark, crop_bbox, crop_offset



def batch_eval():
    img_dir = '/test/image_anguan'
    gt_file = '/test/landmark_anguan.txt'
    model_file = '/test/model/face_68_iter_12000.caffemodel'
    proto_file = '/test/mobilenentv2_back_caijian.prototxt'
    save_dir = './test_crop/'
    #fopen = open('/home//data/landmark/network/test_code/error_list.txt', 'w')  # 替换为你的路径
    output_layer_name = 'fc7_flat'
    norm_input_size = 96
    contour_idx_list = range(0, 17)
    eyebrow_idx_list = range(17, 27)
    eye_idx_list = range(36, 48)
    nose_idx_list = range(27, 36)
    mouth_idx_list = range(48, 68)
    net, transformer = load_model(proto_file, model_file, gpu_id=0)
    pts_dict, bbox_dict = read_landmarktxt(gt_file)

    img_list = os.listdir(img_dir)
    img_list = [p for p in img_list if p.find('.jpg') > 0 or p.find('.jpeg') > 0]

    error_all_list = []
    error_eye_list = []
    error_nose_list = []
    error_mouth_list = []
    error_brow_list = []
    error_contour_list = []
    outside_landmarks_cnt = 0
    for name in img_list:
        img_path = os.path.join(img_dir, name)
        # img = cv2.imread(img_path)
        img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8), -1)
        true_bbox = bbox_dict[name]
        true_pts = np.array(pts_dict[name])

        ''' check if main internal points are all within the given bbox.'''
        left, top0 = true_pts[36]   # 左眼外眼角点
        right, top1 = true_pts[45]  # 右眼外眼角点
        mid, bot = true_pts[57]     # 下嘴唇中点
        top = max([top0, top1])
        left = min([left, mid])
        right = max([right, mid])

        # Does all landmarks fit into this box?
        if top < true_bbox[1] or bot > true_bbox[1] + true_bbox[3] or left < true_bbox[0] or right > true_bbox[0] + true_bbox[2]:
            #print 'landmarkd outside of bbox: ', img_path
            outside_landmarks_cnt += 1
            #print outside_landmarks_cnt
            continue
        # centers　corners　diagonal
        cropped_face, cropped_pts, cropped_box, cropped_offset = crop_face(img, true_bbox, true_pts)
        predict_pts = predict_landmarks(net, transformer, output_layer_name, cropped_face, cropped_offset, norm_input_size)
        error_all = calculate_NME(true_pts, predict_pts, target_parts=None, normalization='centers')
        error_eye = calculate_NME(true_pts, predict_pts, target_parts=eye_idx_list, normalization='centers')
        error_brow = calculate_NME(true_pts, predict_pts, target_parts=eyebrow_idx_list, normalization='centers')
        error_nose = calculate_NME(true_pts, predict_pts, target_parts=nose_idx_list, normalization='centers')
        error_mouth = calculate_NME(true_pts, predict_pts, target_parts=mouth_idx_list, normalization='centers')
        error_contour = calculate_NME(true_pts, predict_pts, target_parts=contour_idx_list, normalization='centers')
        error_all_list.append(error_all)
        error_eye_list.append(error_eye)
        error_brow_list.append(error_brow)
        error_nose_list.append(error_nose)
        error_mouth_list.append(error_mouth)
        error_contour_list.append(error_contour)

        if DEBUG:
            draw_img = cropped_face.copy()
            for pt in cropped_pts:
                cv2.circle(draw_img, (int(pt[0]), int(pt[1])), 2, (255, 255, 0))
            cv2.rectangle(draw_img, (int(cropped_box[0]), int(cropped_box[1])),
                          (int(cropped_box[2] + cropped_box[0]), int(cropped_box[3] + cropped_box[1])),
                          (0, 255, 255), 2)
            # cv2.imwrite(os.path.join(save_dir, '%s_true_crop.jpg' % (os.path.splitext(name)[0])), draw_img)
            cv2.imencode('.jpg', draw_img)[1].tofile(
                os.path.join(save_dir, '%s_true_crop.jpg' % (os.path.splitext(name)[0])))

            draw_img2 = img.copy()
            for i in range(68):
                cv2.circle(draw_img2, (int(predict_pts[i][0]), int(predict_pts[i][1])), 1,
                           (0, 0, 255), 2)

            # cv2.imwrite(os.path.join(save_dir, '%s_predict_src.jpg' % (os.path.splitext(name)[0])), draw_img2)
            cv2.imencode('.jpg', draw_img2)[1].tofile(
                os.path.join(save_dir, '%s_predict_src.jpg' % (os.path.splitext(name)[0])))

    print 'all NME: ', np.mean(error_all_list)
    print 'eye NME: ', np.mean(error_eye_list)
    print 'nose NME: ', np.mean(error_nose_list)
    print 'brow NME: ', np.mean(error_brow_list)
    print 'mouth NME: ', np.mean(error_mouth_list)
    print 'contour NME: ', np.mean(error_contour_list)


if __name__ == '__main__':
    DEBUG = False
    batch_eval()








