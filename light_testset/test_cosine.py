import os
import time
import scipy
from scipy.misc import imsave
import sys
import cv2
import numpy
import numpy as np
import tensorflow as tf
from model import pix2pix

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)

sys.path.append("/home/donny/caffe_center_loss/python")
import caffe

# model_file = "/home/donny/caffe_copy/caffe_dx/original_TSdata/model_cache/original_TSdata/face_train_test_iter_300000.caffemodel"
# deploy_file = "/home/donny/caffe_copy/caffe_dx/original_TSdata/face_deploy.prototxt"
model_file = "face_train_test_iter_174312.caffemodel"
deploy_file = "face_deploy.prototxt"
feat_layer = "fc5"

test_dir = "dlib_aligned"

# caffe.set_device(2)
caffe.set_mode_gpu()

net = caffe.Net(deploy_file, model_file, caffe.TEST)
model = pix2pix(sess)
model.init_test("../checkpoint_bk")



# image = cv2.imread("00.png")
# model.test_one_image(image)

test_donny = test_dir + "/donny_a"
donny_list = os.listdir(test_donny)
donny_feat_list = list()
for filename in donny_list:
    face_img = cv2.imread(test_donny + "/" + filename)

    face_img = model.test_one_image(face_img)
    face_img = (face_img + 1)* 127.5
    face_img = cv2.resize(face_img, (100, 100))
    cv2.imwrite("face2.jpg", face_img)

    height, width = face_img.shape[:2]
    net.blobs['data'].reshape(1, 3, height, width)
    blob_data = net.blobs['data'].data
    face_img = face_img.astype(numpy.float32, copy = False)
    face_img = (face_img - 127.5)/128;
    blob_data[0,0,:,:] = face_img[:,:,0]
    blob_data[0,1,:,:] = face_img[:,:,1]
    blob_data[0,2,:,:] = face_img[:,:,2]
    out = net.forward()
    donny_feat_list.append(numpy.copy(out[feat_layer]))

# print donny_feat_list

test_wang = test_dir + "/wang_a"
wang_list = os.listdir(test_wang)
wang_feat_list = list()
for filename in wang_list:
    face_img = cv2.imread(test_wang + "/" + filename)
    face_img = model.test_one_image(face_img)
    face_img = (face_img + 1)* 127.5
    face_img = cv2.resize(face_img, (100, 100))

    height, width = face_img.shape[:2]
    net.blobs['data'].reshape(1, 3, height, width)
    blob_data = net.blobs['data'].data
    face_img = face_img.astype(numpy.float32, copy = False)
    face_img = (face_img - 127.5)/128;
    blob_data[0,0,:,:] = face_img[:,:,0]
    blob_data[0,1,:,:] = face_img[:,:,1]
    blob_data[0,2,:,:] = face_img[:,:,2]
    out = net.forward()
    wang_feat_list.append(numpy.copy(out[feat_layer]))


cosine_donny = 0.0
cosine_wang = 0.0

for feat1 in donny_feat_list:
    for feat2 in donny_feat_list:
        # print feat1
        # print feat2
        # raw_input("enter")
        print scipy.spatial.distance.cosine(feat1, feat2)
        cosine_donny += 1-scipy.spatial.distance.cosine(feat1, feat2)

cosine_donny = cosine_donny / (len(donny_feat_list)**2 * 1.0)

for feat1 in wang_feat_list:
    for feat2 in wang_feat_list:
        cosine_wang += 1-scipy.spatial.distance.cosine(feat1, feat2)

cosine_wang = cosine_wang / (len(wang_feat_list)**2 * 1.0)

print "donny cosine sim: %f" % cosine_donny
print "wang cosine sim: %f" % cosine_wang
