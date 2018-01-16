# Utility functions for using pretrained models and creating the fcn models
import os
import csv
import math
import random
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import warnings

def tensorflow_check():
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found.')
    else:
      print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vggmodel(session,saved_path):
    # Saved path is the folder containing variables/ and saved_movel.pb
    
    vgg_tag='vgg16'
    vgg_input_name='image_input:0'
    keep_prob_name='keep_prob:0'
    vgg_layer3_out_name='layer3_out:0'
    vgg_layer4_out_name='layer4_out:0'
    vgg_layer7_out_name='layer7_out:0'
       
    model=tf.saved_model.loader.load(session,[vgg_tag],saved_path)
    graph=tf.get_default_graph()
    input_image=graph.get_tensor_by_name(vgg_input_name)
    keep_prob=graph.get_tensor_by_name(keep_prob_name)
    layer3=graph.get_tensor_by_name(vgg_layer3_out_name)
    layer4=graph.get_tensor_by_name(vgg_layer4_out_name)
    layer7=graph.get_tensor_by_name(vgg_layer7_out_name)
    
    return input_image,keep_prob,layer3,layer4,layer7

def convolution_1x1(x,n_classes,x_name):
    return tf.layers.conv2d(inputs=x,filters=n_classes,kernel_size=(1,1),strides=(1,1),name=x_name)

def convolution_upsampling(x,n_classes,k_size,st_size,x_name):
    return tf.layers.conv2d_transpose(inputs=x,filters=n_classes,
                            kernel_size=(k_size,k_size),
                            strides=(st_size,st_size),
                            padding='same',name=x_name)

def model(vgg_layer3,vgg_layer4,vgg_layer7,n_classes):
    # Encoder 1*1 convolutions
    layer_3=convolution_1x1(vgg_layer3,n_classes,'Convolution_layer3')
    layer_4=convolution_1x1(vgg_layer4,n_classes,'Convolution_layer4')
    layer_7=convolution_1x1(vgg_layer7,n_classes,'Convolution_layer7')
    
    deconvolution_1=convolution_upsampling(layer_7,n_classes,k_size=4,st_size=2,x_name='deconvolution_layer_1')
    skip_connection_1=tf.add(deconvolution_1,layer_4,name='deconvolution_layer_2')
    deconvolution_2=convolution_upsampling(skip_connection_1,n_classes,k_size=4,st_size=2,x_name='deconvolution_layer_3')
    skip_connection_2=tf.add(deconvolution_2,layer_3,name='deconvolution_layer_4')
    output=convolution_upsampling(skip_connection_2,n_classes,k_size=16,st_size=8,x_name='deconvolution_output')
    
    return output

def optimizer(layer,ground_truth,learning_rate,n_classes):
    logits=tf.reshape(layer,(-1,n_classes))
    labels=tf.reshape(ground_truth,(-1,n_classes))
    # Cross entropy loss
    cross_entropy_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    optimizer=tf.train.AdamOptimizer(learning_rate)
    train_operation=optimizer.minimize(cross_entropy_loss)
                     
    return logits,cross_entropy_loss,train_operation


def training(session,epochs,batch_size,get_batches_fn,train_operation,
             cross_entropy_loss,input_image,truth_label,keep_prob,learning_rate,keep_prob_value,learning_rate_value):                  
                      
    for epoch in range(epochs):
        loss=None
        losses=[]
        for image,label in get_batches_fn(batch_size):
            feed_dict={input_image:image,
                    truth_label:label,
                    keep_prob:keep_prob_value,
                    learning_rate:learning_rate_value}
            loss,_=session.run([cross_entropy_loss,train_operation],feed_dict=feed_dict)
            losses.append(loss)
        epoch_loss=sum(losses)/len(losses)
        Loss_summary.append(epoch_loss)

        print('Epoch {} of {}: Training Loss {}'.format(epoch+1,epochs,loss) )