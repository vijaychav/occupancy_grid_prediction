"""
import os
def clear_all():
    '''Clears all the variables from the workspace of the spyder application.'''
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue
        del globals()[var]
clear_all()
"""

import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
#from scipy import ndimage
import tensorflow as tf
#from scipy import signal

historyRange = 20
modelName = "./trainedModel.ckpt"
loadModel = 1

#np.set_printoptions(threshold=np.inf)


with h5py.File('/media/vijay/Windows/InvincE/GoogleDrive/Full time app/Dispatch/data.h5','r') as hf:
    print('List of arrays in this file: \n', hf.keys())
    occupancy = hf.get('occupancy')
    groundtruth = hf.get('ground_truth')
    print occupancy.shape

    #occupancyData = occupancy[:,:,inputIndex-historyRange:inputIndex]
    train_test_split = 0.8
    num_images = occupancy.shape[2]
    
    def quantize(x,case):
        if case==0: #occupancy data
            x[np.where(x==0)] = -1 
            x[np.where(x==127)] = 0
            x[np.where(x==255)] = 1
        elif case ==1: #ground truth data
            x[np.where(x==0)] = 0
            x[np.where(x==255)] = 1
        return x
    
    def getDataTrain():
        step =0
        while step+historyRange < int(train_test_split*num_images):
            X_tr = np.array(occupancy[:,:,step:step+historyRange]).transpose([2,1,0]).reshape([historyRange,-1]).astype(int)
            #print X_tr.shape
            Y_tr = np.array(groundtruth[:,:,step+historyRange]).flatten().astype(int)
            #print Y_tr.shape
            X_tr = quantize(X_tr,0)
            Y_tr = quantize(Y_tr,1)
            yield X_tr, Y_tr
            step+=1
            
    def getDataTest(index=None):
        if index:
            step = index-historyRange
        else:
            step = int(train_test_split*num_images)
        while step+historyRange< num_images:
            X_te = np.array(occupancy[:,:,step:step+historyRange]).transpose([2,1,0]).reshape([historyRange,-1]).astype(int)
            #print X_te.shape
            Y_te = np.array(groundtruth[:,:,step+historyRange]).flatten().astype(int)
            #print Y_te.shape
            X_te = quantize(X_te,0)
            Y_te = quantize(Y_te,1)
            yield X_te, Y_te
            step+=1
            
            
    
    print "Number of images:", num_images
    trainData = getDataTrain()
    testData = getDataTest()

    #print type(next(data))
    #print next(trainData)[0].shape

    num_train = int(train_test_split*num_images)

    learning_rate = 0.03
    training_iters = 900*num_train
    num_batch = 1
    max_grad_norm = 10
    lossratio = 5.0
    num_layers = 1
    num_hidden = 128
    
    
    with tf.device('/gpu:0'):
        x = tf.placeholder(tf.float32,[None,20,10000])
        y = tf.placeholder(tf.float32,[None,10000])
        weight = tf.Variable(tf.random_normal([num_hidden, int(y.get_shape()[1])]))
        bias = tf.Variable(tf.random_normal([int(y.get_shape()[1])]))
        cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
        istate = cell.zero_state(1,tf.float32)
        val, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, initial_state= istate)
    
        val = tf.transpose(val, [1,0,2])
        last = tf.gather(val, int(val.get_shape()[0])-1)
        print "shape of last:", last.get_shape()
        #bias = tf.Variable(tf.constant(0.1, shape=[y.get_shape()[1]]))
        logit = tf.nn.xw_plus_b(last, weight,bias)
        #logit_sig = tf.sign(tf.sub(tf.sigmoid(logit),0.5))
        logit_sig = tf.round(tf.sigmoid(logit))
        
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logit,y))
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # Adam Optimizer
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        correct = tf.equal(y,logit_sig)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        #print "y:", y
        #print "logit_sig:", logit_sig
        rmserr = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(y,logit_sig))))
       
    
    
    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    
    if not loadModel:  #training model from stratch
        saver = tf.train.Saver()
    
        sess.run(init_op)
        step=0
        #while step+historyRange< num_images:
        while step+historyRange< num_images:
            train, trainTarget = next(trainData)
            train = [train]; trainTarget = [trainTarget]
            startTime = time.clock()
            sess.run(train_op, feed_dict={x:np.array(train), y:np.array(trainTarget)})
            #output = sess.run(logit_sig, feed_dict={x:np.array(train), y:np.array(trainTarget)})
            acc = sess.run(accuracy, feed_dict={x:np.array(train), y:np.array(trainTarget)})
            rmserror = sess.run(rmserr, feed_dict={x:np.array(train), y:np.array(trainTarget)})
            los = sess.run(loss, feed_dict={x:np.array(train), y:np.array(trainTarget)})
            #print "Shape of output:",output.shape
            print "Train Accuracy: %.3f, loss: %.3f, train rmserr: %.3f, step: %d, time taken: %.5f sec" %(acc,los,rmserror,step, time.clock() - startTime)

            #print "loss:", loss
            step+=1
            if step%10==0:
                steptest = 0
                testacc = []
                while steptest<1:
                    test, testTarget = next(testData)
                    test= [test]; testTarget = [testTarget]
                    testaccperstep = sess.run(accuracy, feed_dict={x:np.array(test), y:np.array(testTarget)})
                    rmserror = sess.run(rmserr, feed_dict={x:np.array(test), y:np.array(testTarget)})
                    testacc.append(testaccperstep)
                    steptest+=1
                print "Test Accuracy, %.3f, Test Rms err: %.3f" %(np.mean(np.array(testacc)),rmserror)
        
        save_path = saver.save(sess, modelName)
        print "Model saved at ",save_path 
    
    else:
        assert len(sys.argv)>1, "Enter input index"
        inputIndex = int(sys.argv[1])
        assert inputIndex > int(train_test_split*num_images) and inputIndex<occupancy.shape[-1], "Input index should be between 800000 and the max number of frames"

        sess.run(init_op)
        testData = getDataTest(inputIndex)
        saver = tf.train.Saver()
        saver.restore(sess, "./trainedModels/"+modelName)
        print "Using saved model......"
        test, testTarget = next(testData)
        #print test
        #plt.figure(3)
        #plt.imshow(test[-1].reshape([100,100]))
        #plt.colorbar(orientation='vertical')
        #plt.title("Occupancy")
        #print "testTarget:", testTarget
        test= [test]; testTarget = [testTarget]
        testTargetShow = 255*(testTarget[0].reshape([100,100]))        
        testacc = sess.run(accuracy, feed_dict={x:np.array(test), y:np.array(testTarget)})
        output = sess.run(logit_sig, feed_dict={x:np.array(test), y:np.array(testTarget)})
        rmserror = sess.run(rmserr, feed_dict={x:np.array(test), y:np.array(testTarget)})
        print "Test Accuracy:", testacc
        print "Test Rms Error:", rmserror
        
        outputShow = 255*(output[0].reshape([100,100]))
        #print "Error distance = ", np.linalg.norm(testTargetShow.flatten()-outputShow.flatten())
        plt.figure(1)
        plt.imshow(testTargetShow)
        plt.colorbar(orientation='vertical')
        plt.title("Ground Truth")
        plt.figure(2)
        plt.imshow(outputShow)
        plt.colorbar(orientation='vertical')
        plt.title("Prediction")
        plt.show()
    
    
    

                

    
    
    
    sess.close()
    hf.close()
    
    
    
    
 
 
    
    
    
    
    