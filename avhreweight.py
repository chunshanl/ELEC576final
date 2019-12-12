from scipy import misc
import numpy as np
import tensorflow as tf
import math
import random
import matplotlib.pyplot as plt
import matplotlib as mp
import time
import os
import skimage as sk
from skimage import transform
from skimage import util
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib.pyplot import imread, imshow, subplots, show

# --------------------------------------------------
# setup
np.random.seed(1)
def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    #initial = tf.truncated_normal(shape, stddev=0.1)
    initial=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)#HE initialization
    W = tf.Variable(initial(shape))
    return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial)
    # IMPLEMENT YOUR BIAS_VARIABLE HERE

    return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')
    return h_max

#-------data augmentation--------
def random_rotation(image_array):
    # pick a random degree of rotation between 20% on the left and 20% on the right
    random_degree = random.uniform(-20, 20)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array):
    return image_array[:, ::-1]

def rotate(x):
    return tf.image.rot90(x,tf.random_uniform(shape=[],minval=0,maxval=4,dtype=tf.int32))

def flip(x):
    x=tf.image.random_flip_left_right(x)
    x=tf.image.random_flip_up_down(x)
    return x

def color(x):
    x=tf.image.random_hue(x,0.08)
    x=tf.image.random_saturation(x,0.6,1.6)
    x=tf.image.random_brightness(x,0.05)
    x=tf.image.random_contrast(x,0.7,1.3)
    return x


start_time = time.time()
ntrain = 1000 # per class
ntest = 100 # per class
nclass = 10 # number of classes
imsize =28
nchannels =1
batchsize =100
BN=True#indicate batch normalization
epsilon=1e-3#small number used in BN

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'C:/Users/linhm/Desktop/2018-19Fall/COMP576/hw/HW2/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = 'C:/Users/linhm/Desktop/2018-19Fall/COMP576/hw/HW2/CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

sess = tf.InteractiveSession()

tf_data = tf.placeholder(tf.float32, [None, 28,28,1], name='x') #tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_labels = tf.placeholder(tf.float32, [None, 10],  name='y') #tf variable for labels

# --------------------------------------------------
# model
#create your model
W_conv1 =weight_variable([5, 5, 1, 32])
b_conv1 =bias_variable([32])
z_conv1=conv2d(tf_data, W_conv1) + b_conv1
if BN:
    batch_mean, batch_var = tf.nn.moments(z_conv1,[0])
    batch_scale = tf.Variable(tf.ones([32]))
    batch_beta = tf.Variable(tf.zeros([32]))
    z_conv1 = tf.nn.batch_normalization(z_conv1,batch_mean,batch_var,batch_beta,batch_scale,epsilon)
h_conv1 =tf.nn.relu(z_conv1)
h_pool1 =max_pool_2x2(h_conv1)

W_conv2 =weight_variable([5, 5, 32, 64])
b_conv2 =bias_variable([64])
z_conv2=conv2d(h_pool1, W_conv2) + b_conv2
if BN:
    batch_mean, batch_var = tf.nn.moments(z_conv2,[0])
    batch_scale = tf.Variable(tf.ones([64]))
    batch_beta = tf.Variable(tf.zeros([64]))
    z_conv2 = tf.nn.batch_normalization(z_conv2,batch_mean,batch_var,batch_beta,batch_scale,epsilon)
h_conv2 =tf.nn.relu(z_conv2)
h_pool2 =max_pool_2x2(h_conv2)


W_fc1 =weight_variable([7*7*64,512])#([7*7*64,1024])
b_fc1 =bias_variable([512])#([1024])
h_pool2_flat =tf.reshape(h_pool2, [-1, 7*7*64])
z_fc1=tf.matmul(h_pool2_flat, W_fc1) + b_fc1
if BN:
    batch_mean, batch_var = tf.nn.moments(z_fc1,[0])
    batch_scale = tf.Variable(tf.ones([512]))#(tf.ones([1024]))
    batch_beta = tf.Variable(tf.zeros([512]))#(tf.zeros([1024]))
    z_fc1 = tf.nn.batch_normalization(z_fc1,batch_mean,batch_var,batch_beta,batch_scale,epsilon)
h_fc1 =tf.nn.relu(z_fc1)
W_fc2 =weight_variable([512,64])#([1024,84])
b_fc2 =bias_variable([64])
z_fc2=tf.matmul(h_fc1, W_fc2) + b_fc2
if BN:
    batch_mean, batch_var = tf.nn.moments(z_fc2,[0])
    batch_scale = tf.Variable(tf.ones([64]))
    batch_beta = tf.Variable(tf.zeros([64]))
    z_fc2 = tf.nn.batch_normalization(z_fc2,batch_mean,batch_var,batch_beta,batch_scale,epsilon)
h_fc2 =tf.nn.relu(z_fc2)
W_sm=weight_variable([64,10])
b_sm=bias_variable([10])
y_conv =tf.nn.softmax(tf.matmul(h_fc2,W_sm)+b_sm, name='yscore')
# AVH---vectorized version-----
wy=tf.tensordot(tf_labels,W_sm,axes=((1,),(1,)))
dotprod=tf.reduce_sum(tf.multiply(h_fc2,wy),axis=1)
xnorm=tf.norm(h_fc2,axis=1)
wynorm=tf.norm(wy,axis=1)
num=tf.acos(tf.divide(tf.divide(dotprod,xnorm),wynorm))
dotprod=tf.tensordot(h_fc2,W_sm,axes=((1,),(0,)))
wnorm=tf.broadcast_to(tf.norm(W_sm,axis=0),shape=[tf.shape(h_fc2)[0],nclass])
xnorm=tf.reshape(xnorm,shape=[-1,1])
xnormb=tf.broadcast_to(xnorm,shape=[tf.shape(h_fc2)[0],nclass])
denom=tf.acos(tf.divide(tf.divide(dotprod,wnorm),xnormb))
denom=tf.reduce_sum(denom,axis=1)
avh=tf.divide(num,denom)
# loss
#set up the loss, optimization, evaluation, and accuracy
#use weighted loss
#cross_entropy =tf.reduce_mean(-tf.reduce_sum(tf_labels * tf.log(y_conv), reduction_indices=[1]))
cross_entropy=-tf.reduce_sum(tf_labels * tf.log(y_conv), reduction_indices=[1])
#include avh in loss
#choice 1: add avh to loss
cross_entropy=cross_entropy+tf.reduce_sum(avh*avh)
#cross_entropy=cross_entropy*tf.exp(avh)
cross_entropy=tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#optimizer=tf.train.RMSPropOptimizer(1e-4).minimize(cross_entropy)
correct_prediction =tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_labels, 1))
accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# summary for avh--------------------------------------------------
avh_mean=tf.reduce_mean(tf.cast(avh,tf.float32))
avh_raw=tf.cast(avh,tf.float32)
tf.summary.scalar(avh_mean.op.name,avh_mean)
#summary for train
tf.summary.scalar(cross_entropy.op.name,cross_entropy)
tf.summary.scalar(accuracy.op.name,accuracy)
filter_summary = tf.summary.image('Filter_1',tf.reshape(W_conv1,(32,5,5,1)),max_outputs=32)

tf.summary.histogram('h_conv1',h_conv1)
tf.summary.histogram('h_conv2',h_conv2)
tf.summary.scalar('mean-h_conv1',tf.reduce_mean(h_conv1))
tf.summary.scalar('mean-h_conv2',tf.reduce_mean(h_conv2))
tf.summary.scalar('max-hconv1',tf.reduce_max(h_conv1))
tf.summary.scalar('max-hconv2',tf.reduce_max(h_conv2))
tf.summary.scalar('min-hconv1',tf.reduce_min(h_conv1))
tf.summary.scalar('min-hconv2',tf.reduce_min(h_conv2))
summary_op=tf.summary.merge_all()

result_dir = 'C:/Users/linhm/Desktop/2018-19Fall/COMP576/proj/results/'
# Create a saver for writing training checkpoints.
saver=tf.train.Saver()

summary_writer = tf.summary.FileWriter(result_dir, sess.graph)


#---------------------------------------------------
'''
# optimization
sess.run(tf.initialize_all_variables())
batch_xs = np.zeros((batchsize*nclass,imsize,imsize,nchannels))#setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros((batchsize*nclass,nclass))#setup as [batchsize, the how many classes]

trainacc_list=[]
testacc_list=[]
trainloss_list=[]
avh_avg_list=[]
nepochs=25
for epoch in range(nepochs):
    perm = np.arange(ntrain*nclass)
    np.random.shuffle(perm)
    for i in range(int(ntrain/batchsize)):
        for j in range(nclass*batchsize):
            batch_xs[j,:,:,:] = Train[perm[i*batchsize+j],:,:,:]
            batch_ys[j,:] = LTrain[perm[i*batchsize+j],:]

        optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys})  # dropout only during training
    trainacc_list.append(accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys}))
    testacc_list.append(accuracy.eval(feed_dict={tf_data:Test, tf_labels:LTest}))
    trainloss_list.append(cross_entropy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys}))
    #avh list
    avh_avg_list.append(avh_mean.eval(feed_dict={tf_data:batch_xs, tf_labels:batch_ys}))
    #save checkpoints
    checkpoint_file = os.path.join(result_dir, 'checkpoint')
    saver.save(sess, checkpoint_file, global_step=epoch)

    print("epoch=",epoch,", training accuracy %g" %accuracy.eval(feed_dict={tf_data:batch_xs, tf_labels:batch_ys}))
    print("mean angle %g" %avh_mean.eval(feed_dict={tf_data:batch_xs, tf_labels:batch_ys}))
#--------------------------------------------------

#plot
plt.plot([i for i in range(len(trainacc_list))],trainacc_list)
plt.title('training accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


plt.plot([i for i in range(len(testacc_list))],testacc_list)
plt.title('test accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

plt.plot([i for i in range(len(trainloss_list))],trainloss_list)
plt.title('training loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


plt.plot([i for i in range(len(avh_avg_list))],avh_avg_list)
plt.title('avh average')
plt.xlabel('epoch')
plt.ylabel('avh')
plt.show()



# test
print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest}))
#stat of act on test images
#summary_str=sess.run(summary_op,feed_dict={tf_data:Test,tf_labels:LTest})
#summary_writer.add_summary(summary_str)
#summary_writer.flush()

stop_time = time.time()
print('The training takes %f second to finish'%(stop_time - start_time))

sess.close()


'''

ckpt = tf.train.get_checkpoint_state(result_dir)
saver.restore(sess, ckpt.model_checkpoint_path)
#saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
print(ckpt)
print("before aug, test acc %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest}))

#Randomly augment the file from one of the 3 transformations
IMG_aug = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
LIMG_aug = np.zeros((ntrain*nclass,nclass))
iaug=-1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'C:/Users/linhm/Desktop/2018-19Fall/COMP576/hw/HW2/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        rnd=np.random.uniform(size=1)
        if rnd<1/3:
            im_aug=random_rotation(im)
        elif rnd>2/3:
            im_aug = random_noise(im)
        else:
            im_aug = horizontal_flip(im)
        im_aug = im_aug.astype(float) / 255
        iaug += 1
        IMG_aug[iaug,:,:,0] = im_aug
        LIMG_aug[iaug,iclass] = 1 # 1-hot lable

avh_aug=sess.run(avh_raw, feed_dict={tf_data:IMG_aug,tf_labels:LIMG_aug})

#seperate the aug picture into 2 group: high avg and low avg

print("the 0.02th quantile of avh:",np.quantile(avh_aug,0.02))
print("the 0.22th quantile of avh:",np.quantile(avh_aug,0.22))
#print("the 0.99th quantile of avh:",np.quantile(avh_aug,0.99))
#plot an easy augmentation and a hard augmentation

'''plot example images of hard and easy augmentation
hard_index=int(np.argwhere(avh_aug>np.quantile(avh_aug,0.5))[0])
print('the avh score is ', avh_aug[hard_index])
print('the img index is ',hard_index,'the label is ',np.argmax(LTrain[hard_index,]))
plt.imshow(IMG_aug[hard_index,:,:,0])
plt.title('hard example (preprocessed)')
plt.show()
plt.imshow(255*IMG_aug[hard_index,:,:,0])
plt.title('hard example (raw)')
plt.show()

plt.imshow(Train[hard_index,:,:,0])
plt.title('original img of hard example (preprocessed)')
plt.show()
plt.imshow(255*Train[hard_index,:,:,0])
plt.title('original img of hard example (raw)')
plt.show()

easy_index=int(np.argwhere(avh_aug<np.quantile(avh_aug,0.05))[0])
print('the avh score is ', avh_aug[easy_index])
print('the img index is ',easy_index,'the label is ',np.argmax(LTrain[easy_index,]))
plt.imshow(IMG_aug[easy_index,:,:,0])
plt.title('esay example (prepocessed)')
plt.show()
plt.imshow(255*IMG_aug[easy_index,:,:,0])
plt.title('esay example (raw)')
plt.show()
plt.imshow(Train[easy_index,:,:,0])
plt.title('original img of easy example (preprocessed)')
plt.show()
plt.imshow(255*Train[easy_index,:,:,0])
plt.title('original img of easy example (raw)')
plt.show()
'''


#retrain
#rebuild the training data
choice='cm'
if choice=='ch':
    aug_havh = np.asarray([IMG_aug[i] for i in range(IMG_aug.shape[0]) if avh_aug[i] > np.quantile(avh_aug, 0.67)])
    Lable_aug_havh = np.asarray(
        [LIMG_aug[i] for i in range(LIMG_aug.shape[0]) if avh_aug[i] > np.quantile(avh_aug, 0.67)])
    newTrain = np.vstack((Train, aug_havh))
    newLTrain = np.vstack((LTrain, Lable_aug_havh))
    print('number of samples=', aug_havh.shape[0], 'accuracy for high avh: ',
          sess.run(accuracy, feed_dict={tf_data: aug_havh, tf_labels: Lable_aug_havh}))
elif choice=='cl':
    aug_lavh = np.asarray([IMG_aug[i] for i in range(IMG_aug.shape[0]) if avh_aug[i] <= np.quantile(avh_aug, 0.35)])
    Lable_aug_lavh = np.asarray(
        [LIMG_aug[i] for i in range(LIMG_aug.shape[0]) if avh_aug[i] <= np.quantile(avh_aug, 0.35)])
    newTrain = np.vstack((Train, aug_lavh))
    newLTrain = np.vstack((LTrain, Lable_aug_lavh))
    print('number of samples=', aug_lavh.shape[0], 'accuracy for low avh: ',
          sess.run(accuracy, feed_dict={tf_data: aug_lavh, tf_labels: Lable_aug_lavh}))

elif choice=='cm':
    aug_mavh = np.asarray([IMG_aug[i] for i in range(IMG_aug.shape[0]) if
                           (avh_aug[i] > np.quantile(avh_aug, 0.02) and avh_aug[i] < np.quantile(avh_aug, 0.22))])
    Lable_aug_mavh = np.asarray([LIMG_aug[i] for i in range(LIMG_aug.shape[0]) if
                                 (avh_aug[i] > np.quantile(avh_aug, 0.02) and avh_aug[i] < np.quantile(avh_aug, 0.22))])
    newTrain=np.vstack((Train,aug_mavh))
    newLTrain=np.vstack((LTrain,Lable_aug_mavh))
    print('number of samples=', aug_mavh.shape[0], 'accuracy for medium avh: ',
          sess.run(accuracy, feed_dict={tf_data: aug_mavh, tf_labels: Lable_aug_mavh}))

elif choice=='cb':
    newTrain = np.vstack((Train, IMG_aug))
    newLTrain = np.vstack((LTrain, LIMG_aug))
    print('number of samples=', IMG_aug.shape[0], 'accuracy for all avh: ',
          sess.run(accuracy, feed_dict={tf_data: IMG_aug, tf_labels: LIMG_aug}))

elif choice=='random':
    perm = np.arange(IMG_aug.shape[0])
    np.random.shuffle(perm)
    #randomly select 20% of augmentated images
    rand_id=perm[1:int(0.2*IMG_aug.shape[0])]
    aug_rand = np.asarray([IMG_aug[i] for i in rand_id])
    Lable_aug_rand = np.asarray([LIMG_aug[i] for i in rand_id])
    newTrain=np.vstack((Train,aug_rand))
    newLTrain = np.vstack((LTrain, Lable_aug_rand))


else:
    pass



nepochs=25
batchsize=batchsize*nclass
ntrain=newTrain.shape[0]
result_aug_dir='C:/Users/linhm/Desktop/2018-19Fall/COMP576/proj/results/augonly/'

batch_xs = np.zeros((batchsize,imsize,imsize,nchannels))#setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros((batchsize,nclass))#setup as [batchsize, the how many classes]
trainacc_list=[]
testacc_list=[]
trainloss_list=[]
testloss_list=[]
avh_avg_list=[]
avh_avg_test_list=[]

for epoch in range(nepochs):
    perm = np.arange(ntrain)
    np.random.shuffle(perm)
    for i in range(int(ntrain/batchsize)):
        for j in range(batchsize):
            batch_xs[j,:,:,:] = newTrain[perm[i*batchsize+j],:,:,:]
            batch_ys[j,:] = newLTrain[perm[i*batchsize+j],:]
        optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys})  # dropout only during training

    #calculate train/lost accuracy
    avhmtp,acctp, losstp = sess.run([avh_mean,accuracy, cross_entropy], feed_dict={tf_data: batch_xs, tf_labels: batch_ys})
#    [testacc_tp, avh_aug_tp] = sess.run([avh_raw,accuracy], feed_dict={tf_data: IMG_aug, tf_labels: LIMG_aug})
    trainacc_list.append(acctp)
    trainloss_list.append(losstp)
    avh_avg_list.append(avhmtp)
    avhmtp, acctp,losstp = sess.run([avh_mean, accuracy, cross_entropy], feed_dict={tf_data: Test, tf_labels: LTest})
    testacc_list.append(acctp)
    avh_avg_test_list.append(avhmtp)
    testloss_list.append(losstp)

    print("epoch=",epoch,", training accuracy %g" %accuracy.eval(feed_dict={tf_data:batch_xs, tf_labels:batch_ys}))
    #print("mean angle %g" %avh_mean.eval(feed_dict={tf_data:batch_xs, tf_labels:batch_ys}))
    if epoch%5==0:
        print("epoch=",epoch,", test accuracy %g" %accuracy.eval(feed_dict={tf_data:Test, tf_labels:LTest}))
        #checkpoint_file = os.path.join(result_aug_dir, 'checkpoint')
        #saver.save(sess, checkpoint_file, global_step=epoch)
        
stop_time = time.time()
print('The training takes %f second to finish'%(stop_time - start_time))
print('aug mode is ', choice, ", test accuracy after aug %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest}))
#plot
plt.plot([i for i in range(len(trainacc_list))],trainacc_list)
plt.title('training accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

plt.plot([i for i in range(len(testacc_list))],testacc_list)
plt.title('test accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

plt.plot([i for i in range(len(trainloss_list))],trainloss_list)
plt.title('training loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.plot([i for i in range(len(testloss_list))],testloss_list)
plt.title('test loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.plot([i for i in range(len(avh_avg_list))],avh_avg_list)
plt.title('training avh average')
plt.xlabel('epoch')
plt.ylabel('avh')
plt.show()

plt.plot([i for i in range(len(avh_avg_test_list))],avh_avg_test_list)
plt.title('test avh average')
plt.xlabel('epoch')
plt.ylabel('avh')
plt.show()

print('train acc: ', trainacc_list)
print('test acc: ',testacc_list)
print('train loss: ', trainloss_list)
print('test loss: ',testloss_list)
print('train avh avg: ',avh_avg_list)
print('test avh avg: ',avh_avg_test_list)

