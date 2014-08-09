IIS_2014
========

Image Recognition with Food Journal Project

This project employs deep learning to extract salient but perhaps non-intuitive features from collected multimodal data sets and use such features to the end of classification tasks. The bulk of the code is written in Python and C++.

I. SYSTEM SETUP

For rbm and other deep learning algorithms we have been using the deepnet implementations whose documentation follows:

(1) DEPENDENCIES
  - Numpy
  - Scipy
  - CUDA Toolkit and SDK.
    Install the toolkit and SDK.
    Set an environment variable CUDA_BIN to the path to the /bin directory of
    the cuda installation and CUDA_LIB to the path to the /lib64 (or /lib)
    directory. Also add them to PATH and LD_LIBARAY_PATH.
   
    For example, add the following lines to your ~/.bashrc file
      export CUDA_BIN=/usr/local/cuda-5.0/bin
      export CUDA_LIB=/usr/local/cuda-5.0/lib64
      export PATH=${CUDA_BIN}:$PATH
      export LD_LIBRARY_PATH=${CUDA_LIB}:$LD_LIBRARY_PATH

  - Protocol Buffers.
    Available from http://code.google.com/p/protobuf/
    Make sure that the PATH environment variable includes the directory that
    contains the protocol buffer compiler - protoc.
	  For example,
      export PATH=/usr/local/bin:$PATH

(2) COMPILING CUDAMAT AND CUDAMAT_CONV
  DeepNet uses Vlad Mnih's cudamat library and Alex Krizhevsky's
  cuda-convnet library. Some additional kernels have been
  added. To compile the library -
  - run make in the cudamat dir.

(3) SET ENVIRONMENT VARIABLES
  - Add the path to cudamat to LD_LIBRARY_PATH. For example if
    DeepNet is located in the home dir,
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/deepnet/cudamat
  - Add the path to DeepNet to PYTHONPATH. For example, if DeepNet is located in the
    home dir,
      export PYTHONPATH=$PYTHONPATH:$HOME/deepnet

(4) RUN AN EXAMPLE
  - Download and extract the MNIST dataset from http://www.cs.toronto.edu/~nitish/deepnet/mnist.tar.gz
    This dataset consists of labelled images of handwritten digits as numpy files.
  - cd to the deepnet/deepnet/examples dir
  - run
    $ python setup_examples.py <path to mnist dataset> <output path>
    This will modify the example models and trainers to use the specified paths.
  - There are examples of different deep learning models. Go to any one and
    execute runall.sh. For example, cd to deepnet/deepnet/examples/rbm and execute
    $ ./runall.sh
    This should start training an RBM model.
  
DeepNet has been tested on Ubuntu 12.04 using CUDA 4.2 and 5.0 on a variety of NVIDIA
GPUs (GTX-280, GTX-580, GTX-690, M2090, K-20x).

For initial feature extraction we also use a visual bag of words model implemented in C++ which additionally requires OpenCV. 

II. DATA SOURCES
There are a few data sets used in training and testing of our rbm implementation. 

-MNIST: Used for preliminary testing, stores as folders of jpegs, and corresponding labels. Mnist is located in /home/huijuan/Deepnet/data and is stored in the form of numpy arrays of pixel data and labels. Its method of access can be viewed in datahandler.py in the Deepnet code base. 

-ImageCLEF:Contains medical images and query images with a ground truth as to whether images from the database are relevant to the query. ImageCLEF is located in /home/huijuan/ImageCLEF2013.



III. HIGH LEVEL DESIGN 

At a high level, the idea of this program is to receive input images in the form of raw pixel data, apply preprocessing to extract initial salient features, and then use the initial extracted features as input to an rbm to generate more sophisticated features. These extracted features will then be used as input to a linear classifier in order to improve the classification of input images. 


IV. IMPLEMENTATION/LOW LEVEL DESIGN

The implementation of this program can be broken into a few main tasks: initial feature extraction, deep learning, and classification. Implementation is given in more detail as follows. 

Initial Feature Extraction:
Written in C++ and employing OpenCV libraries, these classes receive image data in the form of raw pixels and output feature vectors created from the visual bag of words model. These files are stored in /home/huijuan/Deepnet/deepnet/FeatureExtraction.
 
CreateBow.cpp: This file extracts SIFT features from a given number of images and uses the features to build a BOW representation. The clustered features are stored in a .yml file to be processed later.
 
prepKNN.cpp: This file uses the clustered features that were stored in the dictionary to create a BOW descriptor for each image in the data set, defined as a histogram over the features encoded in the dictionary.
 
image_knn.cpp: After having created the training vectors and labels for knn as well as the dictionary for the BOW, we read these in. We also supply a directory containing query images. The queries are fed into knn and the top 100 hits are retrieved. 

Deep Learning:
The current implementation is a modification of the deepnet code base which contains implementations of many deep architectures. The code is run by first setting up the data paths if necessary and then training a model by executing one of the run scripts within the architecture file (runall.sh). runallCustom allows for use of the customized run script which includes some parameter updates. 
The implementation essentially instantiates a trainer, setting data paths and then calling for the training of the model in the superclass neuralnet.py. This works by collecting "batches" of data from the specified set and forward propagating data through the net while also sampling from the resultant distribution. The specifics of the net are handled by the specific deep learning implementation class, such as dbm.py in the case of training an rbm. Data is read in for training from a prespecified path using datahandler.py. The data handler reads in "batchsize" chunks of data from the path and converts it into a cuda matrix form usable by the rest of the program. After each batch is propagated through the net, the net parameters are tested by verification on a held-out data set of labeled images to classify. Currently, when training an rbm using ImageCLEF data, both the training and validation data are coming from the same set. 




Classification:
Currently there is no classifier to receive the output features of the rbm. When one is added it may be tricky to keep it a binary classifier as there are clearly many classes to which an image may belong. One way to handle this might be to create a binary linear classifier that simply outputs whether or not an image is relevant to a particular query. 

V. TESTING

Testing will vary depending on what type of task the output features from the rbm are used for. The most important thing is to verify that the extracted features perform better classification than merely using SIFT or BOW descriptors. This can be done by using the output vectors from the rbm as the input descriptors to compare during querying. Currently little has been actually implemented in the way of testing given that there is no classifier to receive output vectors. 

Testing the rbm itself involves examining training error on labeled sets that is already built into deepnet. This however requires labeled data which is not available in ImageCLEF. 
