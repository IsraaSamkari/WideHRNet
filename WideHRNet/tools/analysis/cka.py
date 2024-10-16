# Copyright (c) OpenMMLab. All rights reserved.
# Source: https://github.com/phrasenmaeher/cka/blob/main/do_nns_learn_the_same%3F.ipynb
import argparse
from functools import partial
import torch
from mmpose.apis.inference import init_pose_model

####
import tensorflow as tf
import numpy as np
import tqdm
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
matplotlib.use( 'tkagg' )
####

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--device', default='cpu', help='Device used for inference') # cuda:0 or cpu
    args = parser.parse_args()
    return args


def get_strategy(xla=0, fp16=0, no_cuda=0):
  '''
  Determines the strategy under which the network is trained.
  
  From https://github.com/huggingface/transformers/blob/8eb7f26d5d9ce42eb88be6f0150b22a41d76a93d/src/transformers/training_args_tf.py
  
  returns the strategy object
  
  '''
  print("TensorFlow: setting up strategy")

  if xla:
    tf.config.optimizer.set_jit(True)

  gpus = tf.config.list_physical_devices("GPU")
    # Set to float16 at first
  if fp16:
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    tf.keras.mixed_precision.experimental.set_policy(policy)

  if no_cuda:
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
  else:
    try:
      tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    except ValueError:
      tpu = None
  
    if tpu:
    # Set to bfloat16 in case of TPU
      if fp16:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_bfloat16")
        tf.keras.mixed_precision.experimental.set_policy(policy)
      tf.config.experimental_connect_to_cluster(tpu)
      tf.tpu.experimental.initialize_tpu_system(tpu)
    
      strategy = tf.distribute.experimental.TPUStrategy(tpu)
    
    elif len(gpus) == 0:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    elif len(gpus) == 1:
      strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    elif len(gpus) > 1:
      # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
      strategy = tf.distribute.MirroredStrategy()
    else:
      raise ValueError("Cannot find the proper strategy! Please check your environment properties.")

  print(f"Using strategy: {strategy}")
  return strategy

def unbiased_HSIC(K, L):
  '''Computes an unbiased estimator of HISC. This is equation (2) from the paper'''

  #create the unit **vector** filled with ones
  n = K.shape[0]
  ones = np.ones(shape=(n))

  #fill the diagonal entries with zeros 
  np.fill_diagonal(K, val=0) #this is now K_tilde 
  np.fill_diagonal(L, val=0) #this is now L_tilde

  #first part in the square brackets
  trace = np.trace(np.dot(K, L))

  #middle part in the square brackets
  nominator1 = np.dot(np.dot(ones.T, K), ones)
  nominator2 = np.dot(np.dot(ones.T, L), ones)
  denominator = (n-1)*(n-2)
  middle = np.dot(nominator1, nominator2) / denominator
  
  
  #third part in the square brackets
  multiplier1 = 2/(n-2)
  multiplier2 = np.dot(np.dot(ones.T, K), np.dot(L, ones))
  last = multiplier1 * multiplier2

  #complete equation
  unbiased_hsic = 1/(n*(n-3)) * (trace + middle - last)

  return unbiased_hsic



def CKA(X, Y):
  '''Computes the CKA of two matrices. This is equation (1) from the paper'''
  
  nominator = unbiased_HSIC(np.dot(X, X.T), np.dot(Y, Y.T))
  denominator1 = unbiased_HSIC(np.dot(X, X.T), np.dot(X, X.T))
  denominator2 = unbiased_HSIC(np.dot(Y, Y.T), np.dot(Y, Y.T))

  cka = nominator/np.sqrt(denominator1*denominator2)

  return cka

    
#................................ Activation comparison ...................................
def calculate_CKA_for_two_matrices(activationA, activationB):
  '''Takes two activations A and B and computes the linear CKA to measure their similarity'''

  #unfold the activations, that is make a (n, h*w*c) representation
  shape = activationA.shape
  activationA = np.reshape(activationA, newshape=(shape[0], np.prod(shape[1:])))

  shape = activationB.shape
  activationB = np.reshape(activationB, newshape=(shape[0], np.prod(shape[1:])))

  #calculate the CKA score
  cka_score = CKA(activationA, activationB)

  del activationA
  del activationB

  return cka_score
 
 
def get_all_layer_outputs_fn(model):
  '''Builds and returns function that returns the output of every (intermediate) layer'''

  return tf.keras.backend.function([model.layers[0].input],
                                  [l.output for l in model.layers[1:]])


def compare_activations(modelA, modelB, data_batch):
  '''
  Calculate a pairwise comparison of hidden representations and return a matrix
  '''
 
  #get function to get the output of every intermediate layer, for modelA and modelB
  intermediate_outputs_A = get_all_layer_outputs_fn(modelA)(data_batch)
  intermediate_outputs_B = get_all_layer_outputs_fn(modelB)(data_batch)

  #create a placeholder array
  result_array = np.zeros(shape=(len(intermediate_outputs_A), len(intermediate_outputs_B)))

  
  i = 0
  for outputA in tqdm.tqdm_notebook(intermediate_outputs_A):
    j = 0
    for outputB in tqdm.tqdm_notebook(intermediate_outputs_B):
      cka_score = calculate_CKA_for_two_matrices(outputA, outputB)
      result_array[i, j] = cka_score
      j+=1
    i+= 1

  return result_array
     
def compare_activations2(intermediate_outputs_A, intermediate_outputs_B):
  #create a placeholder array
  result_array = np.zeros(shape=(len(intermediate_outputs_A), len(intermediate_outputs_B)))

  
  i = 0
  for outputA in tqdm.tqdm_notebook(intermediate_outputs_A):
    j = 0
    for outputB in tqdm.tqdm_notebook(intermediate_outputs_B):
      cka_score = calculate_CKA_for_two_matrices(outputA, outputB)
      result_array[i, j] = cka_score
      j+=1
    i+= 1

  return result_array
  
#...................................... Main .............................         

def main():
    args = parse_args()

    model = init_pose_model(args.config, args.pose_checkpoint, device=args.device.lower())
    
    sim = compare_activations(model, model, 32)
   #plt.figure(figsize=(30, 15), dpi=200)
   #axes = plt.imshow(sim, cmap='magma', vmin=0.0,vmax=1.0)
   #axes.axes.invert_yaxis()
   #plt.savefig("/content/drive/MyDrive/activ_comparison/r50_r50.png", dpi=400)
   
   
   
if __name__ == '__main__':
    main()
