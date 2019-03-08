import numpy as np, sys, unicodedata, requests, os, random, pdb, requests, json, gensim
import matplotlib.pyplot as plt, uuid, time, argparse, pickle, operator
import logging, logging.config, itertools, pathlib
import scipy.sparse as sp

from collections import defaultdict as ddict
from random import randint
from pprint import pprint
from sklearn.metrics import precision_recall_fscore_support

np.set_printoptions(precision=4)

def checkFile(filename):
	"""
  	Check whether file is present or not

        Parameters
        ----------
        filename:       Path of the file to check
        
        Returns
        -------
        """
	return pathlib.Path(filename).is_file()

def getEmbeddings(embed_loc, wrd_list, embed_dims):
	"""
	Gives embedding for each word in wrd_list

	Parameters
	----------
	embed_loc:	Path to embedding file
	wrd_list:	List of words for which embedding is required
	embed_dims:	Dimension of the embedding

	Returns
	-------
	embed_matrix:	(len(wrd_list) x embed_dims) matrix containing embedding for each word in wrd_list in the same order
	"""
	embed_list = []
	model = gensim.models.KeyedVectors.load_word2vec_format(embed_loc, binary=False)

	for wrd in wrd_list:
		if wrd in model.vocab: 	embed_list.append(model.word_vec(wrd))
		else: 			embed_list.append(np.random.randn(embed_dims))

	return np.array(embed_list, dtype=np.float32)

def set_gpu(gpus):
	"""
        Sets the GPU to be used for the run

        Parameters
        ----------
        gpus:           List of GPUs to be used for the run
        
        Returns
        -------    
        """
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def debug_nn(res_list, feed_dict):
	"""
  	Function for debugging Tensorflow model      

        Parameters
        ----------
        res_list:       List of tensors/variables to view
        feed_dict:	Feed dict required for getting values
        
        Returns
        -------
        Returns the list of values of given tensors/variables after execution

        """
	import tensorflow as tf
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	summ_writer = tf.summary.FileWriter("tf_board/debug_nn", sess.graph)
	res = sess.run(res_list, feed_dict = feed_dict)
	pdb.set_trace()

def get_logger(name, log_dir, config_dir):
	"""
        Creates a logger object

        Parameters
        ----------
        name:           Name of the logger file
        log_dir:        Directory where logger file needs to be stored
        config_dir:     Directory from where log_config.json needs to be read
        
        Returns
        -------
        A logger object which writes to both file and stdout
                
        """
	config_dict = json.load(open( config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def partition(inp_list, n):
	"""
        Paritions a given list into chunks of size n

        Parameters
        ----------
        inp_list:       List to be splittted
        n:     		Number of equal partitions needed
        
        Returns
        -------
        Splits inp_list into n equal chunks
	"""
        division = len(inp_list) / float(n)
        return [ inp_list[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def getChunks(inp_list, chunk_size):
	"""
        Splits inp_list into lists of size chunk_size

        Parameters
        ----------
        inp_list:       List to be splittted
        chunk_size:     Size of each chunk required
        
        Returns
        -------
        chunks of the inp_list each of size chunk_size, last one can be smaller (leftout data)
	"""
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def mergeList(list_of_list):
	"""
        Merges list of list into a list

        Parameters
        ----------
        list_of_list:   List of list
        
        Returns
        -------
        A single list (union of all given lists)
	"""
	return list(itertools.chain.from_iterable(list_of_list))