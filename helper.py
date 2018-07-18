import numpy as np, sys, unicodedata, requests, os, random, pdb, requests, json, gensim
import matplotlib.pyplot as plt, uuid, time, argparse, pickle, operator
from random import randint
from pprint import pprint
import logging, logging.config, itertools, pathlib
from sklearn.metrics import precision_recall_fscore_support
import scipy.sparse as sp

np.set_printoptions(precision=4)

def checkFile(filename):
	return pathlib.Path(filename).is_file()

def getEmbeddings(embed_loc, wrd_list, embed_dims):
	embed_list = []
	model = gensim.models.KeyedVectors.load_word2vec_format(embed_loc, binary=False)

	for wrd in wrd_list:
		if wrd in model.vocab: 	embed_list.append(model.word_vec(wrd))
		else: 			embed_list.append(np.random.randn(embed_dims))

	return np.array(embed_list, dtype=np.float32)

def set_gpu(gpus):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def debug_nn(res_list, feed_dict):
	import tensorflow as tf
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	summ_writer = tf.summary.FileWriter("tf_board/debug_nn", sess.graph)
	res = sess.run(res_list, feed_dict = feed_dict)
	pdb.set_trace()

def get_logger(name, log_dir, config_dir):
	config_dict = json.load(open( config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def partition(lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def getChunks(inp_list, chunk_size):
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))