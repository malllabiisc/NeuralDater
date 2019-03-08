from helper import *
import tensorflow as tf

"""
Abbreviations used in variable names:
	et: event-time
	de: dependency parse
Recommendation: View with tab-size 8
"""

class DCT_NN(object):

	def padData(self, data, seq_len):
		"""
		Pads the data in a batch | Used as a helper function by pad_dynamic

		Parameters
		----------
		data:		batch to be padded
		seq_len:	maximum number of words in the batch

		Returns
		-------
		Padded data and mask
		"""
		temp = np.zeros((len(data), seq_len), np.int32)
		mask = np.zeros((len(data), seq_len), np.float32)
		
		for i, ele in enumerate(data):
			temp[i, :len(ele)] = ele[:seq_len]
			mask[i, :len(ele)] = np.ones(len(ele[:seq_len]), np.float32)

		return temp, mask

	def getOneHot(self, data, num_class):
		"""
		Generates the one-hot representation

		Parameters
		----------
		data:		Batch to be padded
		num_class:	Total number of relations 

		Returns
		-------
		One-hot representation of batch
		"""
		temp = np.zeros((len(data), num_class), np.int32)
		for i, ele in enumerate(data):
			temp[i, ele] = 1
		return temp

	def getBatches(self, data, shuffle = True):
		"""
	        Generates batches of multiple bags

	        Parameters
	        ----------
	        data:		Data to be used for creating batches.
	        shuffle:	Decides whether to shuffle the data or not.
	        
	        Returns
	        -------
	        Generator for creating batches. 
	        """
		if shuffle: random.shuffle(data)
		num_batches = len(data) // self.p.batch_size

		for i in range(num_batches):
			start_idx = i * self.p.batch_size
			yield data[start_idx : start_idx + self.p.batch_size]

	def updateEdges(self, data, merge_edges=False):
		"""
	        Merges edge labels or Ignores Edge labels based on cmd arguments

	        Parameters
	        ----------
	        data:		full dataset
	        merge_edges:	Whether to merge Event Time graph edge labels or not
	        
	        Returns
	        -------
	        data:		Updated dataset
	        """

		for dtype in ['train', 'test', 'valid']:
			for i, edges in enumerate(data[dtype]['ETEdges']):
				for j in range(len(edges)-1, -1, -1):
					edge = edges[j]
					lbl  = self.id2ce[edge[2]]
					
					if lbl not in self.n_et2id: del data[dtype]['ETEdges'][i][j]
					else: 			    data[dtype]['ETEdges'][i][j] = (edge[0], edge[1], self.n_et2id[lbl])

			if merge_edges:
				for i, edges in enumerate(data[dtype]['ETEdges']):
					for j, edge in enumerate(edges):
						if   edge[2] == self.n_et2id['BEFORE']: 	data[dtype]['ETEdges'][i][j] = (edge[1], edge[0], self.n_et2id['AFTER'])
						elif edge[2] == self.n_et2id['INCLUDES']: 	data[dtype]['ETEdges'][i][j] = (edge[1], edge[0], self.n_et2id['IS_INCLUDED'])
			
			# Remove dependency edges with negative source/destination ids
			for i, edges in enumerate(data[dtype]['DepEdges']):
				for j in range(len(edges)-1, -1, -1):
					edge = edges[j]
					if edge[0] < 0 or edge[1] < 0:
						del data[dtype]['DepEdges'][i][j]

		if merge_edges: self.num_etLabel -= 2
		return data
	
	def rm_hdeg_docs(self, data):
		"""
	        Remove documents with very large number of edges in Event-Time Graph

	        Parameters
	        ----------
	        data:		full dataset
	        
	        Returns
	        -------
	        data:		Updated dataset
	        """
		rm_idx = {}
		for dtype in ['train', 'test', 'valid']:
			rm_idx[dtype] = set()
			
			for i,vec in enumerate(data[dtype]['ETIdx']):
				if len(vec) > self.p.th_maxet: 
					rm_idx[dtype].add(i)
			
			for i,vec in enumerate(data[dtype]['ET']):
				if len(vec)> self.p.th_seq_len:
					rm_idx[dtype].add(i)

			for i, etIdx in enumerate(data[dtype]['ETIdx']):
				if len(etIdx) == 0: 
					rm_idx[dtype].add(i)
					
		return rm_idx

	def load_data(self):
		"""
		Reads the data from pickle file

		Parameters
		----------
		self.p.dataset: The path of the dataset to be loaded

		Returns
		-------
		self.voc2id:		Mapping of word to its unique identifier
		self.Id2voc:		Inverse of self.voc2id
		self.e2id:		Mapping of event time graph edge label to its unique identifier
		self.n_et2id:		New Mapping of event time graph edge label to its unique identifier
		self.num_etLabel:	Number of edge labels in event-time graph
		self.de2id:		Mapping of dependency graph edge label to its unique identifier
		self.num_deLabel:	Number of edge labels in dependency graph
		self.num_class:		Total number of years to be predicted
		self.wrd_list:		Words in vocabulary
		self.data:		Contains all split of the data train/valid/test
		"""
		data = pickle.load(open(self.p.dataset, 'rb'))

		self.voc2id 	= data['voc2id']
		self.et2id 	= data['et2id']
		self.id2ce  	= dict([(v,k) for k,v in self.et2id.items()])
		self.de2id 	= data['de2id']

		self.n_et2id = {
			'AFTER': 	0,
			'IS_INCLUDED':	1,
			'SIMULTANEOUS':	2,
			'DURING':	2,
			'BEFORE':	3,
			'INCLUDES':	4,
		}

		self.num_etLabel  = len(self.n_et2id)
		self.num_deLabel  = len(self.de2id)
		data 		  = self.updateEdges(data, self.p.merge_edges)			# Merge edge labels
		rm_idx 		  = self.rm_hdeg_docs(data)					# Indexes to be removed

		print('Number of classes {}'.format(len(np.unique(data['train']['Y']))))
		self.num_class = self.p.num_class

		self.logger.info('Removing Train:{}, Test:{}, Valid:{}'.format(len(rm_idx['train']), len(rm_idx['test']), len(rm_idx['valid'])))

		# Get Word List
		self.wrd_list 	= list(self.voc2id.items())					# Get vocabulary
		self.wrd_list.sort(key=lambda x: x[1])						# Sort vocabulary based on ids
		self.wrd_list, _ = zip(*self.wrd_list)
		
		self.data_list = {}
		key_list =  ['X', 'Y', 'ETIdx', 'ETEdges', 'DepEdges']

		for dtype in ['train', 'test', 'valid']:

			if self.p.use_et_labels == False:
				for i, edges in enumerate(data[dtype]['ETEdges']):							# if you want to ignore level information in event time graph
					for j, edge in enumerate(edges): data[dtype]['ETEdges'][i][j] = (edge[0], edge[1], 0)   
				self.num_etLabel = 1

			if self.p.use_de_labels == False:
				for i, edges in enumerate(data[dtype]['DepEdges']):							# if you want to ignore level information in dependency graph
					for j, edge in enumerate(edges): data[dtype]['DepEdges'][i][j] = (edge[0], edge[1], 0)
				self.num_deLabel = 1

			data[dtype]['Y'] = self.getOneHot(data[dtype]['Y'], self.num_class)						# Representing labels by one hot notation

			self.data_list[dtype] = []
			for i in range(len(data[dtype]['X'])):
				if i in rm_idx[dtype]: continue
				self.data_list[dtype].append([data[dtype][key][i] for key in key_list])          			# data_list contains all the fields for train test and valid documents

			self.logger.info('Document count [{}]: {}'.format(dtype, len(self.data_list[dtype])))
		
		self.data = data

	def get_adj(self, edgeList, batch_size, max_nodes, max_labels):
		"""
		Loads adjacency matrix in sparse matrix format, required for feeding to Tensorflow

		Parameters
		----------
		edgeList:	List of list of edges 
		batch_size:	Number of bags in a batch
		max_nodes:	Maximum number of nodes in the graph
		max_labels:	Maximum number of edge labels in the graph 

		Returns
		-------
		adj_mat		Contains dependency/event-time graph for each sentence in the batch
		"""
		adj_main = []
		for edges in edgeList:
			ind, vals, adj   = ddict(list), ddict(list), {}

			for src, dest, lbl in edges:
				ind [lbl].append((dest, src))
				vals[lbl].append(1.0)

			for lbl in range(max_labels):
				if lbl not in ind: 	adj[lbl] = sp.coo_matrix((max_nodes, max_nodes))
				else: 			adj[lbl] = sp.coo_matrix((vals[lbl],  zip(*ind[lbl])),  shape=(max_nodes, max_nodes))

			adj_main.append(adj)

		return adj_main


	def add_placeholders(self):
		"""
		Defines the placeholder required for the model
		"""

		self.input_x  		= tf.placeholder(tf.int32,   shape=[None, None],   name='input_data')		# Words in a document (batch_size x max_words)
		self.input_y 		= tf.placeholder(tf.int32,   shape=[None, None],   name='input_labels')		# Actual document creation year of the document

		self.x_len		= tf.placeholder(tf.int32,   shape=[None],         name='input_len')		# Number of words in each document in a batch
		self.et_idx 		= tf.placeholder(tf.int32,   shape=[None, None],   name='et_idx')		# Index of tokens which are events/time_expressions
		self.et_mask 		= tf.placeholder(tf.float32, shape=[None, None],   name='et_mask')

		# Array of batch_size number of dictionaries, where each dictionary is mapping of label to sparse_placeholder [Temporal graph]
		self.de_adj_mat	= [{lbl: tf.sparse_placeholder(tf.float32,  shape=[None, None]) for lbl in range(self.num_deLabel)}  for _ in range(self.p.batch_size)]

		# Array of batch_size number of dictionaries, where each dictionary is mapping of label to sparse_placeholder [Syntactic graph]
		self.et_adj_mat	= [{lbl: tf.sparse_placeholder(tf.float32,  shape=[None, None]) for lbl in range(self.num_etLabel)}  for _ in range(self.p.batch_size)]

		self.seq_len 		= tf.placeholder(tf.int32, shape=(), name='seq_len')				# Maximum number of words in documents of a batch
		self.max_et 		= tf.placeholder(tf.int32, shape=(), name='max_et')				# Maximum number of events/time_expressions in documents of a batch

		self.dropout 		= tf.placeholder_with_default(self.p.dropout, 	  shape=(), name='dropout')	# Dropout used in GCN Layer
		self.rec_dropout 	= tf.placeholder_with_default(self.p.rec_dropout, shape=(), name='rec_dropout')	# Dropout used in Bi-LSTM

	def pad_dynamic(self, X, et_idx):
		"""
		Pads each batch during runtime.

		Parameters
		----------
		X:		For each sentence in the batch, list of words
		et_idx:		Indices of event-time tokens in the sentence

		Returns
		-------
		x_pad		Padded words 
		x_len		Number of words in each sentence
		et_pad 		padded  event-time token indices
		et_mask 	Mask for et_pad
		seq_len 	Maximum sentence length in the batch
		max_et  	maximum number of event-time tokens in the batch
		"""
		seq_len, max_et = 0, 0

		x_len = np.zeros((len(X)), np.int32)

		for i, x in enumerate(X): 	  
			seq_len  = max(seq_len, len(x))
			x_len[i] = len(x)

		for et in et_idx: max_et  = max(max_et,  len(et))

		x_pad,  _ 	= self.padData(X, seq_len)
		et_pad, et_mask = self.padData(et_idx, max_et)

		return x_pad, x_len, et_pad, et_mask, seq_len, max_et

	def create_feed_dict(self, batch, wLabels=True, dtype='train'):
		"""
		Creates a feed dictionary for the batch

		Parameters
		----------
		batch:		contains a batch of bags
		wLabels:	Whether batch contains labels or not
		split:		Indicates the split of the data - train/valid/test

		Returns
		-------
		feed_dict	Feed dictionary to be fed during sess.run
		"""
		X, Y, et_idx, ETEdges, DepEdges = zip(*batch)

		x_pad, x_len, et_pad, et_mask, seq_len, max_et = self.pad_dynamic(X, et_idx)

		feed_dict = {}
		feed_dict[self.input_x] 		= np.array(x_pad)
		feed_dict[self.x_len] 			= np.array(x_len)
		if wLabels: feed_dict[self.input_y] 	= np.array(Y)

		feed_dict[self.et_idx] 			= np.array(et_pad)
		feed_dict[self.et_mask] 		= np.array(et_mask)

		feed_dict[self.seq_len]			= seq_len
		feed_dict[self.max_et]			= max_et

		et_adj = self.get_adj(ETEdges,  self.p.batch_size, max_et+1,  self.num_etLabel)  # max_et + 1(DCT)
		de_adj = self.get_adj(DepEdges, self.p.batch_size, seq_len, self.num_deLabel)

		for i in range(self.p.batch_size):
			for lbl in range(self.num_etLabel):
				feed_dict[self.et_adj_mat[i][lbl]] = tf.SparseTensorValue( 	indices 	= np.array([et_adj[i][lbl].row, et_adj[i][lbl].col]).T,
											      	values  	= et_adj[i][lbl].data,
												dense_shape	= et_adj[i][lbl].shape)

			for lbl in range(self.num_deLabel):
				feed_dict[self.de_adj_mat[i][lbl]] = tf.SparseTensorValue( 	indices 	= np.array([de_adj[i][lbl].row, de_adj[i][lbl].col]).T,
											      	values  	= de_adj[i][lbl].data,
												dense_shape	= de_adj[i][lbl].shape)
		if dtype != 'train':
			feed_dict[self.dropout]     = 1.0
			feed_dict[self.rec_dropout] = 1.0

		return feed_dict

	def GCNLayer(self, gcn_in, in_dim, gcn_dim, batch_size, max_nodes, max_labels, adj, num_layers=1, name="GCN"):
		"""
		GCN Layer Implementation

		Parameters
		----------
		gcn_in:		Input to GCN Layer
		in_dim:		Dimension of input to GCN Layer 
		gcn_dim:	Hidden state dimension of GCN
		batch_size:	Batch size
		max_nodes:	Maximum number of nodes in graph
		max_labels:	Maximum number of edge labels
		adj:		Adjacency matrix indices
		num_layers:	Number of GCN Layers
		name 		Name of the layer (used for creating variables, keep it different for different layers)

		Returns
		-------
		out		List of output of different GCN layers with first element as input itself, i.e., [gcn_in, gcn_layer1_out, gcn_layer2_out ...]
		"""

		out = []
		out.append(gcn_in)

		for layer in range(num_layers):
			gcn_in    = out[-1]						# out contains the output of all the GCN layers, intitally contains input to first GCN Layer
			if len(out) > 1: in_dim = gcn_dim 				# After first iteration the in_dim = gcn_dim

			with tf.name_scope('%s-%d' % (name,layer)):

				act_sum = tf.zeros([batch_size, max_nodes, gcn_dim])
				
				for lbl in range(max_labels):

					with tf.variable_scope('label-%d_name-%s_layer-%d' % (lbl, name, layer)) as scope:

						w_in   = tf.get_variable('w_in',   [in_dim, gcn_dim],  	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
						b_in   = tf.get_variable('b_in',   [1, gcn_dim],   	initializer=tf.constant_initializer(0.0), 		regularizer=self.regularizer)
						w_out  = tf.get_variable('w_out',  [in_dim, gcn_dim], 	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
						b_out  = tf.get_variable('b_out',  [1, gcn_dim],  	initializer=tf.constant_initializer(0.0), 		regularizer=self.regularizer)
						w_loop = tf.get_variable('w_loop', [in_dim, gcn_dim], 	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)

						if self.p.wGate:
							w_gin  = tf.get_variable('w_gin',  [in_dim, 1], 	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
							b_gin  = tf.get_variable('b_gin',  [1], 	  	initializer=tf.constant_initializer(0.0), 		regularizer=self.regularizer)
							w_gout = tf.get_variable('w_gout', [in_dim, 1], 	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)
							b_gout = tf.get_variable('b_gout', [1], 	  	initializer=tf.constant_initializer(0.0), 		regularizer=self.regularizer)
							w_gloop = tf.get_variable('w_gloop',[in_dim, 1], 	initializer=tf.contrib.layers.xavier_initializer(), 	regularizer=self.regularizer)

					with tf.name_scope('in_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
						inp_in  = tf.tensordot(gcn_in, w_in, axes=[2,0]) + tf.expand_dims(b_in, axis=0)
						in_t    = tf.stack([tf.sparse_tensor_dense_matmul(adj[i][lbl], inp_in[i]) for i in range(batch_size)])
						if self.p.dropout != 1.0: in_t    = tf.nn.dropout(in_t, keep_prob=self.p.dropout)

						if self.p.wGate:
							inp_gin = tf.tensordot(gcn_in, w_gin, axes=[2,0]) + tf.expand_dims(b_gin, axis=0)
							in_gate = tf.stack([tf.sparse_tensor_dense_matmul(adj[i][lbl], inp_gin[i]) for i in range(batch_size)])
							in_gsig = tf.sigmoid(in_gate)
							in_act   = in_t * in_gsig
						else:
							in_act   = in_t

					with tf.name_scope('out_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
						inp_out  = tf.tensordot(gcn_in, w_out, axes=[2,0]) + tf.expand_dims(b_out, axis=0)
						out_t    = tf.stack([tf.sparse_tensor_dense_matmul(tf.sparse_transpose(adj[i][lbl]), inp_out[i]) for i in range(batch_size)])
						if self.p.dropout != 1.0: out_t    = tf.nn.dropout(out_t, keep_prob=self.p.dropout)

						if self.p.wGate:
							inp_gout = tf.tensordot(gcn_in, w_gout, axes=[2,0]) + tf.expand_dims(b_gout, axis=0)
							out_gate = tf.stack([tf.sparse_tensor_dense_matmul(tf.sparse_transpose(adj[i][lbl]), inp_gout[i]) for i in range(batch_size)])
							out_gsig = tf.sigmoid(out_gate)
							out_act  = out_t * out_gsig
						else:
							out_act = out_t

					with tf.name_scope('self_loop'):
						inp_loop  = tf.tensordot(gcn_in, w_loop,  axes=[2,0])
						if self.p.dropout != 1.0: inp_loop  = tf.nn.dropout(inp_loop, keep_prob=self.p.dropout)

						if self.p.wGate:
							inp_gloop = tf.tensordot(gcn_in, w_gloop, axes=[2,0])
							loop_gsig = tf.sigmoid(inp_gloop)
							loop_act  = inp_loop * loop_gsig
						else:
							loop_act = inp_loop


					act_sum += in_act + out_act + loop_act
				gcn_out = tf.nn.relu(act_sum)
				out.append(gcn_out)

		return out

	def gather(self, data, pl_idx, pl_mask, max_len, name=None):
		"""
		Lookup equivalent for tensors with dim > 2 (Can be simplified using tf.batch_gather)

		Parameters
		----------
		data:		Tensor in which lookup has to be performed
		pl_idx:		The indices to be taken
		pl_mask:	For handling padding in pl_idx
		max_len:	Maximum length of indices

		Returns
		-------
		et_vecs * mask_vec:	Extracted vectors at given indices
		
		"""
		idx1  = tf.range(self.p.batch_size, dtype=tf.int32)
		idx1  = tf.reshape(idx1, [-1, 1])
		idx1_ = tf.reshape(tf.tile(idx1, [1, max_len]) , [-1, 1])
		idx_reshape = tf.reshape(pl_idx, [-1, 1])
		indices = tf.concat((idx1_, idx_reshape), axis=1)
		et_vecs = tf.gather_nd(data, indices)
		et_vecs = tf.reshape(et_vecs, [self.p.batch_size, self.max_et, -1])
		mask_vec = tf.expand_dims(pl_mask, axis=2)
		return et_vecs * mask_vec

	def add_model(self):
		"""
		Creates the Computational Graph

		Parameters
		----------

		Returns
		-------
		nn_out:		Logits for each bag in the batch
		"""
		nn_in = self.input_x

		with tf.variable_scope('Embeddings') as scope:
			embed_init = getEmbeddings(self.p.embed_loc, self.wrd_list, self.p.embed_dim)
			embed_init = np.vstack( (np.zeros(self.p.embed_dim, np.float32), embed_init))
			embeddings = tf.get_variable('embeddings', initializer=embed_init, trainable=True, regularizer=self.regularizer)
			embeds     = tf.nn.embedding_lookup(embeddings, self.input_x)

	
		with tf.variable_scope('Bi-LSTM') as scope:
			fw_cell    = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.p.lstm_dim), output_keep_prob=self.rec_dropout)
			bk_cell    = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.p.lstm_dim), output_keep_prob=self.rec_dropout)
			val, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bk_cell, embeds, sequence_length=self.x_len, dtype=tf.float32)
			lstm_out   = tf.concat((val[0], val[1]), axis=2)

		de_in     = lstm_out
		de_in_dim = self.p.lstm_dim*2		# Concatenated output of forward and backward LSTM (Bi-LSTM)

		
		de_out = self.GCNLayer( gcn_in 		= de_in, 		in_dim 	    = de_in_dim, 		gcn_dim    = self.p.de_gcn_dim, 
					batch_size 	= self.p.batch_size, 	max_nodes   = self.seq_len, 		max_labels = self.num_deLabel, 
					adj 		= self.de_adj_mat, 	num_layers  = self.p.de_layers, 	name 	   = "GCN_DE")

		ce_in_dim = self.p.de_gcn_dim
		ce_in 	  = de_out[-1]			# GCNLayer returns list containing output of all layers; last entry is its final output

		et_vecs = self.gather(ce_in, self.et_idx, self.et_mask, self.max_et, name='ET_pick')
		with tf.name_scope('DCT_init'):
			dct_sum  = tf.reduce_sum(et_vecs, axis=1)
			dct_cnt  = tf.reduce_sum(self.et_mask, axis=1)
			dct_init = tf.expand_dims(dct_sum / tf.expand_dims(dct_cnt,axis=1), axis=1)

		et_con = tf.concat( [dct_init, et_vecs], axis=1)
		ce_out = self.GCNLayer( gcn_in 		= et_con, 		in_dim 		= ce_in_dim, 			gcn_dim    = self.p.et_gcn_dim, 
					batch_size 	= self.p.batch_size, 	max_nodes 	= self.max_et+1, 		max_labels = self.num_etLabel,
					adj 		= self.et_adj_mat, 	num_layers	= self.p.et_layers, 		name 	   = "GCN_CE")

		dct_vec = ce_out[-1][:,0]

		con_mean  = tf.reduce_mean(ce_in, axis=1)			# Context  Embedding
		dct_final = tf.concat([dct_vec, con_mean], axis=1)		# Concatenating contextual and temporal embedding
		fc_in_dim = self.p.et_gcn_dim + ce_in_dim 

		with tf.variable_scope('FC1') as scope:
			w = tf.get_variable('w', [fc_in_dim, self.num_class], 	initializer=tf.truncated_normal_initializer(),  regularizer=self.regularizer)
			b = tf.get_variable('b', [self.num_class], 	  	initializer=tf.constant_initializer(0.0), 	regularizer=self.regularizer)
			nn_out = tf.matmul(dct_final, w) + b

		'''
		# debug_nn([dct_final], self.create_feed_dict(self.data_list['train'][0:self.p.batch_size]))
		'''
		return nn_out


	def add_loss(self, nn_out):
		"""
		Computes loss based on logits and actual labels

		Parameters
		----------
		nn_out:		Logits for each bag in the batch

		Returns
		-------
		loss:		Computes loss based on prediction and actual labels 
		"""

		with tf.name_scope('Loss_op'):
			loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=nn_out, labels=self.input_y))
			if self.regularizer != None: loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		return loss

	def add_optimizer(self, loss):
		"""
		Add optimizer for training variables

		Parameters
		----------
		loss:		Computed loss

		Returns
		-------
		train_op:	Training optimizer
		"""
		with tf.name_scope('Optimizer'):
			optimizer = tf.train.AdamOptimizer(self.p.lr)
			train_op  = optimizer.minimize(loss)
		return train_op

	def __init__(self, params):
		"""
		Constructor for the main function. Loads data and creates computation graph. 

		Parameters
		----------
		params:		Hyperparameters of the model

		Returns
		-------
		"""
		self.p  = params
		pprint(vars(params))
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p))
		self.p.batch_size = self.p.batch_size

		if self.p.l2 == 0.0: 	self.regularizer = None
		else: 			self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)

		self.load_data()
		self.add_placeholders()
		
		nn_out = self.add_model()

		self.loss      	= self.add_loss(nn_out)				# Compute the loss
		self.train_op  	= self.add_optimizer(self.loss)			# Update the parameters
		self.logits 	= tf.nn.softmax(nn_out)

		y_pred 	  = tf.argmax(self.logits, 1)				# Predictions by the model
		corr_pred = tf.equal(tf.argmax(self.input_y, 1), y_pred)
		self.corr_pred = tf.reduce_sum(tf.cast(corr_pred, 'int32'))

		self.merged_summ = tf.summary.merge_all()
		self.summ_writer = None

	def predict(self, sess, data, wLabels=True, shuffle=False):
		"""
		Evaluate model on valid/test data

		Parameters
		----------
		sess:		Session of tensorflow
		data:		Data to evaluate on
		wLabels:	Does data include labels or not
		shuffle:	Shuffle data while before creates batches

		Returns
		-------
		losses:		Loss over the entire data
		accuracies:	Overall Accuracy
		y: 		Actual label
		y_pred:		Predicted labels
		logit_list:	Logit list for each bag in the data

		"""
		losses, y_pred, y, logit_list = [], [], [], [], []
		total_correct, total_cnt = 0, 0

		for step, batch in enumerate(self.getBatches(data, shuffle)):
			
			if not wLabels:
				feed   = self.create_feed_dict(batch, wLabels, dtype='test')
				logits, correct = sess.run([self.logits, self.corr_pred] , feed_dict = feed)
			else:
				feed  = self.create_feed_dict(batch, dtype='test')
				loss, logits, correct = sess.run([self.loss, self.logits, self.corr_pred], feed_dict = feed)
				losses.append(loss)

			total_correct += correct
			total_cnt += len(batch)

			pred_ind    = logits.argmax(axis=1)
			logit_list += logits.tolist()
			y_pred   += pred_ind.tolist()
			_, Y, _, _, _ = zip(*batch)
			y += np.array(Y).argmax(axis=1).tolist()

			if step % 5 == 0:
				self.logger.info('Evaluating Test/Valid ({}/{}):\t{:.5}\t{:.5}\t{}'.format(step, len(data)//self.p.batch_size, total_correct/total_cnt, np.mean(losses), self.p.name))

		accuracy = float(total_correct)/total_cnt * 100.0
		self.logger.info('Accuracy: {}'.format(accuracy))

		if wLabels: 	return np.mean(losses), accuracy, y, y_pred, logit_list
		else: 		return 0, accuracy, y, y_pred, logit_list

	def run_epoch(self, sess, data, epoch, shuffle=True):
		"""
		Runs one epoch of training

		Parameters
		----------
		sess:		Session of tensorflow
		data:		Data to train on
		epoch:		Epoch number
		shuffle:	Shuffle data while before creates batches

		Returns
		-------
		losses:		Loss over the entire data
		Accuracy:	Overall accuracy
		"""
		drop_rate = self.p.dropout

		losses = []
		total_correct, total_cnt = 0, 0

		for step, batch in enumerate(self.getBatches(data, shuffle)):
			feed = self.create_feed_dict(batch)
			loss, correct, _= sess.run([self.loss, self.corr_pred, self.train_op], feed_dict=feed)
			if(np.isnan(loss)): 
				print(et_cnt)
				pdb.set_trace()
			total_cnt     += len(batch)
			total_correct += correct

			losses.append(loss)

			if step % 5 == 0:
				self.logger.info('E:{} Train Accuracy ({}/{}):\t{:.5}\t{:.5}\t{}\t{:.5}'.format(epoch, step, len(data)//self.p.batch_size, total_correct/total_cnt, np.mean(losses), self.p.name, self.best_val_acc))

		accuracy = float(total_correct)/total_cnt * 100.0

		self.logger.info('Training Loss:{}, Accuracy: {}'.format(np.mean(losses), accuracy))
		return np.mean(losses), accuracy

	def fit(self, sess):
		"""
		Trains the model and finally evaluates on test

		Parameters
		----------
		sess:		Tensorflow session object

		Returns
		-------
		"""
		self.summ_writer = tf.summary.FileWriter("tf_board/DCT_NN/" + self.p.name, sess.graph)
		self.best_val_acc, self.best_train_acc = 0.0, 0.0
		
		saver = tf.train.Saver()
		save_dir = 'checkpoints/' + self.p.name + '/'
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		save_path = os.path.join(save_dir, 'best_validation')
		if self.p.restore: saver.restore(sess, save_path)

		# Train Model
		for epoch in range(self.p.max_epochs):
			self.logger.info('Epoch: {}'.format(epoch))

			train_loss, train_acc 			 = self.run_epoch(sess,  self.data_list['train'], epoch)
			val_loss, val_acc, y, y_pred, logit_list = self.predict(sess,	self.data_list['valid'])

			if val_acc > self.best_val_acc:
				self.best_val_acc   = val_acc
				self.best_train_acc = train_acc
				saver.save(sess=sess, save_path=save_path)

			self.logger.info('[Epoch {}]: Training Loss: {:.5}, Training Acc: {:.5}, Valid Loss: {:.5}, Valid Acc: {:.5} Best Acc: {:.5}\n'.format(epoch, train_loss, train_acc, val_loss, val_acc, self.best_val_acc))

		# Evaluate on Test
		self.logger.info('Running on Test set')
		test_loss, test_pred, test_acc, y, y_pred, logit_list = self.predict(sess, self.data_list['test'])
		self.logger.info('Test Acc:{}'.format(test_acc))

if __name__== "__main__":

	parser = argparse.ArgumentParser(description='Main Neural Network for Time Stamping Documents')

	parser.add_argument('-data', 	 dest="dataset", 	required=True,			help='Dataset to use')
	parser.add_argument('-class',	 dest="num_class", 	required=True,   type=int, 	help='Number of classes (years/months)')
	parser.add_argument('-gpu', 	 dest="gpu", 		default='0',			help='GPU to use')

	parser.add_argument('-name', 	 dest="name", 		default='test_'+str(uuid.uuid4()),help='Name of the run')
	parser.add_argument('-drop',	 dest="dropout", 	default=1.0,  	type=float,	help='Dropout for full connected layer')
	parser.add_argument('-rdrop',	 dest="rec_dropout", 	default=1.0,  	type=float,	help='Recurrent dropout for LSTM')
	parser.add_argument('-lr',	 dest="lr", 		default=0.001,  type=float,	help='Learning rate')
	parser.add_argument('-batch', 	 dest="batch_size", 	default=64,   	type=int, 	help='Batch size')
	parser.add_argument('-epoch', 	 dest="max_epochs", 	default=50,   	type=int, 	help='Max epochs')
	parser.add_argument('-l2', 	 dest="l2", 		default=0.001, 	type=float, 	help='L2 regularization')
	parser.add_argument('-seed', 	 dest="seed", 		default=1234, 	type=int, 	help='Seed for randomization')
	parser.add_argument('-fix_emb',	 dest="fix_emb",	action='store_true',		help='fix embedding for fast training')
	
	parser.add_argument('-lstm_dim', dest="lstm_dim", 	default=128,   	type=int, 	help='Hidden state dimension of Bi-LSTM')
	parser.add_argument('-de_dim',   dest="de_gcn_dim", 	default=128,   	type=int, 	help='Hidden state dimension of GCN over dependency tree')
	parser.add_argument('-et_dim',   dest="et_gcn_dim", 	default=128,   	type=int, 	help='Hidden state dimension of GCN over ET-graphs')
	parser.add_argument('-fc1_dim',  dest="fc1_dim", 	default=128,   	type=int, 	help='Hidden state dimension of FC layer')
	parser.add_argument('-de_layer', dest="de_layers", 	default=1,   	type=int, 	help='Number of layers in GCN over dependency tree')
	parser.add_argument('-et_layer', dest="et_layers", 	default=2,   	type=int, 	help='Number of layers in GCN over ET-graph')
	parser.add_argument('-th_et',	 dest="th_maxet", 	default=300 , 	type=int,	help='maximum et_nodes')
	parser.add_argument('-th_seq',	 dest="th_seq_len", 	default=800 , 	type=int,	help='maximum de_nodes or sequence_length')

	# Include/Exclude network parts
	parser.add_argument('-no-CE',	 dest="wCE", 		action='store_false', 		help='With or without ET graph')
	parser.add_argument('-noGate', 	 dest="wGate", 		action='store_false', 		help='Use gating in GCN')
	parser.add_argument('-no-et_lbl',dest="use_et_labels", 	action='store_false', 		help='Ignore edge labels in ET-graph')
	parser.add_argument('-merge', 	 dest="merge_edges", 	action='store_true', 		help='Merge edge labels in ET-graph')
	parser.add_argument('-de_lbl', 	 dest="use_de_labels", 	action='store_true', 		help='Use edge labels in dependency tree')

	parser.add_argument('-restore',	 dest="restore", 	action='store_true', 		help='Restore from the previously saved model')
	parser.add_argument('-logdir',	 dest="log_dir", 	default='./log/', 		help='Log directory')
	parser.add_argument('-config',	 dest="config_dir", 	default='./config/', 		help='Config directory')
	parser.add_argument('-embed_loc',dest="embed_loc", 	default='./glove/glove.6B.300d_word2vec.txt', 		help='Log directory')
	parser.add_argument('-embed_dim',dest="embed_dim", 	default=300, type=int,		help='Dimension of embedding')

	args = parser.parse_args()
	if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

	tf.set_random_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	set_gpu(args.gpu)

	model  = DCT_NN(args)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		model.fit(sess)