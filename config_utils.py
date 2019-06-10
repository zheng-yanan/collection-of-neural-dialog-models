class Seq2SeqConfig(object):

	update_limit = 3000  # the number of mini-batch before evaluating the model
	sent_type = "rnn" # bi_rnn, bow, rnn

	dec_keep_prob = 1.0  # do we use word drop decoder [Bowman el al 2015]
	cell_type = "gru"  # gru or lstm
	
	embed_size = 128  		# word embedding size
	sent_cell_size = 300 	#300  # utterance encoder hidden size
	dec_cell_size = 400 	#400  # response decoder hidden size
	attention_dim = 200

	backward_size = 10  # how many utterance kept in the context window
	step_size = 1  # internal usage
	max_utt_len = 40  # max number of words in an utterance
	num_layer = 1  # number of context RNN layers

	op = "adam"
	grad_clip = 5.0  # gradient abs max cut
	init_w = 0.08  # uniform random from [-init_w, init_w]
	batch_size = 30  # mini-batch size
	init_lr = 0.001  # initial learning rate
	lr_hold = 1  # only used by SGD
	lr_decay = 0.6  # only used by SGD
	keep_prob = 1.0  # drop out rate

	improve_threshold = 0.996  # for early stopping
	patient_increase = 2.0  # for early stopping
	early_stop = True

	max_epoch = 60  # max number of epoch of training
	grad_noise = 0.0  # inject gradient noise?



class HREDConfig(object):

	update_limit = 3000  # the number of mini-batch before evaluating the model
	sent_type = "rnn" # bi_rnn, bow, rnn

	dec_keep_prob = 1.0  # do we use word drop decoder [Bowman el al 2015]
	cell_type = "gru"  # gru or lstm

	embed_size = 128  # word embedding size
	cxt_cell_size = 200
	sent_cell_size = 300 #300  # utterance encoder hidden size
	dec_cell_size = 400 #400  # response decoder hidden size
 

	backward_size = 10  # how many utterance kept in the context window
	step_size = 1  # internal usage
	max_utt_len = 40  # max number of words in an utterance
	num_layer = 1  # number of context RNN layers

	# Optimization parameters
	op = "adam"
	grad_clip = 5.0  # gradient abs max cut
	init_w = 0.08  # uniform random from [-init_w, init_w]
	batch_size = 30  # mini-batch size
	init_lr = 0.001  # initial learning rate
	lr_hold = 1  # only used by SGD
	lr_decay = 0.6  # only used by SGD
	keep_prob = 1.0  # drop out rate
	improve_threshold = 0.996  # for early stopping
	patient_increase = 2.0  # for early stopping
	early_stop = True
	max_epoch = 60  # max number of epoch of training
	grad_noise = 0.0  # inject gradient noise?



class VHREDConfig(object):


	update_limit = 3000  # the number of mini-batch before evaluating the model
	sent_type = "rnn"

	# latent variable (gaussian variable)
	latent_size = 200 # 200  # the dimension of latent variable
	full_kl_step = 10000 # 10000  # how many batch before KL cost weight reaches 1.0
	dec_keep_prob = 1.0  # do we use word drop decoder [Bowman el al 2015]

	# Network general
	cell_type = "gru"  # gru or lstm
	embed_size = 128  # word embedding size
	cxt_cell_size = 200
	sent_cell_size = 300 #300  # utterance encoder hidden size
	dec_cell_size = 400 #400  # response decoder hidden size

	backward_size = 10  # how many utterance kept in the context window
	step_size = 1  # internal usage
	max_utt_len = 40  # max number of words in an utterance
	num_layer = 1  # number of context RNN layers

	# Optimization parameters
	op = "adam"
	grad_clip = 5.0  # gradient abs max cut
	init_w = 0.08  # uniform random from [-init_w, init_w]
	batch_size = 30  # mini-batch size
	init_lr = 0.001  # initial learning rate
	lr_hold = 1  # only used by SGD
	lr_decay = 0.6  # only used by SGD
	keep_prob = 1.0  # drop out rate
	improve_threshold = 0.996  # for early stopping
	patient_increase = 2.0  # for early stopping
	early_stop = True
	
	max_epoch = 60  # max number of epoch of training
	grad_noise = 0.0  # inject gradient noise?


class KGCVAEConfig(object):

 
	update_limit = 3000  # the number of mini-batch before evaluating the model
	sent_type = "rnn"

	latent_size = 200 # 200  # the dimension of latent variable
	full_kl_step = 10000 # 10000  # how many batch before KL cost weight reaches 1.0
	dec_keep_prob = 1.0  # do we use word drop decoder [Bowman el al 2015]

	# Network general
	cell_type = "gru"  # gru or lstm
	embed_size = 128  # word embedding size
	cxt_cell_size = 200
	sent_cell_size = 300 #300  # utterance encoder hidden size
	dec_cell_size = 400 #400  # response decoder hidden size

	backward_size = 10  # how many utterance kept in the context window
	step_size = 1  # internal usage
	max_utt_len = 40  # max number of words in an utterance
	num_layer = 1  # number of context RNN layers


	topic_embed_size = 30  # topic embedding size
	da_embed_size = 30  # dialog act embedding size
	
	op = "adam"
	grad_clip = 5.0  # gradient abs max cut
	init_w = 0.08  # uniform random from [-init_w, init_w]
	batch_size = 30  # mini-batch size
	init_lr = 0.001  # initial learning rate
	lr_hold = 1  # only used by SGD
	lr_decay = 0.6  # only used by SGD
	keep_prob = 1.0  # drop out rate
	improve_threshold = 0.996  # for early stopping
	patient_increase = 2.0  # for early stopping
	early_stop = True
	
	max_epoch = 60  # max number of epoch of training
	grad_noise = 0.0  # inject gradient noise?