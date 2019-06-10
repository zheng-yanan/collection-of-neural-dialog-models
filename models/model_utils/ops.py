import tensorflow as tf
import numpy as np


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
	kld = -0.5 * tf.reduce_sum(1 + (recog_logvar - prior_logvar)
							   - tf.div(tf.pow(prior_mu - recog_mu, 2), tf.exp(prior_logvar))
							   - tf.div(tf.exp(recog_logvar), tf.exp(prior_logvar)), 
							   reduction_indices=1)
	return kld


def norm_log_liklihood(x, mu, logvar):
	return -0.5 * tf.reduce_sum(
		tf.log(2*np.pi) + logvar + tf.div(tf.pow((x-mu), 2), tf.exp(logvar)), 
		reduction_indices=1)



def sample_gaussian(mu, logvar):
	epsilon = tf.random_normal(tf.shape(logvar), name="epsilon")
	std = tf.exp(0.5 * logvar)
	z= mu + tf.multiply(std, epsilon)
	return z


def get_bow(embedding, avg=False):
	# [batch_size, seq_len, embedding_dim]

	embedding_size = embedding.get_shape()[2].value
	if avg:
		return tf.reduce_mean(embedding, reduction_indices=[1]), embedding_size
	else:
		return tf.reduce_sum(embedding, reduction_indices=[1]), embedding_size


def get_rnn_encode(embedding, cell, length_mask=None, scope=None, reuse=None):
	# zero padding
	# [batch_size, seq_len, embedding_dim]

	with tf.variable_scope(scope, 'single-rnn-encoding', reuse=reuse):
		if length_mask is None:
			length_mask = tf.reduce_sum(
				tf.sign(tf.reduce_max(tf.abs(embedding), reduction_indices=2)),
				reduction_indices=1)
			length_mask = tf.to_int32(length_mask)
		_, encoded_input = tf.nn.dynamic_rnn(
			cell, 
			embedding, 
			sequence_length=length_mask, 
			dtype=tf.float32)

		return encoded_input, cell.state_size


def get_bi_rnn_encode(embedding, f_cell, b_cell, length_mask=None, scope=None, reuse=None):
	# zero padding
	# [batch_size, seq_len, embedding_dim]
	
	with tf.variable_scope(scope, 'bi-rnn-encoding', reuse=reuse):
		if length_mask is None:
			length_mask = tf.reduce_sum(
				tf.sign(tf.reduce_max(tf.abs(embedding), reduction_indices=2)),
				reduction_indices=1)
			length_mask = tf.to_int32(length_mask)

		_, states = tf.nn.bidirectional_dynamic_rnn(
			f_cell, 
			b_cell, 
			embedding, 
			sequence_length=length_mask, 
			dtype=tf.float32)

		encoded_input = tf.concat(states, 1)
		

		return encoded_input, f_cell.state_size + b_cell.state_size