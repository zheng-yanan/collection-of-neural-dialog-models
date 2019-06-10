import argparse
from random import randint
import numpy as np
import gensim
import random

def read_data(input_path, repeat=10):

	file = open(input_path, "r")

	data = []
	block = []
	for line in file:
		if line != "\n":
			block.append(line.replace("\n", "").strip())
		else:
			if len(block) <= repeat+3:
				pass
			else:
				data.append(block)
			block = []

	# print(len(data))


	target_list = []
	sample_list = []

	for block in data:
		target = block[-(repeat+1)].replace("Target >> ", "")
		target_list.append(target)

		sample = block[-repeat:]
		sample = [item[12:] for item in sample]
		num = random.randint(0, len(sample) - 1)
		sample_list.append(sample[num])

	return sample_list, target_list


def distinct_1(lines):
	'''Computes the number of distinct words divided by the total number of words.

	Input:
	lines: List of strings.
	'''
	words = ' '.join(lines).split(' ')
	num_distinct_words = len(set(words))
	return float(num_distinct_words) / len(words)


def distinct_2(lines):
	'''Computes the number of distinct bigrams divided by the total number of words.

	Input:
	lines: List of strings.
	'''
	all_bigrams = []
	num_words = 0

	for line in lines:
		line_list = line.split(' ')
		num_words += len(line_list)
		bigrams = zip(line_list, line_list[1:])
		all_bigrams.extend(list(bigrams))

	return len(set(all_bigrams)) / float(num_words)


def avg_len(lines):
	'''Computes the average line length.

	Input:
	lines: List of strings.
	'''
	return(len([w for s in lines for w in s.strip().split()])/len(lines))


def bleu(target_lines, gt_lines, DEBUG=False):
	# https://cloud.tencent.com/developer/article/1042161

	'''Computes the average BLEU score.
	
	Input:
	target_lines: List of lines produced by the model.
	gt_lines: List of ground-truth lines corresponding to each line produced by the model.
	'''

	# This import is in here because it is really slow, so only do it if we have to.
	from nltk.translate.bleu_score import sentence_bleu

	assert len(target_lines) == len(gt_lines)

	avg_bleu_1 = 0
	avg_bleu_2 = 0
	avg_bleu_3 = 0
	avg_bleu_4 = 0

	avg_bleu = 0

	num_refs = len(gt_lines)
	for i in range(num_refs):

		reference = [gt_lines[i].lower().split()]
		candidate = target_lines[i].lower().split()
	
		bleu = sentence_bleu(reference, candidate, weights = (0.5, 0.5))
		bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
		bleu_2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
		bleu_3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
		bleu_4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))

		#print('Individual 1-gram: %f' % bleu_1)
		#print('Individual 2-gram: %f' % bleu_2)
		#print('Individual 3-gram: %f' % bleu_3)
		#print('Individual 4-gram: %f' % bleu_4)

		if DEBUG == 2:
			print('CAND: ',target_lines[i])
			print('GT	: ',gt_lines[0][i])
			print('BLEU:', bleu)

		avg_bleu += bleu

		avg_bleu_1 += bleu_1
		avg_bleu_2 += bleu_2
		avg_bleu_3 += bleu_3
		avg_bleu_4 += bleu_4


	avg_bleu = avg_bleu / len(target_lines)

	avg_bleu_1 = avg_bleu_1 / len(target_lines)
	avg_bleu_2 = avg_bleu_2 / len(target_lines)
	avg_bleu_3 = avg_bleu_3 / len(target_lines)
	avg_bleu_4 = avg_bleu_4 / len(target_lines)



	return(avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4, avg_bleu)





"""
Everything below this comment was borrowed from https://github.com/julianser/hed-dlg-truncated/blob/master/Evaluation/embedding_metrics.py
(with some slight modifications)

Word embedding based evaluation metrics for dialogue.

This method implements three evaluation metrics based on Word2Vec word embeddings, which compare a target utterance with a model utterance:
1) Computing cosine-similarity between the mean word embeddings of the target utterance and of the model utterance
2) Computing greedy meatching between word embeddings of target utterance and model utterance (Rus et al., 2012)
3) Computing word embedding extrema scores (Forgues et al., 2014)

We believe that these metrics are suitable for evaluating dialogue systems.

Example run:

		python embedding_metrics.py path_to_ground_truth.txt path_to_predictions.txt path_to_embeddings.bin

The script assumes one example per line (e.g. one dialogue or one sentence per line), where line n in 'path_to_ground_truth.txt' matches that of line n in 'path_to_predictions.txt'.

NOTE: The metrics are not symmetric w.r.t. the input sequences. 
			Therefore, DO NOT swap the ground truths with the predicted responses.

References:

A Comparison of Greedy and Optimal Assessment of Natural Language Student Input Word Similarity Metrics Using Word to Word Similarity Metrics. Vasile Rus, Mihai Lintean. 2012. Proceedings of the Seventh Workshop on Building Educational Applications Using NLP, NAACL 2012.

Bootstrapping Dialog Systems with Word Embeddings. G. Forgues, J. Pineau, J. Larcheveque, R. Tremblay. 2014. Workshop on Modern Machine Learning and Natural Language Processing, NIPS 2014.


"""

"""
def cosine_similarity(vector1, vector2):
	dot_product = 0.0
	normA = 0.0
	normB = 0.0
	for a, b in zip(vector1, vector2):
		dot_product += a * b
		normA += a ** 2
		normB += b ** 2
	if normA == 0.0 or normB == 0.0:
		return 0
	else:
		return round(dot_product / ((normA ** 0.5) * (normB ** 0.5)), 4)
"""

def greedy_score(sentence1, sentence2, w2v):
	'''
			:param sentence1: a list of words in a sentence, ['i', 'am', 'fine', '.']
			:param sentence2: a list of words in a sentence, ['i', 'am', 'fine', '.']
			:param w2v: dict of word to np.array
			:return: a scalar, it's value is in [0, 1]
	'''
	cosine_list = []
	word_count = 0
	dim = 300

	Y = []
	for tok in sentence2:
		if tok in w2v:
			Y.append(w2v[tok])
	if len(Y) == 0:
		return 0.0

	for tok in sentence1:
		if tok in w2v:
			i_vector = w2v[tok]
		else:
			continue

		word_count += 1
		cosine_list.append(max(w2v.cosine_similarities(i_vector, Y)))
		
	if word_count == 0:
		return 0.0

	score = float(sum(cosine_list) / word_count)
	return score


def greedy_match(sentence_list1, sentence_list2, w2v):

	greedy_match_score = []
	for i in range(len(sentence_list1)):

		sentence1 = sentence_list1[i].strip().split(' ')
		sentence2 = sentence_list2[i].strip().split(' ')

		greedy_1 = greedy_score(sentence1, sentence2, w2v)
		greedy_2 = greedy_score(sentence2, sentence1, w2v)

		greedy_match_score.append((greedy_1 + greedy_2) / 2)

		# print('{} line has done'.format(i))
		
	return np.mean(greedy_match_score), \
				 1.96 * np.std(greedy_match_score) / float(len(greedy_match_score)), \
				 np.std(greedy_match_score)



def extrema_score(r1, r2, w2v):
	scores = []

	for i in range(min(len(r1), len(r2))):
		tokens1 = r1[i].strip().split(" ")
		tokens2 = r2[i].strip().split(" ")
		X= []
		for tok in tokens1:
			if tok in w2v:
				X.append(w2v[tok])
		Y = []
		for tok in tokens2:
			if tok in w2v:
				Y.append(w2v[tok])

		# if none of the words have embeddings in ground truth, skip
		if np.linalg.norm(X) < 0.00000000001:
			continue

		# if none of the words have embeddings in response, count result as zero
		if np.linalg.norm(Y) < 0.00000000001:
			scores.append(0)
			continue

		xmax = np.max(X, 0)	# get positive max
		xmin = np.min(X,0)	# get abs of min
		xtrema = []
		for i in range(len(xmax)):
			if np.abs(xmin[i]) > xmax[i]:
				xtrema.append(xmin[i])
			else:
				xtrema.append(xmax[i])
		X = np.array(xtrema)	 # get extrema

		ymax = np.max(Y, 0)
		ymin = np.min(Y,0)
		ytrema = []
		for i in range(len(ymax)):
			if np.abs(ymin[i]) > ymax[i]:
				ytrema.append(ymin[i])
			else:
				ytrema.append(ymax[i])
		Y = np.array(ytrema)

		o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)

		scores.append(o)

	scores = np.asarray(scores)
	return np.mean(scores), 1.96*np.std(scores)/float(len(scores)), np.std(scores)


def average_embedding_score(r1, r2, w2v):
	# dim = int(w2v.dim())	# dimension of embeddings
	dim = 300
	scores = []

	for i in range(min(len(r1), len(r2))):
		tokens1 = r1[i].strip().split(" ")
		tokens2 = r2[i].strip().split(" ")
		X= np.zeros((dim,))
		for tok in tokens1:
			if tok in w2v:
				X+=w2v[tok]
		Y = np.zeros((dim,))
		for tok in tokens2:
			if tok in w2v:
				Y += w2v[tok]

		# if none of the words in ground truth have embeddings, skip
		if np.linalg.norm(X) < 0.00000000001:
			continue

		# if none of the words have embeddings in response, count result as zero
		if np.linalg.norm(Y) < 0.00000000001:
			scores.append(0)
			continue

		X = np.array(X)/np.linalg.norm(X)
		Y = np.array(Y)/np.linalg.norm(Y)
		o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)

		scores.append(o)

	scores = np.asarray(scores)
	return np.mean(scores), 1.96*np.std(scores)/float(len(scores)), np.std(scores)



if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()

	# parser.add_argument('--ground_truth', help="ground truth text file, one example per line")
	parser.add_argument('--input_file', help="predicted text file")
	parser.add_argument('--embeddings', default="GoogleNews-vectors-negative300.bin", help="embeddings bin file")
	args = parser.parse_args()

	"""
	with open(args.ground_truth, "rb") as gt_file:
		gt_texts = gt_file.readlines()
		gt_texts = [line.replace("\n", "").lower().strip() for line in gt_texts]

	with open(args.predicted, "rb") as pre_file:
		pre_texts = pre_file.readlines()
		pre_texts = [line.replace("\n", "").lower().strip() for line in pre_texts]
	"""


	pre_texts, gt_texts = read_data(args.input_file)

	print(len(pre_texts))
	print(len(gt_texts))

	print("\n")
	print("Results:")


	r = distinct_1(pre_texts)
	print("Distinct-1: %f" % r)

	r = distinct_2(pre_texts)
	print("Distinct-2: %f" % r)

	r = avg_len(pre_texts)
	print("Avg_len: %f" % r)

	bleu_1,bleu_2,bleu_3,bleu_4, avg_bleu = bleu(gt_texts, pre_texts)
	print('BLEU 1-gram: %f' % bleu_1)
	print('BLEU 2-gram: %f' % bleu_2)
	print('BLEU 3-gram: %f' % bleu_3)
	print('BLEU 4-gram: %f' % bleu_4)
	print('Cultimative BLEU 2-gram: %f' % avg_bleu)



	print("loading embeddings file...")
	# w2v = gensim.models.KeyedVectors.load_word2vec_format(args.embeddings, binary=True)
	# w2v = gensim.models.Word2Vec.load(args.embeddings)
	# print(type(w2v))
	# print(w2v.shape)
	import gensim.models.keyedvectors as word2vec
	w2v = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

	r = average_embedding_score(gt_texts, pre_texts, w2v)
	print("Embedding Average Score: %f +/- %f ( %f )" %(r[0], r[1], r[2]))

	r = greedy_match(gt_texts, pre_texts, w2v)
	print("Greedy Match Score: %f +/- %f ( %f )" %(r[0], r[1], r[2]))

	r = extrema_score(gt_texts, pre_texts, w2v)
	print("Extrema Score: %f +/- %f ( %f )" %(r[0], r[1], r[2]))
