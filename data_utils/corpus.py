from collections import Counter
import nltk
import numpy as np
import pickle as pkl


# utterance object
utt_act_id = 0
utt_emot_id = 1


# DailyDialog Info:
# meta info: topic
# utt info: act, emotion, 
class DailyDialogCorpus(object):

	def __init__(self, corpus_path="data/dailydialog/dailydialog_split.plk", 
						max_vocab_cnt=20000, 
						word2vec=None, 
						word2vec_dim=None):

		self.word_vec_path = word2vec
		self.word2vec_dim = word2vec_dim
		self.word2vec = None

		data = pkl.load(open(corpus_path, "rb"))
		self.train_data = data["train"]
		self.valid_data = data["valid"]
		self.test_data = data["test"]

		print("DailyDialog Statistics: ")
		print("train data size: %d" % len(self.train_data))
		print("valid data size: %d" % len(self.valid_data))
		print("test data size: %d" % len(self.test_data))

		self.train_corpus = self.process(self.train_data)
		self.valid_corpus = self.process(self.valid_data)
		self.test_corpus = self.process(self.test_data)

		self.build_vocab(max_vocab_cnt)
		self.load_word2vec()
		print("Done loading DailyDialog corpus.")


	def process(self, data):

		new_meta = []
		new_dialog = []
		all_lenes = []
		new_utts = []
		bod_utt = ["<s>", "</s>"]

		for l in data:
			
			topic = l["topic"]
			dial = l["utts"]

			lower_utts = [
				(
					item["floor"],
					["<s>"] + nltk.WordPunctTokenizer().tokenize(item["text"].strip()) + ["</s>"],
					(item["act"], item["emot"])
				 ) 
				for item in dial]

			all_lenes.extend([len(u) for c, u, f in lower_utts])
			
			new_utts.extend([bod_utt] + [utt for floor, utt, feat in lower_utts])

			dialog = [(bod_utt, 0, None)] + [(utt, floor, feat) for floor, utt, feat in lower_utts]
			new_dialog.append(dialog)

			meta = (topic,)
			new_meta.append(meta)

		print("Max utt len %d, Min utt len %d, mean utt len %.2f" % \
			(np.max(all_lenes),np.min(all_lenes), float(np.mean(all_lenes))))

		return {"dialog": new_dialog, "meta": new_meta, "utts": new_utts}


	def build_vocab(self, max_vocab_cnt):

		print("\n building word vocabulary...")
		all_words = []
		for tokens in self.train_corpus["utts"]:
			all_words.extend(tokens)
		vocab_count = Counter(all_words).most_common()
		raw_vocab_size = len(vocab_count)
		discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
		vocab_count = vocab_count[0:max_vocab_cnt]
		self.vocab = ["<pad>", "<unk>"] + [t for t, cnt in vocab_count]
		self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
		self.unk_id = self.rev_vocab["<unk>"]
		
		print("Raw vocab size %d, vocab size %d, at cut_off frequent %d OOV rate %f"
			  % (raw_vocab_size, 
			  	len(vocab_count), 
			  	vocab_count[-1][1], 
			  	float(discard_wc) / len(all_words)))
		print("<pad> index %d" % self.rev_vocab["<pad>"])
		print("<unk> index %d" % self.rev_vocab["<unk>"])
		print("<s> index %d" % self.rev_vocab["<s>"])
		print("</s> index %d" % self.rev_vocab["</s>"])
		print("\n")


		print("\n building topic vocabulary...")
		all_topics = []
		for topic, in self.train_corpus["meta"]:
			all_topics.append(topic)
		self.topic_vocab = [t for t, cnt in Counter(all_topics).most_common()]
		self.rev_topic_vocab = {t: idx for idx, t in enumerate(self.topic_vocab)}
		print("%d topics in train data" % len(self.topic_vocab))
		print(self.topic_vocab)
		print("\n")

		print("\n building act vocabulary...")
		all_dialog_acts = []
		all_emots = []
		for dialog in self.train_corpus["dialog"]:
			all_dialog_acts.extend([feat[0] for floor, utt, feat in dialog if feat is not None])
			all_emots.extend([feat[1] for floor, utt, feat in dialog if feat is not None])

		self.dialog_act_vocab = [t for t, cnt in Counter(all_dialog_acts).most_common()]
		self.rev_dialog_act_vocab = {t: idx for idx, t in enumerate(self.dialog_act_vocab)}
		print("%d dialog acts in train data" % len(self.dialog_act_vocab))
		print(self.dialog_act_vocab)
		print("\n")

		print("\n building emotion vocabulary...")
		self.dialog_emot_vocab = [t for t, cnt in Counter(all_emots).most_common()]
		self.rev_dialog_emot_vocab = {t: idx for idx, t in enumerate(self.dialog_emot_vocab)}
		print("%d dialog emots in train data" % len(self.dialog_emot_vocab))
		print(self.dialog_emot_vocab)
		print("\n")


	def load_word2vec(self):
		if self.word_vec_path is None:
			print("no word2vec.")
			return
		with open(self.word_vec_path, "rb") as f:
			lines = f.readlines()
		raw_word2vec = {}
		for l in lines:
			w, vec = l.split(" ", 1)
			raw_word2vec[w] = vec
		
		self.word2vec = []
		oov_cnt = 0
		for v in self.vocab:
			str_vec = raw_word2vec.get(v, None)
			if str_vec is None:
				oov_cnt += 1
				vec = np.random.randn(self.word2vec_dim) * 0.1
			else:
				vec = np.fromstring(str_vec, sep=" ")
			self.word2vec.append(vec)
		print("word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab)))


	def get_utt_corpus(self):
		def _to_id_corpus(data):
			results = []
			for line in data:
				results.append([self.rev_vocab.get(t, self.unk_id) for t in line])
			return results
		
		id_train = _to_id_corpus(self.train_corpus["utts"])
		id_valid = _to_id_corpus(self.valid_corpus["utts"])
		id_test = _to_id_corpus(self.test_corpus["utts"])

		return {'train': id_train, 'valid': id_valid, 'test': id_test}


	def get_dialog_corpus(self):
		def _to_id_corpus(data):
			results = []
			for dialog in data:
				temp = []
				# convert utterance and feature into numeric numbers
				for utt, floor, feat in dialog:
					if feat is not None:
						id_feat = list(feat)
						id_feat[0] = self.rev_dialog_act_vocab[feat[0]]
						id_feat[1] = self.rev_dialog_emot_vocab[feat[1]]
					else:
						id_feat = None
					temp.append(([self.rev_vocab.get(t, self.unk_id) for t in utt], floor, id_feat))
				results.append(temp)
			return results

		id_train = _to_id_corpus(self.train_corpus["dialog"])
		id_valid = _to_id_corpus(self.valid_corpus["dialog"])
		id_test = _to_id_corpus(self.test_corpus["dialog"])

		return {'train': id_train, 'valid': id_valid, 'test': id_test}


	def get_meta_corpus(self):
		def _to_id_corpus(data):
			results = []
			for (topic,) in data:
				results.append((self.rev_topic_vocab[topic]))
			return results

		id_train = _to_id_corpus(self.train_corpus["meta"])
		id_valid = _to_id_corpus(self.valid_corpus["meta"])
		id_test = _to_id_corpus(self.test_corpus["meta"])
		
		return {'train': id_train, 'valid': id_valid, 'test': id_test}







# SWDA Info:
# meta info: topic/A profile/B profile
# utt info: act
class SWDADialogCorpus(object):

	dialog_act_id = 0

	def __init__(self, 
				 corpus_path="data/switchboard/full_swda_clean_42da_sentiment_dialog_corpus.p", 
				 max_vocab_cnt=20000, 
				 word2vec=None, 
				 word2vec_dim=None):

		self.word_vec_path = word2vec
		self.word2vec_dim = word2vec_dim
		self.word2vec = None

		data = pkl.load(open(corpus_path, "rb"))
		self.train_data = data["train"]
		self.valid_data = data["valid"]
		self.test_data = data["test"]

		print("**" * 30)
		print("SWDA Statistics: ")
		print("train data size: %d" % len(self.train_data))
		print("valid data size: %d" % len(self.valid_data))
		print("test data size: %d" % len(self.test_data))

		self.train_corpus = self.process(self.train_data)
		self.valid_corpus = self.process(self.valid_data)
		self.test_corpus = self.process(self.test_data)

		self.build_vocab(max_vocab_cnt)
		self.load_word2vec()

		print("\nDone loading SWDA corpus")


	def process(self, data):

		new_dialog = []
		new_meta = []
		new_utts = []
		bod_utt = ["<s>", "</s>"]
		all_lenes = []

		for l in data:
			lower_utts = [(
							floor, 
							["<s>"] + nltk.WordPunctTokenizer().tokenize(utt.lower()) + ["</s>"], 
							feat) for floor, utt, feat in l["utts"]]

			all_lenes.extend([len(u) for c, u, f in lower_utts])

			### prepare meta info
			a_age = float(l["A"]["age"])/100.0
			b_age = float(l["B"]["age"])/100.0
			a_edu = float(l["A"]["education"])/3.0
			b_edu = float(l["B"]["education"])/3.0
			vec_a_meta = [a_age, a_edu] + ([0, 1] if l["A"]["sex"] == "FEMALE" else [1, 0])
			vec_b_meta = [b_age, b_edu] + ([0, 1] if l["B"]["sex"] == "FEMALE" else [1, 0])
			meta = (vec_a_meta, vec_b_meta, l["topic"])
			new_meta.append(meta)


			dialog = [(bod_utt, 0, None)] + [(utt, int(floor=="B"), feat) for floor, utt, feat in lower_utts]
			new_dialog.append(dialog)


			new_utts.extend([bod_utt] + [utt for floor, utt, feat in lower_utts]) # list of list of utterance
			
			

		print("Max utt len %d, Min utt len %d, mean utt len %.2f" % \
			(np.max(all_lenes), np.min(all_lenes), float(np.mean(all_lenes))))

		return {"dialog": new_dialog, "meta": new_meta, "utts": new_utts}



	# build vocabulary for words/meta_info/utt_info
	def build_vocab(self, max_vocab_cnt):

		print("\n")
		print("building word vocabulary...")
		all_words = []
		for tokens in self.train_corpus["utts"]:
			all_words.extend(tokens)
		vocab_count = Counter(all_words).most_common()
		raw_vocab_size = len(vocab_count)
		discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
		vocab_count = vocab_count[0:max_vocab_cnt]

		self.vocab = ["<pad>", "<unk>"] + [t for t, cnt in vocab_count]
		self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
		self.unk_id = self.rev_vocab["<unk>"]

		print("raw vocab size %d, vocab size %d, at cut_off freq %d, with OOV rate %f"
			% (raw_vocab_size, 
			   len(vocab_count) + 2, 
			   vocab_count[-1][1], 
			   float(discard_wc) / len(all_words)))

		print("<pad> index %d" % self.rev_vocab["<pad>"])
		print("<unk> index %d" % self.rev_vocab["<unk>"])
		print("<s> index %d" % self.rev_vocab["<s>"])
		print("</s> index %d" % self.rev_vocab["</s>"])
		print("\n")

		
		print("\n")
		print("building topic vocabulary...")
		all_topics = []
		for a, b, topic in self.train_corpus["meta"]:
			all_topics.append(topic)
		self.topic_vocab = [t for t, cnt in Counter(all_topics).most_common()]
		self.rev_topic_vocab = {t: idx for idx, t in enumerate(self.topic_vocab)}
		print("%d topics in train data" % len(self.topic_vocab))
		print(self.topic_vocab)
		print("\n")


		print("\n")
		print("building act vocabulary...")
		all_dialog_acts = []
		for dialog in self.train_corpus["dialog"]:
			all_dialog_acts.extend([feat[self.dialog_act_id] for floor, utt, feat in dialog if feat is not None])
		self.dialog_act_vocab = [t for t, cnt in Counter(all_dialog_acts).most_common()]
		self.rev_dialog_act_vocab = {t: idx for idx, t in enumerate(self.dialog_act_vocab)}
		print("%d dialog acts in train data" % len(self.dialog_act_vocab))
		print(self.dialog_act_vocab)
		print("\n")


	def load_word2vec(self):
		if self.word_vec_path is None:
			print("\n no word2vec")
			return
		with open(self.word_vec_path, "rb") as f:
			lines = f.readlines()
		raw_word2vec = {}
		for l in lines:
			w, vec = l.split(" ", 1)
			raw_word2vec[w] = vec

		self.word2vec = []
		oov_cnt = 0
		for v in self.vocab:
			str_vec = raw_word2vec.get(v, None)
			if str_vec is None:
				oov_cnt += 1
				vec = np.random.randn(self.word2vec_dim) * 0.1
			else:
				vec = np.fromstring(str_vec, sep=" ")
			self.word2vec.append(vec)
		print("\n word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab)))


	def get_utt_corpus(self):
		def _to_id_corpus(data):
			results = []
			for line in data:
				results.append([self.rev_vocab.get(t, self.unk_id) for t in line])
			return results
		# convert the corpus into ID
		id_train = _to_id_corpus(self.train_corpus["utts"])
		id_valid = _to_id_corpus(self.valid_corpus["utts"])
		id_test = _to_id_corpus(self.test_corpus["utts"])
		return {'train': id_train, 'valid': id_valid, 'test': id_test}

	def get_dialog_corpus(self):
		def _to_id_corpus(data):
			results = []
			for dialog in data:
				temp = []
				for utt, floor, feat in dialog:
					if feat is not None:
						id_feat = list(feat)
						id_feat[self.dialog_act_id] = self.rev_dialog_act_vocab[feat[self.dialog_act_id]]
					else:
						id_feat = None
					temp.append(([self.rev_vocab.get(t, self.unk_id) for t in utt], floor, id_feat))
				results.append(temp)
			return results
		id_train = _to_id_corpus(self.train_corpus["dialog"])
		id_valid = _to_id_corpus(self.valid_corpus["dialog"])
		id_test = _to_id_corpus(self.test_corpus["dialog"])

		return {'train': id_train, 'valid': id_valid, 'test': id_test}

	def get_meta_corpus(self):
		def _to_id_corpus(data):
			results = []
			for m_meta, o_meta, topic in data:
				results.append((m_meta, o_meta, self.rev_topic_vocab[topic]))
			return results

		id_train = _to_id_corpus(self.train_corpus["meta"])
		id_valid = _to_id_corpus(self.valid_corpus["meta"])
		id_test = _to_id_corpus(self.test_corpus["meta"])
		
		return {'train': id_train, 'valid': id_valid, 'test': id_test}

