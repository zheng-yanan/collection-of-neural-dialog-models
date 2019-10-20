## Collections of Neural Dialogue Models

A collection of Tensorflow implementation for several milestone neural dialogue generation works. It includes: 

 - The **Seq2Seq Model (SEQ2SEQ)** proposed in [**Sequence to Sequence Learning with Neural Networks**](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf).
 - The **Hierarchical Neural Network Model (HRED)** proposed in [**Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models**](https://arxiv.org/pdf/1507.04808.pdf).
 - The **Hierarchical Latent Variable Encoder-Decoder Model (VHRED)** presented in [**A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues**](https://arxiv.org/pdf/1605.06069.pdf).
 - The **Knowledge-Guided Conditional Variational Model (KGCVAE)** presented in [**Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders**](https://arxiv.org/pdf/1703.10960.pdf).
 
 
 ### Usage: 
 
First download GoogleNews word2vec embeddings from https://code.google.com/archive/p/word2vec/downloads and put it in the root directoy. The default setting use 300 dimension.
 
 To train a model: 
 
	python run_models.py --model <model_name> -- dataset <dataset_name>
where <model_name> can be **seq2seq, hred, vhred** or **kgcvae**, and <dataset_name> can be **dailydialog or switchboard**. It will save models to ./save/model_dir.

 To test an existing model: 
 
	python run_models.py --model <model_name> -- dataset <dataset_name> --forward True --test_path <model_dir>
where <model_dir> is in the format of **"run_<model_name>_<dataset_name>_others"**. A file containing predicted responses will be generated in **<model_dir>/test.txt** at the same time.

 To evaluate an existing model using automatic metrics: 
 
	python eval_utils.py --input_file <model_dir>/test.txt
It provides commonly used evaluation tools for dialog generation, including the **per-word perplexity, distinct-1, distinct-2, bleu-N, embedding-based metrics (average & greedy & extrema)**.


### Datasets

Two datasets are provided, respectively [**dailydialog**](https://arxiv.org/abs/1710.03957) and [**SwitchBoard**](http://compprag.christopherpotts.net/swda.html). Both datasets contains dialogue instances, along with rich meta information. Please refer to [daily_readme.txt](https://github.com/zheng-yanan/variational-neural-dialog-models/blob/master/data/dailydialog/ReadMe.txt) and [swda_readme.txt](https://github.com/zheng-yanan/variational-neural-dialog-models/blob/master/data/switchboard/ReadMe.txt) for details.

### Prerequisites
 - TensorFlow 1.4.0
 - Python 2.7

### Acknowledgements

Several code snippets are reused from [**NeuralDialog-CVAE**](https://github.com/snakeztc/NeuralDialog-CVAE). I appreciate all the authors who made their code public, which greatly facilitates this project. This repository would be continuously updated.
