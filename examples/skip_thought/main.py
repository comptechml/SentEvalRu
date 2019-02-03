import os, time, re, sys, theano, argparse, errno
import skip_thought.vocab, skip_thought.train, skip_thought.tools
import numpy as np

from sklearn.externals import joblib
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from classifiers import MLPClassifier
from itertools import groupby

theano.config.floatX = 'float32'

def tokenize(s):
	"""
	This function takes a string, splits it into tokens and returns a line of tokens separated by space charachers.
	This tokenizer ignores any punctuation. It also ignores words that only contain one letter.
	:param s: input string to tokenize
	:return tokenized string 
	"""
	s = ''.join([ch for ch in s if ord(ch) != 769]) #removing accent signs
	s = ' '.join(re.findall(r'(?u)\b\w\w+\b', s))
	return s

def load_corpus(main_dir):
	"""
	This iterator takes a path to a directory, finds every file in its subdirectories,
	splits its lines into sentences and iterates over those sentences.
	We use the corpus to build a vocabulary 
	and then to train a model. You can store it like this:
	
	directory/A/file1
	directory/A/file2
	directory/B/file1
	etc.

	The iterator ignores lines that look like xml-markup or contain no space characters.

	:param main_dir: the directory which contains the training corpus split into sub-directories
	:yield the next line of the corpus 
	"""
	
	for sub in os.listdir(main_dir):
		if os.path.isdir('%s/%s' % (main_dir, sub)):
			for f in os.listdir('%s/%s' % (main_dir, sub)):
				print("File %s/%s/%s in process" % (main_dir, sub, f))
				with open("%s/%s/%s" % (main_dir, sub, f), encoding = 'utf-8') as f:
					for line in f.read().split('\n'):
						if line[:4] != '<doc' and line[:5] != '</doc' and line != '' and ' ' in line:
							for subline in re.split('[\.\?\!]+ *', line)[:-1]:
									yield tokenize(subline)

def save_vocab(corpus, dict):
	"""
	This function uses a corpus to build a vocabulary and stores it where the variable dict says.
	:param corpus: an iterable that yields sentences of a corpus.
	:param dict: path to where we should store the resulting vocabulary
	"""						
	X = load_corpus(corpus)
	worddict, wordcount = vocab.build_dictionary(X)
	vocab.save_dictionary(worddict, wordcount, dict)


def training_set(model, training_files):
	"""
	This procedure takes an iterable of text file paths, turns the found texts into a dataset, and returns it.
	:param model: the model that turns sentences into vectors.
	:param training_files: an iterable of text file paths:
	:return X_train: a 2-dimensional array of features
	:return y_train: a vector of class labels
	"""
	X_train = []
	y_train = []
	
	for counter_train, train in enumerate(training_files):
		print('Processing file %s, %d / %d. Training...' % (train, counter_train +1, len(training_files)))
		with open(train, encoding = 'utf-8') as f:
			train_lines = f.readlines()
		train_lines.pop(0)
		train_lines = [line[:-1].split('\t') for line in train_lines]
		assert len(list({len(line) for line in train_lines})) == 1
		for counter_word, (key, word) in enumerate(groupby(train_lines, key = lambda x: x[1])):
			print("Processing contexts of the word %s (%d / %d). Training..." % (key, counter_word + 1, len({x[1] for x in train_lines})))
			lines_word = list(word)
			context_word = [tokenize(context) for context_id, word, gold_sense_id, predict_sense_id, positions, context in lines_word]
			labels = [gold_sense_id for context_id, word, gold_sense_id, predict_sense_id, positions, context in lines_word]
			vectors = tools.encode(model, context_word)
			print('Encoding complete')
			X_train_0 = []
			X_train_1 = []
			for number, (label1, vector1) in enumerate(zip(labels, vectors)):
				for label2, vector2 in zip(labels, vectors):
					if not np.array_equal(vector1, vector2):
						if label1 == label2:
							X_train_1.append(np.hstack((vector1, vector2)))
						else:
							X_train_0.append(np.hstack((vector1, vector2)))
			
			if len(X_train_0) > 0:
				X_train_0 = np.vstack(X_train_0)
			if len(X_train_1) > 0:
				X_train_1 = np.vstack(X_train_1)
			if len(X_train_0) > 0 and len(X_train_1) > 0:
				X_train_local = np.vstack((X_train_0, X_train_1))
				X_train.append(X_train_local)
				
				y_train_0 = np.zeros(X_train_0.shape[0])
				y_train_1 = np.ones(X_train_1.shape[0])
				y_train_local = np.hstack((y_train_0, y_train_1))
				y_train.append(y_train_local)
			elif len(X_train_0) > 0:
				X_train_local = X_train_0
				X_train.append(X_train_local)
				y_train_local = np.zeros(X_train_0.shape[0])
				y_train.append(y_train_local)
			elif len(X_train_1) > 0:
				X_train_local = X_train_1
				X_train.append(X_train_local)
				y_train_local = np.ones(X_train_1.shape[0])
				y_train.append(y_train_local)
			
			print('Word processing complete')

	X_train = np.vstack(X_train)
	y_train = np.hstack(y_train)
	
	if X_train.shape[0] != y_train.shape[0]:
		raise Exception("X_train and y_train have inconsistent shapes: %s and %s" % (X_train.shape, y_train.shape))
	if len(X_train.shape) != 2:
		raise Exception("X_train is expected to have 2 dimensions, but it has %d" % (len(X_train.shape)))
	if len(y_train.shape) != 1:
		raise Exception("y_train is expected to have 1 dimension, but it has %d" % (len(X_train.shape)))
	
	return X_train, y_train

def create_metric(X_train, y_train):
	"""
	This function trains an MLP Classifier to tell sentences with close meanings from other sentences.
	It returns a function which estimates how close vectors of 2 sentences are.
	"""
	if X_train.shape[0] != y_train.shape[0]:
		raise Exception("X_train and y_train have inconsistent shapes: %s and %s" % (X_train.shape, y_train.shape))
	if len(X_train.shape) != 2:
		raise Exception("X_train is expected to have 2 dimensions, but it has %d" % (len(X_train.shape)))
	if len(y_train.shape) != 1:
		raise Exception("y_train is expected to have 1 dimension, but it has %d" % (len(X_train.shape)))
	if not set(y_train).issubset({0, 1, 0.0, 1.0}):
		if len(set(y_train)) > 5:
			raise Exception("labels 0.0 and 1.0 expected, %d different labels found" % len(set(y_train)))
		else:
			raise Exception("labels 0.0 and 1.0 expected, %s found" % set(y_train))
	
	clf = MLPClassifier(verbose = True, hidden_layer_sizes = (100, 100, 100), validation_fraction = 0.2).fit(X_train, y_train)
	metric = lambda vector1, vector2: -clf.predict_log_proba(np.hstack((vector1, vector2)).reshape(1, -1))[0][1]
	joblib.dump(clf, 'clf.dat')
	return metric

def test_set(test_file, model, metric, output_file):
	"""
	This function takes a path to a test file and tries to clusterize it using a metric.
	If the test file contains gold standard ids, we also evaluate the ARI.
	:param test_file: a string that contains a path.
	:param model: the model that tools.encode uses to turns sentences into vectors
	:param metric: a callable that takes two vectors and returns the distance between them.
	:param output_file: a path to the file where we store the result of the clustering
	"""
	
	with open(test_file, encoding = 'utf-8') as f:
		test_lines = f.readlines()
	test_lines.pop(0)
	test_lines = [line[:-1].split('\t') for line in test_lines]
	
	if {len(line) for line in test_lines} != {6}:
		raise Exception("Unexpected length of some lines in the test file: only lines with length 6 expected, lines with lengths %s found" % {len(line) for line in test_lines})
	
	result = 'context_id\tword\tgold_sense_id\tpredict_sense_id\tpositions\tcontext'
	
	for counter, (key, word) in enumerate(groupby(test_lines, key = lambda x: x[1])):
		print("Processing contexts of the word %s, %d / %d. Testing..." % (key, counter + 1, len({x[1] for x in test_lines})))
		lines_word = list(word)
		context_word = [tokenize(context) for context_id, word, gold_sense_id, predict_sense_id, positions, context in lines_word]
		vectors = tools.encode(model, context_word)
		clst = DBSCAN(metric = metric).fit(vectors)
		predict = [str(label) for label in clst.labels_]
		output = ['\t'.join([context_id, word, gold_sense_id, predict[counter], positions, context]) for counter, (context_id, word, gold_sense_id, predict_sense_id, positions, context) in enumerate(lines_word)]
		output = '\n'.join(output)
		result += '\n' + output
		f = open(output_file, 'w', encoding = 'utf-8')
		f.write(result)
		f.close()
		print('Output stored to', output_file)

	if {gold_sense_id for context_id, word, gold_sense_id, predict_sence_id, positions, context in test_lines} != '':
		print("Gold sence id detected. We can evaluate the quality.")
		from evaluate import evaluate
		evaluate(output_file)

def get_args():
	"""
	This function returns parsed arguments the file has been executed with.
	:return a argparse.Namespace object
	"""
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dictionary', default = 'dict.pkl', help = 'Path to the dictionary file. If the path does not exist, we train a new dictionary and store it there.')
	parser.add_argument('-nv', '--new_vocab', action = 'store_true', help = 'Use this argument to train a new vocabulary.')
	parser.add_argument('-ov', '--old_vocab', action = 'store_true', help = 'Use this argument to load an existing vocabulary or raise an error.')
	parser.add_argument('-m', '--model', default = 'models/wiki_model.dat.npz', help = 'Path to your model. If the path does not exist, we train a new model and store it there.')
	parser.add_argument('-nm', '--new_model', action = 'store_true', help = 'Use this argument to train a new model.')
	parser.add_argument('-om', '--old_model', action = 'store_true', help = 'Use this argument to load an existing model or raise en error.')
	parser.add_argument('-w2v', '--word2vec', default = 'word2vec.w2v', help = 'Path to your word2vec file.')
	parser.add_argument('-t', '--text', action = 'store_true', help = 'Use this argument to specify that the word2vec file is in text format (i.e., not binary).')
	parser.add_argument('-c', '--corpus', default = 'wiki-ru', help = 'The directory which contains texts to train the model.')
	parser.add_argument('-tr', '--train', nargs = '*', default = ['data/main/wiki-wiki/train.csv'], help = 'The training dataset.')
	parser.add_argument('-ts', '--test', default = 'data/main/wiki-wiki/test.csv', help = 'The training dataset.')
	parser.add_argument('-o', '--output', default = 'output.csv', help = 'The output file to store the result.')
	namespace = parser.parse_args()
	
	for path in namespace.train:
		if not os.path.exists(path):
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
	
	if not os.path.exists(namespace.test):
		raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), namespace.test)
	
	if namespace.new_vocab and namespace.old_vocab:
		raise Exception("You cannot use arguments old_vocab and new_vocab at the same time")
	
	if namespace.new_model and namespace.old_model:
		raise Exception("You cannot use arguments old_model and new_model at the same time")

	return namespace

def main():
	args = get_args()
	
	if args.new_vocab:
		save_vocab(args.corpus, args.dict)
	elif args.old_vocab:
		if not os.path.exists(args.dictionary):
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.dictionary)
	elif not os.path.exists:
		save_vocab(args.corpus, args.dict)
	
	if args.new_model:
		train.trainer(list(load_corpus(args.wiki)), saveto = args.model)
	elif args.old_model:
		if not os.path.exists(args.model):
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.model)
	elif not os.path.exists(args.model):
		train.trainer(list(load_corpus(args.wiki)), saveto = args.model)
	
	print("Loading the pre-trained model...")
	embed_map = tools.load_googlenews_vectors(args.word2vec, not args.text)
	model = tools.load_model(args.model, args.dictionary, args.word2vec, embed_map)
	
	print("Creating a dataset")
	X_train, y_train = training_set(args.train, model)
	
	print('Fitting a metric')
	metric = create_metric(X_train, y_train)
	test_set(args.test, model, metric, args.output)
	
	
if __name__ == '__main__':
	main()
