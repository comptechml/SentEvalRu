# SentEvalRu

[Russian](https://github.com/comptechml/SentEvalRu/blob/master/READMEen.md)|[English](https://github.com/comptechml/SentEvalRu/blob/master/README.md)

This project was dedicated to creating a library for evaluating the quality of [sentence embeddings](https://en.wikipedia.org/wiki/Sentence_embedding) for the russian language. We assess their generalization power by using them as features on a broad and diverse set of tasks. SentEvalRu currently includes 17 NLP tasks. 

**Our goal** is to evaluate different algorithms of text representation with datasets in russian. This is the first approach of such task for the russian language.
We were inspired to create this library by [SentEval](https://arxiv.org/abs/1803.05449)[1].

This project was implemented in the context of winter school [ComptechNsk'19](http://comptech.nsk.su/), the idea of creating SentEvalRu belongs to  [MIPT](https://mipt.ru/english/)'s Neural Networks and Deep Learning Lab who develops artificial intelligence system  [iPavlov](https://ipavlov.ai/).

Project participants:
- [Mosolova Anna](https://github.com/anya-bel) (project manager)
- [Obukhova Alisa](https://github.com/lbdlbdlbdl) (technical writer)
- [Pauls Aleksey](https://github.com/AlekseyPauls) (engineer)
- [Stroganov Mikhail](https://github.com/MikhailStroganov) (engineer)
- [Timasova Ekaterina](https://github.com/KaterinaTimasova) (researcher)
- [Shugalevskaya Natalya](https://github.com/nshugalevkaia) (researcher)

### What tasks does it help to solve?
*Sentence embeddings* are used in a wide range of tasks where NLP systems are required. For example:
- intent classifier;
- [QA systems](https://en.wikipedia.org/wiki/Question_answering);
- [sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis);
- [machine translation](https://en.wikipedia.org/wiki/Machine_translation);
- [document clustering](https://en.wikipedia.org/wiki/Document_clustering).

Our tool helps to evaluate sentence embeddings, and that could be useful for everyone, who solves these tasks or analyses embeddings' quality for russian scientifically.

### Available models of text representation

Our project currently includes following models of text representation for the russian language:
- [Bert](https://arxiv.org/pdf/1810.04805.pdf) [2]
- [FastText](https://fasttext.cc/) [4]
- [FastText](https://fasttext.cc/)+[IDF](https://en.wikipedia.org/wiki/Tf–idf) [4] [5]
- [Skip-Thought](https://arxiv.org/abs/1506.06726) [6]

## Evaluation and tasks
There is no way to evaluate embeddings' quality direclty so we can only solve some NLP tasks using these embeddings and evaluate them depending on the results of these systems.

For example, we can use following tasks:
- sentiment analysis;
- named-entity recognition;
- topic modelling;
etc.

We suggest evaluating embeddings by means of these tasks:

|Tag| Task     	| Type                       | Description                       |
|----------------|-------------|---------------------------|--------------------------------|
|MRPC| [MRPC](https://github.com/Koziev/NLP_Datasets/tree/master/ParaphraseDetection/Data) | paraphrase detection | Detect whether one sentence is the paraphrase of another one |
|SST-3| [SST/dialog-2016](http://www.dialog-21.ru/evaluation/2016/sentiment/) | ternary sentiment analysis | Detect a text sentiment (positive (1), neutral (0), negative (-1))|
|SST-2| [SST/binary](http://study.mokoron.com/) | binary sentiment analysis | Detect a text sentiment (positive (1), negative (-1)) |
|TagCl| [Tags classifier](https://tatianashavrina.github.io/taiga_site/downloads) | tag classifier | Detect a tag of news from Interfax Corpus |
|ReadabilityCl| [Readability classifier](https://tatianashavrina.github.io/taiga_site/downloads) | Readability Classifier | Detect a readibility grade of a text (1-10) |
|PoemsCl| [Poems classifier](https://tatianashavrina.github.io/taiga_site/) | genre classifier | Detect poem's genre |
|ProzaCl| [Proza classifier](https://tatianashavrina.github.io/taiga_site/) | genre classifier | Detect prose's genre |
|TREC| [TREC](http://cogcomp.cs.illinois.edu/Data/QA/QC/) (translated) | question-type classification | Detect a type of a question (about entity, human, description, location etc.) |
|SICK| [SICK-E](http://clic.cimec.unitn.it/composes/sick.html) (translated) | natural language inference | Detect whether a second sentence is an entailment, a contradiction, or neutral of the first one) |
|STS| [STS](https://www.cs.york.ac.uk/semeval-2012/task6/) (translated) | semantic textual similarity | Detect a semantic similarity grade of two texts |

Futher information is available in **/data**

---

## Prerequisites

You should install all required modules before the start:

* Python 3 with [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/)
* [Pytorch](http://pytorch.org/)>=0.4
* [scikit-learn](http://scikit-learn.org/stable/index.html)>=0.18.0
* [TensorFlow](https://www.tensorflow.org/) >=1.12.0
* [Keras](https://keras.io/) >=2.2.4
* ...
We reccomend using [Anaconda](https://www.anaconda.com/distribution/) package or you can just run the following command:
```
pip3 install -r requirements.txt
```

## Setup
```
git init
git clone https://github.com/comptechml/SentEvalRu.git
cd SentEvalRu
```
You should store your datasets in */data*, and you could add your examples (new embeddings) to */examples*.

## Examples

Available tasks for russian are situated in  */examples*.

## How to use SentEval

To evaluate your sentence embeddings, SentEval requires that you implement two functions:

1. **prepare** (sees the whole dataset of each task and can thus construct the word vocabulary, the dictionary of word vectors etc)
2. **batcher** (transforms a batch of text sentences into sentence embeddings)


### 1.) prepare(params, samples) (optional)

*batcher* only sees one batch at a time while the *samples* argument of *prepare* contains all the sentences of a task.

```
prepare(params, samples)
```
* *params*: senteval parameters.
* *samples*: list of all sentences from the tranfer task.
* *output*: No output. Arguments stored in "params" can further be used by *batcher*.

*Example*: in bow.py, prepare is is used to build the vocabulary of words and construct the "params.word_vect* dictionary of word vectors.


### 2.) batcher(params, batch)
```
batcher(params, batch)
```
* *params*: senteval parameters.
* *batch*: numpy array of text sentences (of size params.batch_size)
* *output*: numpy array of sentence embeddings (of size params.batch_size)

*Example*: in bow.py, batcher is used to compute the mean of the word vectors for each sentence in the batch using params.word_vec. Use your own encoder in that function to encode sentences.

### 3.) evaluation on transfer tasks

After having implemented the batch and prepare function for your own sentence encoder,

1) to perform the actual evaluation, first import senteval and set its parameters:
```python
import senteval
params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
```

2) (optional) set the parameters of the classifier (when applicable):
```python
params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
```
You can choose **nhid=0** (Logistic Regression) or **nhid>0** (MLP) and define the parameters for training.

3) Create an instance of the class SE:
```python
se = senteval.engine.SE(params, batcher, prepare)
```

4) define the set of transfer tasks and run the evaluation:
```python
transfer_tasks = ['MR', 'SICKEntailment', 'STS14', 'STSBenchmark']
results = se.eval(transfer_tasks)
```
The current list of available tasks is:
```python
['SST2', 'SST3', 'MRPC', 'ReadabilityCl', 'TagCl', 'PoemsCl', 'ProzaCl', 'TREC', 'STS', 'SICK']
```

## SentEval parameters
Global parameters of SentEval:
```bash
# senteval parameters
task_path                   # path to SentEval datasets (required)
seed                        # seed
usepytorch                  # use cuda-pytorch (else scikit-learn) where possible
kfold                       # k-fold validation for MR/CR/SUB/MPQA.
```

Parameters of the classifier:
```bash
nhid:                       # number of hidden units (0: Logistic Regression, >0: MLP); Default nonlinearity: Tanh
optim:                      # optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
tenacity:                   # how many times dev acc does not increase before training stops
epoch_size:                 # each epoch corresponds to epoch_size pass on the train set
max_epoch:                  # max number of epoches
dropout:                    # dropout for MLP
```


## References

[1] A. Conneau, D. Kiela, [*SentEval: An Evaluation Toolkit for Universal Sentence Representations*](https://arxiv.org/abs/1803.05449)

[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
](https://arxiv.org/abs/1810.04805)

[3] Daniel Cer, Yinfei Yang, Sheng-yi Kong, ... [*Universal Sentence Encoder*](https://arxiv.org/abs/1803.11175)

[4] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [*Bag of Tricks for Efficient Text Classification*](https://arxiv.org/abs/1607.01759)

[5] Martin Klein, Michael L. Nelson, [*Approximating Document Frequency with Term Count Values*](https://arxiv.org/abs/0807.3755)

[6] Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, and Sanja Fidler, [*Skip-Thought Vectors*]((https://arxiv.org/abs/1506.06726))
