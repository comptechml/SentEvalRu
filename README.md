Read this in other languages: [English](README.md), [Русский](Readme.ru.md)

# What is it and why is it?

SentEvalRu is a library for evaluating the quality of sentence embeddings for russian texts. We assess their generalization power by using them as features on a broad and diverse set of *tasks*.

Our goal is to evaluate the algorithms, intended mostly for the English language, to Russian — prepare datasets and training models. As part of the project, we were limited in time, so it was decided to use the library [SentEval](https://arxiv.org/abs/1803.05449) as a benchmark.

## Development 

This project was made during the [ComptechNsk'19](http://comptech.nsk.su/) winter school. The customer was the [MIPT](https://mipt.ru/english/) laboratory [iPavlov](https://ipavlov.ai/) on behalf of Ivan Bondarenko.


## Tasks (data and models)
SentEval allows you to evaluate sentence embeddings as features for the following [tasks](\data\README.md)

More details on the tasks see **\examples**, more details about task data see **\data**

## System Requirements
**ATTENTION!** We advise you have more that 4GB RAM for training.

## Installation

Clone this repo and add your dataset to *data* folder, add your processing code to  *examples* folder.

## Usage examples

More information about implemented tasks for russian language see at **\examples**

## Usage

To evaluate your sentence embeddings by adding your data and a vectorization algorithm, SentEval requires that you implement two functions — *prepare* and *batcher*. After implementation you should set SentEval parameters. Then SentEval takes care of the evaluation on the transfer tasks using the embeddings as features.

### 1) prepare(params, samples) (optional)

**Prepare** sees the whole dataset of each task and can thus construct the word vocabulary, the dictionary of word vectors etc., depending on a task.

 ```python
 prepare(params, samples) 
 ```
 - *params*: SentEval parameters;
 - *samples*: list of all sentences from the task;
 - no *output*. Arguments stored in *params* can futher be used by **batcher**.

*Example*: in bow.py, *prepare* is is used to build the vocabulary of words and construct the "params.word_vect* dictionary of word vectors.

### 2) batcher(params, batch)

**Batcher** transforms a batch of text sentences into sentence embeddings. **Batcher** only sees one batch of sentences at a time.
 ```python
 batcher(params, batch)
 ```
- *params*: SentEval parameters;
- *batch*: numpy array of text sentences (of size params.batch_size)
- *output*: numpy array of sentence embeddings (of size params.batch_size)

*Example*: in bow.py, batcher is used to compute the mean of the word vectors for each sentence in the batch using params.word_vec. Use your own encoder in that function to encode sentences.

### 3) Evaluation on transfer tasks

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
tasks = ['MRPC', 'SST2']
results = se.eval(tasks)
```
The current list of available tasks is:
```python
['MRPC', 'SST2', 'SST3' ]
```

5) RUN! 
---
To get a proxy of the results while reducing computation time, we suggest the **prototyping config**, which will results in a 5 times speedup for classification tasks:
```python
params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
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

## Dependencies

This code is written in python. The dependencies are:

* Python 3 with [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/)
* [Pytorch](http://pytorch.org/)>=0.4
* [scikit-learn](http://scikit-learn.org/stable/index.html)>=0.18.0
* [TensorFlow](https://www.tensorflow.org/) >=1.12.0
* [Keras](https://keras.io/) >=2.2.4

You can download [Anaconda](https://www.anaconda.com/distribution/) that includes all necessary libraries. 
## References

### SentEval: An Evaluation Toolkit for Universal Sentence Representations

[1] A. Conneau, D. Kiela, [*SentEval: An Evaluation Toolkit for Universal Sentence Representations*](https://arxiv.org/abs/1803.05449)

```
@article{conneau2018senteval,
  title={SentEval: An Evaluation Toolkit for Universal Sentence Representations},
  author={Conneau, Alexis and Kiela, Douwe},
  journal={arXiv preprint arXiv:1803.05449},
  year={2018}
}
```