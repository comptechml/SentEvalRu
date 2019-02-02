# Datasets information

Read this in other languages: [English](README.md), [Русский](README.ru.md)

SentEval_Ru allows you to evaluate your sentence embeddings as features for the following tasks:

| Task     	| Type                         	| model |	
|----------	|------------------------------	|-----------|
| [MRPC](https://github.com/Koziev/NLP_Datasets/tree/master/ParaphraseDetection/Data) | paraphrase detection ||
| [SST/dialog-2016](http://www.dialog-21.ru/evaluation/2016/sentiment/) |third-labeled sentiment analysis  	||
| [SST/binary](http://study.mokoron.com/) |binary sentiment analysis  	||
|[_](https://tatianashavrina.github.io/taiga_site/downloads)| tags classifier ||
|[_](https://tatianashavrina.github.io/taiga_site/downloads)| readability classifier ||
| [TREC](http://cogcomp.cs.illinois.edu/Data/QA/QC/) (translated to Russian) | question-type classification 	|| 
| [SICK-E](http://clic.cimec.unitn.it/composes/sick.html) (translated to Russian) | natural language inference ||
---
In the folder with each task there are datasets presented in *.csv* format. Test datasets contain the following:

#### MRPC
Tab separated input files with `s1 | s2 | label` structure. (s1, s2 – sentences)

The system participating in this task should compute semantic similarity `s1` and `s2` are, returning a similarity score — 0 or 1.

#### SST/dialog-2016 
Tab separated input files with `id | sentence | label` structure. 

The system participating in this task should classify the *polarity* of a given `sentence` at the document — is it positive (1), negative (-1) or neutral (0).

#### SST/binary 
Tab separated input files with `sentence | label` structure.

The system participating in this task should classify the *polarity* of a given `sentence` at the document — is it positive (1) or negative (-1).

#### TREC
Tab separated input files with `sentence | label` structure.

The system participating in this task should classify what answer type have a question `sentence`.

See possible variations of `labels` at [paper](http://cogcomp.org/Data/QA/QC/definition.html)

Read more about [Learning Question Classifiers](http://aclweb.org/anthology/C02-1150)

*Example*: **Q**: What Canadian city has the largest population?, the hope is to classify this question as having
*answer type* **city**.

#### SICK-E
Tab separated input files with `s1 | s2 | label` structure. (s1, s2 – sentences)

The system participating in this task should compute *how similar* semantically `s1` and `s2` are, returning a similarity score in range [1..5].
