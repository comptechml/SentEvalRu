# Datasets information

Read this in other languages: [English](README.md), [Русский](README.ru.md)

SentEval_Ru allows you to evaluate your sentence embeddings as features for the following tasks:

| Task     	| Type                         	| #train 	| #test 	|
|----------	|------------------------------	|-----------:|:----------:|
| [MRPC](https://github.com/Koziev/NLP_Datasets/tree/master/ParaphraseDetection/Data) | paraphrase detection  | ?k | ?k 
| [SST](http://www.dialog-21.ru/evaluation/2016/sentiment/) |third-labeled sentiment analysis  	| ?k     	| ?k   
| [SST](http://study.mokoron.com/) |binary sentiment analysis  	| ?k     	| ?k   

---
In the folder with each task there are datasets presented in *.csv* format. Test datasets contain the following:

#### MRPC
Tab separated input files with `s1 | s2 | label` structure. (s1, s2 – sentences)

The system participating in this task should compute *how similar* `s1` and `s2` are, returning a similarity score.

#### SST/dialog-2016 
Tab separated input files with `id | sentence | label` structure. (s – sentence)

The system participating in this task should classify the *polarity* of a given `sentence` at the document — is it positive, negative or neutral.

#### SST/binary 
Tab separated input files with `sentence | label` structure. (s – sentence)

The system participating in this task should classify the *polarity* of a given `sentence` at the document — is it positive or negative.

