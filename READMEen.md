# SentEvalRu

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
- FastText+IDF [4] [5]
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
