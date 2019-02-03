# Что это и зачем оно нужно?

SentEvalRu это библиотека для оценки качества [эмбеддингов предложений](https://en.wikipedia.org/wiki/Sentence_embedding)  для русских текстов. С её помощью мы можем оценить обобщающую возможность эмбеддингов, используя их как признаки (features) для разнообразных *задач*.

**Наша цель** — оценить алгоритмы векторизации текстов, предназначенные в основном для английского языка, для подготовленных русских наборов данных. Мы не нашли готового решения для русского языка, и решили написать своё.

### Эмбеддинги предложений
*Эмбеддинги предложений* используются везде, где решаются или используются задачи компьютерной лингвистики, например:
- встраивание в чат-боты для классификации намерения пользователей (intent classifier);
- использование в вопросно-ответных системах для поиска ответа на вопрос пользователя наиболее близкий по смыслу;
- встраивание в парсеры веб-контента для анализа тональности текста (sentiment analyses);
- задачи машинного перевода;
- задачи [кластеризации документов](https://ru.wikipedia.org/wiki/Кластеризация_документов).

Наш инструмент помогает оценивать эмбеддинги предложений, что может быть полезным для каждого, кто решает одну из вышеперечисленных задач или исследует качество эмбеддингов для русского языка с научной точки зрения.

### USE и BERT
Исследователи из компании Google придумали ряд интересных алгоритмов
дистрибутивной семантики, среди которых стоит отметить [USE (Universal
Sentence Encoder)](https://arxiv.org/pdf/1803.11175.pdf) и [Bert](https://arxiv.org/pdf/1810.04805.pdf). 

Так как мы были ограниченны по времени, то было принято решение использовать библиотеку [SentEval](https://arxiv.org/abs/1803.05449) как эталон.

## Разработка 

Этот проек был разработан в рамках зимней школы [ComptechNsk'19](http://comptech.nsk.su/). Заказчиком была лаборатория [iPavlov](https://ipavlov.ai/) из [МФТИ](https://mipt.ru/english/)  от имени Ивана Бондаренко.

Список участников:
- Иван Бондаренко (куратор проекта)
- Мосолова Анна
- Обухова Алиса (технический писатель)
- Паульс Алексей
- Строганов Михаил
- Тимасова Екатерина
- Шугалевская Наталья

## Оценка качества и Задачи 
Определить качество эмбеддинга можно лишь косвенно, например, в следующих задачах:
- анализ тональности текста;
- распознавание именованных сущностей;
- тематическое моделирование и т.д.
  
SentEval позволяет вам оценить эмбеддинги для предложений как признаков для следующих [задач](\data\README.md):
| Задача     	| Тип                         	|
|----------	|------------------------------	|
| [MRPC](https://github.com/Koziev/NLP_Datasets/tree/master/ParaphraseDetection/Data) | paraphrase detection | |
| [SST/dialog-2016](http://www.dialog-21.ru/evaluation/2016/sentiment/) |third-labeled sentiment analysis  	||
| [SST/binary](http://study.mokoron.com/) |binary sentiment analysis  	||
|[Tags classifier](https://tatianashavrina.github.io/taiga_site/downloads)| tags classifier ||
|[Readability classifier](https://tatianashavrina.github.io/taiga_site/downloads)| readability classifier ||
|[Poems classifier]()| tag classifier||
|[Proza classifier]()|tag classifier||
|[Genre classification]()|tag classifier||
| [TREC](http://cogcomp.cs.illinois.edu/Data/QA/QC/) (переведенный) | question-type classification ||
| [SICK-E](http://clic.cimec.unitn.it/composes/sick.html) (переведенный) | natural language inference ||
| [STS](https://www.cs.york.ac.uk/semeval-2012/task6/) (переведенный)| semantic textual similarity||

Больше деталей о моделях, использованных для решения задач, вы можете найти в **\examples**, больше деталей о данных вы можете найти в **\data**

---
## Системные требования
**Внимание!** Мы советуем вам иметь больше 4 ГБ оперативной памяти для тренировки модели skip_thought.

## Установка

Склонируйте этот репозиторий и добавьте ваши данные в папку *data*, добавьте ваш код в папку  *examples*.

## Примеры использования

Больше информации о реализованых задачах для русского языка смотрите в **\examples**

## Использование

Чтобы оценить ваши эмбеддинги для предложений, добавляя ваши данные и алгоритм векторизации, SentEval требует, чтобы вы реализовали две функции: *prepare* and *batcher*. После реализации вы должны задатьпараметры SentEval. Далее SentEval позаботится об оценке задач transfer learning, используя эмбеддинги как признаки.

### 1) prepare(params, samples) (optional)

**Prepare** просматривает весь датасет для каждой из задач и составляет словарь слов или векторов слов, в зависимости от ваших задач.

 ```python
 prepare(params, samples) 
 ```
 - *params*: параметры SentEval;
 - *samples*: список всех предложений из задачи;
 - нет *output*. Аргументы, собранные в *params* могут быть использованы в **batcher**.

*Example*: в bow.py, *prepare* используется для создания словаря слов и конструирования "params.word_vect* словаря векторов слов.

### 2) batcher(params, batch)

**Batcher** преобразует  партию(batch) текстовых предложений в  эмбеддинги слов. **Batcher** просматривает только одны партию за раз.
 ```python
 batcher(params, batch)
 ```
- *params*: параметры SentEval;
- *batch*: numpy массив текстовых предложений (размера params.batch_size)
- *output*: numpy массив эмбеддингов предложений (размера params.batch_size)

*Example*: в bow.py, batcher используется для вычисления усреднения векторов для словдля каждого предложения в партии используя params.word_vec. Используя свой собственный кодировщик для кодировки предложений.

### 3) Оценка на transfer tasks

После реализации the batch and prepare function for your own sentence encoder,

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

