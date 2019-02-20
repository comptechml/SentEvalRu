
# SentEvalRu

В рамках данного проекта была разработана библиотека для оценки качества [эмбеддингов предложений](https://en.wikipedia.org/wiki/Sentence_embedding) русских текстов. С её помощью мы можем оценить обобщающую способность эмбеддингов, используя их как признаки (features) для решения разнообразных *задач компьютерной лингвистики*.

**Наша цель** — оценить алгоритмы векторизации текстов, предназначенные в основном для английского языка, на подготовленных русскоязычных наборах данных. Мы не нашли готового решения для русского языка, и решили написать своё.

Так как мы были ограничены по времени, было принято решение использовать библиотеку [SentEval](https://arxiv.org/abs/1803.05449) [1] как эталон.


Этот проек был разработан в рамках зимней школы [ComptechNsk'19](http://comptech.nsk.su/). 
Заказчиком была лаборатория нейронных систем и глубокого обучения [МФТИ](https://mipt.ru/english/), реализующая проект по созданию разговорного искусственного интеллекта  [iPavlov](https://ipavlov.ai/).


Список участников:
- Мосолова Анна (руководитель группы)
- Обухова Алиса (технический писатель)
- Паульс Алексей (инженер)
- Строганов Михаил (инженер)
- Тимасова Екатерина (исследователь)
- Шугалевская Наталья (исследователь)

### Какие задачи поможет решать
*Эмбеддинги предложений* используются везде, где решаются или используются задачи компьютерной лингвистики, например:
- классификация намерений пользователей в чат-ботах (intent classifier);
- использование в вопросно-ответных системах для поиска ответа на вопрос пользователя наиболее близкий по смыслу;
- встраивание в парсеры веб-контента для анализа тональности текста (sentiment analyses);
- задачи машинного перевода;
- задачи [кластеризации документов](https://ru.wikipedia.org/wiki/Кластеризация_документов).

Наш инструмент помогает оценивать эмбеддинги предложений, что может быть полезным для каждого, кто решает одну из вышеперечисленных задач или исследует качество эмбеддингов для русского языка с научной точки зрения.

### Проверяемые модели векторизации текстов

Мы включили примеры использования следующих моделей для русского языка:
- [Bert](https://arxiv.org/pdf/1810.04805.pdf) [2]
- [FastText](https://fasttext.cc/) [4]
- FastText+IDF [4] [5]
- [Skip-Thought](https://arxiv.org/abs/1506.06726) [6]

## Оценка качества и Задачи 
Определить качество эмбеддинга можно лишь косвенно, оценивая, насколько эффективно решается ряд задач компьютерной лингвистики с использованием этой модели эмбеддингов. 

Например, в следующих задачах:
- анализ тональности текста;
- распознавание именованных сущностей;
- тематическое моделирование и т.д.

SentEvalRu позволяет вам оценить эмбеддинги предложений для следующих задач:

|Тэг| Задача     	| Тип                       | Описание                       |
|----------------|-------------|---------------------------|--------------------------------|
|MRPC| [MRPC](https://github.com/Koziev/NLP_Datasets/tree/master/ParaphraseDetection/Data) | Обнаружение перефраза|Является ли одно предложение перефразом второго|
|SST-3| [SST/dialog-2016](http://www.dialog-21.ru/evaluation/2016/sentiment/) |Трехклассовый анализ тональности|Определение тональности текста по трем классам: позитивный (1), нейтральный (0), негативный (-1)|
|SST-2| [SST/binary](http://study.mokoron.com/) | Двухклассовый анализ тональности |Определение тональности текста по двум классам: позитивный (1), негативный (-1)|
|TagCl| [Tags classifier](https://tatianashavrina.github.io/taiga_site/downloads) | Классификатор тэгов |Определение тэга новости из корпуса новостей Interfax|
|ReadabilityCl| [Readability classifier](https://tatianashavrina.github.io/taiga_site/downloads) | Классификатор сложности текста | Определение сложности чтения и восприятия текста от 0 до 10 |
|PoemsCl| [Poems classifier](https://tatianashavrina.github.io/taiga_site/) | Классификатор жанра | Определение жанра стихотворения |
|ProzaCl| [Proza classifier](https://tatianashavrina.github.io/taiga_site/) |Классификатор жанра | Определение жанра прозаических произведений |
|TREC| [TREC](http://cogcomp.cs.illinois.edu/Data/QA/QC/) (переведенный) | Классификация по типу вопроса | Определение типа вопроса (о сущности, о человеке, об описании, о месте и т.д.) |
|SICK| [SICK-E](http://clic.cimec.unitn.it/composes/sick.html) (переведенный) | Логический вывод естественного языка | Определение, является ли второе предложение логическим выводом, логическим противоречием или нейтральным по отношению к первому |
|STS| [STS](https://www.cs.york.ac.uk/semeval-2012/task6/) (переведенный) | Семантиическое сходство текстов | Оценка схожести текстов от 0 до 5 |

Более подробную информацию о данных вы можете найти в **/data**

---
## Системные требования

**Внимание!** Внимание! Не менее 4 ГБ оперативной памяти требуется для тренировки моделей. Для модели skip_thought требуется более 4 ГБ.

## Подготовка к установке

Перед началом использования установите следующие системы и компоненты:

* Python 3 with [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/)
* [Pytorch](http://pytorch.org/)>=0.4
* [scikit-learn](http://scikit-learn.org/stable/index.html)>=0.18.0
* [TensorFlow](https://www.tensorflow.org/) >=1.12.0
* [Keras](https://keras.io/) >=2.2.4

Вы можете скачать [Anaconda](https://www.anaconda.com/distribution/), которая включает все эти зависимости.

Более детальный список зависимостей приведён в файле ``requirements.txt``. 

## Установка

Склонируйте этот репозиторий и добавьте ваши данные в папку */data*, добавьте ваш код в папку */examples*.

## Примеры использования

Больше информации о реализованых задачах для русского языка смотрите в */examples*

## Использование

Чтобы оценить ваши эмбеддинги предложений, добавляя ваши данные и алгоритм векторизации, SentEvalRu требует, чтобы вы реализовали две функции: *prepare* and *batcher*. После реализации вы должны задать параметры SentEvalRu. Далее SentEvalRu позаботится об оценке задач, переданных на обучение, используя эмбеддинги как признаки.

### 1) prepare(params, samples) (optional)

**Prepare** просматривает весь датасет для каждой из задач и составляет словарь слов или векторов слов, в зависимости от ваших задач.

 ```python
 prepare(params, samples) 
 ```
 - *params*: параметры SentEvalRu;
 - *samples*: список всех предложений задачи;
 - нет *output*. Аргументы, собранные в *params* могут быть использованы в **batcher**.

*Example*: в bow.py, *prepare* используется для создания словаря слов и конструирования "params.word_vect* словаря векторов слов.

### 2) batcher(params, batch)

**Batcher** преобразует  партию (batch) текстовых предложений в  эмбеддинги слов. **Batcher** просматривает только одну партию за раз.
 ```python
 batcher(params, batch)
 ```
- *params*: параметры SentEvalRu;
- *batch*: numpy массив текстовых предложений (размера params.batch_size)
- *output*: numpy массив эмбеддингов предложений (размера params.batch_size)

*Пример*: в bow.py, batcher используется для вычисления усреднения векторов для слов в каждом батче, используя params.word_vec. 

### 3) Оценка задач

После реализации функций batch и prepare для вашего кодировщика предложений, выполните следующие шаги:

1) Чтобы выполнить фактическую оценку, сначала импортируйте SentEvalRu и задайте его параметры:
```python
import SentEvalRu
params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
```

2) (опционально) задайте параметры классификатора:
```python
params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
```
Вы можете выбрать **nhid=0** (Логистическая регрессия) или **nhid>0** (MLP) и определить параметры для тренировки (training).

3) Создайте экземпляр класса SE:
```python
se = SentEvalRu.engine.SE(params, batcher, prepare)
```

4) Определите набор Задач:
```python
tasks = ['MRPC', 'SST2']
results = se.eval(tasks)
```
Текущий список доступных задач:
```python
[SST2, SST3, MRPC, ReadabilityCl, TagCl, PoemsCl, TREC, STS, SICK ]
```
5) Запускайте!
---

## SentEvalRu параметры
Глобальные параметры SentEvalRu:
```bash
# SentEvalRu parameters
task_path                   # path to SentEvalRu datasets (required)
seed                        # seed
usepytorch                  # use cuda-pytorch (else scikit-learn) where possible
kfold                       # k-fold validation for MR/CR/SUB/MPQA.
```

Параметры классификатора:
```bash
nhid:                       # number of hidden units (0: Logistic Regression, >0: MLP); Default nonlinearity: Tanh
optim:                      # optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
tenacity:                   # how many times dev acc does not increase before training stops
epoch_size:                 # each epoch corresponds to epoch_size pass on the train set
max_epoch:                  # max number of epoches
dropout:                    # dropout for MLP
```

## Ссылки на научные публикации

[1] A. Conneau, D. Kiela, [*SentEval: An Evaluation Toolkit for Universal Sentence Representations*](https://arxiv.org/abs/1803.05449)

[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
](https://arxiv.org/abs/1810.04805)

[3] Daniel Cer, Yinfei Yang, Sheng-yi Kong, ... [*Universal Sentence Encoder*](https://arxiv.org/abs/1803.11175)

[4] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [*Bag of Tricks for Efficient Text Classification*](https://arxiv.org/abs/1607.01759)

[5] Martin Klein, Michael L. Nelson, [*Approximating Document Frequency with Term Count Values*](https://arxiv.org/abs/0807.3755)

[6] Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, and Sanja Fidler, [*Skip-Thought Vectors*]((https://arxiv.org/abs/1506.06726))
