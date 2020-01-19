# Forschungspraktikum Political Polarization - Propaganda Group <!-- omit in toc -->


Based on the Task SI (Span Identification) of the [Semeval2020 Task 11](https://propaganda.qcri.org/semeval2020-task11/index.html), this work contains the software implementation of the machine learning based propaganda detector prototype, as authored by the team *stayfoolish*.

The implementation has achieved an accuracy of xx%.


- [The task](#the-task)
- [Approach](#approach)
- [Methodology](#methodology)
  - [Designing the learning features](#designing-the-learning-features)
  - [Modeling the text as vectors](#modeling-the-text-as-vectors)
  - [Modeling the features as vectors](#modeling-the-features-as-vectors)
  - [Training an LSTM neural network on propagandistic and non propagandistic sentences](#training-an-lstm-neural-network-on-propagandistic-and-non-propagandistic-sentences)
- [Installation and usage](#installation-and-usage)
    - [Scaling](#scaling)
  - [Credits](#credits)

## The task
The challenge of the Task SI is to build a model capable of detecting spans of text that contain one of [these 18 propaganda techniques](https://propaganda.qcri.org/annotations/definitions.html), based on a training corpus of 371 news articles.

## Approach
In order to build an intelligence capable of detecting spans that contain propaganda, it was assumed that a deep knowledge of the textual features of propaganda will be necessary.

News articles that use propaganda as one of their persuasive techniques are assumed to use similar elements. These defining features should distinguish a news article that was written in the intent of spreading propaganda, and one that wasn't.

Elements that we came with include:

+ **Entities**: `ner_*` It was assumed that many spans of text that use one the propaganda techniques should contain some recognized entity. As an example, if there is one, it is very easy for a human to spot a propaganda technique around a sentence where the president Trump is being mentioned. 
  
  
  Entities that we focused on in the context of this project are *persons*, *organizations* and *countries*. These tend to be used the most with propaganda texts.

  An initial representation of spotted persons, organizations and countries in the spans that are labeled as propaganda reveal that the most mentioned entities are Trump, the Church and the US respectively.

+ **Question marks**: `qm_size` The use of questions is also assumed to be a shared feature among spans that contain propaganda. Some rethorical questions are meant to induce an emotional response around a subject rather than to simply seek knowledge.
  
+ **Exclamation marks**: `em_size` As a broad indicator of emotional writing, the usage of exclamation marks could be a shared property and a useful learning feature to consider.
  
+ **Sentiment**: `sentiment` As defined by Standford CoreNLP (0 very negative to 4 being very positive), a certain level of sentiment could also help detecting the spans.

+ **Loaded language**: `loaded_language` The aim here is to see if scoring the loaded language usage could help in detecting the propagandistic spans.
  
+ **Polarity and subjectivity**: `polarity - subjectivity` Measuring the amount of polarity as well as subjectivity in a given span could prove to be helpful.  [TextBlop API](https://textblob.readthedocs.io/en/dev/api_reference.html) 

+ **Readability**: `readability` We also added readability as a measure to detect possible spans.



## Methodology
### Designing the learning features

In `feature_engineering/`, all the features are being collected from the different sources and APIs through `annotate.py` and saved locally in `annotated.csv`, using sentences stored in `train_table.csv` as input and calling the different scripts in `annotators/`.


### Modeling the text as vectors

One of the options we thought about in order to model the text is using the [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) model. 

If we can get all the unique words out of the corpus and create a vector having the number of uniques as a dimension, then we can model each word/sequence of words as a vector of the said dimension, having 1 in the positions where these words take place.

As an example, if the corpus is "John is idiotic and stupid because he only says stupid things", then the unique words are {"John", "is", "idiotic", "and", "stupid", "because", "he", "only", "says", "things"}, which can be modeled as :
[-1 -1 -1 -1 -1 -1 -1 -1 -1 -1]. The sequence "John says things" can be modeled as [1 -1 -1 -1 -1 -1 -1 -1 1 1].

Input sequences are being fed using the principle of sequencing all possible combination of consecutive words. For example, the sentence "John is idiotic." would be split to "John", "John is", "John is idiotic", and fed all together to the LSTM network.

Words are being modeled as vectors in `embedding/` using `transform.py`, that is based on functions defined on `embedding/BoW/BoW.py`. The result is stored in `embedding/test.csv`. Since the number of unique words is large (18933), *we modeled words based on their position in the vector*, rather than the full vector.


### Modeling the features as vectors

Following the same principle of Bag of Words modeling, we want to transform the textual content of NER features to mathematical vectors. This is achieved through loading `embedding/BoW_NER_features.py`, which stores the result in `embedding/BoW/annotated_NERtoBoW.csv`

### Training an LSTM neural network on propagandistic and non propagandistic sentences

In `machine_learning/`, `deeplearning.py` implements the process of training an LSTM neural network.

The input data is a vector `X` consisting of all the sentences in the 371 news articles, embedded as a bag of words representation; which is split to 70% training data `X_data` and 30% test data `X_test`. 

The ``features`` vector consists of the features that are being used to train the LSTM. 

The label data `y` consists of the `{0, 1}` labels, 0 being for non propaganda and 1 for being propaganda. If a sentence doesn't contain a propaganda span (as defined in the Task), it is labeled 0. If a sentence contains both propaganda and non propaganda span, the non propaganda span is being ignored and only the propaganda part is taken into the input. Similarly, it is being split to `y_train` and `y_test`.



## Installation and usage

To install, make sure you have [Python 3.7](https://www.python.org/downloads/release/python-370/) and clone the repository:

```bash
git clone https://github.com/IliassAymaz/Propaganda-detector-political-polarization.git
cd Propaganda-detector-political-polarization
```
Then, to install in a virtual environment, under linux/macOS use:
```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
Under Windows, use:
```cmd
python -m virtualenv venv
.\venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```
If a physical [CUDA compatible](https://developer.nvidia.com/cuda-gpus) Nvidia GPU is available, we recommend to use it, as the computation runs faster than on CPU. For this, ``tensorflow-gpu`` and CUDA v10.0 are necessary. To make tensorflow GPU support possible, use:

```bash
pip install tensorflow-gpu==1.14
```
And install CUDA **v10.0** from [the official archive](https://developer.nvidia.com/cuda-10.0-download-archive) for your OS.

More information about tensorflow GPU support and requirements is available [here](https://www.tensorflow.org/install/gpu).

To train the model on the available data, use:
```bash
cd machine_learning
python deeplearning.py
```

#### Scaling
Training the model on a different dataset can be achieved through adding new articles in the same format as in ``data/datasets/train-articles``, with the title being in the first line, followed by sentences in lines.


### Credits

As part of : Research lab "Online Political Polarization", University of Koblenz Landau.

Data is taken from Semeval2020 Task 11.
