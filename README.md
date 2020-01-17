# Forschungspraktikum Political Polarization - Propaganda Group 


Based on the Task SI (Span Identification) of the [Semeval2020 Task 11](https://propaganda.qcri.org/semeval2020-task11/index.html), this work contains the software implementation of the machine learning based propaganda detector prototype, as authored by the team *stayfoolish*.

The implementation has achieved an accuracy of xx%.

## 1. The task
The challenge of the Task SI is to build a model capable of detecting spans of text that contain one of [these 18 propaganda techniques](https://propaganda.qcri.org/annotations/definitions.html), based on a training corpus of 371 news articles.

## 2. Approach
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



## 3. Methodology
+ designing the learning features
+ modeling the text as vectors
  + converting sentences into all possible combinations of words
+ modeling the features as vectors
+ training an LSTM neural network on propagandistic and non propagandistic sentences
+ 

### Credits

As part of : Research lab "Online Political Polarization", University of Koblenz Landau.

Data is taken from Semeval2020 Task 11.
