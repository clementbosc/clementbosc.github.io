---
layout: post
title:  "Build a personalized information retrieval system"
date:   2019-06-17
summary: >-
  todo
tags: [information_retrieval, tweets, lucene, word2vec]
author: Clément Bosc
---

During my last year of Master degree at Paul Sabatier university in Toulouse (FR), some colleagues and I, worked on building a personalized information retrieval system for retriving specific tweets.

_**Personalized information retrieval systems**_ are search engines that are able to build and store a user profile based on user interaction and behaviour in order to rerank the query results to better match the user wishes.

Our project was composed of several blocks :
* **Step 1 :** Build the Lucene index
* **Step 2 :** Build the user profile mechanism
* **Step 3 :** Personalized inforamtion retrieval system with PyLucene, relevent tweets based on a query and the user profile

![General architecture of our project](/assets/img/2019_06_17_global_schema.png)

## The working corpus

The corpus we worked on for our experimental project is a 6M+ tweets from 2009 to 2016 about `#iot`. The corpus was already labeled with information about tweet's author using automatic methods and NLP processings :
* sentiment : Opinion / point of view expressed in the tweet towards IoT (`neutral`, `positive`, `negative`)
* topicID : topic modeling analysis to group tweets in 6 topics, from `0` to `5`
* country : `fr`, `us`, `ca` ...
* gender : gender of the user (`andy` -androgynous-, `male`, `female`, `mostly_male`, `mostly_female`)

<br/>
![Head of the pandas dataframe](/assets/img/2019_06_17_pandas_df.png)
*Sample of our corpus*
<br/>

## How to build a simple user profile ?

Our approach was pretty simple :
#### 1. We turned all tweets to Word2Vec vectors

Word2Vec is a powerful model that is used to produce word embeddings. It's trained on huge corpuses (we took the GNews pre-trained model) and produce a vector representation of each individual word. Usually, vectors produced have 50 to 300 dimensions. Word vectors are positioned in a vector space such that words that share common contexts in the corpus are located in close proximity to one another in the space.
We choose to use a very simple vector representation for a tweet : we **`avg`** the words vectors which compose it.

#### 2. We built models to predict user characteristics from a tweet vector

The idea was to get user characteristics (gender, topic, sentiment, country) from a w2v vector. So we trained 4 different models to predict each characteristic of the tweet vector : SVM for gender and topicId, MLPClassifier for sentiment and NaiveBayes for country prediction.

#### 3. User profile representation and update

The user profile we designed was composed of a w2v vector and the 4 characteristics. Each time the user uses the search engine and interact with query results (by clicking or ❤-ing) we update the profile :
* The w2v vector is the **`avg`** of all liked and clicked vector by the user in the current session
* The 4 characteristics are predicted with the models using the avg w2v vector.

<br/>
![Our user profile representation](/assets/img/2019_06_17_user profile.png)
*Our user profile representation*
<br/>

## Rerank the query results

Now that our profile is built, the idea is to use it !

#### OneHotEncoder
For that, we first used a `OneHotEncoder`. The first step is to train the OneHotEncoder model with all different possible values for gender, sentiment, country in order to convert these categorical variables into dummy variables.

<br/>
![OneHotencoder](/assets/img/2019_06_17_onehotencoder.png)
*The OneHotEncoder process*
<br/>

#### Reranking
The produced OneHotEncoder vector from the user's characteristics is appended to the user's w2v vector : this new big vector is actually the real profile of the user. The final step or our personalized information retrieval system is to rerank results to match the user's profile.
* We take the `n` top results retrieved by Lucene and build w2v+OneHotEncoder vectors
* Perform **Cosine similairty** between the user's profile vector and all top `n` results vectors

<br/>
![Rerank of the results](/assets/img/2019_06_17_rerank.png)
*Rerank of the results*
<br/>

That's all ! We have created a very simple personnalized search engine ✌
