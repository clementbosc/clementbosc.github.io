---
layout: post
title:  "Tutoriel Machine Learning sur un cas pratique : les survivants  du Titanic"
date:   2019-06-25
summary: >-
  todo
tags: [machine_learning]
author: Clément Bosc, Cédric Lespagnol, Sarrah Khammassi
---

![](/assets/img/ml_dummies/titanic.jpeg)

## Qu'est-ce que le Machine Learning ?

#### Les origines

"Le Machine Learning (ML) a émergé dans la seconde moitié du XXème siècle du domaine de l’intelligence artificielle et correspond à l’élaboration d’algorithmes capables d’accumuler de la connaissance et de l’intelligence à partir d’expériences, sans être humainement guidés au cours de leur apprentissage, ni explicitement programmés pour gérer telle ou telle expérience ou donnée spécifique."

Ces algorithmes se basent tous ou presque sur des concepts issus du monde des statistiques ou mathématiques en général.

De nombreux algorithmes présentés dans ce tutoriel ont été imaginés à la fin du siècle dernier mais ont seulement été popularisés à partir des années 2010. Cela est principalement dû aux moyens à disposition des data scientists (puissance de calcul) et de la quantité de données produites en permanence sur le web.

#### Méthodologie

![](/assets/img/ml_dummies/workflow_small.png)
<br/>

Les principales étapes d'un processus classique de Machine Learning sont les suivantes :
* **1. Analyse** : réaliser des statistiques descriptives sur les données, faire quelques hypothèses pour mieux saisir la nature du problème.
* **2. Préparation des données** : Etape très importante et déterminante pour la réussite du processus complet : il s'agit de sélectionner les traits importants ("features") qui permettront d'effectuer une bonne classification par les algos choisis.
* **3. Choix et entrainement des modèles** : procédé itératif qui consiste à affiner les paramètres du ou des modèles pour obtenir le meilleur résultat possible.
* **4. Evaluation** : test et validation du modèle le plus performant.

#### Supervisé vs Non supervisé

Il existe deux principales tâches en machine learning : de l’apprentissage supervisé et non supervisé.

* **Apprentissage supervisé** : nous avons à notre disposition un jeu de données labellisées, découpé en trois sous-ensembles avec pour objectif de prédire correctement le label des données. Ce label peut être une classe dans le cas d’un problème de classification (spam / non-spam par ex) ou une valeur numérique dans le cas  d’une régression (prédire le prix d’un appartement par ex).

    Les trois sous-ensembles sont :
    * Un ensemble d’apprentissage (“train”), qui doit être utilisé pour créer vos modèles d'apprentissage. Pour cet ensemble, une “vérité terrain” ou “ground truth” est fournie, autrement dit, un label ou classe pour chaque ligne. Le modèle sera basé sur des «caractéristiques» (“features”), et vous pouvez également utiliser le procédé de “feature engineering” pour créer de nouvelles caractéristiques (nous détaillerons ce procédé plus tard).
    * Un ensemble de “validation” qui doit être utilisé pour calculer un score pour votre modèle en comparant les classes prédites avec la vérité terrain. Votre modèle doit être amélioré itérativement jusqu’à l’obtention d’un score satisfaisant.
    * Un ensemble de “test”, pour lequel on prédit les classes pour obtenir le score final de votre modèle. Il ne doit être utilisé qu’une fois pour que les données n’aient jamais été vues par votre modèle, donc pour que le modèle ne soit pas spécialisé sur cet ensemble précis (éviter le phénomène de sur-apprentissage).
* **Apprentissage non supervisé** : dans ce cas, nous n’avons pas de données labellisées et l’objectif pour le modèle est de grouper les données similaires par lui même. On parle de “clustering” : par exemple regrouper des articles ou des textes portant sur des sujets communs.

En pratique, il est assez long et coûteux d’obtenir des données labellisées : on utilise soit des jeux de données publiques ou communautaires, soit de l’apprentissage non supervisé qui est bien moins précis que l’apprentissage supervisé dans la plupart des cas.

#### Types de caractéristiques

Les caractéristiques utilisées dans les modèles de Machine Learning sont des variables que l'on peut classer selon deux grands types :

* Variables numériques :
    Ces variables sont des nombres et peuvent être décomposées en deux sous-ensembles :
    * Variables continues : elles peuvent prendre n’importe quelle valeur comprise dans un ensemble de nombres réels. (ex : age)
    * Variables discrètes : ce sont des valeurs entières, elles ne peuvent pas être une fraction entre une valeur possible et la suivante. (ex : nombre de roues d’un véhicule)
* Variables catégorielles :
    Les valeurs de ces variables sont définies à partir d’un petit groupe de catégories possibles. Il existe également deux sous-ensembles de variables :
    * Variables ordinales : les valeurs de ces variables sont des catégories qui peuvent être logiquement ordonnées. (ex : faible / moyen / haut)
    * Variables nominales : ce sont des catégories qui ne peuvent pas être organisées en séquence logique. (ex : masculin / féminin)



## “Titanic: Machine Learning from Disaster”
Pour présenter notre cas pratique, nous nous appuyons sur le Challenge Kaggle [“Titanic: Machine Learning from Disaster”](https://www.kaggle.com/c/titanic).

#### Kaggle ?

Kaggle est une plateforme web organisant des compétitions en science des données. Sur cette plateforme, les entreprises proposent des problèmes en science des données et offrent un prix aux concurrents obtenant les meilleures performances.

#### En quoi consiste le challenge Titanic ?

Le challenge Titanic est un peu le “HelloWorld” de la Data Science. L’objectif est de prédire, à partir du manifeste des passagers du fameux bateau coulé en 1912, qui sont ceux qui ont survécu.

Vous avez à votre disposition deux jeux de données :
* un ensemble d’apprentissage (“train”), qui doit être utilisé pour créer vos modèles d'apprentissage. Pour cet ensemble, un label 0 ou 1 pour indiquer si le passager a survécu est fourni. On pourra baser le model sur des “features” telles que le sexe et l’âge des passagers.
* un ensemble de “test”, qui doit être utilisé pour voir si votre modèle fonctionne correctement. Dans le cadre de ce tutoriel, nous avons caché les classes de cet ensemble et proposons un outil de soumission pour évaluer vos résultats. C'est à vous de prédire ces résultats !

#### Alors, le challenge du Titanic,  apprentissage supervisé ou non supervisé ?

Supervisé bien sûr ! Nous avons un ensemble d'entraînement qui est labellisé et l’objectif est de prédire la classe pour de nouvelles données non labellisées.

#### Présentation des données



| Variable 	| Définition                                   	| Valeurs possible                               	| Type de donnée 	|
|----------	|----------------------------------------------	|------------------------------------------------	|----------------	|
| survival 	| Survival                                     	| 0 = No, 1 = Yes                                	| Classe à prédire 	|
| pclass   	| Classe du ticket                             	| 1 = 1st, 2 = 2nd, 3 = 3rd                      	| Ordinale       	|
| name     	| Nom du passager                              	| . 	|                	|
| sex      	| Sexe                                         	| male, female                                   	| Nominale      	|
| Age      	| Age en année                                 	| .	| Continue       	|
| sibsp    	| Nombre de frères et soeurs ou conjoints à bord 	| . 	| Discrète       	|
| parch    	| Nombre de parents et enfants à bord          	| . 	| Discrète       	|
| ticket   	| Numéro du ticket                             	| . 	|                	|
| fare     	| Tarif passager                               	| . 	| Continue       	|
| cabin    	| Numéro de cabine                             	| . 	|                	|
| embarked 	| Port d'Embarquement                       	| C = Cherbourg, Q = Queenstown, S = Southampton 	| Nominale      	|

## Les choses sérieuses commencent

### Quelques imports de base

* **`pandas`** : il s'agit d'une librairie qui permet de manipuler des jeux de données (appelés dataframe)
* **`matplotlib.pyplot`** : permet de faire de la visualisation de données


{% highlight python %}
import pandas as pd
import matplotlib.pyplot as plt
{% endhighlight %}


Et maintenant nos deux jeux de données


{% highlight python %}
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
{% endhighlight %}

Voyons voir d'un peu plus près ce que contient notre ensemble d'entrainement :


{% highlight python %}
train.head()
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



### 1. Observations et statistiques descriptives


{% highlight python %}
train.info()
{% endhighlight %}

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB


Comme on peut le voir, certaines "features" contiennent des **données nulles, vides ou manquantes**. C'est la cas de l'age (714 valeurs non nulles sur 891 lignes en tout), du numéro de cabine et du pont d'embarquement. Il faudra remédier à ce problème.

#### Quelle est la distribution des valeurs numériques ?


{% highlight python %}
train.describe()
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



* Le jeu d'entrainement contient 891 entrées; soit environ 40% du nombre de passagers réels à bord du Titanic (2224).
* Environ 38% des personnes de notre échantillon ont survécu, ce qui est représentatif car ce nombre est de 32% sur l'ensemble du Titanic.
* La plupart des passagers ( > 75%) n'ont pas voyagé avec des parents ou des enfants.
* Environ 30 % des passagers ont des frères et soeurs et/ou un.e conjoint.e.
* Il n'y a pas beaucoup de personnes agées et le prix des billets est relativement faible ( au moins 75% ont payé moins de 31$ mais certains ont payé jusqu'à 512$).

#### Quelle est la distribution des valeurs catégorielles ?


{% highlight python %}
train.describe(include=['O'])
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Sex</th>
      <th>Ticket</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>204</td>
      <td>889</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>891</td>
      <td>2</td>
      <td>681</td>
      <td>147</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Morrow, Mr. Thomas Rowan</td>
      <td>male</td>
      <td>1601</td>
      <td>B96 B98</td>
      <td>S</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>577</td>
      <td>7</td>
      <td>4</td>
      <td>644</td>
    </tr>
  </tbody>
</table>
</div>



* Les noms sont uniques dans ce dataset (count=unique=891)
* Il y a plus d'hommes que de femmes (65%)
* Les numéros de cabines ont plusieurs doublons, ce qui veut dire que des passagers partagaient une cabine
* La plupart des passagers ont embarqué au port S
* Il y a beaucoup de doublons (22%) parmi les numéros de ticket (unique=681)

### 2. Nettoyage et feature engineering

#### Suppression des features inutiles

On suppose que l'on peut supprimer le numéro de ticket de notre échantillon, car il contient beaucoup de doublons et on peut raisonnablement penser qu'il n'y a pas de corrélation avec la survie du passager. On peut également supprimer le numéro de cabine car il y a beaucoup de valeurs manquantes. La valeur `PassengerId` peut aussi être supprimée car celle-ci n'apporte pas d'information. Nous ne la supprimons pas de l'ensemble de test car nous en avons besoin pour la soumission et l'évaluation des résultats.


{% highlight python %}
train = train.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)
combine = [train, test]
{% endhighlight %}

#### Création d'une nouvelle feature "Title" à partir du nom du passager

On remarque que les noms des passagers contiennent un titre (Mrs., Mr., etc). Il serait judicieux d'extraire cette information pour en faire une nouvelle feature.


{% highlight python %}
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Sex</th>
      <th>female</th>
      <th>male</th>
    </tr>
    <tr>
      <th>Title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Capt</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Col</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Countess</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Don</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Dr</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Jonkheer</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Lady</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Major</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Master</th>
      <td>0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Miss</th>
      <td>182</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mlle</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mme</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mr</th>
      <td>0</td>
      <td>517</td>
    </tr>
    <tr>
      <th>Mrs</th>
      <td>125</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ms</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Rev</th>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Sir</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Nous pouvons regrouper les titres qui apparaissent peu sous une seule catégorie. (`Rare`)


{% highlight python %}
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Master</td>
      <td>0.575000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Miss</td>
      <td>0.702703</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mr</td>
      <td>0.156673</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mrs</td>
      <td>0.793651</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rare</td>
      <td>0.347826</td>
    </tr>
  </tbody>
</table>
</div>



Nous pouvons maintenant supprimer la colonne `Name` qui est inutile en soit.


{% highlight python %}
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)
combine = [train, test]
{% endhighlight %}

#### Substitution des valeurs manquantes

Comme nous l'avions souligné plus tôt dans notre analyse, les features `Age`, `Embarked` et `Fare` présentent des valeurs manquantes. Une technique courante est de remplacer ces valeurs par une valeur par défaut.

Pour le port d'embarquement, il y a deux valeurs manquantes et les valeurs possibles sont S, Q et C. On peut simplement remplacer ces valeurs manquantes par la valeur la plus fréquente.


{% highlight python %}
freq_port = train.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
{% endhighlight %}

Pour l'age et le tarif du ticket, nous proposons de remplacer les valeurs manquantes par la médiane.


{% highlight python %}
for dataset in combine:
    dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)
    dataset['Age'].fillna(dataset['Age'].dropna().median(), inplace=True)
{% endhighlight %}

#### Conversion des valeurs catégorielles vers nominales

Nous pouvons maintenant convertir les features qui contiennent des chaines de caractères en valeurs numériques. Cette étape est nécessaire pour la très grande majorité des algorithmes de Machine Learning.


{% highlight python %}
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for dataset in combine:
    dataset['Sex'] = le.fit_transform(dataset['Sex'])
    dataset['Title'] = le.fit_transform(dataset['Title'])
    dataset['Embarked'] = le.fit_transform(dataset['Embarked'])
{% endhighlight %}

Nous avons utilisé un `LabelEncoder` qui attribue un chiffre à chaque valeur unique. Cependant, il est possible  que le modèle interprète ces valeurs comme des valeurs ordinales. Il pourra être interessant d'utiliser un `OneHotEncoder`.

Voyons voir à nouveau notre jeu de données préparé et nettoyé


{% highlight python %}
train.head()
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Pas mal ! 👌

### 3. Enfin un peu de Machine Learning

Dans ce tutoriel, nous utiliserons la librairie open source `sklearn` (ou `scikit-learn`) qui implémente de très nombreux algorithmes de Machine Learning.
Dans un 1er temps, nous allons créer nos ensembles train/test/validation. Nous utiliserons le jeu de test fourni et créerons l'ensemble de validation en prenant 20% de l'ensemble de train. Nous utiliserons pour cela la fonction `train_test_split` déjà implémentée dans `sklearn`.

Les algorithmes sont tous basés sur la même structure et attendent en entrée de la fonction d'apprentissage (`.fit(X_train, y_train)`):
* **X_train**, qui est une matrice ($$nb\_features * nb\_exemples$$)
* **y_train**, qui est un tableau de taille $$nb\_exemples$$ contenant les classes pour chaque exemple

Une fois la phase d'apprentissage terminée (très rapide pour des petits ensembles comme les notres), il convient d'évaluer le modèle construit avec l'ensemble de validation. On peut utiliser une métrique élémentaire, la justesse (`accuracy`) qui consiste à calculer le ratio $$\frac{nb\_bien\_classés}{nb\_total}$$. L'accuracy peut être calculée avec le fonction `.score(X_val, y_val)` du modèle.

Enfin, une fois que le modèle construit est satisfaisant, on peut prédire les classes de notre ensemble de test pour une évaluation non biaisée (afin d'éviter le phénomène de sur-apprentissage). Pour cela on utilise la fonction `.predict(X_test)`.


{% highlight python %}
from sklearn.model_selection import train_test_split

X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape
{% endhighlight %}




    ((712, 8), (712,), (179, 8), (179,), (418, 8))



#### KNeighborsClassifier

L’algorithme des KNN ("K plus proches voisins" en français) attribue à un exemple la classe majoritaire parmi ses K plus proches voisins. K est un paramètre à optimiser par l’utilisateur, il est préférable de choisir un K impair pour de la classification binaire pour éviter les égalités. On utilise communément la distance euclidienne entre deux vecteurs de features pour calculer la proximité entre 2 exemples.

La phase d’apprentissage consiste ici seulement à stocker les vecteurs de features et les labels associés.

Un problème avec le "vote majoritaire" apparaît lorsque les classes ont une répartition inégale. En effet, si une classe est sur-représentée, elle pourra être plus représentée dans les plus proches voisins. Une méthode pour contrer ce problème est d’attribuer un poids plus important aux voisins les plus proches (paramètre `weight='distance'` dans l’implémentation de sklearn).

Voici un schéma pour illustrer un cas où K = 5 :

![](/assets/img/ml_dummies/knn.png)


{% highlight python %}
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
acc_knn = round(knn.score(X_val, y_val) * 100, 2)
acc_knn
{% endhighlight %}




    74.3



#### DecisionTreeClassifier

Pour faire une prédiction, les Decision Trees utilisent un ensemble de règles de décision « If Then Else » sur les features. Cette méthode permet de décomposer un ensemble de données en sous-ensembles de plus en plus petits. On peut ainsi assigner aux sous-ensemble finaux une classe (0 ou 1 pour une classification binaire). Le but du modèle va être de créer des sous-ensembles homogènes (contenant des exemples de même classe) pour minimiser l’erreur de ses prédictions.


{% highlight python %}
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(max_depth=3)
decision_tree.fit(X_train, y_train)
acc_decision_tree = round(decision_tree.score(X_val, y_val) * 100, 2)
print(acc_decision_tree)

import pydotplus
from sklearn import tree
from IPython.display import Image  

# Create DOT data
dot_data = tree.export_graphviz(decision_tree, out_file=None,
                                feature_names=train.columns[1:],
                                class_names=["dead", "alive"])
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  
#Show graph
Image(graph.create_png(), width=800)
{% endhighlight %}

    81.01





![](/assets/img/ml_dummies/decision_tree_output.png)



#### Random Forest

Les modèles Random Forest sont composés de plusieurs Decision Trees ("forêt" d'arbres de décision). Le résultat retourné par un Random Forest est la classe majoritaire retournée par les Decision Trees qui le composent. Ces Decision Trees sont chacuns entraînés avec un échantillon des données d’entraînement pris au hasard et un sous-ensemble des features pris au hasard également.

![](/assets/img/ml_dummies/random_forest.png)


{% highlight python %}
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, max_depth=3)
random_forest.fit(X_train, y_train)
acc_random_forest = round(random_forest.score(X_val, y_val) * 100, 2)
acc_random_forest
{% endhighlight %}




    80.45



#### Logistic Regression

La régression logistique fut l'un des premiers algorithmes de Machine Learning (première mention en 1944). C'est un algorithme qui est utilisé quand la classe à prédire est catégorielle. L'objectif est de séparer l'espace vectoriel en deux "régions", une pour chaque classe, par une frontière linéaire. Par exemple pour deux dimensions, la frontière est une droite; pour 3 dimensions, la frontière est un plan, etc...  

Pour que cela ait du sens, il faut bien-sûr que les points de l'ensemble soient **linéairement séparables** (ce qui n'est pas forcement facile à savoir en pratique).

![](/assets/img/ml_dummies/linear_non_linear.PNG)
<br/>
Dans notre cas, (deux classes) on parle de **Binomial Logistic Regression**. Dans les cas où l'on a plus de classes, on parle de **Multinomial Logistic Regression**.


{% highlight python %}
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
acc_log = round(logreg.score(X_val, y_val) * 100, 2)
acc_log
{% endhighlight %}




    79.89



#### Gaussian Naive Bayes

Les algorithmes de type Naive Bayes se basent sur le [théorème de Bayes](https://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_de_Bayes) :  $$P(c \vert e) = \frac{P(e \vert c) * P(c)}{P(e)}$$

Avec :
* $$P(c \vert e)$$ = probabilité d’assigner la classe $$c$$ sachant l’exemple $$e$$
* $$P(e \vert c)$$ = probabilité d’avoir l’exemple $$e$$ sachant qu’on a la classe $$c$$
* $$P(c)$$ = probabilité de la classe $$c$$ (nombre d’exemples de classe $$c$$ / nombre d’exemples)
* $$P(e)$$ = probabilité de l’exemple $$e$$ (nombre d’exemples $$e$$ / nombre d’exemples)

La classe $$c$$ assignée à l’exemple $$e$$ sera la classe pour laquelle $$P(c \vert e)$$ est maximum. Si l’on souhaite seulement obtenir une assignation, on peut enlever $$P(e)$$ du calcul qui ne sert qu’à normaliser.

Un exemple $$e$$ est représenté par ses features $$f_1$$, $$f_2$$, $$f_n$$... La probabilité $$P(e \vert c)$$ est donc $$P(f_1, f_2, ..., f_n  \vert c)$$. Les algorithmes Naive Bayes supposent que les features n’ont pas de liens entre elles donc $$P(f_1, f_2, ..., f_n  \vert c) = P(f_1 \vert c) * P(f_2 \vert c) * ... * P(f_n \vert c)$$.

Un modèle Naive Bayes stocke donc les probabilités d’apparition de chaque valeur pour chaque feature et chaque classe en se basant sur leur fréquence d’apparition. Nous utilisons ici la variante Gaussienne qui n’utilise pas les fréquences d’apparition mais la densité de probabilité de la [loi normale](https://fr.wikipedia.org/wiki/Loi_normale), donc elle doit stocker la moyenne et l’écart-type de chaque feature sachant chaque classe.


{% highlight python %}
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
acc_gaussian = round(gaussian.score(X_val, y_val) * 100, 2)
acc_gaussian
{% endhighlight %}




    76.54



#### Support Vector Machines (SVM)

L'agorithme de Support Vector Machines (plus communément appelés SVM ou "machines à vecteurs de support") est un algorithme supervisé utilisé pour la classification.

Cet algo a pour but de séparer des groupes de données différents avec des **hyperplans**. Un hyperplan peut être vu comme une ligne qui sépare et classifie linéairement un jeu de données.

Le SVM est capable de réaliser 2 types de séparation :
* **linéaire** : quand les données sont linéairement séparables.
* **non linéaire** : on utilise un "noyau" (kernel) qui va transformer les features existantes et en créer de nouvelles afin de pouvoir représenter les données de manière linéairement séparable. Cette opération est transparente pour l'utilisateur. Il existe plusieurs types de noyaux en fonction de la répartition spatiale des données : Noyau linéaire, polynomial, sigmoïde, "Radial Basic Function" (RBF)...

<br/>
![](/assets/img/ml_dummies/svm.png)
<br/>


{% highlight python %}
from sklearn.svm import SVC

svc = SVC(gamma='auto')
svc.fit(X_train, y_train)
acc_svc = round(svc.score(X_val, y_val) * 100, 2)
acc_svc
{% endhighlight %}




    73.18



#### Comparaison des modèles

Bon ! Nous avons testé les grands classiques des modèles de machine learning, lequel s'en sort le mieux ?


{% highlight python %}
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Decision Tree</td>
      <td>81.01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Random Forest</td>
      <td>80.45</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Logistic Regression</td>
      <td>79.89</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Naive Bayes</td>
      <td>76.54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNN</td>
      <td>74.30</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Support Vector Machines</td>
      <td>73.18</td>
    </tr>
  </tbody>
</table>
</div>



## Evaluation finale!


{% highlight python %}
# Prédiction de la classe pour l'ensemble de test
y_pred = decision_tree.predict(X_test)

# Création d'un dataframe de submission avec une colonne PassengerId et une colonne Survived
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })

# Enregistrement au format csv
submission.to_csv('output/submission.csv', index=False)
{% endhighlight %}

Et maintenant, vérifions l'accuracy de notre modèle sur l'ensemble de test


{% highlight python %}
from toolbox import submit, confusion_matrix
final_score = submit('output/submission.csv')
{% endhighlight %}

    Votre accuracy score est de 0.7751196172248804.
    Pas mal, c'est un bon début ! Tu peux encore faire un peu mieux 🙂


### Matrice de confusion

Pour visualiser les prédictions de notre modèle, nous pouvons afficher la **matrice de confusion**, qui indique comment sont classées les valeurs prédites par rapport aux valeurs réelles. Cette matrice nous apporte des précisions par rapport au calcul de l'accuracy.

![](/assets/img/ml_dummies/confusion_matrix.png)

On y retrouve des informations comme le nombre de TP (True Positive), FP (False Positive), FN (False Negative) et TN (True Negative), à partir desquelles ont peut calculer le **rappel**, la **précision** et l'**accuracy**.


{% highlight python %}
pd.DataFrame(confusion_matrix(y_pred), index = ["Predicted survived", "Predicted dead"], columns=["Really survived", "Really dead"])
{% endhighlight %}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Really survived</th>
      <th>Really dead</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Predicted survived</th>
      <td>212</td>
      <td>48</td>
    </tr>
    <tr>
      <th>Predicted dead</th>
      <td>46</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
</div>



## Et maintenant, c'est à vous !

Ce tutoriel présente les principaux algorithmes de Machine Learning. Nous avons construit un modèle se basant sur un arbre de décision et nous avons obtenu un score de 0.775 dans la phase d'évaluation finale sur notre ensemble de test. Mais ça ne veut pas dire que l'on ne peut pas faire mieux, bien au contraire !!
Maintenant, c'est à vous de jouer, l'objectif est d'améliorer le score.

Pour cela voici quelques pistes :
* Tester plusieurs valeurs différentes pour les hyperparamètres des modèles présentés. Pour ce tutoriel nous avions utilisé les valeurs par défaut mais il existe d'autres paramètres qui, selon les cas, peuvent améliorer l'accuracy. La fonction `sklearn.model_selection.GridSearchCV` pourra vous y aider.
* Tester d'autres modèles non présentés ici (voir liste complète sur [scikit-learn.org](https://scikit-learn.org/)).
* Créer de nouvelles features à partir de celles existantes : voyage_seul (binaire) si  `SibSp > 1 OU Parch > 1`, grouper les ages ou les prix par tranches (voir la fonction `pd.cut(train['Age'], nb_bands)`) ... à vous d'être inventifs 😁.
* Utiliser un `OneHotEncoder` à la place d'un `LabelEncoder`.
* Utiliser une k-fold Cross-Validation (ça ne permettra pas d'améliorer votre score mais c'est une bonne pratique à connaitre)


#### Références

* [Challenge Kaggle "Titanic: Machine Learning from Disaster"][titanic_kaggle]
* [Titanic Data Science Solutions][kaggle_kernel_sol]
* [Understanding Confusion Matrix](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62)
* [Une petite histoire du Machine Learning](https://www.quantmetry.com/une-petite-histoire-du-machine-learning)

[titanic_kaggle]: https://www.kaggle.com/c/titanic
[kaggle_kernel_sol]: https://www.kaggle.com/startupsci/titanic-data-science-solutions
