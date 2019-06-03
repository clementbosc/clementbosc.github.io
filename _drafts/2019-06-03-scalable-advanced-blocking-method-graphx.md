---
layout: post
title:  "A scalable, advanced blocking method with Spark GraphX"
date:   2019-06-03
summary: >-
  todo
tags: [recordlinkage, blocking, spark, graphx, graphframe, bigdata]
---


## GraphX ?

GraphX is a component of Spark which allow parallel graph processing. It is based on RDD concepts (*Resilient Distributed Datasets*). GraphX includes a collection of basic graph algorithms such as connected components,  PageRank, Triangle Counting, etc...

A graph is composed of nodes (*vertex*) and edges on which we can attach properties and therefore see them as DB tables : vertices table contain the id of the node and its attributes and edges table contain a source node (srcId), a target node (dstId) and its attributes.


## Use case : a record linkage blocking method

When I was in internship, I worked on a project for the french airline company Air France : the idea was to recognize an anonymous customer from a flight to another. So basically use multiple source of data to compare and merge

{% highlight plain_text %}
+---+-----------+--------+----------+------------------------+----------+
|id |firstName  |lastName|phone     |mail                    |lastFlight|
+---+-----------+--------+----------+------------------------+----------+
|1  |JEAN       |MART    |0630169073|jean@michel.fr          |SGBGSL    |
|2  |JEAN-MICHEL|MARTIN  |0561402726|jean@michel.fr          |J285RQ    |
|3  |JEAN       |MARTIN  |0630169073|jojo@martin.fr          |J285RQ    |
|4  |JEAN-MICH  |MARTIN  |0731020503|jojo@martin.fr          |YTU545    |
|5  |ROBERT     |DUPONT  |0807060803|rob.dup@gmail.com       |RUHFDS    |
|6  |ROBERT     |DUPOND  |0807060803|robert.dupond@hotmail.fr|JHGFDS    |
|7  |ROB        |DUPOND  |0304067328|rob.dup@gmail.com       |IUYTRE    |
|8  |RBERT      |DUPOND  |0304067328|robert.dupond@hotmail.fr|POIJK5    |
+---+-----------+--------+----------+------------------------+----------+
{% endhighlight %}

{% highlight plain_text %}
+-----+-----+--------------------------+
|srcId|dstId|key                       |
+-----+-----+--------------------------+
|1    |2    |M:jean@michel.fr          |
|1    |3    |P:0630169073              |
|3    |4    |M:jojo@martin.fr          |
|2    |3    |F:J285RQ                  |
|5    |6    |P:0807060803              |
|6    |8    |M:robert.dupond@hotmail.fr|
|8    |7    |P:0304067328              |
|5    |7    |M:rob.dup@gmail.com       |
+-----+-----+--------------------------+
{% endhighlight %}

{% highlight scala linenos %}
// Création des arrêtes, à noter la tranformation en RDD[(Long, Long)].
val edges = edges.rdd.map(row => (row.getAs[Long](0), row.getAs[Long](1)))

// Création du graphe à partir de la liste des arrêtes. La création des sommets est implicite avec cette méthode.
// On pourrait aussi construire notre graphe en conservant les propriétés liés aux noeuds et arrêtes.
val graph = Graph.fromEdgeTuples(edges)
graph.connectedComponents().vertices.forEach(e => println((e._1, e._2)))

// OUTPUT :
// 1, 1
// 3, 1
// 4, 1
// 5, 5
// 6, 5
// 7, 5
// 8, 5
{% endhighlight %}

![Graph Clustering](/assets/img/2019_06_03_graph_connected_components.png)
