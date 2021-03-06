---
layout: post
title:  "A scalable, advanced blocking method with Spark GraphX"
date:   2019-06-03
summary: >-
  todo
tags: [recordlinkage, blocking, spark, graphx, graphframe, bigdata]
author: Clément Bosc
---


## GraphX ?

GraphX is a component of Spark which allow parallel graph processing. It is based on RDD concepts (*Resilient Distributed Datasets*) and includes a collection of basic graph algorithms such as connected components,  PageRank, Triangle Counting, etc...

A graph is composed of nodes (*vertex*) and edges on which we can attach properties and therefore see them as DB tables : vertices table contains the id of the node and its attributes and edges table contains a source node (*srcId*), a target node (*dstId*) and edge attributes.

<br/>
![Graph exemple showing links between persons from a same university](/assets/img/2019_06_03_graphx_tables.png)
*Example graph showing links between people within a university*
<br/>

## Use case : a record linkage blocking method

When I was in internship, I worked on a project for an airline company : the idea was to recognize an anonymous customer from a flight to another. So basically we used multiple data sources to compare and merge records that represent the same person in real life. This kind of process is called *Record Linkage*.

A typical *Record linkage* algorithm can be decomposed in 3 steps :
* **Schema alignement** : prepare, clean and standardize the data to work with from the different sources
* **Blocking** (or *clustering*) : drasticly reduce the number of pair of records to compare by applying a blocking function or use a specific blocking key : in practice who group the records that are potentially the same entity
* **Comparison/Fusion** of the records : probabilistic or decision-tree based algorithm that take the decision to merge two records that are considered refering to the same entity.

#### What about GraphX ?

Good question ! GraphX take place in the blocking step : we create a graph where each node is a customer (or traveller), linked to the other if they share at least one common attribute (email adress, phone number, flight booking, etc...). Let's take the following example :


{% highlight plain_text %}
+---+-----------+----------+------------+------------------------+----------+
|id |firstName  |lastName  |phone       |mail                    |lastFlight|
+---+-----------+----------+------------+------------------------+----------+
|1  |HUGO       |REIEES    |+33630169073|hugo@reyes.com          |ABCDEF    |
|2  |HUGO HURLEY|REYES     |+33561402726|hugo@reyes.com          |TYUIOP    |
|3  |HUGO       |REYES     |+33630169073|h.reyes@gmail.com       |TYUIOP    |
|4  |HURLEY     |REYES     |+33731020503|h.reyes@gmail.com       |YTU545    |
|5  |ROBERT     |SMITH     |+33807060803|rob.smith@gmail.com     |XXXXXX    |
|6  |ROBERT     |SMITH     |+33807060803|robert.smith@hotmail.fr |YYYYYY    |
|7  |ROB        |SMITH JR. |+33304067328|rob.smith@gmail.com     |IUYTRE    |
|8  |RBERT      |SMITH     |+33304067328|robert.smith@hotmail.fr |POIJK5    |
+---+-----------+----------+------------+------------------------+----------+
{% endhighlight %}

Our dataset (very simplified for the purpose of this article) is composed of customers/travellers with an email, a phone number and their last flight booking. These records represent the nodes of our graph.

Then we generate a Dataset with the links between our records with a simple `flatMap` process. These are the edges of our graph. The `key` column references the commun attribute between the *src* and *dst* records. In our application the orientation of the graph is not important.

{% highlight java linenos %}

dataset
    // 1. For each record generate the keys
    .flatMap((FlatMapFunction<Row, Row>) row -> {
          List<Row> result = new ArrayList<>();

          result.add(RowFactory.create("M:"+row.getAs("mail"), row.getAs("id")));
          result.add(RowFactory.create("P:"+row.getAs("phone"), row.getAs("id")));
          result.add(RowFactory.create("F:"+row.getAs("lastFlight"), row.getAs("id")));

          return result.iterator();
      }, RowEncoder.apply(YOUR_ROW_KEY_STRUCT))  // key | id custom row struct

    // 2. Group by key
    .groupByKey((MapFunction<Row, String>) row -> row.getAs("key"), Encoders.STRING())

    // 3. Generate edges
    .flatMapGroups((FlatMapGroupsFunction<String, Row, Row>) (key, values) -> {
          List<Row> result = new ArrayList<>();
          int srcId = null;
          int i = 0;

          while(values.hasNext()){
              Row value = values.next();
              if (i == 0)
                  srcId = value.getAs("id");

              if (!srcId.equals(value.getAs("id")))
                  result.add(srcId, value.getAs("id"), key);
          }

          return result.iterator();
      }, RowEncoder.apply(YOUR_ROW_GRAPH_STRUCT))  // srcId | dstId | key custom row struct

    // Show the result dataset !
    .show();

// +-----+-----+--------------------------+
// |srcId|dstId|key                       |
// +-----+-----+--------------------------+
// |1    |2    |M:hugo@reyes.com          |
// |1    |3    |P:+33630169073            |
// |3    |4    |M:h.reyes@gmail.com       |
// |2    |3    |F:TYUIOP                  |
// |5    |6    |P:+33807060803            |
// |6    |8    |M:robert.smith@hotmail.fr |
// |8    |7    |P:+33304067328            |
// |5    |7    |M:rob.smith@gmail.com     |
// +-----+-----+--------------------------+
{% endhighlight %}

Here, we can clearly see that we have to distinct customers : Hugo Reyes and Robert Smith.
So, how can we group together these two entities ? Using the **connected components** of course !

Some reminders from graph theory : A non directed graph *G=(V,E)* is connected when it has at least one vertex and there is a path between every pair of vertices. In our case, we want to find every connected subgraph in our graph.

{% highlight scala linenos %}
// Getting the edges from our edges dataset, note the RDD cast : RDD[(Long, Long)].
val edges = edges.rdd.map(row => (row.getAs[Long](0), row.getAs[Long](1)))

// Creation of the graph from the list or edges. The node creation is implicit with this method.
// We could also have build our graph keeping the edges and nodes attributes.
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

It's that simple !
The connected components algorithm from GraphX labels each connected component of the graph with the ID of its lowest-numbered vertex.
Note the id of the node must be a `Long` and the implementation must be done in Scala (but a call from Java code is feasible).

![Graph Clustering](/assets/img/2019_06_03_graph_connected_components.png)

We can now compare and merge each record from a same cluster because they potentially represent the same person and drasticly reduce the compute time.

Special thanks to [David Gougaud][david-gougaud-linkedin] who designed this blocking method during his engineer graduation thesis.

[david-gougaud-linkedin]: https://fr.linkedin.com/in/david-gougaud-ab87b9b6/en
