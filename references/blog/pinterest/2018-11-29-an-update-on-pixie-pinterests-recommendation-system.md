# An update on Pixie, Pinterest’s recommendation system

*Source: https://medium.com/pinterest-engineering/an-update-on-pixie-pinterests-recommendation-system-6f273f737e1b  
Published: 2018-11-29  
Fetched: 2026-09-21 via wayback*

# An update on Pixie, Pinterest’s recommendation system

Pinterest Engineering

·

Follow

Published in

Pinterest Engineering Blog

·

7 min read

·

Nov 29, 2018

--

Pong Eksombatchai | Software Engineer, Applied Science

We recently welcomed engineers to our offices for our annual Mad Science Fair, where I shared the latest on Pixie, an advanced graph-based recommendation system. Here’s a recap of that talk, and you can find the video here.

As Pinterest grows to more than 250 million people, we’re constantly scaling the number of Pins saved and matching them to people with overlapping interests, serving more than 10 billion recommendations every day.

To make this process as efficient as possible, we built Pixie. Pixie is a flexible, graph-based system for making personalized recommendations in real-time (you might have read about it when we launched it last year). When we designed Pixie, the goal was to create a system that could provide relevant and narrow recommendations at SOS response rate as Pinners scroll through the home feed (just one of the many product surfaces where we can apply Pixie).

Now we’re sharing how we scale Pixie to support this growing number of users. Since we deployed Pixie online, we’ve seen a vast improvement in probabilities. In the pre-Pixie era, we needed to recycle popular content to Pinners because chances were that they would like it too. However, once we deployed Pixie we found, we were much more efficient and able to recommend more relevant content, increasing engagement by 37x. Where we had been serving Pins with a median of 90,000 saves, we were now serving Pins with a median of 1,000 saves. Despite recommending all content, including content that is much less popular, we saw much higher engagement.

# Serving personalization

As Pinterest’s main recommender system, Pixie is applied across all of our product surfaces. To give you an idea of how good of a recommender system Pixie has become, imagine you save a Pin of a delicious “Healthy Chocolate Strawberry Shake” to one of your boards.

Using visual signals, Pixie then suggests ten other smoothie or shake Pins all based on “Healthy Chocolate Strawberry Shake,” but it may not know yet exactly what other kinds of shakes you want. As the query gets more complicated, Pixie will know that you also save Pins featuring “Healthy Chocolate Muffins” and “Ultimate Healthy Chocolate Chip Cookies.” Pixie then narrows down the content to Pins related to chocolate, cookie, dessert shakes, all with a focus on healthy ingredients.

Now, when you return to your home feed, you’ll see Pins that are exactly what you’re looking for, and you can save them to your “Healthy Shakes and Smoothies” board. This is one of the main reasons people come to Pinterest — to get recommendations for things they love but may not have even known existed.

# Let’s take a random walk

Today, Pixie powers over 60 percent of all engagement on Pinterest. That means that +250M users are relying on our recommender system being a success. How do we do it?

We start with the Pinterest object graph (the graph between Pins and boards). The dataset is highly unique as it’s created from how people describe and organize Pins and boards, and it results in countless Pins that have been added hundreds of thousands of times. From this dataset, we know two valuable things: how those Pins are organized based on the context people add as they save and the Pinner’s interests. The challenge then becomes making personalized recommendations for each of those hundreds of millions of users, in milliseconds, from a set of billions of Pins.

To unravel this challenge, let’s go back to that same “Healthy Chocolate Strawberry Shake” Pin and imagine it was saved to three different boards — “Smoothie,” “Strawberries,” and “Yummm.”

*Figure: This framework maps directly to Pixie — the black circle is the Pin, and the red squares are the boards.*

With more than 175 billion Pins in the system, we’re working with a huge bipartite graph.

One of the biggest challenges of our recommendation problem is figuring out how to narrow down the best Pin for the best person at the best time. This is where the graph-based recommender system comes in: we know a set of nodes that are already interesting to a Pinner, so we start graph traversal from there.

*Figure: “Q” here is a single node, but this can be generalized to a set of query nodes as well.*

Pixie then finds the Pins most relevant to the user by applying a random walk algorithm for 100,000 steps. At each step, it selects a random neighbor and visits the node, incrementing node visit counts as it visits more random neighbors. We also have a probability Alpha, set at 0.5, to restart at node Q so our walks do not stray too far. We continue randomly sampling the neighboring boards and nodes for 100,000 steps.

The nodes that have been visited 14 and 16 times are the ones that are most closely related to the query node.

Once the random walks are complete, we know the nodes which have been visited most frequently are the ones most closely related to the query node. Pixie continuously repeats this process in real-time as the data grows, so our users are always able to keep narrowing down their searches and find the exact ideas they’re looking for to pursue their goal.

Our “Healthy Chocolate Strawberry Shake” example is from a Pin-to-Pin recommendation, but Pixie also supports two other main clusters (Pin-to-Board and Pin-to-Ads). Furthermore, instead of just one starting point, Pixie also operates with multiple starting nodes where we assign different weights based on the different actions a user can take on a Pin, whether it’s zooming in, saving the Pin, or something else. The degree of the query Pins matters as well. For example, for the difference between a query Pin with ten thousand degrees and a query Pin with five degrees, we’d allocate more random walk steps to the Pin with ten thousand.

# Optimizing Pixie

Since we created Pixie, we’ve developed many optimizations to suit our needs, such as Early Stopping. In an ideal world, we’d only want to retrieve the top 1,000 most visited nodes, so we wouldn’t need to walk the complete 100,000 steps every time. To accomplish this, we keep walking until the rank 1,000 candidate gets at least 20 visits. From this optimization, we’re able to gain a 2x boost in performance.

Another optimization we created is Graph Pruning. The full Pinterest graph has over 100 billion edges, which is way more than we actually use, but we can remove some of those edges to make Pixie suit our needs. To prune the graph, we downscale the effect of popular Pins by implementing a function that provides a cap for the number of neighbors a Pin may have. We can also prune by getting ahead of users who may accidentally save something to the wrong board (which happens to the best of us). If we can identify those edges, we can remove them. Last but not least, another optimization is to remove diverse boards (those with Pins from multiple different ideas).

# A look back at building Pixie

Now that you know the theory behind Pixie, let’s dig into how we built it.

The ultimate goal is to fit the entire Pinterest graph in memory. Obviously we can’t fit the entire 100 billion edges in RAM, but once we prune down to 20 billion edges, we end up with a graph that’s 150GB. That’s still big, but it’s definitely manageable.

We use Hive Jobs to pull the data in a streamlined graph compilation process, and then our graph compilation machines take the raw data and parse it into a more compact serving format. Once the data is on the machine, we can serve the random walk requests.

# Why is Pixie cool?

We love Pixie because it’s easy to use, looks good, and most of all enables radical personalization in real time. Each person who comes to Pinterest is counting on personalization to help drive them closer to discovering a great new idea. Pixie’s ease of generalization to different types of tasks makes this possible, whether it’s creating many types of graphs, recommending different objects, or tuning the graphs to capture the absolute right content. Now with Pixie and recent optimizations, Pinterest is streamlined to serve Pinners relevant ideas in real time.

Interested in joining the Pinterest team? Check out our open roles here!
