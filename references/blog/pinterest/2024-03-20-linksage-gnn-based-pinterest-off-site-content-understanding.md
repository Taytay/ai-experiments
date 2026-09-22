# LinkSage: GNN-based Pinterest Off-site Content Understanding

*Source: https://medium.com/pinterest-engineering/linksage-gnn-based-pinterest-off-site-content-understanding-fca14b0d1141  
Published: 2024-03-20  
Fetched: 2026-09-21 via wayback*

# LinkSage: GNN-based Pinterest Off-site Content Understanding

Pinterest Engineering

·

Follow

Published in

Pinterest Engineering Blog

·

9 min read

·

5 days ago

--

Adopted by Pinterest multiple user facing surfaces, Ads, and Board.

Jianjin Dong | Staff Machine Learning Engineer, Content Quality; Michal Giemza| Machine Learning Engineer, Content Quality; Qinglong Zeng | Senior Engineering Manager, Content Quality; Andrey Gusev | Director, Content Quality; Yangyi Lu | Machine Learning Engineer, Home Feed; Han Sun | Staff Machine Learning Engineer, Ads Conversion Modeling; William Zhao | Software Engineer, Boards Foundation, Jay Ma | Machine Learning Engineer, Ads Lightweight Ranking

*Figure: LinkSage: Graph Neural Network based model for Pinterest off-site content semantic embeddings*

# Background

Pinterest is the visual inspiration platform where Pinners come to search, save, and shop the best ideas in the world for all of life’s moments. Most of the Pins are linked to off-site content to provide Pinners with inspiration and actionability. It is critical to understand off-site content (images, text, structure), because understanding their semantics is an important factor in assessing how safe (e.g. community guidelines), functional, relevant, and actionable (e.g. Ads and Shopping) the off-site content is. More importantly, Pinterest can have a better understanding of Pinterest users through users’ click through events. Both of the above can improve overall engagement and monetization of Pinterest contents. To achieve it, we developed LinkSage, which is a Graph Neural Network (GNN) based model that learns the semantics of landing page contents.

*Figure: Figure 1: Off-site content understanding and its applications*

# Motivation and Introduction

To make full use of Pinterest off-site content to improve Pinners’ engagement and shopping experience, we established the following goals:

- Unified semantics embedding: Provide a unified semantic embedding of all the Pinterest off-site content. All the landing pages related to downstream models can leverage LinkSage embedding as a key input.

- Graph based model: Leverage the Pinner’s curation data to build a heterogeneous graph that supports different types of entities. The GNN can learn from nearby landing pages/nodes to improve accuracy.

- XSage ecosystem: Make the LinkSage embedding compatible with all the XSage embedding space.

- Multi-dimensional representation: Provide a multi-dimensional representation of the LinkSage embedding so consumers would have a flexibility of choosing performance vs cost.

- Impact on engagement and monetization: Improve both engagement (e.g. long clicks) and shopping/ads experience (e.g. CVR) through a better understanding of Pinterest content and Pinner profile.

In this blog, we touch on:

- Technical design

- Key innovations

- Offline results

- Online results

# Technical Design

**Data**

Most Pins are associated with a landing page. We treat “(Pin, landing page):” as a positive pair if the Pin and its associated landing page have similar semantics, and we leverage Pinterest Cohesion ML signal to evaluate the semantic similarity between a Pin and its landing page. We also label a “(Pin, landing page)” pair as positive if the Cohesion score is higher than a certain threshold.

For negative pairs, we include both batch and random negatives. In the case of batch negatives, we use Pins that are paired with other landing pages in the same batch. In the case of random negatives, we use random Pins across Pinterest, which may not be seen in the positive pairs. This helps to train a model generic to new contents.

In the latter version of LinkSage, we would leverage Pinner onsite engagement data and Pinner off-site conversion data to enrich our training targets.

**Graph**

We leverage Pinner’s curated data to build the graph. Graph compilation and random walk is conducted using Pinterest XPixie, which supports heterogeneous graphs of different types of entities. In our case, a heterogeneous graph is built by using “(Pin, landing page)” pairs. We leverage Pinterest Cohesion ML signal to filter out non-cohesive pairs, similar to training data generation. Thus, all the “(Pin, landing page)” pairs used in the graph have similar semantics. To increase the graph density, we leverage Pinterest Neardup ML signal to cluster similar Pin images to an image cluster. Graph pruning is done on both graph nodes and edges to ensure graph connections are not skewed on certain popular landing pages or Pins. In this graph, landing pages with similar semantics are connected with Pins that are cohesive to the landing pages.

After the random walk, for each landing page, we get a list of its neighbor landing pages and their visit counts. Random walk is configurable based on the node entity type.

In our latter version, we fully utilize the heterogeneous graph feature of XPixie that we add more different types of entities, including Pinterest Boards and link clusters.

**Features**

There are three types of features: self landing page features, neighbor landing page features, and graph structure features.

For both self landing pages and neighbor landing pages, we use two types of content features: landing page text embedding (which summarize the semantics of title, description, main body text), and visual embedding of each crawled image. We perform a weighted aggregation of all the crawled images by their size to reduce the calculation while keeping the main crawled images’ information of the landing pages.

For graph structure features, we use graph node visit counts and self degree to represent the topological structure of the graph. Graph node visit counts represent the importance of the neighbor landing pages, while self degree represents the popularity of the self landing page in the graph.

**Model**

The model leverages a Transformer encoder to learn the cross attention of self landing page features, neighbor landing page features, and graph structure features.

The text and crawled image features are split in the transformer encoder to let the model learn the cross attention of them. The neighbors are reverse sorted by the visited counts so the top neighbors would be more important than the bottom ones. Together with position embeddings, our model can learn the importance of different neighbors. The number of neighbors is chosen to balance computational cost and model performance.

In the latter version, we split crawled images and treat them as separate tokens in the transformer encoder, which would provide the model with more accurate visual information of the landing pages.

*Figure: Figure 2: Model schematics of LinkSage*

# Key Innovations

**Multi-dimensional representation**

Downstream teams would consume different dims of embedding based on their preference between performance and computational cost. Instead of training five different models separately, we leverage the research of Matryoshka Representation Learning to provide five dims of LinkSage in place by training one model. Shorter dims would capture a coarse representation of the landing pages, and more details are embedded in the longer ones.

*Figure: Figure 3: Schematic of the loss function of multi-dimensional representation*

**Compatibility of XSage**

The compatibility of the embedding space between LinkSage and XSage (e.g. PinSage) would make the downstream usage easier. Downstream teams can even use proximity in embedding space to compare the similarity of different contents across Pinterest, like Pins and their landing pages. To achieve this, we leverage PinSage as the representation of the Pins in our training target.

**Incremental serving**

Pinterest has tens of billions of landing pages associated with Pins. To serve all the landing pages, it would take a huge amount of computational cost and time. To solve it, we apply incremental serving that we only run serving of daily crawled landing pages. After daily inference, we merge today’s inference results with the previous ones. Our incremental serving not only saves a large amount of unnecessary computations but also keeps the same accuracy and coverage as the full corpus serving.

# Offline Results

**Recall**

Recall is the most commonly used metric for ranking tasks. When given a query landing page, it evaluates how good the model can retrieve the positive candidate Pins among all the negatives. Higher recall means a better model.

*Figure: Table 1: Recall of LinkSage across different serving dimensions.*

From the table above, by using 256 dims of LinkSage, the probability of fetching the positive candidate Pins is 72.9% from the top 100 ranking results. By using 64 dims of it, it saves 75% of the cost and the performance only drops by 8.3%.

**Score distribution**

Score distribution is plotted to show the distribution of cosine similarity scores between (1) query landing page and positive candidate Pins, and (2) query landing page and negative candidate Pins

*Figure: Figure 4: Score distribution of LinkSage positive and negative pairs*

From the histogram below, almost all the negative pairs have a score < 0.25 and the mean value is close to 0. On the other hand, more than 50% of the positive pairs have a score > 0.25.

**Kurtosis**

Kurtosis is used to evaluate the ability of the embedding to distinguish between different landing pages.

For embedding pairwise cosine similarity distribution, a smaller kurtosis is preferable because a wide-spread distribution tends to have better “resolution” to distinguish between queries (aka landing pages) of different relevance.

The Kurtosis of LinkSage is 1.66.

*Figure: Figure 5: Kurtosis analysis of LinkSage*

**Visualization**

Given a landing page, the top k ranked Pins can be fetched and visualized to check whether the landing page and Pins have similar semantics.

# Online Results

We launched A/B experiments in multiple user facing surfaces, Ads, and Boards.

**User facing surfaces**

Multiple user facing surface teams have adopted LinkSage into their ranking model to improve the understanding of both candidate Pins and user profiles (through User Sequence).

On Pinterest, “repin, long click, engaged sessions” are the key indicators of positive user engagement. On the other hand, “hide” is the key indicator of negative user engagements on the platform. We observed significant gains on all the metrics.

*Figure: Table 2: LinkSage gains on user facing surface ranking model: from candidate Pins (top) and user sequence (bottom)*

**Ads**

Ads has adopted LinkSage into their Conversion ranking model and Engagement ranking model.

On Pinterest Ads, conversion rate per impression (iCVR), conversion volume, long click through rate (GCTR30), and cost per click (CPC) are the key indicators of user conversion and engagement. We observed significant gains on all the metrics.

*Figure: Table 3: Combined gains with LinkSage on Ads conversion (top) and engagement ranking model (bottom)*

**Board**

LinkSage use in the Boarding ranking model (or called Board Picker) has improved the understanding of external links. Significant gains have been observed:

*Figure: Table 4: LinkSage gains on Board ranking model*

# Summary

We developed LinkSage, a Graph Neural Network-based model, which is trained using a heterogeneous graph that supports different types of entities (e.g. Pins and landing pages). It leverages Pinner curated data to build the graph and training targets. It uses Pinterest ML signals (e.g. Cohesion and Neardup) to prune the graph/target and improve the graph density. It incorporates Pinterest ML signals (e.g. PinSage) into training to make its embedding space compatible with XSage. It applies cutting edge research of Matryoshka Representation Learning to provide multi-dimensional representation. It applies incremental serving to serve all the Pinterest landing pages corpus with a low computational cost and time.

We comprehensively evaluated the quality of LinkSage embeddings with offline metrics and online A/B experiments on surface ranking models. We have seen substantial online gains across multiple user facing surfaces, Ads, and Board, which covers all the key surfaces of Pinterest.

This work fills the blank of all the Pinterest off-site content understanding. It supercharges the backend of all the other landing pages signals’ development (e.g. Link Quality). It enriches Pinterest’s understanding of Pins, Pinterest users, and powers the future of ads and shopping at Pinterest.

If you are interested in this type of work we do, join Pinterest!

# Future work

In the latter version of LinkSage, we would improve the graph generation, feature engineering, and model architecture. We would incorporate more Pinterest entities in the heterogeneous graph to increase graph density. We would split crawled images as separate input to the transformer’s encoder to reduce information dilution. We would explore FastTransformer to save computation time and cost.

In addition to batch serving, we would establish a Near Real Time (NRT) infrastructure to serve LinkSage in real time. Pinterest has leveraged Apache Flink for NRT serving; for example, NRT Neardup successfully reduces the latency to sub-seconds instead of hours. We would establish a similar streaming pipeline to increase the coverage of fresh contents without compromising accuracy.

# Acknowledgement

Contributors to LinkSage development and adoption:

- ATG (GraphSage framework)

- Search Infrastructure (XPixie)

- Home Feed

- Ads Conversion

- Content Curation

- Notification

- Search

- Related Pins

- Ads Signal

- Ads Engagement

- Ads Relevance

*To learn more about engineering at Pinterest, check out the rest of our **Engineering Blog** and visit our **Pinterest Labs** site. To explore and apply to open roles, visit our **Careers** page.*
