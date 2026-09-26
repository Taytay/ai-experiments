# Large-scale User Sequences at Pinterest

*Source: https://medium.com/pinterest-engineering/large-scale-user-sequences-at-pinterest-78a5075a3fe9  
Published: 2023-05-02  
Fetched: 2026-09-21 via wayback*

# Large-scale User Sequences at Pinterest

Pinterest Engineering

·

Follow

Published in

Pinterest Engineering Blog

·

11 min read

·

May 2, 2023

--

User Understanding team: Zefan Fu, Minzhe Zhou, Neng Gu, Leo Zhang, Kimmie Hua, Sufyan Suliman | Software Engineer, Yitong Zhou | Software Engineering Manager

Index Core Entity team: Dumitru Daniliuc, Jisong Liu, Kangnan Li | Software Engineer, Shunping Chiu | Software Engineering Manager

Understanding and responding to user actions and preferences is critical to delivering a personalized, high quality user experience. In this blog post, we’ll discuss how multiple teams joined together to build a new large-scale, highly-flexible, and cost-efficient user signal platform service, which indexes the relevant user events in near real-time, constructs them into user sequences, and makes it super easy to use both for online service requests and for ML training & inferences.

# Background & Context

User sequence is one type of ML feature composed as a time-ordered list of user engagement activities. The sequence captures one’s recent actions in real-time, reflecting their latest interests as well as their shift of focus. This kind of signal plays a critical role in various ML applications, especially for large-scale sequential modeling applications (see example).

To make the real-time user sequence more accessible within the Pinterest ML ecosystem, and to empower our daily metrics improvement, we list the following key features to deliver for ML applications:

- **Real-time**: on average < 2 seconds latency from a user’s latest action to the service response

- **Flexibility**: data can be fetched and reused by a mix-and-use pattern to enable faster iterations for ML engineers focusing on quick development time

- **Platform**: serve all different needs and requests with a uniform data API layer

- **Cost Efficient**: improve infra shareability and reusability, and avoid duplications in storage or computation wherever possible

Taxonomy:

- **Signal**: the data inputs for downstream applications specifically in machine learning applications

- **User Sequence**: a specific kind of user signals that arranges user’s past activities in a strict temporal order and joins each activity with enrichment data

- **Unified Feature Representation**: or “UFR” is a feature format for all Pinterest model features

# Infrastructure Design

Our infrastructure adopts a lambda architecture: the real-time indexing pipeline, the offline indexing pipeline, and the serving side components.

## Real-Time Indexing Pipeline

The main goal of the real-time indexing pipeline is to enrich, store, and serve the last few relevant user actions as they come in. At Pinterest, most of our streaming jobs are built on top of Apache Flink, because Flink is a mature streaming framework with a lot of adoption in the industry. So our user sequence real-time indexing pipeline is composed of a Flink job that reads the relevant events as they come into our Kafka streams, fetches the desired features for each event from our feature services, and stores the enriched events into our KV store system. We set up a separate dataset for each event type indexed by our system, because we want to have the flexibility to scale these datasets independently. For example, if a user is much more likely to click on pins than to repin them, it might be enough to store the last 10 repins per user, and at the same time we might want to store the last 100 “close-ups.”

It’s worth noting that the choice of the KV store technology is extremely important, because it can have a big impact on the overall efficiency (and ultimately, cost) of the entire infrastructure, as well as the complexity of the real-time indexing job. In particular, we wanted our KV store datasets to have the following properties:

- **Allows inserts.** We need each dataset to store the last N events for a user. However, when we process a new event for a user, we do not want to read the existing N events, update them, and then write them all back to the respective dataset. This is inefficient (processing each event takes O(N) time instead of O(1)), and it can lead to concurrent modification issues if two hosts process two different events for the same user at the same time. Therefore, our most important requirement for our storage layer was to be able to handle inserts.

- **Handles out-of-order inserts.** We want our datasets to store the events for each user ordered in reverse chronological order (newest events first), because then we can fetch them in the most efficient way. However, we cannot guarantee the order in which our real-time indexing job will process the events, and we do not want to introduce an artificial processing delay (to order the events), because we want an infrastructure that allows us to immediately react to any user action. Therefore, it was imperative that the storage layer is able to handle out-of-order inserts.

- **Handles duplicate values.** Delegating the deduplication responsibility to the storage layer has allowed us to run our real-time indexing job with “at least once” semantic, which has greatly reduced its complexity and the number of failure scenarios we needed to address.

Fortunately, Pinterest’s internal wide column storage system (built on top of RocksDB) could satisfy all these requirements, which has allowed us to keep our real-time indexing job fairly simple.

## Cost Efficient Storage

In the ML world, there is no gain that can be sustained without taking care of the cost. No matter how fancy an ML model is, it must function within reasonable infrastructure costs. In addition, a cost saving infra usually comes with optimized computing and storage which in turn contribute to the stableness of the system.

When we designed and implemented this system, we kept cost efficiency in mind from day one. To build up this system, the cost comes from two parts: computing and storage. We implemented various ways to reduce the cost from these two parts without sacrificing system performance.

- **Computing cost efficiency**: During indexing time, at a high level, Flink jobs should consume from the latest new events and apply these updates to the existing storage, representing the historical user sequence. Instead of read, modify and write back, our Flink job is designed to only append new events to the end of user sequence and rely on storage periodical clean-up thread to maintain user sequence length under limitation. Compared with read-modify-write, which has to load all previous user sequence into Flink job, this approach uses far less memory and CPU. This optimization also allows this job to handle more volume when we want to index more user events.

- **Storage cost efficiency**: To chase down storage costs, we encourage data sharing across different use sequence use cases and only store the enrichment of a user event when multiple use cases need it. For example, let’s say use case 1 needs to click_event and view_event with enrichment A and B, and use case 2 needs to click_event with enrichment A only. Use case 1 and 2 will fetch click_event from the same dataset, and only enrichment A is built-in. Use case 1 needs to fetch view_event from another dataset and fetch enrichment B in the serving time. This principle helps us maximize the data sharing across different use cases.

## Offline Indexing Pipeline

Having a real-time indexing pipeline is critical, because it allows us to react to user actions and adjust our recommendations in real-time. However, it has some limitations. For example, we cannot use it to add new signals to the events that were already indexed. That is why we also built an offline pipeline of Spark jobs to help us:

- **Enrich and store events daily. **If the real-time pipeline missed or incorrectly enriched some events (due to some unexpected issues), the offline pipeline will correct them.

- **Bootstrap a dataset for a new relevant event type**. Whenever we need to bootstrap a dataset for a new event type, we can run the offline pipeline for that event type for the last N days, instead of waiting for N days for the real-time indexing pipeline to produce data.

- **Add new enrichments to indexed events. **Whenever a new feature becomes available, we can easily update our offline indexing pipeline to enrich all indexed events with the new feature.

- **Try out various event selection algorithms. **For now, our user sequences are based on the last N events of a user. However, in the future, we’d like to experiment with our event selection algorithm (for example, instead of selecting the last N events, we could select the “most relevant” N events). Since our real-time indexing pipeline needs to enrich and index events as fast as possible, we might not be able to add sophisticated event selection algorithms to it. However, it would be very easy to experiment with the event selection algorithm in our offline indexing pipeline.

Finally, since we want our infrastructure to provide as much flexibility as possible to our product teams, we need our offline indexing pipeline to enrich and store as many events as possible. At the same time, we have to be mindful of our storage and operational costs. For now, we have decided to store the last few thousand events for each user, which makes our offline indexing pipeline process PBs of data. However, our offline pipeline is designed to be able to process much more data, and we can easily scale up the number of events stored per user in the future, if needed.

## Serving Layer

Our API is built on top of the Galaxy framework (i.e. Pinterest’s internal signal processing and serving stack) and offers two types of responses: Thrift and UFR . Thrift allows for greater flexibility by allowing the return of raw or aggregated features. UFR is ideal for direct consumption by models.

Our serving layer has several features that make it useful for experiments and testing new ideas. Tenant separation ensures that use cases are isolated from each other, preventing problems from propagating. Tenant separation is implemented in feature registration, logging and signal level logic isolation. We ensure the heavy processing of one use case does not affect others. While features can be easily shared, the input parameters are strictly tied to feature definition so no other use case can mess up the data. Health metrics and built-in validations ensure stability and reliability. The serving layer is also flexible, allowing for easy experimentation at low cost. Clients can test multiple approaches within a single experiment and quickly iterate to find the best solution. We provide tuning configurations in many ways, different sequence combinations, feature length, filtering thresholds, etc, all of which can change immediately on-the-fly.

More specifically, at the serving layer, decoupled modules handle different tasks during the processing of a request. The first module retrieves key-value data from the storage system. This data is then passed through a filter, which removes any unnecessary or duplicate information. Next, the enricher module adds additional embedding to the data by joining from various sources. The sizer module trims the data to a consistent size, and the featurizer module converts the data into a format that can be easily consumed by models. By separating these tasks into distinct modules, we can more easily maintain and update the serving layer as needed.

# What Are the Tradeoffs

The decision to enrich embedding data at indexing time or serving time can have a significant impact on both the size we store in kv and the time it takes to retrieve data during serving. This trade-off between indexing time and serving time is essentially a balancing act between storage cost and latency. Moving heavy joins to indexing time may result in smaller serving latency, but it also increases storage cost.

Our decision-making rules have evolved to emphasize cutting storage size as follows:

- If it’s an experimental user sequence, it is added to the serving time enricher

- If it’s not shared with multiple surfaces, it is also added to the serving time enricher

- If a timeout is reached during serving time, it is added to the indexing time enricher

# Collaboration Model

Building and effectively using a generic infrastructure of this scale requires commitment from multiple teams. Traditionally, product engineers need to be exposed to the infra complexity, including data schema, resource provisions, and storage allocations, which involves multiple teams. For example, when product engineers want to make use of a new enrichment in their models, they need to work with the indexing team to make sure that the enrichment is added to the relevant data, and in turn, the indexing team needs to work with the storage team to make sure that our data stores have the required capacity. Therefore, it is important to have a collaboration model that hides the complexity by clearly defining the responsibilities of each team and the way teams communicate requirements to each other.

Reducing the number of dependencies for each team is key to making that team as efficient as possible. This is why we have divided our user sequence infrastructure into multiple horizontal layers, and we devised a collaboration model that requires each layer to talk only to the layer directly above and the one directly below.

In this model, the User Understanding team takes ownership of the serving-side components and is the only team that interacts with the product teams. On one hand, we hide the complexity of this infrastructure from the product teams and provide the product teams with a single point of contact for all their requests. On the other hand, it gives the User Understanding team visibility into all product requirements, which allows them to design generic serving-side components that can be reused by multiple product teams. Similarly, if a new product requirement cannot be satisfied on the serving side and needs some indexing-side changes, the User Understanding team is responsible for communicating those requirements to the Indexing Core Entities team, which owns the indexing components. The Indexing Core Entities team then communicates with the “core services” teams as needed, in order to create new datasets, provision more processing resources, etc., without exposing all these details to the teams higher up in the stack.

Having this “collaboration chain” (rather than a tree or graph of dependencies at each level) also makes it much easier for us to keep track of all work that needs to be done to onboard new use cases onto this infrastructure: at any point in time, any new use case is blocked by one and only one team, and once that blocker is resolved, we automatically know which team needs to work on the next steps.

# User Sequence ML Application

UFR logging is often used both for model training and model serving. Most models keep the data at serving time and use it for training purposes to make sure they are the same.

Inside Model structure, user sequence features are fed into sequence transformer and merged at feature cross layer

For more detail information, please check out this engineering article on HomeFeed model taking in User Sequence and boost Engagement Volume

# Conclusion

In this blog, we presented a new user sequence infra that introduces significant improvements on **real-time responsiveness**, **flexibility**, **and** **cost efficiency. **Different than our previous real-time user signal infra, this platform has been much more scalable and maximizes storage reusability. We’ve had successful adoptions such as in homefeed recommendation driving significant user engagement gains. This platform is also a key component for PinnerFormer work providing real-time user sequence data.

For future work, we are looking into both more efficient and scalable data storage solutions, such as event compression or online-offline lambda architecture, as well as more scalable online model inference capability integrated into the streaming platform. In the long run, we envision the real-time user signal sequence platform serving as an essential infrastructure foundation for all recommendation systems at Pinterest.

# Acknowledgements

Contributors to user sequence adoption:

- HomeFeed Ranking

- HomeFeed Candidate Generation

- Notifications Relevance

- Activation Foundation

- Search Ranking and Blending

- Closeup Ranking & Blending

- Ads Whole Page Optimization

- ATG Applied Science

- Ads Engagement

- Ads Ocpm

- Ads Retrieval

- Ads Relevance

- Home Product

- Galaxy

- KV Storage Team

- Realtime Data Warehouse Team

*To learn more about engineering at Pinterest, check out the rest of our **Engineering Blog** and visit our **Pinterest Labs** site. To explore life at Pinterest, visit our **Careers** page.*
