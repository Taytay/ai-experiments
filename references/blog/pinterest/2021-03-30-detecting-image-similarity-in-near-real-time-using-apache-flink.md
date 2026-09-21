# Detecting Image Similarity in (Near) Real-time Using Apache Flink

*Source: https://medium.com/pinterest-engineering/detecting-image-similarity-in-near-real-time-using-apache-flink-723ce072b7d2  
Published: 2021-03-30  
Fetched: 2026-09-21 via wayback*

# Detecting Image Similarity in (Near) Real-time Using Apache Flink

Pinterest Engineering

·

Follow

Published in

Pinterest Engineering Blog

·

6 min read

·

Mar 30, 2021

--

Shaji Chennan Kunnummel| Software Engineer, Content Quality

Iaroslav Tymchenko| Software Engineer, Content Quality

Pinterest is a visual platform at its core, so the need to understand and act on images is paramount. A couple of years ago, the Content Quality team designed and implemented our own batch pipeline to detect similar images. The similarity signal is widely used at Pinterest for use cases varying from improving recommendations based on similar images to taking down spam and abusive content. However, it was taking several hours for the signal to be computed for newly created images, which was a long window for spammers and abusers to harm the platform. So recently, the team implemented a streaming pipeline to detect similar images in near-real-time.

Given the platform’s scale, identifying duplicate images has been difficult, and doing it in real-time is even more challenging. This blog post focuses on the work the Content Quality team did recently to leverage Apache Flink to detect duplicate images in (near) real-time.

The project’s goal was to reduce the latency to sub-seconds instead of the hours-long latency the batch pipeline takes without compromising accuracy and coverage.

Specifically, we wanted to solve the following two problems:

- Given an image, find if the same image (or a slight variation, aka *NearDup*) had been used at Pinterest before

- Given an image, find the list of all similar images used on Pinterest

For practical reasons, the entire universe of images used at Pinterest is broken down into a set of non-overlapping clusters. Note the similarity relation is not transitive, hence an approximate relation is used to partition the images. For every cluster, a representative member is picked (at random) and used as a cluster-ID. More specifically, we use the following relations between images to represent the disjoint clusters:

- Image (aka *cluster member*) to canonical image (aka *cluster head*)

- Canonical image to the list of cluster members

The rest of the article focuses on the design and implementation of the real-time pipeline. Note that this article is not about detecting image similarity but about *how* to do it in real-time. The details of how to detect image similarity using a Locality-Sensitive Hashing (LSH) search and a TensorFlow-based classifier are explained in detail in this previous blog post named “Detecting image similarity using Spark, LSH and TensorFlow” and these articles.

# Challenges

The sheer volume of images on Pinterest poses a set of challenges in terms of scalability and robustness. The numbers given below give glimpses of the scale we are dealing with:

- Number of Pins saved across Pinterest: **300B**

- Rate of image creation per second: **~100 (and 200 at peak)**

- Number of cluster members: **6 on an average but as high as 1.1M for a handful of clusters**

Given the importance of the signal and the impact that it could have if the signal is delayed/corrupted, we had to bake the following aspects into the system right from the beginning:

- Easiness of debugging

- Explainability of the signal

- Real-time and long term monitoring of the health of the signal

- Capability to reprocess a subset of images in case of catastrophic failures

- Ability to make a switch from the batch pipeline to the new pipeline as seamlessly as possible

# Design and Implementation

For every newly created image, we run the following steps to detect similar images:

- Extract LSH terms from the visual embeddings

- Query custom search engine (bootstrapped with the LSH terms index) to identify a set of potential candidates. The candidates are sorted based on the number of terms matched with the image in question.

- Evaluate the set of candidates using a TensorFlow-based classifier. We use an empirically determined threshold to filter out non-matching images

- Identify the cluster if a similar image is detected and update storage.

The entire system is built as an Apache Flink workflow. At a high level, the similarity computation is triggered as soon as the embeddings are ready. The media team at Pinterest has made notifications available through Kafka.

## Architecture Diagram

The diagram given in this section captures the essence of the architecture of the pipeline.

## Stream-Stream Join

The similarity computation uses different embeddings (partly for historical purposes) for LSH and machine learning evaluation. Typically embeddings are made available within a few seconds, and the pipeline uses a stream-stream join to synchronize the availability of multiple embeddings.

## Manas: Custom Search Engine

We use Manas (Pinterest’s configurable search engine) to find potential candidates through LSH term matching. The details of how the LSH terms are used in identifying similar images are explained in the previous blog post.

Since we need the candidates to be sorted based on their number of overlapping terms,

the search cluster has been optimized for correctness over latency. Unlike traditional search engines, our use case typically requires the entire corpus to be scanned, and the results with the highest term overlappings are expected to be returned. The extensive document scanning does stress the search infrastructure and requires strict rate-limiting to regulate the rate of search queries.

Once the similarity score is computed, the search index also gets updated to make the newly created images searchable.

## TensorFlow Model Serving

We leverage Pinterest’s ML serving infrastructure named Scorpion to evaluate the selected set of candidates. Given the problem’s scale (at the peak, nearly 500k instances are getting evaluated per second), the model serving uses non-trivial optimizations like GPUs and micro batching for better performance.

## Storage and Serving

If a duplicate image is detected, underlying storage needs to be updated to serve the mappings. As mentioned above, there are two relationships we persist in the storage:

- Image to cluster head mapping

- Cluster head to the list of cluster members

Image to cluster head mapping is simple and stored in a homegrown variation of rocksdb, which provides us low latency and linear scalability.

However, the cluster head to the list of members relation is a more complex relation to maintain as the cluster size is heavily skewed (average size is six but goes to a million or so for a few clusters). The cluster head to the list of members relation is stored as a graph (nodes being images, and edges representing the cluster head to image mapping) in Pinterest’s own graph storage system called Zen. The primary reason for using graph storage is to take advantage of its pagination support for fetching edges (without pagination there would be K-V pairs with very large sizes of V which would limit its use in the online K-V systems).

The relations are served through the generalized signal delivery system called Galaxy, which provides low latency fetching of signals.

## Bootstrapping Existing Relationships

We leveraged Flink’s file watcher feature to bootstrap rocksdb and Zen graph storage. Historical data was transformed into the schema that the Flink workflow understands, and saved in a directory on AWS S3. A file watcher operator is added to the workflow to watch the S3 location and bulk upload data into the storage systems.

# Operability of the Pipeline

The pipeline has been designed and implemented with operability aspects.

## Debuggability

Since the pipeline is complex, we have implemented special debugging data propagation through the Flink operators. The debug details are pushed to Kafka queues and persisted using Pinterest’s own scalable Kafka materialization infrastructure named Merced. There is also capability built into the system to selectively ingest image IDs into the pipeline and inspect the intermediate results in real-time for better and easy debugging.

## Monitoring and Alerting

Apart from using the standard metrics provided by Flink, we also have many custom metrics to measure the pipeline’s health. There are also hourly jobs running over the materialized Kafka logs to measure the coverage and other standard metrics to detect model skew, etc.

## Dealing with Failures

We have built the following tools to handle failures and bugs:

- Tools to rollback to a good state in case of failures to any of the major components in the pipeline

- Tools to fix false positives by forcefully changing the image to the cluster head mapping

# Future Work

What started as an image-centric pipeline found applications beyond static images to dynamic Pins like videos and Story Pins. In addition, it can also be generalized to handle any kind of data where near-dup relation exists, creating opportunities for more efficiencies in the future.

*Acknowledgments: This post summarizes several quarter’s work that involved multiple teams. Thanks to Michael Mi, Andrey Gusev, Haibin Xie Saurabh Joshi, Sheng Cheng, Bin Shi, Ankit Patel, Qingxian Lai, Karthik Anantha Padmanabhan, Lu Niu, Heng Zhang, Nilesh Gohel, Teja Thotapalli, Nick DeChant*
