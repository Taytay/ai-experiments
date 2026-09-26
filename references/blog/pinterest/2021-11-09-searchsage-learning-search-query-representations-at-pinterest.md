# SearchSage: Learning Search Query Representations at Pinterest

*Source: https://medium.com/pinterest-engineering/searchsage-learning-search-query-representations-at-pinterest-654f2bb887fc  
Published: 2021-11-09  
Fetched: 2026-09-21 via direct*

Data Science

Pinner Experience

Pinterest

Engineering

# SearchSage: Learning Search Query Representations at Pinterest

Pinterest Engineering

10 min read

·

Nov 9, 2021

--

Nikil Pancha | Software Engineer; Andrew Zhai | Software Engineer; Chuck Rosenberg | Head of Advanced Technologies Group; and Jure Leskovec | Chief Scientist, Advanced Technologies Group

Pinterest surfaces billions of ideas to people every day, and the neural modeling of embeddings for content, users, and search queries are key in the constant improvement of these machine learning-powered recommendations. Good embeddings — representations of discrete entities as vectors of numbers — enable fast candidate generation and are strong signals to models that classify, retrieve and rank relevant content.

We began our representation learning workstream with Visual Embeddings, a convolutional neural network (CNN) based Image representation, then moved toward PinSage, a graph-based multi-modal Pin representation. We expanded into more use cases such as PinnerSage, a user representation based on clustering a user’s past Pin actions, and have since worked with even more entities including search queries, Idea Pins, shopping items and content creators.

In this blog post we focus on SearchSage, our search query representation, and detail how we built and launched SearchSage for search retrieval and ranking to increase relevance of recommendations and engagement in search across organic Pins, Product Pins, and ads. Now used for 15+ use cases, this embedding is one of the most important features in both our organic and ads relevance models, and has led to metric wins such as an **11%** increase in 35s+ click-throughs on product Pins in search, and a **42%** increase in related searches.

*Figure: Search result blending*

When starting out, we aimed to solve both (1) a funnel efficiency problem with search retrieval and (2) to leverage user feedback and state-of-the-art language modeling to improve the quality of our search results. Pinterest is a visual platform, and when Pinners search on Pinterest, they’ll often make an initial judgement based on the image, not the text associated with it. Despite this, most search retrieval at Pinterest is based on exact text matching, where there are 3 stages:

- Candidates are generated based on token matches to a search query

- Those candidates are scored using a lightweight model, and the top N are retained

- These N results are ranked by a combination of factors, including their predicted relevance, engagement, and conversion likelihood

Previously, to incorporate visual results into search candidates, we followed an indirect approach: we would map search queries to the most engaged Pins for those queries (called cover Pins), fetch those Pin embeddings, cluster them, and then issue several approximate nearest neighbor (ANN) queries against a Hierarchical Navigable Small World (HNSW) index. The ANN results were then blended with text-based results before step 3, allowing ranking to score both text-based and embedding-based candidates.

Press enter or click to view image in full size

*Figure: Cover pin retrieval vs SearchSage*

While this existing approach had some notable benefits (e.g. handling query polysemy), the system was not learned end-to-end. Hence we aimed to build a learned retrieval system for search by leveraging the *(search query, pin) *engagement feedback coming from our users.

## Two-tower models

Press enter or click to view image in full size

*Figure: Two tower retrieval*

Traditionally, embeddings used for candidate generation are learned with two towers: one to embed a query, and another to embed items from the corpus used for retrieval. The similarity between these queries and items can be learned through traditional metric learning loss functions, such as a margin-based triplet loss, or sampled softmax.

Our aim in the development of SearchSage, however, was slightly different. Across Pinterest, many of our models already include PinSage embeddings as features. Similarly, we have built HNSW approximate nearest neighbor indices of PinSage embeddings to enable fast Pin-to-Pin candidate generation for Related Pins and home feed. Because of the existing infrastructure centered around PinSage (and the infra cost of storing extra 256d fp16 embeddings), we developed a learned embedding that’s compatible with PinSage. In other words, we have a two-tower model, but the candidate tower is frozen to be our PinSage embedding. This comes at the cost of model performance (more degrees of freedom usually leads to better models), but in terms of facilitating adoption, it was a clear choice.

## Training Data

Press enter or click to view image in full size

*Figure: Training data examples*

To train this model, we start with pairs of the form (search query, engaged Pin). We limit ourselves to Pin saves and long (35 sec+) click-throughs under the assumption that they carry more signal than weaker forms of engagement such as closeups (when a Pin is clicked but the user does not click through the webpage behind the Pin), or shorter clicks. For simplicity, we don’t gather explicit negative examples.

One heuristic that we’ve found to be useful for producing relevant models is capping of positive appearance counts. We sample data so that each engaged Pin may appear at most N times in our training data, which empirically helps to remove outlier Pins which may receive engagement for many different queries, and cause the model training to be unstable. Intuitively, this sampling strategy limits the model’s ability to learn to always retrieve certain popular Pins, as a single positive example may only be seen a limited number of times per epoch.

## Evaluation

As a proxy for online performance, we compute Recall@k (most often Recall@10), which we define as the proportion of (query, positive) pairs for which the engaged Pin is retrieved into the top k among an index of 1M random Pins drawn from the full corpus we plan on indexing. In Search, there are two tabs: the Explore tab (aimed towards content discovery), and the Shop tab (aimed towards helping users find content to purchase). In practice, we often retrieve from both organic and product corpora, so we measure performance on two evaluation datasets:

- Pin saves, evaluated against an index sampled from all Pins (“Organic engagement”)

- Long clicks of product Pins, evaluated against an index of all products (“Shopping engagement”)

## Model architecture

Press enter or click to view image in full size

*Figure: Model Architecture*

To embed queries, we use a small Transformer model, initialized to pretrained weights provided in Huggingface’s transformers package (distilbert-base-multilingual-cased). We considered and experimented with other architectures, including a CLSM-like architecture, bag of ngrams/character trigrams, and LSTMs, but found that Transformers are sufficiently performant to serve online, easy to train, and outperformed other architectures (if and only if fine-tuned end to end). We apply a single linear readout layer to the [CLS] token of the model (alternate pooling strategies didn’t improve performance, including max/sum/avg, or weighted combination of each layer’s [CLS] embedding). Despite the intimidating O(num_tokens²) cost of inferring a Transformer, we didn’t see substantially higher latency than with other deep architectures, as search queries typically contain less than 10 tokens.

## Loss function

*Figure: Visual depiction of softmax loss*

As a loss function, we use softmax over batch positive examples. In the literature, this method is used to decrease computation and complexity, as it only requires pairs of queries and positive examples, and no computation of embeddings for negative samples is required [Yi et al, 2019]. This amounts to treating our training as a classification problem, where we wish to predict a label of 1 for the engaged Pin, and 0 for all other Pins in the batch (normalization of query embeddings does not lead to substantially different performance). For the more mathematically inclined, this loss function can be described as follows:

Press enter or click to view image in full size

*Figure: Equation for loss softmax loss*

We’ve found this approach to be more effective than either using a margin loss or using more sophisticated negative sampling strategies with a triplet loss (akin to [Wu et al, 2017]) when mapping into a fixed embedding space.

## Multitask Learning

Our training data takes on a similar form to the evaluation; intuitively, we believe that training on data related to both of our evaluation metrics will give a reasonable tradeoff between shopping and organic performance. To evaluate this, we trained models on these three datasets:

- All Pin saves (organic)

- Product long click-throughs (shopping)

- A 50/50 blend of (1) and (2) (even_blend)

*Figure: Multi-task performance*

Above is a plot of Recall@10 (as defined above), measured on the organic and shopping evaluation datasets, with each color representing a different training dataset. We see that solely optimizing for shopping performance substantially degrades organic metrics without improving our shopping evaluation over a 50/50 blend. Optimizing solely for organic does increase metrics on organic tasks, but the improvement is relatively marginal, and comes at a larger cost to shopping metrics. This verifies that a 50/50 multi-task setup outperforms a single-task learning setup offline.

## Serving

To efficiently serve the model, we use an internal C++ multi-DNN framework inference platform built on top of TensorFlow Serving, which supports dynamically batching requests online. Every 5 ms (or when the max batch size of requests is accumulated), we group batch_size pending requests together for inference, allowing for significant throughput increases at slight latency cost.

One challenge to overcome is that commonly for text models the input preprocessing and model inference are separate. Preprocessing consists of the tokenization — taking raw text and producing outputs such as character trigrams, unigrams, wordpieces, or sentence pieces. This can lead to some efficiency improvements as preprocessing can happen asynchronously; however, because the preprocessing commonly is custom code, maintenance becomes a problem, especially aligning training and serving preprocessing methods. Specifically, we train in a Python environment and serve in a C++ only environment. To simplify this maintenance we opted for building a custom PyTorch C++ operator for text preprocessing. This allows us to have a single, self-contained artifact, taking an input string List[str] of length N and returning a (N, D) tensor of embeddings. During training this custom op is exposed as a Python Pytorch operator for us to leverage.

## Results

Because SearchSage was so different from what was in production, we tuned our model based on our offline metrics and then ran the best model we found offline in an A/B experiment to verify gains over the cover Pin based approach.

Despite training the model only on engagement, we saw increases in product-only search relevance (**+2 percentage points (pp) P@8** weighted by volume, +8pp P@8 if all queries are assigned equal weight), and overall search relevance for shopping-related queries (**+1pp P@25** weighted by volume).

We also observed an **11%** increase in product long click throughs with an **8%** increase in product impressions in search, increasing engagement rate, which is an indication that the content retrieved through SearchSage offered more utility to users than the previous approach.

SearchSage generally works well on both head and tail queries. Below are a few examples of ANN results retrieved from our full corpus of shopping pins:

Press enter or click to view image in full size

*Figure: Example approximate nearest neighbor results*

## Summary

In this post, we gave an overview of SearchSage, a query embedding designed to replace indirect solutions. The previous setup represented a query as a set of clusters of PinSage embeddings produced by clustering historical engagement for that query. We believed that a model that explicitly produced query embeddings compatible with this Pin embedding space from raw text would perform better than the former approach, and this was validated through online experiments, showing increases in search engagement and relevance.

In the future, we’ll look into better representing both queries and Pins in this model, as initial experiments have shown large performance improvements by allowing the candidate embeddings to be learned as well. We will also continue to explore ways to improve our query representation, following some of the newer literature, which points to promising results when including a graph structure in the query embedding model.

## *Acknowledgements*

The authors would like to thank the following people for their contributions: Laksh Bhasin, Yan Wang, Cosmin Negruseri, Pak Ming Cheung, Yinrui Li, Kurchi Subhra Hazra, Vijai Mohan, Rajat Raina, Pong Eksombatchai, Zhiyuan Zhang

*To learn more about engineering at Pinterest, check out the rest of our **Engineering Blog**, and visit our **Pinterest Labs** site. To view and apply to open opportunities, visit our **Careers** page.*

## Referenced Works

- Unifying visual embeddings for visual search at Pinterest | by Pinterest Engineering | Pinterest Engineering Blog

- PinSage: A new graph convolutional neural network for web-scale recommender systems

- PinnerSage: Multi-Modal User Embedding Framework for Recommendations at Pinterest

- Y. A. Malkov and D. A. Yashunin, “Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs,” IEEE transactions on pattern analysis and machine intelligence, vol. 42, no. 4, pp. 824–836, 2018.

- T. Wolf et al., HuggingFace’s Transformers: State-of-the-art Natural Language Processing. 2019.

- Y. Shen, X. He, J. Gao, L. Deng, and G. Mesnil, “A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval,” in Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management, 2014, pp. 101–110. doi: 10.1145/2661829.2661935.

- P. Nigam et al., “Semantic Product Search,” in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2019, pp. 2876–2885. doi: 10.1145/3292500.3330759.

- X. Yi et al., “Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations,” in Proceedings of the 13th ACM Conference on Recommender Systems, 2019, pp. 269–277. doi: 10.1145/3298689.3346996.

- C.-Y. Wu, R. Manmatha, A. J. Smola, and P. Krahenbuhl, “Sampling matters in deep embedding learning,” in Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 2840–2848.
