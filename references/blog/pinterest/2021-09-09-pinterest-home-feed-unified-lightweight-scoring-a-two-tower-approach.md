# Pinterest Home Feed Unified Lightweight Scoring: A Two-tower Approach

*Source: https://medium.com/pinterest-engineering/pinterest-home-feed-unified-lightweight-scoring-a-two-tower-approach-b3143ac70b55  
Published: 2021-09-09  
Fetched: 2026-09-21 via wayback*

# Pinterest Home Feed Unified Lightweight Scoring: A Two-tower Approach

Pinterest Engineering

8 min read

·

Sep 9, 2021

--

Dafang He | Software Engineer, Home Candidate Generation; Andrew Liu; Dhruvil Deven Badani | Software Engineer, Homefeed Ranking; Poorvi Bhargava; Sangmin Shin |Engineering Manager, Home Ranking; Duo Zhang | Engineering Manager, Candidate Generation; and Jay Adams | Software Engineer, Inspire

# Intro

Pinterest is a place where users (Pinners) can save and discover content from both web and mobile platforms, and where increasingly Creators can publish native content right to Pinterest. We hold billions of content (Pins) in our corpus and serve personalized recommendations that inspire Pinners to create a life they love. One of the key and most complicated surfaces for Pinterest is the home feed, where Pinners will see personalized feeds based on their engagement and interests. In this blog, we will discuss how we unify our light-weight scoring layer across the various candidate generators that power home feed recommendations.

# Motivations

Home feed is what Pinners see first when they open the Pinterest app. To give relevant and diverse recommendations, we use a recommendation system comprising many different sources. One major source, for example, is Pixie[1], which is based on a random walk of the bipartite pin-board graph. Based on the Pixie platform, we are able to generate multiple different sources, some directly returning pins from a random walk based on the engagement history, and some based on the pins retrieving from boards that returned from Pixie random walk. In addition to Pixie, we also have recommendation sources that take in the topics or based on embeddings. These candidate generators usually have their own light-weight scoring model, which ranks and selects the most relevant candidates sent to final ranking. For example, we have a gbdt light-weight scoring model used in production after random walk[2]. The overall picture of home feed recommendation engines is in Fig. 1.

Most of these recommendation sources have:

- Individual training data generation pipelines

- Different online serving approaches

- Different light-weight models which have different feature-set available during serving.

It takes a tremendous amount of engineering effort to develop a single machine learning model that serves only one specific candidate generator. This greatly limits the development speed for ML engineers.

In addition to the development speed, we are also seeing an increased online serving cost if we increase the feature set in order to improve the light-weight models. This is because each candidate source will need to do a relatively complicated online model computation on a large set of candidates (Pins). For example, for one of the sources that is powered by pixie light weight scoring [2], it uses a gbdt model in serving and needs to compute thousands of pins’ rankings per request. This computational overhead makes it hard to conduct feature engineering without introducing significant serving cost.

Press enter or click to view image in full size

*Figure: Fig. 1. *Overview of current home feed recommendation pipeline. *Home feed recommendation is powered by many different candidate generators. Each of them serves a unique role and has its own light-weight ranking layer.*

Last but not least, many features are also not available during the online request phase for certain candidate sources. Adding these features will introduce infrastructure overhead and make it hard to justify the performance gain before actually collecting data and running experiments. This pain point was mentioned in the Pixie lws blog post as well.

All these factors motivate us to **unify** our machine learning modeling and serving approach for these candidate generators to provide a more personalized set of candidates to the final ranking model. Thus we decided to move to use a unified two-tower model[3], [4] for the light-weight modeling.

# Two-Tower Architecture

In this section, we are going to describe our two-tower modeling approach for a unified light scoring with the goal of resolving the above pain points.

The overview of the two tower architecture is shown in Fig. 2. It has a separate user tower and a Pin tower with a final dot product that computes the similarity between a given user and a given Pin. The Pin tower takes the features from the given Pin and generates a Pin embedding for it. The features used include dense features such as Pin’s recent performance as well as sparse features such as category. The sparse feature will be passed into an embedding layer before sending to the pin’s MLP for final embedding computation. The user tower takes the engagement history features as input and generates a user-specific embedding. Finally, we do a simple dot product based on the two embeddings as a measure of how likely the Pinner will engage with the Pin.

Press enter or click to view image in full size

*Figure: Figure 2. *The overview of two-tower architecture used in unified light-weight scoring. On the left side, it is the pin tower that computes the pin’s embedding, and on the right side we have the user tower that generates the user embedding.**

# Training

For the optimization of the two-tower model on the lightweight ranking layer, we must treat it differently than the recommendation on the ranking layer, since compared with the Pins that will be used for ranking, we will be facing much more negative candidates in serving on candidate generator level. Thus we apply the in-batch negative sampling approach [5] and use a batch size of 6000 in our training so as to make each positive candidate coupled with enough negative candidates generated for free. We found by applying this, we are able to achieve much better offline metrics as measured by recall at top k.

# Serving

In this section, we are going to describe how we serve the machine learning model to support online candidate retrieval recommendations. The serving architecture will include two parts: the Pin embedding serving and the user embedding serving. We will discuss the two parts separately here.

## **Pin Embedding Serving**

Home feed powers recommendation for all possible candidate Pins in Pinterest. So we have to compute all Pins’ embeddings. Usually for a recommendation engine the content distribution will follow a long-tail rule and it is important for us to avoid re-computation of the same pin embedding each time for an online request, as many of the requests will be attributed to the same pin. In another aspect, pins’ contents don’t shift much, so their embeddings are relatively stable. Thus we should be able to compute the embeddings for most of the pins in an offline workflow. In the online requests, we only need to retrieve the precomputed embedding instead of doing a recomputation. For fresh Pins (Pins that entered the Pinterest eco-system within hours), we will need to do online inference for them as they will not be picked up by the offline workflow. So we separate out Pins’ embedding computation into two different pipelines, as shown in Fig. 3.

## **User Embedding Serving**

Pinners tend to change their status instantly. For example, if a Pinner engaged with a cat Pin, they are likely to engage with another cat Pin in the near future. Thus it is important for us to capture the status change which reflects their realtime interests. In order to do so, we enable online user embedding computation whenever there is a new user request. Adding something like this won’t be a costly computation, as we only calculate once per request.

Press enter or click to view image in full size

*Figure: Figure 3. *The overall serving pipeline for candidate retrieval with two-tower embedding computation. Most of the existing pins will be inserted into candidate generators with an offline workflow that computes pin embeddings, while new pins will be inserted to be served after they get online pin embedding computation*.*

# Wins

In order to evaluate the performance of the unified lightweight scoring approach, we did an online experiment with the light-weight scoring layer applied to all the candidate generators. We see the gains in the following aspects.

## **Engagement Win**

We see a huge engagement lift by applying the light-weight ranking layer to all the candidate generators. For example, total saves and closeups both increased 2–3%. We also see total hides drop 3–4% as well. These metrics wins demonstrate the relevance improvement while using the two-tower architecture to replace the old light-weight ranking approach. We think this is because the two-tower approach can leverage all relevant features available for the Pin and the Pinner and thus we can get a better embedding representation for both.

## **Diversity**

One concern of applying the same model to all candidate generators is that it will make our recommendation less diverse. As a platform that aims at inspiring Pinners, we don’t want to drop our recommendation diversity. However, in our online experiment, we actually see increased diversity in adoption (save, closeup, etc). We believe this is because (1) the generated user embedding can encode diverse interests based on engagement history. (2) a better recommendation powered by the two-tower approach filtered out unrelated Pins and the recommended items are more likely to be adopted.

## **Infra cost**

Online computation with thousands of pins for each request is costly. This is especially the case considering the number of candidate generators we are working with. By applying a simple dot product at the light-weight ranking layer with the two-tower approach, we significantly dropped our online serving cost for each request. Considering a dot product with two 64 dimensional embedding vectors, we only need n multiplication operations with n add operations. This is significantly cheaper than, for example, a logistic regression model with online feature transformation, bucketization. Usually this ends up with a long vector with huge online serving cost in our original serving system.

# **Summary & Future Works**

In this post, we gave an overview of our unified lightweight ranking layer currently used in Homefeed. We focus on the major motivations for these efforts and the learnings from them. To summarize, in a machine learning modeling perspective, a densely encoded learned embedding vector gives better recommendation and we achieved gains not just in engagement, but also in adoption diversity and infra cost saving. At the same time, we are able to apply the same model across different serving infrastructures used by different candidate generators.

For future work, we will try to keep improving the light-weight ranking model with better features and ML modeling. We will also try applying the model to some other newly added candidate generators to further unify our light-weight scoring layer.

# **Acknowledgements:**

Bowen Deng, Zhaohui Wu, Haibin Xie, Wangfan Fu, Angela Sheu, Se Won Jang, Kent Jiang, Michael Mi, Zheng Liu, Dylan Wang, Zhiyuan Zhang, Sihan Wang, Bee-chung Chen, Liang Zhang.

*To learn more about engineering at Pinterest, check out the rest of our **Engineering Blog**, and visit our **Pinterest Labs** site. To view and apply to open opportunities, visit our **Careers** page.*

## **References:**

[1] Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time

[2] Pixie lws blog post: https://medium.com/pinterest-engineering/improving-the-quality-of-recommended-pins-with-lightweight-ranking-8ff5477b20e3

[3] Deep neural networks for youtube recommendations. Covington, P., Adams, J., and Sargin, E.

[4] Efficient training on very large corpora via gramian estimation. Krichene, W., Mayoraz, N., Rendle, S., Zhang, L., Yi, X., Hong, L., Chi, E., and Anderson, J

[5] Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations. Xinyang Yi Ji Yang Lichan Hong Derek Zhiyuan Cheng Lukasz Heldt Aditee Ajit Kumthekar Zhe Zhao Li Wei Ed Chi.
