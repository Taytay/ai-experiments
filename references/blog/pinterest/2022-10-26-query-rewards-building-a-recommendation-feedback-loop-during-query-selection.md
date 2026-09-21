# Query Rewards: Building a Recommendation Feedback Loop During Query Selection

*Source: https://medium.com/pinterest-engineering/query-rewards-building-a-recommendation-feedback-loop-during-query-selection-70b4d20e5ea0  
Published: 2022-10-26  
Fetched: 2026-09-21 via wayback*

# Query Rewards: Building a Recommendation Feedback Loop During Query Selection

Pinterest Engineering

·

Follow

Published in

Pinterest Engineering Blog

·

4 min read

·

Oct 26, 2022

--

Bella Huang | Software Engineer, Home Candidate Generation; Raymond Hsu | Engineer Manager, Home Candidate Generation; Dylan Wang | Engineer Manager, Home Relevance

In Homefeed, ~30% of recommended pins come from pin to pin-based retrieval. This means that during the retrieval stage, we use a batch of query pins to call our retrieval system to generate pin recommendations. We typically use a user’s previously engaged pins, and a user may have hundreds (or thousands!) of engaged pins, so a key problem for us is: how do we select the right query pins from the user’s profile?

# User Profiling with PinnerSAGE Overview

At Pinterest, we use PinnerSAGE as the main source of a user’s pin profile. PinnerSAGE generates clusters of the user’s engaged pins based on the pin embedding by grouping nearby pins together. Each cluster represents a certain use case of the user and allows for diversity by selecting query pins from different clusters. We sample the PinnerSAGE clusters as the source of the queries.

# Homefeed Query Composition Before Adding Query Reward

Previously, we sampled the clusters based on raw counts of actions in the cluster. However, there are several drawbacks for this basic sampling approach:

- The query selection is relatively static if no new engagements happen. The main reason is that we only consider the action amount when we sample the clusters. Unless the user takes a significant number of new actions, the sampling distribution remains roughly the same.

- No feedback is used for the future query selection. During each cluster sampling, we don’t consider the downstream engagements from the last request’s sampling results. A user may have had positive or negative engagement on the previous request, but don’t take that into account for their next request.

- It cannot differentiate between the same action types aside from their timestamp. For example, if the actions inside the same cluster all happened around the same time, the weight of each action will be the same.

*Figure: *Figure 1. Previous query selection flow**

# Homefeed Query Composition After Adding Query Reward

*Figure: *Figure 2. Current query selection flow with query reward**

To address the shortcomings of the previous approach, we added a new component to the Query Selection layer called Query Reward. Query Reward consists of a workflow that computes the engagement rate of each query, which we store and retrieve for use in future query selection. Therefore, we can build a feedback loop to reward the queries with downstream engagement.

Here’s an example of how Query Reward works. Suppose a user has two PinnerSAGE clusters: one large cluster related to Recipes, and one small cluster related to Furniture. We initially show the user a lot of recipe pins, but the user doesn’t engage with them. Query Reward can capture that the Recipes cluster has many impressions but no future engagement. Therefore, the future reward, which is calculated by the engagement rate of the cluster, will gradually drop and we will have a greater chance to select the small Furniture cluster. If we show the user a few Furniture pins and they engage with them, Query Reward will increase the likelihood that we select the Furniture cluster in the future. Therefore, with the help of Query Reward, we are able to build a feedback loop based on users’ engagement rates and better select the query for candidate generation.

Some clusters may not have any engagement (e.g. an empty Query Reward). This could be because:

- The cluster was engaged a long time ago so it did not have a chance to be selected recently

- The cluster is a new use case for users, so we don’t have much record in the reward

When clusters do not have any engagement, we will give them an average weight so that there will still be a chance for them to be exposed to the users. After the next run of the Query Reward workflow, we will get more information about the unexposed clusters and decide whether we will select them next time.

*Figure: *Figure 3. Building a feedback loop based on Query Reward**

# Next Step / Future Directions

- Pinterest, as a platform to bring inspirations, would like to give Pinners personalized recommendations as much as we can. Taking users’ downstream feedback like both positive and negative engagements is what we want to prioritize. In the future iterations, we will consider more engagement types rather than repin to build a user profile.

- In order to maximize the Pinterest usage efficiency, instead of building the offline Query Reward, we want to move to a realtime version to enrich the signal for profiling among online requests. This would allow the feedback loop to be more responsive and instant, potentially responding to a user in the same Homefeed session as they browse.

- Besides the pin based retrieval, we can easily adopt a similar method on any token-based retrieval method.

# Acknowledgements

Thanks to our collaborators who contributed through discussions, reviews, and suggestions: Bowen Deng, Xinyuan Gui, Yitong Zhou, Neng Gu, Minzhe Zhou, Dafang He, Zhaohui Wu, Zhongxian Chen

*To learn more about engineering at Pinterest, check out the rest of our **Engineering Blog**, and visit our **Pinterest Labs** site. To explore life at Pinterest, visit our **Careers** page.*
