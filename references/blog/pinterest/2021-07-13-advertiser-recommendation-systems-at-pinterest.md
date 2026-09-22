# Advertiser Recommendation Systems at Pinterest

*Source: https://medium.com/pinterest-engineering/advertiser-recommendation-systems-at-pinterest-ccb255fbde20  
Published: 2021-07-13  
Fetched: 2026-09-21 via wayback*

# Advertiser Recommendation Systems at Pinterest

Pinterest Engineering

·

Follow

Published in

Pinterest Engineering Blog

·

6 min read

·

Jul 13, 2021

--

Erika Sun & Keshava Subramanya | Ads Intelligence Team

The Ads Intelligence team at Pinterest is charged with building and maintaining machine learning and algorithm-driven recommendations and the Recommendations Ranker to provide advertisers with the best experience in reaching relevant Pinners and help them reach their campaign goals. As part of this work, we leverage data in various algorithms to generate recommendations. These recommendations then appear on multiple surfaces, such as the external-facing Ads Manager and internal tools for our sales team. We use the insight and feedback found in past recommendation adoptions to improve future recommendation generation logic. The following diagram illustrates the lifecycle of our recommendation system.

In the next section, we will introduce a few recommendations and discuss how they are implemented. Then, we will go over a recommendation-ranking system that enables us to surface the top recommendations for our advertisers.

# Recommendations

# Bid Recommendation

## **What does the recommendation look like?**

“Increasing the bid for *adgroup 1* from $0.50 to $0.60 could help drive 25% more impressions.”

## How is the recommendation generated?

There are multiple bid products on Pinterest. They can be divided into two categories: automatic bid (autobid) and manual bid.

Autobid is a product that can help advertisers achieve the lowest-cost bid given certain budget constraints. The bid is adjusted automatically in real time by the system.

Manual bid is a product where advertisers need to manually set the bid. Compared to autobid, manual bid allows advertisers to have control on the price they pay per bid. However, setting the manual bid correctly can be tricky. If it is set too low, the ad would win very few auctions, thus failing the campaign goal. On the other hand, if the bid is set too high, with the same budget, the number of winning auctions would again decrease as the bid increases. The following diagram shows the relationship between bid change percent and winning auctions.

*Figure: **Figure 1: Winning Auctions against Bid Change***

In the recommendation algorithm, we use auction data to form the above chart for each ad group. More specifically, for a given ad group, we have the data for all of its participating auctions. For each losing auction, we could calculate how much it would take to win that auction.

Then we could accumulate this data to get the following cumulative additional winning auction counts against bid increase percent.

Similarly, for all the winning auctions, we could calculate the minimum bid decrease to lose that auction. Then we could plot the cumulative additional losing auction counts against bid decrease percent.

Then, combining the additional winning and losing auction counts data, we have Figure 1: Winning Auctions against Bid Change. We recommend the bid change corresponding to highest winning auctions to advertisers.

# Budget Recommendation

## What does the recommendation look like?

“Increasing the budget for *adgroup 1* from $50 to $60 could help drive 20% more impressions.”

## How is the recommendation generated?

Campaigns can begin with sub-optimal settings. By adopting various recommendations, campaigns are tuned to meet or even exceed campaign goals. For the performing campaigns, advertisers would like to increase the budget to have more reach under the same cost efficiency. For such campaigns, we build the budget recommendation.

For each campaign, advertisers usually create multiple ad groups with different pin images and targeting specifications. The spend of an adgroup can be limited by the adgroup level budget and/or campaign level budget.

Leveraging auction data, we could calculate the potential impressions and clicks for different budget amounts. Advertisers can choose a point from the diagram that has the best cost-reach trade off.

# Expanded Targeting Recommendation

## What does the recommendation look like?

“Enabling expanded targeting for *adgroup 1* could help drive 20% more impressions.”

## How is the recommendation generated?

Neural networks are a subset of machine learning and are at the heart of deep learning algorithms. As a recommendation engine with a powerful interest graph, we use neural networking for home feed personalization (organically recommending products and ideas) as well as for ads. We use neural networks to improve the quality of ads seen by users and the relevance it brings to advertisers.

Usually, advertisers have a certain reach goal for each campaign. For example, awareness campaigns are measured by number of impressions, traffic campaigns are measured by number of clicks, and conversion campaigns are measured by number of conversions. Sometimes, ad groups and campaigns face under-delivery issues. In other words, the actual impressions/clicks/conversions are far less than expected. One of the reasons may be narrow targeting.

On the Pinterest Ads platform, targeting configurations include demographics, user interests, and search keywords. Interests and keywords settings are more difficult to set than demographic settings. Advertisers need to select relevant and popular interests and keywords for their Pin images.

To solve this pain point, we launched the expanded targeting product. By enabling expanded targeting, advertisers no longer need to set interests and keywords. Instead, the ads retrieval neural network model would automatically present relevant Pin images to users by leveraging pin image and user embeddings.

This recommendation is shown to advertisers who could benefit by enabling expanded targeting. After applying certain necessary business rule filters, we use a Gradient Boosting Decision Tree (GBDT) regression model to predict the impressions for each adgroup, assuming expanded targeting is enabled. We then recommend expanded targeting to the ad groups whose predicted impressions are significantly larger than the current impressions.

# Other Recommendations

Additional recommendations available on the Pinterest platform can be based on business rules, while others are based purely on algorithms, and the rest are based on a combination of the two. Figure 2: Sample Recommendations on Ads Manager shows some example recommendations on Ads Manager.

# Recommendation Ranking

In Pinterest, one of the most important recommendation surfaces is the recommendation page in Ads Manager.

With the ever-growing number of new recommendation types, an advertiser could have hundreds of recommendations available. A ranking layer is needed to prioritize the most interesting recommendations for the advertisers on top. This helps prevent cognitive overload and keeps engagement with recommendation high.

We built a GBDT classification model to predict positive click event probability and descendingly rank notifications based on model scores. The modeling features can be classified into two categories: user features and notification features.

User features include advertisers, demographics and other properties, and the entity (adgroup or campaign for the specific notifications) properties. Notification features include notification properties and historical performance.

# Conclusion

The Automated Recommendation and Ranking efforts support a larger area of Advertiser Guidance and Experience at Pinterest. The Ads Intelligence team builds and maintains Machine Learning and algorithm- driven recommendations and the Recommendations Ranker. The vision is to present advertisers with the most impactful information and help them get the most out of using Pinterest products.

# We’re hiring!

If these topics interest you, you may want to check out our job post here!

# Acknowledgements

## **Automated Recommendations and Ranking Crew under The Ads Intelligence Team:**

Engineering: Erika Sun, Ruixin Qiang, Kelvin Jiang, Danilo Nunes Dos Santos, Keshava Subramanya, Mao Ye, Roelof van Zwol

Product: Perrye Ogunwole

## **Guidance Team:**

Engineering: Argun Alparslan, Bryan Chu, Chong Yao, Dena M, Jayanth Mettu, Kate Liu, Leo Lam, Mauricio Valderrama, Niket Arora, Priyanka Patil, Reynold Jesus Morel Diaz, Sowmya Prakash, Tulio Araujo De Oliveira, Calvin Chhour, Giancarlo Hernandez, Gustavo Krause, Kay Ashaolu, Kleber Infante, Ky Statham, Rassan Walker, Prathibha Deshikachar, Juan Carlos Rojas Vargas, Edison Sanchez

Product: Meera Srinivasan

Design: Darren Barone, Prianka Rayamajhi

UX Researcher: Shilpa Banerjee

Data Scientist: Rajat Tasgaonkar, Raveena Mathur

Product Writer: Joan Barnard

## **Workbench Team:**

Engineering: Phil Price

Product: Dan Marantz
