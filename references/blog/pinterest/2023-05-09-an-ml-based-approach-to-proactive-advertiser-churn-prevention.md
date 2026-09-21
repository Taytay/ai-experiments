# An ML based approach to proactive advertiser churn prevention

*Source: https://medium.com/pinterest-engineering/an-ml-based-approach-to-proactive-advertiser-churn-prevention-3a7c0c335016  
Published: 2023-05-09  
Fetched: 2026-09-21 via wayback*

# An ML based approach to proactive advertiser churn prevention

Pinterest Engineering

·

Follow

Published in

Pinterest Engineering Blog

·

6 min read

·

May 9, 2023

--

Erika Sun ML Engineer | Advertiser Growth Modeling Team; Ogheneovo Dibie Engineering Manager | Advertiser Growth Modeling Team

*Figure: Photo by Jason Blackeye on Unsplash*

# Summary

In this blog post, we describe a Machine Learning (ML) powered proactive churn prevention solution that was prototyped with our small & medium business (SMB) advertisers. Results from our initial experiment suggest that we can detect future churn with a high degree of predictive power and consequently empower our sales partners in mitigating churn. ML-powered proactive churn prevention can achieve better results than traditional reactive manual effort.

# Introduction

Like many ads-based businesses, at Pinterest, we are intently focused on minimizing advertiser churn on our platform. Traditionally, advertiser churn is addressed reactively. Specifically, a sales person reaches out to an advertiser only after they have churned. This approach is challenging because it is incredibly difficult to “resurrect” a customer once they leave the platform. To address the challenges with addressing churn reactively, we present a ML-powered proactive approach to advertiser churn reduction. Specifically, we developed a model that can predict the likelihood of advertiser churn in the near future and empowered our sales team with insights from this model to prevent at risk accounts from churning.

In this blog, we cover the:

- Churn prediction model’s design and implementation

- Experimentation in the managed North America SMB segment

# Churn Prediction Model

Our team built a ML model to predict advertiser’s churn likelihood in the next 14 days. We use the Shapely Additive Explanation (SHAP) package to estimate the model’s features’ contribution to the churn prediction. We provide the model churn prediction along with top contributing features to sales. Sales uses this information to prioritize their effort to mitigate churn for advertisers at risk. We will talk about each component in more detail in the following subsections.

## Model Architecture

The initial version of our model is based on a snapshot Gradient Boosting Decision Tree (GBDT) architecture. We chose GBDT for the following reasons:

- GBDT is a widely used model with good performance on small to medium sized tabular data* (our data fits in this description).

- SHAP works well with GBDT to estimate features’ contributions.

- Model feature importance is easy to generate with GBDT.

- It can also serve as a good baseline model for future model improvements, e.g. a sequential model.

**Snapshot means we use all the information available up to a given timestamp to predict the churn probability in the next 14 days with respect to that timestamp.*

## Target Variable

After thorough analysis and consultation on the business needs, we decided to use the following target variable definition (see Figure 1).

*Figure: **Figure 1: Target Variable Definition***

For our use case, we distinguish between an active and churned advertiser as follows:

- Active advertiser: spent in the last 7 days

- Churned advertiser: no spend in the last 7 days

We only predict the churn likelihood for active advertisers. Specifically, we predict if they will churn in the next 14 days.

## Features

There are over 200 features used in the model. These features are aggregated across different statistical measures–e.g. min, avg, max etc — over a range of time windows such as the past week / month prior to the inference dates. We also include week over week and month over month change features to reflect recent trends. These features can be grouped in the following categories:

- Performance: impressions**, clicks, conversions, conversion values, spend, cost per 1000 impressions, cost per click, clickthrough rate

- Goal: goal attainment ratio, distance to goal

- Budget: budget and utilization

- Ads manager activities: creates, edits, archives, custom reports

- Property: sales channel, country, industry, tenure, size, spend history

- Campaign configuration: targeting, bid strategy, objective type, campaign end date

***View more than 1 second.*

## Feature Contribution

We use the SHAP library to estimate the feature contribution to model probability output. Sigmoid of the sum of the features’ SHAP contribution is equal to model probability. From SHAP feature contribution, we can know what the key drivers are of high churn probability. We then highlight them for the Sales team to prevent churn.

# Model Usage

We use an offline trained model to infer active advertisers’ churn probability on a daily basis.

## Churn Risk Category

To help the Sales team better understand the meaning of the model output, we classify accounts into three categories based on their churn probability: high, medium, and low churn risk. *High churn risk* captures the accounts that are mostly likely to churn with high precision. *Medium churn risk* captures the accounts that have a lower likelihood of churn. *Low churn risk *contains the ‘healthy’ accounts that are unlikely to churn in the next 14 days. We select the thresholds to define different churn risk categories according to the Sales team’s request of desired precision and recall. More details can be found in Experiment Result.

# Experimentation with North America SMB

Our first experiment was focused on SMB accounts in North America that are managed by Sales Account Managers (AMs). We split the advertisers randomly into treatment and control groups within the experiment population. For the control group, we do not make any changes to the existing Sales team procedures. For the treatment group, we supported the Sales team to prevent churn with the following information:

- Churn Risk Category: High / medium / low churn risk

- Churn Reason Category. We classified the detailed churn reasons into coarse churn categories to ease understanding. The Sales team performed investigations using churn categories as directions.

*Figure: **Figure 2: Churn Information Widget***

## Experiment Success Metrics

Our experiment was evaluated based on the following criteria:

- Model predictive power, i.e. how well our model is able to identify advertisers that are likely to churn

- Efficacy of churn prediction in churn reduction

# Experiment Result

## Model Predictive Power

In order to determine the model’s predictive power, we compared its online performance on the control group (i.e. AMs who didn’t have access to the churn predictions) to what we had observed offline during development (i.e. our out-of-sample evaluation). Specifically, we measured model performance based on:

- Model quality: We compared the AUC-ROC and AUC-PR observed online to offline.

- Churn risk segmentation: In consultation with sales, we determined thresholds for high, medium, and low churn risk categories so that:

- Recall in high and medium churn risk should be above 70%.

- Precision in high churn risk should be around 70%.

This enables sales to capture most accounts at risk of churning while also prioritizing how to work through them, i.e. high churn risk first (highest precision).

With respect to model quality, our results indicate that the AUC-ROC observed online is within 1% of the offline AUC-ROC and the online AUC-PR is within 3% of the offline AUC-PR. This indicates that the model’s predictive power in identifying at-risk accounts is comparable to what we observed offline.

In terms of churn risk segmentation, our model’s precision, recall, and proportion of the population captured within the high and medium risk churn categories were consistently within 2–3% of our offline evaluation. This indicates that the segmentation of account risk based on churn likelihood were consistent with our offline evaluation and sales expectations.

## Efficacy of Churn Prediction in Advertiser Churn Reduction

We observed a 24% (statistically significant) reduction in the churn rate of high tier pods*** in our experiment treatment group compared to the control. This indicates that accounts whose churn risks were exposed to AMs were less likely to churn than those that were not.

**** In high tier pods, AMs manage about 50–70 accounts on average.*

# Conclusion & Future Work

In this blog post, we illustrated the development and implementation of an ML-based solution for proactive churn prevention at Pinterest. We are also actively investigating sequential model architectures such as Long short-term memory (LSTM) and Transformers, which may better capture the usage behaviors of advertisers and minimize the need for manual feature engineering such as week-over-week or month-over-month feature aggregation used in our current model.

# Acknowledgments

**Advertiser Growth Modeling Team**

- Engineering**: **Erika Sun, Ogheneovo Dibie, Keshava Subramanya, Mao Ye

- Product: Shailini Pandya

- Product Analytics/Data Science: Alex Simons

**Sales Team**

- Product: Wesley Kwiecien, Grace Yun

- Sales Managers: Abby (Fromm) Lubarsky

**Salesforce Team**

- Engineering**: **Gayathri Varadarangan (She Her), Murthy Tumuluri, Phani Chimata, Gabriela Mihaila, Richard Wu

**Optimization Workbench Team**

- Engineering: Phil Price, Jordan Boaz, Lucilla Chalmer

- Product: Dan Marantz

# References

[1] When and Why Tree-Based Models (Often) Outperform Neural Networks | by Andre Ye | Towards Data Science

*To learn more about engineering at Pinterest, check out the rest of our **Engineering Blog** and visit our **Pinterest Labs** site. To explore life at Pinterest, visit our **Careers** page.*
