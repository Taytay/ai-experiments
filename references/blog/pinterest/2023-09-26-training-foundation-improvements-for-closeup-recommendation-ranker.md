# Training Foundation Improvements for Closeup Recommendation Ranker

*Source: https://medium.com/pinterest-engineering/training-foundation-improvements-for-closeup-recommendation-ranker-67d90603426e  
Published: 2023-09-26  
Fetched: 2026-09-21 via wayback*

# Training Foundation Improvements for Closeup Recommendation Ranker

Pinterest Engineering

8 min read

·

Sep 26, 2023

--

Fan Jiang | Software Engineer, Closeup Candidate Retrieval; Liyao Lu | Software Engineer, Closeup Ranking & Blending; Laksh Bhasin | Software Engineer, Core ML Foundations; Chen Yang | Software Engineer, Core ML Foundations; Shivin Thukral | Software Engineer, Closeup Ranking & Blending; Travis Ebesu | Software Engineer, Closeup Ranking & Blending; Kent Jiang | Software Engineer, Core Serving Infra; Yan Sun | Engineering Manager, Closeup Ranking & Blending; Huizhong Duan | Engineering Manager, Closeup Relevance

# Introduction

Pinterest’s mission is- to bring everyone the inspiration to create a life they love. The closeup team helps with this mission by providing a feed of relevant and context-and-user-aware recommendations when a Pinner closes up on any Pin.

The recommendations are powered by innovative and cutting-edge machine learning technologies. We have published a detailed blog post of its modeling architecture. While adopting the newest architectures improves a model’s capabilities, building a solid training foundation stabilizes the model and further up-levels the model’s potential.

Training foundations cover a lot of aspects, from training preparation (training data logging, feature freshness, sampling strategies, hyperparameter tuning, etc), to training efficiency optimization (distributed training, model refreshes, GPU training, etc), to post training validation (offline replay, etc).

In this post, we are going to take a deeper look into three areas for closeup ranking model, specifically:

- Training data logging and generation

- Various sampling configurations and learnings

- Periodical and automatic model refreshes with in-house auto-retraining framework

# Logging Foundation and Improvements

## Hybrid logging

The closeup surface handles a large number of Pin impressions and engagements. While it is blessed with an abundance of data for training, it is also crucial to maintain a high data storage efficiency. Therefore, we adopted a hybrid data logging approach, with which the data is logged through both the backend service and the frontend clients. The process is captured in Figure 1.

The frontend logging system tracks the Pins that have been impressed by the Pinner and keeps a low percentage of the impressions and all positive engagements. For the sampled Pins, it reads the context and candidate cache, which are populated by the backend service, and calls the deduping service for further pruning. Then the frontend logging service calls the inference service to log the Pins with the full set of features. At the end of this pipeline, the data with training features are ingested in the database.

*Figure: Figure 1: hybrid logging for features*

On a daily basis, the features are joined with the labels to produce the final training dataset. Last year, we migrated the dataset from Thrift format to tabular format, which largely reduced the data size and improved development velocity due to its better data inspection capability.

By leveraging the hybrid logging approach, the pipeline avoids logging data without impressions, which drastically reduces the logging volume to achieve the same level of training data efficiency.

## Randomized Traffic

We keep a small amount of traffic to have completely randomly ordered candidates. For this stream of traffic, we log all the candidates that have been served to Pinners instead of just the impressed candidates. The randomized training data has proven to be helpful for multiple purposes, including offline replay experimentation, calibration, and model evaluations.

# Sampling Foundation and Improvements

Undoubtedly, training data is one of the most important components of model training. What the model learns largely depends on what data the model has seen. The bias of the model can be caused by the biases from the training data. And the training data can be measured by different segmentations, including positive and negative label ratios, content type distribution, user / context distributions, etc.

As covered in the data logging pipeline section, initially we only had a simple sampling strategy, which was to downsample the impressed candidates and keep all candidates with positive actions. Essentially, we were under-utilizing the opportunities contained in the data. Therefore we constructed a sampling job as part of the training data generation pipeline.

*Figure: Figure 2: Current training pipeline. Joiner is a Pyspark Job that joins features and labels to format a complete machine learning instance in full training data. Tabularizer is another Pyspark Job that converts full training data to TabularML format. Sampler reads in full training data and outputs the sampled training data for the downstream pytorch trainer job to consume.*

The sampler is a Pyspark job that reads in petabyte level of full training data, applies sampling logic, and then outputs sampled training data in hundreds of terabytes level. Users can pass their customized sampling logic via sampling configs to the sampler and then a downstream trainer will consume the sampled training data. Thanks to Pinterest’s Ezflow framework, datasets generated in the workflow are managed and cached by Ezflow’s lineage tracking mechanism, such that for multiple training jobs adopting the same sampling logic, sampled training data can be reused and we don’t have to rerun the sampler.

The overall goal for sampling foundation is to:

- Increase topline engagement

- Enhance content safety

At the current stage we have experimented with several sampling configurations, and the results shown below come from our online A/B experiment on the Closeup surface. Even though users are triggered in the experiment only when they visit the closeup surface, the engagement impact is not limited to the surface only. At the site-wide level, we see broad-based engagement metric gains across different actions and content types.

*Figure: Table 1: Site-wide Engagement Gain in A/B experiments*

## Future Work on Sampling

In addition to the exploration of more sophisticated sampling logic, we have a big opportunity in training efficiency. Current sampling logic is implemented as a standalone Pyspark batch processing job and we have ongoing effort to integrate sampling logic in the Ray dataloader. We believe it will significantly speed up the training workflow runtime as sampling and training can be coordinated in parallel. In addition, for two different but similar sampling logics, we no longer have to generate two different sampled training dataset, saving hundreds of terabytes of storage cost for each training workflow.

*Figure: Figure 3: Future Training Pipeline*

# Auto-Retraining Framework

## Overview

A deep neural network model’s performance can degrade as time goes on. For example, a model may be trained to give accurate outputs for specific input feature distributions, but these distributions can drift over time. More broadly, seasonal factors and user trends can change what users find useful and inspiring on Pinterest.

To keep our models fresh and avoid degradation, teams across Pinterest make use of the Auto-Retraining Framework (ARF), which allows for the automated training and re-training of models on a specified cadence.

ARF includes two main components:

- An offline Airflow workflow that trains, validates, and registers models for use in serving. Models must pass both an absolute validation check, where their evaluation metrics must exceed a threshold, and a relative validation check, where they must not regress on the previous production model’s metrics.

- An online model deploy Spinnaker pipeline that releases new model versions in serving, with validation on online metrics such as model latencies, resource usages, and predicted scores.

*Figure: *Figure 4: Components involved in ARF. Offline components (left) are run within an Airflow DAG. Model artifacts are registered in an MLflow run (center), which a Spinnaker deploy pipeline (right) then reads from and deploys to all users.**

With ARF, teams across Pinterest have a validated infrastructure to train on Pinners’ latest interactions, thereby continually improving our ranking models.

# Extending ARF for Closeup Ranking Model

## Hypothesis on Model Refreshes

We conducted learning experiments and refreshed the closeup ranking model on daily, tri-daily, weekly, and bi-weekly cadences. We consistently found that model refreshes bring in better performance across all refresh cadence. Though an increased cadence of refreshing yields better results, the maintenance overhead is not trivial. Since the closeup model utilizes knowledge distillation, a bad teacher model can be populated faster and make the investigation harder. Therefore, the weekly retraining strikes a good balance between model refreshness and maintenance.

It is not trivial to set up the auto-retrain experiments without the support of ARF. We need to separately maintain the data generation pipeline, model training flow, manually model deployment, and manually update the holdout experiment. ARF provides a configuration interface as the single entrypoint and contract between the client and the auto-retrain process. The data, model, deployment, and maintenance are handled with minimal human intervention. Onboarding to ARF greatly improved velocity from 3+ hours to 30 minutes.

## Customized Components

The closeup ranking model was one of the earliest adopters for ARF. Customized components need to be supported for the closeup ranking model use case.

The closeup ranking model leverages knowledge distillation, where the previous production model acts as the teacher model and the model scores are used in the loss function. As part of the data processing, we utilize batch inference to get model scores from the previous version of the production model and enrich the training dataset.

We also calibrate the scores of our ranking model, and the training pipeline produces both an uncalibrated and calibrated model, wherein we use the calibrated model as the serving production model, and the uncalibrated model as the teacher model for knowledge distillation. Whenever the ranking model is retrained, it is important that both these models are updated simultaneously. To allow this, ARF infrastructure extends support for the multi-model case so that both the calibrated and uncalibrated models are trained and deployed in sync.

## Performance Validation

We validate the auto-retraining quality at two places throughout the pipeline. The first place is the data validation, where we examine the features and labels and make sure there is no large shift in the distribution. Once we make sure the training data is valid, we also check the offline model evaluation metrics to make sure that the model performance is not degrading.

It is important to keep track of the model performance since the framework updates the model in production. In addition to real-time engagement alerts we have in place, we also set up a holdout experiment to track the performance. Every time the workflow successfully retrains a new model and publishes the model to production, the workflow will automatically reversion the experiment so that the experiment is always tracking the up-to-date production model with its previous version. By looking at the holdout experiment, we can conclude that in a relatively long term (usually a month), the auto-retraining brings in consistent gains for the core metrics.

# Conclusions

In this blog post, we shed some light on a few training foundations that powers the machine learning technical stack for the closeup recommendation system. Throughout the work, we found out that:

- By leveraging a hybrid data logging approach, we are able to achieve very high data storage efficiency.

- By providing a configuration based sampling mechanism, we can easily experiment with various strategies. Sampling can be a powerful lever to mitigate the system biases and improve pinner experience.

- By adopting the auto-retraining framework, we are able to refresh the production model with confidence and adapt to trends and shifts with high efficiency.

Machine learning training foundations can be as powerful as the machine learning techniques to drive pinner experiences. We are always looking for opportunities to improve the experience throughout the whole tech stack.

# Acknowledgements

The above work cannot be accomplished without the help from Olafur Gudmundsson, Pong Eksombatchai, Abhishek Tayal, Serena Rao, Chen Chen, Andrew Zhai, Bo Fu, Mingda Li. We would like to thank them for their support and contributions along the way.

*To learn more about engineering at Pinterest, check out the rest of our **Engineering Blog** and visit our **Pinterest Labs** site. To explore and apply to open roles, visit our **Careers** page.*
