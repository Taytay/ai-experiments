# Deep Multi-task Learning and Real-time Personalization for Closeup Recommendations

*Source: https://medium.com/pinterest-engineering/deep-multi-task-learning-and-real-time-personalization-for-closeup-recommendations-1030edfe445f  
Published: 2023-06-13  
Fetched: 2026-09-21 via direct*

Pinterest

Engineering

Pinner Experience

# Deep Multi-task Learning and Real-time Personalization for Closeup Recommendations

Pinterest Engineering

10 min read

·

Jun 13, 2023

--

Haomiao Li | Software Engineer, Closeup Ranking & Blending; Travis Ebesu | Software Engineer, Closeup Ranking & Blending; Fan Jiang | Software Engineer, Closeup Candidates; Jay Adams | Software Engineer, Pinner Growth & Signals; Olafur Gudmundsson | Software Engineer, Pinner Discovery; Yan Sun | Engineering Manager, Closeup Ranking & Blending; Huizhong Duan | Engineering Manager, Closeup Relevance

Press enter or click to view image in full size

## Introduction

At Pinterest, Closeup recommendations (aka Related Pins) is typically a feed of recommended content (primarily Pins) that we serve on any pin closeup. Closeup recommendations generate the largest amount of impressions among all recommendation surfaces at Pinterest and are uniquely critical for our users’ inspiration-to-realization journey. It’s important that we surface qualitative, relevant, and context-and-user-aware recommendations for people on Pinterest.

To achieve our goals of user engagement and satisfaction, the Closeup relevance team has been innovating and applying state-of-the-art machine learning techniques. Specifically, we have designed deep neural network (DNN) models that deeply embed multi-task predictions for user outcomes. We’ve introduced sequential features that capture a user’s most recent actions, as well as employed a personalized, context-aware blending model that combines all predictions into final ranking in real-time. In this blog post, we will touch on:

- How we got started on multi-task prediction

- How we further improved multi-task prediction in our DNN architecture using Multi-gate Mixture of Experts (MMoE)

- How we introduced teacher-student regularization to stabilize ranking model predictions

- How we incorporated general user signals as well as real-time user sequence signals to capture users’ long term and short term interest

- How we leveraged utility blending to further model users’ real-time, query-specific preferences

## Closeup Ranking Model Evolution

The Closeup “ranking” model is somewhat of a misnomer today. When it was first introduced, it was meant to be the one model that determines the ranking of recommendations for Closeup recommendations. Since then, the model itself, as well as its usage, has evolved a lot. Some noteworthy changes include the use of xgboost model, transition to DNN, adoption of AutoML¹ , but most notably, switching from single output to multi-task prediction. In this new paradigm, the “ranking” model no longer directly determines the final order for the recommendations; rather, it outputs the likelihood for different actions a user may take, including closing up, repin, click, etc. This has led to significant flexibility in optimization as well as significant improvement in the prediction quality. However, we needed to “deepen” the multi-task modeling further into our DNN architecture through MMOE, so that we unleashed the potential of multi-task modeling, where each expert/task shared learnings to the maximum extent. Figure 1 is a quick view of our overall DNN architecture.

Press enter or click to view image in full size

*Figure: Figure 1: Model Architecture for Closeup Feed Ranking Model*

The Closeup ranking model consists of a list of major components as shown in Figure 1 including:

- Representation Layers: pre-processes different types of features (embedding table lookup for categorical features, log transformation, and normalization for continuous features, etc.)

- - One highlight is that we employed a transformer encoder (shown in Figure 2) to preprocess user sequence signals, context features, and candidate Pin features:

- ▹*User’s most recent 100 engagement actions (repin, closeup, hide, etc.)

- *▹*User’s most recent 100 engaged pins’ pinSage embeddings

- *▹*Context signals such as query Pin embeddings and Pinner embeddings

- *▹*Candidate Pin embeddings*

Press enter or click to view image in full size

*Figure: Figure 2: Transformer Encoder for User Sequence Signals Preprocessing*

- Summarization layer: groups features that are similar together (i.e. user annotations from different sources such as search queries, board, etc.) into a single feature by passing through a MLP, representing each feature group in a lower dimensional latent space

- Transformer mixer: performs self-attention over groups of features

- MMoE: combines the results of independent “experts” to produce predictions for each task

Below we will highlight some of the components in additional detail.

### Multi-task Predictions

The tasks that the model is trying to predict are repin, closeup, clicks, and long-clicks. The model learned the probability through a binary entropy loss for each task, and the loss is averaged per batch during each training step. Currently the loss weight for each task is equal, but during the data preparation stage, we apply various weight adjustments so that each training example is properly represented in the loss function. The loss function is captured below, where b = (1, … B) from B examples in the batch, and h = (1, … H) from H tasks.

Press enter or click to view image in full size

### Score Regularization

In the past, we encountered model instability where predictions across two models with the same configuration vary significantly leading to an inconsistent user experience from unnecessary permutations in ranking order. Therefore we introduced score regularization⁴ (formulation is shown in Figure 3) to distill knowledge from the teacher model (the previous production model) and stabilize model predictions distribution. The inference for the teacher model is run during student model training, and we add the regularization term to total loss and tuned the coefficient 𝜆 to control the weight of this regularization term.

Press enter or click to view image in full size

*Figure: Figure 3: Formulation of Score Regularization*

### Multi-gate Mixture of Experts (MMOE)

MMoE was originally proposed in this paper² and demonstrated the ability to explicitly learn task relationships from data as opposed to the traditional shared-bottom model structure. The intuition is that in a share-bottom structure, model parameters are tightly shared among tasks, where inherent conflict among the tasks can harm the predictions for one or more tasks.

An MMoE module consists of multiple MLP experts and multiple corresponding softmax gates. Each expert in this module is a MLP that specializes in learning specialized task representations, and the corresponding gate will learn the weights for each expert’s task output. Then the final output is a weighted sum of the outputs from the experts and gates, passed through a linear transformation. The placement for the MMoE module is shown in Figure 4 below:

Press enter or click to view image in full size

*Figure: Figure 4: Mixture of Experts Architecture*

Some implementation details include:

- Concatenating transformer mixer output to expert output: this idea is similar to ResNet, where we not only pass the output from the transformer mixer as the input of the experts and gates, but also concatenate it to the output of the experts. This helps to preserve the full information from the transformer mixer and further boosts model performance.

- Applying 20% dropout in expert layers helps to avoid model overfitting

- Extensive parameter tuning to find the optimal set of hyperparameters: we performed a grid search on three hyperparameters [num_experts, expert_hidden_sizes, tfmr_output_dim]. From the tuning, we learned that:

- - Within a reasonable range, the more experts we use, the better the model performs offline. But in order to make sure the experts are not under-utilized, we produced Figure 5 below to visualize how each expert is specialized at modeling tasks.

Press enter or click to view image in full size

*Figure: Figure 5: Plot Average Weights From Gates Output*

— Simpler expert module performs better than wider or deeper experts, i.e., [256, 256] gives better performance than [512, 512] or [256, 256, 256]. This could be because we already have a relatively large number of experts, so the experts don’t need to be complex.

Here we show some offline and online results for applying the MMoE to ranking model:

- Offline Evaluation: as shown in Table 6, for the closeup surface, we aim at improve the HIT@3 and AUC for the four actions: repin (**most important one**), closeup, click and long-click as mentioned in Figure 1

Press enter or click to view image in full size

*Figure: Table 6: Offline Evaluation Metrics (relative change to baseline)*

- Online Experiment Results: as shown in Table 7, for online A/B experiment, we observed that for overall users and P5 countries (US, UK, CA, FR and DE) users, the repin volume increased by 4% and closeup volume increased by 1%, aligning with the offline evaluation.

Press enter or click to view image in full size

*Figure: Table 7: Online Experiment Metrics*

## Closeup Blending Model Evolution

After the ranking layer predictions, we employ a blending layer where the order of Pin recommendations is determined. Here, we introduced another ML model, which builds upon the multi-objective optimization framework and leverages the user and query Pin features to make real-time decisions on what to prioritize and how much we want to optimize them, in order to best serve users’ needs as well as to accommodate various business requirements. Currently, the layer provides a good balance between the organic content, which optimizes for organic engagements, and shopping content, which optimizes for shopping conversion.

The organic content objective is currently represented as a weighted sum between hand-tuned coefficients and each task’s prediction by the ranking model due to its Pareto optimality. Historically, the team has been using Bayesian optimization techniques to tune the blending weights through online experiments. But this generic approach lacks robustness as we need to tune the weights each time the ranking model score distribution shifts, and the feedback loop is long. Therefore, we launched a model-based approach to learn personalized weights, which we call Learned Utility.

### Learned Utility Model

We formulate learning these optimal blender parameters (policy) into an offline supervised learning setting. For a slice of users, we randomly vary their blender parameters and log the corresponding outcome. Next, we define a reward function which assigns a value to the corresponding engagement we observed (e.g. closeup reward = 1 and hide reward = -2). Then we learn a model that predicts the expected reward for a given request. We use a model that can be factored permitting access to the learned optimal blender parameters as shown in Figure 8. At serving time, we use only the part of the model that predicts the optimal blender parameters as shown in Figure 9.

Press enter or click to view image in full size

*Figure: Figure 8: Training process of the Learned Utility Model*

*Figure: Figure 9: Serving the Learned Utility model*

More formally, Learned Utility attempts to find a set of blending parameters {w₁, … , wₙ} that optimizes a given reward, R. We can formulate this as a binary classification task with a reward weighted cross-entropy loss denoted as R * l(g(x, r), y). Each training instance is comprised of (R, x, r, y) , where user, context and query level features denoted as x; r the randomized blender parameters that led to the user’s engagement behavior y resulting in the reward R and our model g(x, r). Our model is parameterized via a multi-layer perceptron f(x) = w₁…. wₙ. To calculate the reward of the predicted blender parameters we compute the inner product with the randomized blender parameters, ie g(x, r) = (rᵀf(x) + b), where b is a learnable global bias and is the logistic sigmoid function. This formulation allows us to factorize the model g(.) and obtain our desired blender parameters f(.).

Noise introduced during the collection of the randomized logging policy makes it difficult for the model to properly learn a good set of parameters. Therefore we place informative Gaussian priors on our blender parameters wᵢ ~N(sᵢ, σᵢ²) where the sᵢ denotes the iᵗʰ known production parameter and a hyperparameter σ² to control the variance. Performing an MAP estimation will give us an equivalent L2 regularizer leading to our final objective

where we simplify 𝜆ᵢ= 1/2σᵢ² and in experiments we use a global 𝜆 = 2 .

### Online Experiment Results

The results shown below come from our online A/B experiment for the closeup stream surface ranking and blending stage. This is the stream experience triggered when a user closes up on a natively published video Pin³. The key metrics for this surface are 10s full screen view (FSV), duration and time spent, and from Table 10, we have seen significant improvements in these metrics.

Press enter or click to view image in full size

*Figure: Table 10: Online Experiment Results for Learned Utility*

## Summary & Future Works

Our work of adopting and innovating upon multi-task learning with advanced features and state-of-art model architecture in the Closeup recommendation system has effectively improved quality of content and led to significant benefits to pinners’ engagements.

As for next steps, we are working with cross team efforts on:

- Adopting a richer and longer real time user sequence signal

- Improving GPU model serving performance

- Model architecture iterations

- Adoption of learned utility in other surfaces such as Homefeed

## Acknowledgements

This work represents a result of collaboration across multiple teams at Pinterest.

And many thanks to the following people that contributed to this work:

Closeup team: Minzhen Yi , Bo Fu, Chen Chen

ATG team: Yi-Ping Hsu, Paul Baltecsu, Pong Eksombatchai, Jiajing Xu

ML Platform team: Nazanin Farahpour, Se Won Jang, Zhiyuan Zhang

User Sequence Support team: Zefan Fu, Shun-ping Chiu, Jisong Liu, Yitong Zhou,Jiacheng Hong

Homefeed team: Yaron Greif, Ruimin Zhu

Core Serving Infra team: Kent Jiang,Zheng Liu

Search team: Cosmin Negruseri

## References

¹E. Wang, How we use AutoML, Multi-task learning and Multi-tower models for Pinterest Ads

²J. Ma, etc “Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts”, KDD 2018, August 19–23, 2018

³“Pinterest introduces Idea Pins globally and launches new creator discovery features”

https://newsroom.pinterest.com/en/post/pinterest-introduces-idea-pins-globally-and-launches-new-creator-discovery-features

⁴R. Li, et al “Stabilizing Neural Search Ranking Models”, WWW 2020

*To learn more about engineering at Pinterest, check out the rest of our **Engineering Blog** and visit our **Pinterest Labs** site. To explore life at Pinterest, visit our **Careers** page.*
