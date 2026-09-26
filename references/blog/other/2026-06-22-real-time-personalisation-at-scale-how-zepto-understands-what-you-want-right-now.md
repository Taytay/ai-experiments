# Real-Time Personalisation at Scale: How Zepto Understands What You Want, Right Now

*Source: https://blog.zepto.com/real-time-personalisation-at-scale-how-zepto-understands-what-you-want-right-now-d743b13367c8  
Published: 2026-06-22  
Fetched: 2026-09-21 via rss*

Zepto started as a quick-commerce platform for groceries. Today it’s where people shop for electronics, home essentials, gifts and festive needs. The catalogue has grown dramatically, but the app is still the same screen. That same real estate now needs to serve a far wider range of intents, in real-time, to millions of users a day.

This is what makes in-session personalisation critical. A user browsing at 8 AM on a Tuesday (rushing to add staples before leaving for work) is operating very differently from the same user at 11:30 PM on a Friday, exploring snacks and beverages after a long week. While the historical user profile is identical, the immediate intent has completely shifted. To keep the shopping experience seamless, our ranking systems must instantly adapt to these context switches.

In this post, we describe the system we built to do exactly that: a real-time dual sequence ranker that combines what a user is doing right now with what they have done before and serves scores within strict milliseconds latency budgets.

### The Problem: One Screen, Many Intents

Personalisation at Zepto is never a one-size-fits-all *“Recommended for You”* widget. Our app must seamlessly adapt to a wide spectrum of user intents in real-time from habit driven repurchases to deeply contextual cross sells.

> ***The core issue:**** User behaviour varies significantly across assets. A model that treats every surface the same, or compresses all interactions into a single static user vector, will overlook these differences — and those mistakes directly translate into lost conversions.*

### What Came Before: And Why It Was Not Enough

Before this system, recommendation surfaces at Zepto relied on a combination of Collaborative Filtering (CF), popularity weighted ranking and cohort level models. These approaches had real strengths: they were fast to train, highly interpretable and worked well for head items and well understood user segments.

But as our catalogue expanded and user journeys became increasingly complex, three structural limitations became glaringly visible:

- **No session awareness:** Prior systems had no visibility into what a user was doing *in the current visit*. A user who opened the app specifically to restock party supplies got the exact same ranking as a user doing a routine weekly shop.

- **Time blindness:** CF treats interactions from three months ago the same as interactions from three days ago. For fast moving categories such as snacks, beverages and seasonal items, recency matters.

- **Static user representation:** Compressing a user’s entire history into a single vector is a heavily lossy operation. It works for stable preferences, but completely misses the contextual signals that drive in-session decisions.

> ***The Goal:**** Build a model that understands both who you are (long term habits) and what you are doing right now (short term session intent) — producing a deeply personalised score for every candidate Stock Keeping Unit (SKU) at inference time.*

### System Overview: Architecting the Dual Sequence ReRanker

To address these historical gaps, we built a custom **Dual Sequence ReRanker** — inspired by Alibaba’s *Deep Interest Network (DIN)* and *Behaviour Sequence Transformer (BST)*. Instead of relying on a static snapshot, the new architecture dynamically balances what a user *usually* does with what they are doing *right now*.

> ***The Architectural Edge**** Separate encoders ensure neither time horizon drowns the other out. The model leans on history for new sessions, but instantly adapts when in-session intent is strong. Coupled with ****target aware pooling**** — which rebuilds the user profile per candidate item — and a ****hybrid loss function****, the model captures a highly precise, context aware signal.*

In production, this architecture delivered measurable bottom line improvements across our recommendation surfaces: **Conversion Lift, Tail Item Discovery and Lightning Fast Serving (P99 in low single digit ms).**

### Building Intuition: The Full Architecture

*Figure: Ranker end-to-end architecture: Token enrichment (item + action + temporal), dual encoders, target aware pooling, fusion gate and ranking head.*

Ranker end-to-end architecture: Token enrichment (item + action + temporal), dual encoders, target aware pooling, fusion gate and ranking head.

> ***End-to-End Execution**** At inference, the model ingests a user’s history alongside their live in-session interactions to score a set of candidate SKUs. It encodes both timelines through separate transformer stacks, computes a candidate specific user representation via target aware pooling, dynamically blends these signals through a learned gate and outputs a final conversion probability — enriched by real-time popularity trends and calendar context.*

### 1. Token Enrichment: Richer Than a Product ID

A sequence item isn’t just an ID; it’s a structured token combining three core signals:

- **Item Embedding (128 d):** A dense SKU vector, initialised from pre-trained embeddings and fine tuned during training.

- **Action Embedding:** Differentiates interaction types. An *Add-to-Cart (ATC)* carries heavier intent than a mere *view (click)*.

- **Temporal Positional Encoding:** Instead of static position indexes, we feed the *exact time elapsed (Δhours)* between the interaction and the current request into a small MLP.

> *This grants the model a continuous, non-parametric sense of recency — perfect for evaluating both rapid in-session clicks and days old historical purchases.*

### 2. Dual Transformer Encoders

We use two separate transformer stacks to encode long term history and in-session sequences.

> ***Why separate them?**** Because sharing weights forces the same attention heads to simultaneously model “bought this category every Friday for six months” and “tapped three items in the last two minutes.”*

By decoupling them, we respect their vastly different lengths, sparsities and temporal dynamics. Both utilise *pre-norm layers* for enhanced stability under noisy, weak supervision.

### 3. Target Aware Pooling

Standard pooling creates a rigid, static user vector. We replace this with a **soft attention mechanism** where the candidate SKU acts as the *query* and sequence positions act as *keys/values*. Attention weights depend on:

- **Product Similarity:** *Chips* scores high against *Cold Coffee*; *Detergent* scores low.

- **Time Decay (Δhours):** Recent interactions heavily outweigh interactions from months ago.

- **Action Type:** ATCs receive a learnable repeat_boost, reflecting stronger purchase intent.

> ***The Result:**** The user profile is dynamically recomputed per candidate. When scoring Cold Coffee, the profile up weights Nachos. When scoring Detergent, it pivots to highlight household items.*

### 4. Neural Fusion Gate

At any moment, one sequence holds more value. An empty session relies entirely on history, whereas a user actively browsing snacks for five minutes signals strong in-session intent.

We use a learned fusion gate that ingests pooled representations and metadata (lengths, repeat counts, similarities) to output a scalar weight (**g**).

> ***Fused Vector = (g × session_repr) + ((1 − g) × history_repr)***

*(Note: Sensible fallbacks are hard coded. If a sequence is entirely masked, the gate collapses completely to the active side.)*

### 5. Ranking Head with Real-Time Signals

The final step is a lightweight MLP. It takes the fused user vector, projects the candidate SKU and enriches it with dense, real-time context to output a final conversion probability:

- **Per User Counters:** Historical frequency of this user interacting with this specific SKU.

- **Global 28 Day Counters:** Platform wide item popularity.

- **2 Hour Sliding Window:** Real-time trending signals to catch intra-day spikes.

- **Calendar Context:** Learnable embeddings for *hour-of-day*, *day-of-week* and *week-of-month* to capture meal time peaks and weekend trends.

### 6. Loss Function: Hybrid Supervision

Standard Binary Cross Entropy (BCE) on isolated positive/negative pairs is information lean. To capture a richer signal, we use a hybrid loss combining two distinct sources:

- **In-Session Listwise Loss:** Treats ATCs as positives and un-purchased in-session views as negatives. This teaches the model how to rank *locally* within a single visit.

- **Batch Sampled Softmax Negatives:** Uses random interactions from other users in the mini-batch as negatives, providing an efficient, catalogue wide contrastive signal.

*Figure: Hybrid loss function*

***Custom ******WeightedBCELoss:**** In quick commerce, replenishment is just as vital as discovery. We dynamically upweight the loss penalty for positive samples if a user has a high historical **repeat_count for that item, forcing the model to aggressively prioritise restocking behaviours.*

### Engineering Hurdles & Optimisations

Building the architecture was only half the battle. Scaling it to Zepto’s volume required clearing several massive engineering bottlenecks:

**The Serving Latency Trap** > Naively running a transformer pass for 60 candidate items takes hundreds of milliseconds. We fixed this by decoupling encoding from scoring: user sequences are encoded exactly once and torch.expand() creates zero-copy views across all candidates, dropping our P99 latency to single digit milliseconds.

**Spark Array OOMs** > Standard PySpark explode/collect_list joins on massive historical sequence arrays caused severe cluster memory pressure and OOMs. We eliminated this by rewriting the pipeline to use optimised range based window joins, avoiding large intermediate array materialisation entirely.

**The Cold Start Problem** > New SKUs lack interaction history, meaning a standard zero vector embedding ruins their ranking. We built a dynamic, four level fallback chain (SKU → Subcategory → Category → Global Average) to guarantee robust performance even on brand new inventory.

**Position Bias & Label Noise** > Items ranked at the top get clicked simply for being at the top. To combat this, we strictly use Add-to-Cart (ATC) as our primary positive signal and sample hard negatives from proximity window views to filter out position induced noise.

### Learnings & Takeaways

**1. Static embeddings are a ceiling, not a floor:** Target aware pooling — dynamically rebuilding the user vector per candidate — proved to be our single largest modelling improvement.

**2. Feature engineering trumps model depth:** Injecting real-time 2-hour trending counters into the ranking head delivered a far greater A/B lift than simply stacking extra transformer layers.

**3. Retrieval sets the ultimate ceiling:** A reranker is only as good as the candidates it receives, making high quality upstream retrieval (like our semantic search architecture) the true binding constraint.

**4. Evaluate by asset, not just in aggregate:** Segmenting A/B metrics by specific surfaces (e.g., cart vs. home feed) is crucial to prevent aggregate conversion numbers from masking localised regressions.

**5. Online/offline parity is non-negotiable:** Absolute synchronisation between Flink (real-time) and Spark (batch) feature definitions is mandatory to prevent the model from learning a distribution it will never see in production.

**6. Training infrastructure is a first class concern:** Investing early in optimisations like bucketed dataloaders and async prefetch queues drastically reduced our epoch times, enabling the rapid iteration required at this scale.

### Limitations & The Road Ahead

While this system set a new baseline for personalisation at Zepto, it has known constraints that pave the way for our next architectural evolution.

- **Asset Blindness & Popularity Bias:** *Because the current model uses the same parameters for the home feed as it does for the cart, it occasionally lets global popularity override deep contextual relevance.*

- **Memory & Freshness Constraints:** *Relying on daily batch retraining and truncated sequences means we can slightly lag on hyper-fast trends and drop older signals for our most active users.*

- **Heterogeneous Sequences:** *We plan to integrate explicit search queries directly into the behavioural sequence, building on our foundation of **correcting multilingual queries with LLMs**.*

- **Continuous & Asset Aware Training:** *Moving to near real-time streaming weight updates and injecting surface specific adapter layers will allow the model to react instantly and appropriately per asset.*

- **Advanced Representations & Debiasing:** *Exploring graph based modelling (e.g., LightGCN) and Inverse Propensity Scoring (IPS) will help us enrich sparse user histories and explicitly unlearn display position biases.*

### Final Thoughts

Zepto’s mission is to make shopping as fast as delivery. The infrastructure for that is not just logistics; it is also making sure that when a user opens the app, the most relevant item is at the top of the list, whether they know exactly what they want or they are still figuring it out.

> *In production, the Dual-Sequence ReRanker delivered on this promise by driving a significant uplift in add-to-cart rates and surfacing highly relevant long-tail items. Crucially, through our encode-once serving optimisations and embedding fallback chains, we achieved this with robust cold start performance and blazing fast, single digit millisecond latency.*

The Dual-Sequence ReRanker is just one layer of that effort. If you found this post useful and want to go deeper on any component — the Spark data pipeline, the TorchServe serving architecture, or the Flink feature stream — let us know in the comments.

Real-Time Personalisation at Scale: How Zepto Understands What You Want, Right Now was originally published in Zepto TechXPress on Medium, where people are continuing the conversation by highlighting and responding to this story.
