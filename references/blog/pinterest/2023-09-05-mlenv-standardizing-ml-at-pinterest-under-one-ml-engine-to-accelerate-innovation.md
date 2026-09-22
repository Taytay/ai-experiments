# MLEnv: Standardizing ML at Pinterest Under One ML Engine to Accelerate Innovation

*Source: https://medium.com/pinterest-engineering/mlenv-standardizing-ml-at-pinterest-under-one-ml-engine-to-accelerate-innovation-e2b30b2f6768  
Published: 2023-09-05  
Fetched: 2026-09-21 via wayback*

# MLEnv: Standardizing ML at Pinterest Under One ML Engine to Accelerate Innovation

Pinterest Engineering

11 min read

·

Sep 5, 2023

--

Pong Eksombatchai | Principal Engineer; Karthik Anantha Padmanabhan | Manager II, Engineering

Press enter or click to view image in full size

*Figure: *Image from **https://unsplash.com/photos/w7ZyuGYNpRQ**

Pinterest’s mission is to bring everyone the inspiration to create a life they love. We rely on an extensive suite of AI powered products to connect over 460M users to hundreds of billions of Pins, resulting in hundreds of millions of ML inferences per second, hundreds of thousands of ML training jobs per month by just a couple of hundreds of ML engineers.

In 2021, ML was siloed at Pinterest with 10+ different ML frameworks relying on different deep learning frameworks, framework versions, and boilerplate logic to connect with our ML platform. It was a major bottleneck for ML innovation at Pinterest because the amount of engineering resources spent by each ML team to maintain their own ML stack was immense and there was limited knowledge sharing across teams.

To fix these problems we introduced MLEnv — a standardized ML engine at Pinterest now leveraged by 95% of ML jobs at Pinterest (starting from <5% in 2021). Since launching our platform we have:

- Observed a 300% increase in the number of training jobs, world-class 88 Net Promoter Score (NPS) for MLEnv and a 43% increase in ML Platform NPS

- Shifted the paradigm for ML innovations and delivered aggregate gains in Pinner engagement on the order of mid-double digit percentages

Press enter or click to view image in full size

*Figure: Growth of MLEnv over all of Pinterest ML jobs over time*

# Siloed State of ML Development

When we started working on the project, ML development at Pinterest was in a siloed state where each team would own most of their own unique ML stack. With standardization in tooling and popular ML libraries more or less offering the same functionalities, maintaining multiple ML stacks in a company at Pinterest scale is suboptimal for ML productivity and innovation. Both ML and ML platform engineers felt the full brunt of this issue.

For ML Engineers, this would mean:

- *Having to maintain their own environment* including work to ensure code quality and maintainability, the runtime environment and CI/CD pipeline. Questions that the team has to answer and consistently maintain include how to enable unit/integration testing, how to ensure consistency between training and serving environment, what coding best practices to enforce, etc.

- *Handling integrations to leverage tools and frameworks* that are critical for developer velocity. Heavy engineering work is needed for basic quality of life functionalities. For example, the project needs to integrate with MLFlow to track training runs, with Pinterest internal ML training and serving platform to train and serve models at scale, etc.

- *Enabling advanced ML capabilities* to properly develop state of the art ML at scale. ML has had an explosion of innovations in recent years, especially with the prominence of large language models and generative AI, and are much more complicated than just training the model on one GPU and serving on CPU. Teams need to spend an inordinate amount of time and resources to reinvent the wheels for different platforms to enable distributed training, re-implement state-of-the art algorithms on TensorFlow, optimize serving, etc.

- Worst of all is that *everything is done in a silo*. There is a lot of repeated work by each team to maintain their own environments and handle various integrations. All the effort put into enabling advanced ML capabilities can only be applied to an individual project due each project having a unique ML stack.

Press enter or click to view image in full size

*Figure: The diagram summarizes important pillars that are crucial for ML productivity and for it to work at scale in which teams spend substantial resources and repeated efforts in maintaining their own ML stacks.*

Press enter or click to view image in full size

*Figure: Teams struggle to maintain/enable all functionalities in the pillars due to how much resource and effort each of them requires.*

For Platform Engineers, this would mean:

- *Major struggles in the creation and adoption of platform tools* which severely limited the value that could be added by platform teams to ML engineers. It is very difficult for platform engineers to build good standardized tools that fit diverse ML stacks. The platform team also needs to work closely with ML stacks one by one in order to integrate offerings from ML Platform — tools like a distributed training platform, automated hyperparameter tuning etc. took much longer than needed since the work had to be repeated for every team.

- *Having to build expertise in both TensorFlow and PyTorch* stretched ML platform engineering resources to the limit. The nuances of the underlying deep learning framework needs to be considered in order to build a high-performance ML system. The platform team spent multiple times the effort needed due to having to support multiple deep learning frameworks and versions (PyTorch vs TensorFlow vs TensorFlow2).

- *Inability to drive software and hardware upgrades.* Individual teams were very far behind in ML-related software upgrades even though each upgrade brings a lot of new functionalities. Rather than the upgrade process being handled by platform engineers, most teams ended up using a very old version of TensorFlow, CUDA etc. because of how cumbersome the upgrade process usually is. Similarly, it is also very difficult to drive hardware upgrades which limits Pinterest’s ability to take advantage of the latest NVIDIA accelerators. Hardware upgrades usually require months of collaboration with various client teams to get software versions that are lagging behind up-to-date.

# Introducing MLEnv: The Pinterest ML Engine

Press enter or click to view image in full size

*Figure: MLEnv architecture diagram with major components*

In mid 2021, we gained alignment from various ML stakeholders at Pinterest and built the ML Environment (MLEnv), which is a full-stack ML developer framework that aims to make ML engineers more productive by abstracting away technical complexities that are irrelevant to ML modeling. MLEnv directly addresses the various issues mentioned in the previous section and provides four major components for ML developers.

## Code Runtime and Build Environment

MLEnv provides a standardized code runtime and build environment for its users. MLEnv maintains a monorepo (single code repository) for all ML projects, a single shared environment for all ML projects that training and serving are executed on by leveraging Docker and the CI/CD pipeline that customers can leverage powerful components that are not easily available such as GPU unit tests and ML trainer integration tests. Platform engineers handle the heavy lifting work of setting them up once for every ML project at Pinterest to easily re-use.

## ML Dev Toolbox

MLEnv provides ML developers with the ML Dev toolbox of commonly used tools that helps them be more productive in training and deploying models. Many are regular 3rd party tools such as MLFlow, Tensorboard and profilers, while others are internal tools and frameworks that are built by our ML Platform team such as our model deployment pipeline, ML serving platform and ML training platform.

The toolbox allows ML engineers to use dev velocity tools through an interface and skip integrations which are usually very time consuming. One tool to highlight is the training launcher CLI which makes the transition between local development and training the model at scale on Kubernetes through our internal training platform seamless. All the tools combined created a streamlined ML development experience for our engineers where they are able to quickly iterate on their ideas, use various tools to debug, scale training and deploy the model for inference.

## Advanced Functionalities

MLEnv gives customer access to advanced functionalities that were in the past only available internally to the team developing them because of our previous siloed state. ML projects now have access to a portfolio of training techniques that help speed up their training like distributed training, mixed precision training and libraries such as Accelerate, DeepSpeed etc. Similarly on the serving side, ML projects have access to highly optimized ML components for online serving as well as newer technologies such as GPU serving for recommender models.

## Native Deep Learning Library

With the previous three components combined, ML developers can focus on the interesting part which is the logic to train their model. We took extra care to not add any abstraction to the modeling logic which can pollute the experience of working with well-functioning deep learning libraries such as TensorFlow2 and PyTorch. In our framework, what ends up happening is that ML engineers have full control over the dataset loading, model architecture and training loop implemented using native deep learning libraries while having access to complementary components outlined above.

# The Golden Age of ML at Pinterest

After MLEnv general availability in late 2021, we entered a very interesting time period where there were rapid advancements in ML modeling and the ML platform at Pinterest which resulted in huge improvements in recommendation quality and our ability to serve more inspiring content to our Pinners.

## ML Development Velocity

The direct impact of MLEnv is a *massive improvement in ML dev velocity* at Pinterest of ML engineers. The capabilities to offload most of the ML boilerplate engineering work, access to a complete set of useful ML tools through an easy-to-use interface and easy access to advanced ML capabilities are game changers in developing and deploying state of the art ML models.

ML developers are very satisfied with the new tooling. MLEnv maintains an NPS of 88 which is world-class and is a key contributor in improving ML Platform NPS by 43%. In one of the organizations that we work with, the NPS improved by 93 points once MLEnv had been fully rolled out.

Teams are also much more productive as a result. We see multiple times growth in the amount of ML jobs (i.e. offline experiments) that each team runs even though the number of ML engineers are roughly the same. They can now also take models to online experimentation in days rather than months resulting in a multiple times improvement of the number of online ML experiments.

Press enter or click to view image in full size

*Figure: Explosion in the number of ML jobs over time due to developer velocity improvements*

## ML Platform 2.0

MLEnv made the ML Platform team much more productive by allowing the team to focus on a single ML environment. The ML Platform team can now *build standardized tools and cutting-edge ML capabilities, and drive adoption through a single integration* with MLEnv.

An example on the ML training platform side is Training Compute Platform (TCP), which is our in-house distributed training platform. Before MLEnv, the team struggled to maintain the platform due to having to support diverse ML environments with different deep learning framework libraries and setup. The team also struggled with adoption due to having to onboard various client teams one by one with varying needs to the platform. However, with MLEnv, the team was able to greatly reduce maintenance overhead by narrowing down to a single unified environment while gaining explosive growth in the number of jobs on the platform. With the much reduced maintenance overhead the team was able to focus on natural extensions to TCP. More advanced functionalities like distributed training, automated hyperparameter tuning and distributed data loading through Ray became straightforward for the team to implement and are released through MLEnv for client teams to adopt and use with minimal effort.

Press enter or click to view image in full size

Press enter or click to view image in full size

*Figure: Charts showing explosive growth in TCP functionalities adoption after MLEnv GA in Jan 2022*

Similarly on the ML serving platform side, MLEnv has allowed the team to make a 100x improvement in ML serving efficiency company-wide through GPU serving. The project would have been impossible in the previous paradigm due to it being a challenging combination of ML modeling improvements, GPU computation and distributed systems — requiring unique solutions to each ML stack. However, with MLEnv, we were able to form a cross functional team of Advanced Technology Group (ATG) and the ML Serving platform team to focus on a single unified environment and deliver the first production launch on Pinterest’s home feed within 6 months. Once the capability was available on MLEnv and showcased vastly improved business metrics, other major ML projects were able to quickly leverage the technology to scale their models within a couple months.

Press enter or click to view image in full size

*Figure: GPU Serving requires the combination of both the ML modeling and server architecture side optimizations*

## ML Development Paradigm Shift

MLEnv created a new paradigm at Pinterest where *ML and ML Platform engineers are working towards a single goal of advancing the ML capabilities of Pinterest* while blurring the boundaries of each ML team.

Successful modeling architectures and changes now propagate quickly across Pinterest product surfaces once one team has proven it to work because of the unified ML stack. One example is the home feed ranking team’s work on the Realtime User Actions Sequence. The team was able to improve business metrics significantly by integrating realtime user action sequences coupled with transformer and DCNv2 architectures. After the home feed team productionized the model architecture, other major ML projects were able to experiment with the same architecture changes within days and launch similar scale of improvements to production.

Press enter or click to view image in full size

*Figure: Successful model architectures and learnings propagating quickly to various ML use cases at Pinterest*

The new paradigm also encouraged teams to collaborate to work on fundamental changes because the impact scales horizontally across all ML use cases at Pinterest. We now have cross functional teams that target improving ML training and serving efficiency — delivering cost savings while allowing ML at Pinterest to scale by orders of magnitude. We formed a cross functional ML Modeling workgroup working on sophisticated ML model improvements such as large embedding tables, graph convolutional neural networks, etc. Last but not least, teams contribute to the ML framework by building standardized tooling such as ML feature importance through integrated gradients, ML training orchestration framework, etc. to share with each other.

Press enter or click to view image in full size

*Figure: ML teams contribute resources to the three pillars to advance Pinterest ML capabilities*

# Conclusion

The standardization of ML at Pinterest has led to immense ML developer velocity improvements and many ML innovations for the business. ML engineers are able to focus solely on modeling improvements because the majority of the system and infrastructure challenges are offloaded to our unified ML engine. On the other hand, platform engineers are able to focus on building and iterating on a single cutting-edge world-class ML stack. The work also shifted the culture of ML development at Pinterest by encouraging collaboration across teams — successful ML modeling and infrastructure changes developed by individual teams now propagate quickly to all product surfaces at Pinterest. All of this led to immense improvements in business metrics for Pinterest in the past year and we are excited to see what future ML innovations can bring to our Pinners!

# Acknowledgements

- ATG — Pong Eksombatchai, Prabhat Agarwal, Paul Baltescu, Po-Wei Wang, Yi-Ping Hsu, Andrew Zhai, Jiajing Xu, Chuck Rosenberg

- Core Infra — Kent Jiang

- ML Training Team — Chia-Wei Chen, Karthik Anantha Padmanabhan

- ML Serving Team — Nazanin Farahpour, Saurabh Vishwas Joshi, Zhiyuan Zhang

- ML Data Team — Se Won Jang

- Home feed Ranking — Xue Xia, Dhruvil Deven Badani

*To learn more about engineering at Pinterest, check out the rest of our **Engineering Blog** and visit our **Pinterest Labs** site. To explore and apply to open roles, visit our **Careers** page.*
