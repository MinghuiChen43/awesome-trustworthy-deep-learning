# Awesome Trustworthy Deep Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

The deployment of deep learning in real-world systems calls for a set of complementary technologies that will ensure that deep learning is trustworthy. The list covers different topics in emerging research areas including but not limited to out-of-distribution generalization, adversarial examples, backdoor attack, model inversion attack, machine unlearning, etc.

## Table of Contents

- [Survey](#survey)
- [Out-of-Distribution Generalization](#out-of-distribution-generalization)
- [Evasion Attacks and Defenses](#evasion-attacks-and-defenses)
- [Poisoning Attacks and Defenses](#poisoning-attacks-and-defenses)
- [Interpretability](#interpretability)
- [Causality](#causality)
- [Privacy](#privacy)
- [Fairness](#fairness)
- [Uncertainty](#uncertainty)
- [Ownership](#ownership)
- [Environmental Well-being](#environmental-well-being)
- [Interactions with Blockchain](#interactions-with-blockchain)
- [Others](#others)

## Survey

## Survey: 2022

- Trustworthy Graph Neural Networks: Aspects, Methods and Trends. [[paper]](https://arxiv.org/abs/2205.07424)
  - He Zhang, Bang Wu, Xingliang Yuan, Shirui Pan, Hanghang Tong, Jian Pei.
  - Key Word: Survey; Graph Neural Networks.
  - Digest: We propose a comprehensive roadmap to build trustworthy GNNs from the view of the various computing technologies involved. In this survey, we introduce basic concepts and comprehensively summarise existing efforts for trustworthy GNNs from six aspects, including robustness, explainability, privacy, fairness, accountability, and environmental well-being. Additionally, we highlight the intricate cross-aspect relations between the above six aspects of trustworthy GNNs. Finally, we present a thorough overview of trending directions for facilitating the research and industrialisation of trustworthy GNNs.

- A Survey on AI Sustainability: Emerging Trends on Learning Algorithms and Research Challenges. [[paper]](https://arxiv.org/abs/2205.03824)
  - Zhenghua Chen, Min Wu, Alvin Chan, Xiaoli Li, Yew-Soon Ong.
  - Key Word: Survey; Sustainability.
  - Digest: The technical trend in realizing the successes has been towards increasing complex and large size AI models so as to solve more complex problems at superior performance and robustness. This rapid progress, however, has taken place at the expense of substantial environmental costs and resources. Besides, debates on the societal impacts of AI, such as fairness, safety and privacy, have continued to grow in intensity. These issues have presented major concerns pertaining to the sustainable development of AI. In this work, we review major trends in machine learning approaches that can address the sustainability problem of AI.

- State of AI Ethics Report (Volume 6, February 2022). [[paper]](https://arxiv.org/abs/2202.07435)
  - Abhishek Gupta, Connor Wright, Marianna Bergamaschi Ganapini, Masa Sweidan, Renjie Butalid.
  - Key Word: Report; Ethics.
  - Digest: This report from the Montreal AI Ethics Institute (MAIEI) covers the most salient progress in research and reporting over the second half of 2021 in the field of AI ethics. Particular emphasis is placed on an "Analysis of the AI Ecosystem", "Privacy", "Bias", "Social Media and Problematic Information", "AI Design and Governance", "Laws and Regulations", "Trends", and other areas covered in the "Outside the Boxes" section. The two AI spotlights feature application pieces on "Constructing and Deconstructing Gender with AI-Generated Art" as well as "Will an Artificial Intellichef be Cooking Your Next Meal at a Michelin Star Restaurant?".

### Survey: 2021

- A Survey on AI Assurance. [[paper]](https://arxiv.org/abs/2111.07505)
  - Feras A. Batarseh, Laura Freeman. *Journal of Big Data*
  - Key Word: Survey; Validation; Verification; Testing; Assurance.
  - Digest: Artificial Intelligence (AI) algorithms are increasingly providing decision making and operational support across multiple domains. AI includes a wide library of algorithms for different problems. One important notion for the adoption of AI algorithms into operational decision process is the concept of assurance. The literature on assurance, unfortunately, conceals its outcomes within a tangled landscape of conflicting approaches, driven by contradicting motivations, assumptions, and intuitions. Accordingly, albeit a rising and novel area, this manuscript provides a systematic review of research works that are relevant to AI assurance, between years 1985 - 2021, and aims to provide a structured alternative to the landscape.

- Trustworthy AI: From Principles to Practices. [[paper]](https://arxiv.org/abs/2110.01167)
  - Bo Li, Peng Qi, Bo Liu, Shuai Di, Jingen Liu, Jiquan Pei, Jinfeng Yi, Bowen Zhou.
  - Key Word: Survey.
  - Digest: In this review, we strive to provide AI practitioners a comprehensive guide towards building trustworthy AI systems. We first introduce the theoretical framework of important aspects of AI trustworthiness, including robustness, generalization, explainability, transparency, reproducibility, fairness, privacy preservation, alignment with human values, and accountability. We then survey leading approaches in these aspects in the industry. To unify the current fragmented approaches towards trustworthy AI, we propose a systematic approach that considers the entire lifecycle of AI systems, ranging from data acquisition to model development, to development and deployment, finally to continuous monitoring and governance.

- Ethics of AI: A Systematic Literature Review of Principles and Challenges. [[paper]](https://arxiv.org/abs/2109.07906)
  - Arif Ali Khan, Sher Badshah, Peng Liang, Bilal Khan, Muhammad Waseem, Mahmood Niazi, Muhammad Azeem Akbar.
  - Key Word: Survey; Ethics.
  - Digest: We conducted a systematic literature review (SLR) study to investigate the agreement on the significance of AI principles and identify the challenging factors that could negatively impact the adoption of AI ethics principles. The results reveal that the global convergence set consists of 22 ethical principles and 15 challenges. Transparency, privacy, accountability and fairness are identified as the most common AI ethics principles.

- The State of AI Ethics Report (Volume 5). [[paper]](https://arxiv.org/abs/2108.03929)
  - Abhishek Gupta, Connor Wright, Marianna Bergamaschi Ganapini, Masa Sweidan, Renjie Butalid.
  - Key Word: Report; Ethics.
  - Digest: This report from the Montreal AI Ethics Institute covers the most salient progress in research and reporting over the second quarter of 2021 in the field of AI ethics with a special emphasis on "Environment and AI", "Creativity and AI", and "Geopolitics and AI." The report also features an exclusive piece titled "Critical Race Quantum Computer" that applies ideas from quantum physics to explain the complexities of human characteristics and how they can and should shape our interactions with each other. The report also features special contributions on the subject of pedagogy in AI ethics, sociology and AI ethics, and organizational challenges to implementing AI ethics in practice.

- Ethical AI for Social Good. [[paper]](https://arxiv.org/abs/2107.14044)
  - Ramya Akula, Ivan Garibay.
  - Key Word: Survey; Social Good.
  - Digest: The concept of AI for Social Good(AI4SG) is gaining momentum in both information societies and the AI community. Through all the advancement of AI-based solutions, it can solve societal issues effectively. To date, however, there is only a rudimentary grasp of what constitutes AI socially beneficial in principle, what constitutes AI4SG in reality, and what are the policies and regulations needed to ensure it. This paper fills the vacuum by addressing the ethical aspects that are critical for future AI4SG efforts.

- Trustworthy AI: A Computational Perspective. [[paper]](https://arxiv.org/abs/2107.06641)
  - Haochen Liu, Yiqi Wang, Wenqi Fan, Xiaorui Liu, Yaxin Li, Shaili Jain, Yunhao Liu, Anil K. Jain, Jiliang Tang.
  - Key Word: Survey.
  - Digest: We present a comprehensive survey of trustworthy AI from a computational perspective, to help readers understand the latest technologies for achieving trustworthy AI. Trustworthy AI is a large and complex area, involving various dimensions. In this work, we focus on six of the most crucial dimensions in achieving trustworthy AI: (i) Safety & Robustness, (ii) Non-discrimination & Fairness, (iii) Explainability, (iv) Privacy, (v) Accountability & Auditability, and (vi) Environmental Well-Being.

- Ethics Sheets for AI Tasks. [[paper]](https://arxiv.org/abs/2107.01183)
  - Saif M. Mohammad. *ACL 2022*
  - Key Word: Ethics; Natural Language Processing.
  - Digest: In this position paper, I make a case for thinking about ethical considerations not just at the level of individual models and datasets, but also at the level of AI tasks. I will present a new form of such an effort, Ethics Sheets for AI Tasks, dedicated to fleshing out the assumptions and ethical considerations hidden in how a task is commonly framed and in the choices we make regarding the data, method, and evaluation.

- AI-Ethics by Design. Evaluating Public Perception on the Importance of Ethical Design Principles of AI. [[paper]](https://arxiv.org/abs/2106.00326)
  - Kimon Kieslich, Birte Keller, Christopher Starke.
  - Key Word: Ethics; Public Perception.
  - Digest: In this study, we investigate how ethical principles (explainability, fairness, security, accountability, accuracy, privacy, machine autonomy) are weighted in comparison to each other. This is especially important, since simultaneously considering ethical principles is not only costly, but sometimes even impossible, as developers must make specific trade-off decisions. In this paper, we give first answers on the relative importance of ethical principles given a specific use case - the use of AI in tax fraud detection.

- The State of AI Ethics Report (Volume 4). [[paper]](https://arxiv.org/abs/2105.09060)
  - Abhishek Gupta, Alexandrine Royer, Connor Wright, Victoria Heath, Muriam Fancy, Marianna Bergamaschi Ganapini, Shannon Egan, Masa Sweidan, Mo Akif, Renjie Butalid.
  - Key Word: Report; Ethics.
  - Digest: The 4th edition of the Montreal AI Ethics Institute's The State of AI Ethics captures the most relevant developments in the field of AI Ethics since January 2021. This report aims to help anyone, from machine learning experts to human rights activists and policymakers, quickly digest and understand the ever-changing developments in the field. Through research and article summaries, as well as expert commentary, this report distills the research and reporting surrounding various domains related to the ethics of AI, with a particular focus on four key themes: Ethical AI, Fairness & Justice, Humans & Tech, and Privacy.

- The State of AI Ethics Report (January 2021). [[paper]](https://arxiv.org/abs/2105.09059)
  - Abhishek Gupta, Alexandrine Royer, Connor Wright, Falaah Arif Khan, Victoria Heath, Erick Galinkin, Ryan Khurana, Marianna Bergamaschi Ganapini, Muriam Fancy, Masa Sweidan, Mo Akif, Renjie Butalid.
  - Key Word: Report; Ethics.
  - Digest: It aims to help anyone, from machine learning experts to human rights activists and policymakers, quickly digest and understand the field's ever-changing developments. Through research and article summaries, as well as expert commentary, this report distills the research and reporting surrounding various domains related to the ethics of AI, including: algorithmic injustice, discrimination, ethical AI, labor impacts, misinformation, privacy, risk and security, social media, and more.

- AI and Ethics -- Operationalising Responsible AI. [[paper]](https://arxiv.org/abs/2105.08867)
  - Liming Zhu, Xiwei Xu, Qinghua Lu, Guido Governatori, Jon Whittle.
  - Key Word: Survey; Ethics; Responsibility.
  - Digest: In the last few years, AI continues demonstrating its positive impact on society while sometimes with ethically questionable consequences. Building and maintaining public trust in AI has been identified as the key to successful and sustainable innovation. This chapter discusses the challenges related to operationalizing ethical AI principles and presents an integrated view that covers high-level ethical AI principles, the general notion of trust/trustworthiness, and product/process support in the context of responsible AI, which helps improve both trust and trustworthiness of AI for a wider set of stakeholders.

- Ethics and Governance of Artificial Intelligence: Evidence from a Survey of Machine Learning Researchers. [[paper]](https://arxiv.org/abs/2105.02117)
  - Baobao Zhang, Markus Anderljung, Lauren Kahn, Noemi Dreksler, Michael C. Horowitz, Allan Dafoe.
  - Key Word: Survey; Ethics; Governance.
  - Digest: To examine these researchers' views, we conducted a survey of those who published in the top AI/ML conferences (N = 524). We compare these results with those from a 2016 survey of AI/ML researchers and a 2018 survey of the US public. We find that AI/ML researchers place high levels of trust in international organizations and scientific organizations to shape the development and use of AI in the public interest; moderate trust in most Western tech companies; and low trust in national militaries, Chinese tech companies, and Facebook.

- Causal Learning for Socially Responsible AI. [[paper]](https://arxiv.org/abs/2104.12278)
  - Lu Cheng, Ahmadreza Mosallanezhad, Paras Sheth, Huan Liu. *IJCAI 2021*
  - Key Word: Survey; Bias Mitigation; Transparency.
  - Digest: To make AI address ethical challenges and shun undesirable outcomes, researchers proposed to develop socially responsible AI (SRAI). One of these approaches is causal learning (CL). We survey state-of-the-art methods of CL for SRAI. We begin by examining the seven CL tools to enhance the social responsibility of AI, then review how existing works have succeeded using these tools to tackle issues in developing SRAI such as fairness. The goal of this survey is to bring forefront the potentials and promises of CL for SRAI.

### Survey: 2020

- The State of AI Ethics Report (October 2020). [[paper]](https://arxiv.org/abs/2011.02787)
  - Abhishek Gupta, Alexandrine Royer, Victoria Heath, Connor Wright, Camylle Lanteigne, Allison Cohen, Marianna Bergamaschi Ganapini, Muriam Fancy, Erick Galinkin, Ryan Khurana, Mo Akif, Renjie Butalid, Falaah Arif Khan, Masa Sweidan, Audrey Balogh.
  - Key Word: Report; Ethics.
  - Digest: This report aims to help anyone, from machine learning experts to human rights activists and policymakers, quickly digest and understand the ever-changing developments in the field. Through research and article summaries, as well as expert commentary, this report distills the research and reporting surrounding various domains related to the ethics of AI, including: AI and society, bias and algorithmic justice, disinformation, humans and AI, labor impacts, privacy, risk, and future of AI ethics.

- Technologies for Trustworthy Machine Learning: A Survey in a Socio-Technical Context. [[paper]](https://arxiv.org/abs/2007.08911)
  - Ehsan Toreini, Mhairi Aitken, Kovila P. L. Coopamootoo, Karen Elliott, Vladimiro Gonzalez Zelaya, Paolo Missier, Magdalene Ng, Aad van Moorsel.
  - Key Word: Survey.
  - Digest:  In this paper we provide an overview of technologies that support building trustworthy machine learning systems, i.e., systems whose properties justify that people place trust in them. We argue that four categories of system properties are instrumental in achieving the policy objectives, namely fairness, explainability, auditability and safety & security (FEAS). We discuss how these properties need to be considered across all stages of the machine learning life cycle, from data collection through run-time model inference.

- The State of AI Ethics Report (June 2020). [[paper]](https://arxiv.org/abs/2006.14662)
  - Abhishek Gupta, Camylle Lanteigne, Victoria Heath, Marianna Bergamaschi Ganapini, Erick Galinkin, Allison Cohen, Tania De Gasperis, Mo Akif, Renjie Butalid.
  - Key Word: Report; Ethics.
  - Digest: Our goal is to help you navigate this ever-evolving field swiftly and allow you and your organization to make informed decisions. This pulse-check for the state of discourse, research, and development is geared towards researchers and practitioners alike who are making decisions on behalf of their organizations in considering the societal impacts of AI-enabled solutions. We cover a wide set of areas in this report spanning Agency and Responsibility, Security and Risk, Disinformation, Jobs and Labor, the Future of AI Ethics, and more.

- Trust in Data Science: Collaboration, Translation, and Accountability in Corporate Data Science Projects. [[paper]](https://arxiv.org/abs/2002.03389)
  - Samir Passi, Steven J. Jackson.
  - Key Word: Survey; Data Science.
  - Digest: The trustworthiness of data science systems in applied and real-world settings emerges from the resolution of specific tensions through situated, pragmatic, and ongoing forms of work. Drawing on research in CSCW, critical data studies, and history and sociology of science, and six months of immersive ethnographic fieldwork with a corporate data science team, we describe four common tensions in applied data science work: (un)equivocal numbers, (counter)intuitive knowledge, (in)credible data, and (in)scrutable models. We show how organizational actors establish and re-negotiate trust under messy and uncertain analytic conditions through practices of skepticism, assessment, and credibility.

- Artificial Intelligence for Social Good: A Survey. [[paper]](https://arxiv.org/abs/2001.01818)
  - Zheyuan Ryan Shi, Claire Wang, Fei Fang.
  - Key Word: Survey; Social Good.
  - Digest: Artificial intelligence for social good (AI4SG) is a research theme that aims to use and advance artificial intelligence to address societal issues and improve the well-being of the world. AI4SG has received lots of attention from the research community in the past decade with several successful applications.

### Survey: 2019

- The relationship between trust in AI and trustworthy machine learning technologies. [[paper]](https://arxiv.org/abs/1912.00782)
  - Ehsan Toreini, Mhairi Aitken, Kovila Coopamootoo, Karen Elliott, Carlos Gonzalez Zelaya, Aad van Moorsel. *FAT 2020*
  - Key Word: Survey; Social Science.
  - Digest: To build AI-based systems that users and the public can justifiably trust one needs to understand how machine learning technologies impact trust put in these services. To guide technology developments, this paper provides a systematic approach to relate social science concepts of trust with the technologies used in AI-based services and products. We conceive trust as discussed in the ABI (Ability, Benevolence, Integrity) framework and use a recently proposed mapping of ABI on qualities of technologies. We consider four categories of machine learning technologies, namely these for Fairness, Explainability, Auditability and Safety (FEAS) and discuss if and how these possess the required qualities.

- Artificial Intelligence for Social Good. [[paper]](https://arxiv.org/abs/1901.05406)
  - Gregory D. Hager, Ann Drobnis, Fei Fang, Rayid Ghani, Amy Greenwald, Terah Lyons, David C. Parkes, Jason Schultz, Suchi Saria, Stephen F. Smith, Milind Tambe.
  - Key Word: Report; Social Good.
  - Digest: The Computing Community Consortium (CCC), along with the White House Office of Science and Technology Policy (OSTP), and the Association for the Advancement of Artificial Intelligence (AAAI), co-sponsored a public workshop on Artificial Intelligence for Social Good on June 7th, 2016 in Washington, DC. This was one of five workshops that OSTP co-sponsored and held around the country to spur public dialogue on artificial intelligence, machine learning, and to identify challenges and opportunities related to AI. In the AI for Social Good workshop, the successful deployments and the potential use of AI in various topics that are essential for social good were discussed, including but not limited to urban computing, health, environmental sustainability, and public welfare. This report highlights each of these as well as a number of crosscutting issues.

### Survey: 2018

- A Survey of Safety and Trustworthiness of Deep Neural Networks: Verification, Testing, Adversarial Attack and Defence, and Interpretability. [[paper]](https://arxiv.org/abs/1812.08342)
  - Xiaowei Huang, Daniel Kroening, Wenjie Ruan, James Sharp, Youcheng Sun, Emese Thamo, Min Wu, Xinping Yi. *Computer Science Review*
  - Key Word: Survey.
  - Digest: This survey paper conducts a review of the current research effort into making DNNs safe and trustworthy, by focusing on four aspects: verification, testing, adversarial attack and defence, and interpretability. In total, we survey 202 papers, most of which were published after 2017.

## Out-of-Distribution Generalization

### Out-of-Distribution Generalization: 2022

- Stable learning establishes some common ground between causal inference and machine learning. [[paper]](https://www.nature.com/articles/s42256-022-00445-z)
  - Peng Cui, Susan Athey. *Nature Machine Intelligence*
  - Key Word: Stable Learning; Causal Inference.
  - Digest: With the aim of bridging the gap between the tradition of precise modelling in causal inference and black-box approaches from machine learning, stable learning is proposed and developed as a source of common ground. This Perspective clarifies a source of risk for machine learning models and discusses the benefits of bringing causality into learning.

- CrossMatch: Cross-Classifier Consistency Regularization for Open-Set Single Domain Generalization. [[paper]](https://openreview.net/forum?id=48RBsJwGkJf)
  - Ronghang Zhu, Sheng Li. *ICLR 2022*
  - Key Word: Single Domain Generalization, Open-Set Recognition.
  - Digest: We propose a challenging and untouched problem: Open-Set Single Domain Generalization (OS-SDG), where target domains include unseen categories out of source label space. The goal of OS-SDG is to learn a model, with only one source domain, to classify a target sample with correct class if it belongs to source label space, or assign it to unknown classes. We design a CrossMatch approach to improve the performance of SDG methods on identifying unknown classes by leveraging a multi-binary classifier.

- Invariant Causal Representation Learning for Out-of-Distribution Generalization. [[paper]](https://openreview.net/forum?id=-e4EXDWXnSn)
  - Chaochao Lu, Yuhuai Wu, José Miguel Hernández-Lobato, Bernhard Schölkopf. *ICLR 2022*
  - Key Word: Out-of-Distribution Generalization; Invariant Causal Prediction; Causal Representation Learning.
  - Digest: We propose invariant Causal Representation Learning (iCaRL), an approach that enables out-of-distribution (OOD) generalization in the nonlinear setting (i.e., nonlinear representations and nonlinear classifiers). It builds upon a practical and general assumption: the prior over the data representation (i.e., a set of latent variables encoding the data) given the target and the environment belongs to general exponential family distributions, i.e., a more flexible conditionally non-factorized prior that can actually capture complicated dependences between the latent variables.

- Improving Robustness against Real-World and Worst-Case Distribution Shifts through Decision Region Quantification. [[paper]](https://arxiv.org/abs/2205.09619)
  - Leo Schwinn, Leon Bungert, An Nguyen, René Raab, Falk Pulsmeyer, Doina Precup, Björn Eskofier, Dario Zanca. *ICML 2022*
  - Key Word: Decision Region Quantification; Corruption Robustness; Distribution Shift.
  - Digest: We propose the Decision Region Quantification (DRQ) algorithm to improve the robustness of any differentiable pre-trained model against both real-world and worst-case distribution shifts in the data. DRQ analyzes the robustness of local decision regions in the vicinity of a given data point to make more reliable predictions. We theoretically motivate the DRQ algorithm by showing that it effectively smooths spurious local extrema in the decision surface.

- Uncertainty Modeling for Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2202.03958) [[code]](https://github.com/lixiaotong97/dsu)
  - Xiaotong Li, Yongxing Dai, Yixiao Ge, Jun Liu, Ying Shan, Ling-Yu Duan. *ICLR 2022*
  - Key Word: Out-of-Distribution Generalization; Uncertainty.
  - Digest: We improve the network generalization ability by modeling the uncertainty of domain shifts with synthesized feature statistics during training. Specifically, we hypothesize that the feature statistic, after considering the potential uncertainties, follows a multivariate Gaussian distribution. Hence, each feature statistic is no longer a deterministic value, but a probabilistic point with diverse distribution possibilities. With the uncertain feature statistics, the models can be trained to alleviate the domain perturbations and achieve better robustness against potential domain shifts.

- Handling Distribution Shifts on Graphs: An Invariance Perspective. [[paper]](https://arxiv.org/abs/2202.02466) [[code]](https://github.com/qitianwu/graphood-eerm)
  - Qitian Wu, Hengrui Zhang, Junchi Yan, David Wipf. *ICLR 2022*
  - Key Word: Distribution Shift; Graph Neural Networks.
  - Digest: We formulate the OOD problem on graphs and develop a new invariant learning approach, Explore-to-Extrapolate Risk Minimization (EERM), that facilitates graph neural networks to leverage invariance principles for prediction. EERM resorts to multiple context explorers (specified as graph structure editers in our case) that are adversarially trained to maximize the variance of risks from multiple virtual environments.

### Out-of-Distribution Generalization: 2021

- Why Stable Learning Works? A Theory of Covariate Shift Generalization. [[paper]](https://arxiv.org/abs/2111.02355)
  - Renzhe Xu, Peng Cui, Zheyan Shen, Xingxuan Zhang, Tong Zhang.
  - Key Word: Stable Learning.
  - Digest: Covariate shift generalization, a typical case in out-of-distribution (OOD) generalization, requires a good performance on the unknown testing distribution, which varies from the accessible training distribution in the form of covariate shift. Recently, stable learning algorithms have shown empirical effectiveness to deal with covariate shift generalization on several learning models involving regression algorithms and deep neural networks. However, the theoretical explanations for such effectiveness are still missing. In this paper, we take a step further towards the theoretical analysis of stable learning algorithms by explaining them as feature selection processes.

- Benchmarks for Corruption Invariant Person Re-identification. [[paper]](https://arxiv.org/abs/2111.00880) [[code]](https://github.com/MinghuiChen43/CIL-ReID)
  - Minghui Chen, Zhiqiang Wang, Feng Zheng. *NeurIPS 2021*
  - Key Word: Corruption Robustness; Benchmark; Person Re-Identificaiton.
  - Digest: We comprehensively establish six ReID benchmarks for learning corruption invariant representation. In the field of ReID, we are the first to conduct an exhaustive study on corruption invariant learning in single- and cross-modality datasets, including Market-1501, CUHK03, MSMT17, RegDB, SYSU-MM01. After reproducing and examining the robustness performance of 21 recent ReID methods, we have some observations: 1) transformer-based models are more robust towards corrupted images, compared with CNN-based models, 2) increasing the probability of random erasing (a commonly used augmentation method) hurts model corruption robustness, 3) cross-dataset generalization improves with corruption robustness increases. By analyzing the above observations, we propose a strong baseline on both single- and cross-modality ReID datasets which achieves improved robustness against diverse corruptions.

- Towards Out-Of-Distribution Generalization: A Survey. [[paper]](https://arxiv.org/abs/2108.13624)
  - Zheyan Shen, Jiashuo Liu, Yue He, Xingxuan Zhang, Renzhe Xu, Han Yu, Peng Cui.
  - Key Word: Survey; Out-of-Distribution Generalization.
  - Digest: Out-of-Distribution (OOD) generalization problem addresses the challenging setting where the testing distribution is unknown and different from the training. This paper serves as the first effort to systematically and comprehensively discuss the OOD generalization problem, from the definition, methodology, evaluation to the implications and future directions.

- Learning to Diversify for Single Domain Generalization. [[paper]](https://arxiv.org/abs/2108.11726) [[code]](https://github.com/busername/learning_to_diversify)
  - Zijian Wang, Yadan Luo, Ruihong Qiu, Zi Huang, Mahsa Baktashmotlagh. *ICCV 2021*
  - Key Word: Corruption Robustness; Single Domain Generalization.
  - Digest: To tackle this problem, we propose a style-complement module to enhance the generalization power of the model by synthesizing images from diverse distributions that are complementary to the source ones. More specifically, we adopt a tractable upper bound of mutual information (MI) between the generated and source samples and perform a two-step optimization iteratively.

- StyleAugment: Learning Texture De-biased Representations by Style Augmentation without Pre-defined Textures. [[paper]](https://arxiv.org/abs/2108.10549)
  - Sanghyuk Chun, Song Park.
  - Key Word: Corruption Robustness; Shape-Texture; Style Transfer; Data Augmentation.
  - Digest: Stylized ImageNet approach has two drawbacks in fidelity and diversity. First, the generated images show low image quality due to the significant semantic gap betweeen natural images and artistic paintings. Also, Stylized ImageNet training samples are pre-computed before training, resulting in showing the lack of diversity for each sample. We propose a StyleAugment by augmenting styles from the mini-batch.

- Impact of Aliasing on Generalization in Deep Convolutional Networks. [[paper]](https://arxiv.org/abs/2108.03489)
  - Cristina Vasconcelos, Hugo Larochelle, Vincent Dumoulin, Rob Romijnders, Nicolas Le Roux, Ross Goroshin. *ICCV 2021*
  - Key Word: Corruption Robustness; Anti-Aliasing.
  - Digest: Drawing insights from frequency analysis theory, we take a closer look at ResNet and EfficientNet architectures and review the trade-off between aliasing and information loss in each of their major components. We show how to mitigate aliasing by inserting non-trainable low-pass filters at key locations, particularly where networks lack the capacity to learn them.

- Using Synthetic Corruptions to Measure Robustness to Natural Distribution Shifts. [[paper]](https://arxiv.org/abs/2107.12052)
  - Alfred Laugros, Alice Caplier, Matthieu Ospici.
  - Key Word: Corruption Robustness; Benchmark.
  - Digest: We propose a methodology to build synthetic corruption benchmarks that make robustness estimations more correlated with robustness to real-world distribution shifts. Using the overlapping criterion, we split synthetic corruptions into categories that help to better understand neural network robustness.

- Just Train Twice: Improving Group Robustness without Training Group Information. [[paper]](https://arxiv.org/abs/2107.09044) [[code]](https://github.com/anniesch/jtt)
  - Evan Zheran Liu, Behzad Haghgoo, Annie S. Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, Chelsea Finn. *ICML 2021*
  - Key Word: Invariant Learning; Robust Optimization; Fairness.
  - Digest: We propose a simple two-stage approach, JTT, that first trains a standard ERM model for several epochs, and then trains a second model that upweights the training examples that the first model misclassified. Intuitively, this upweights examples from groups on which standard ERM models perform poorly, leading to improved worst-group performance.

- Visual Representation Learning Does Not Generalize Strongly Within the Same Domain. [[paper]](https://arxiv.org/abs/2107.08221) [[code]](https://github.com/bethgelab/InDomainGeneralizationBenchmark)
  - Lukas Schott, Julius von Kügelgen, Frederik Träuble, Peter Gehler, Chris Russell, Matthias Bethge, Bernhard Schölkopf, Francesco Locatello, Wieland Brendel. *ICLR 2022*
  - Key Word: Out-of-Distribution Generalization; Disentanglement; Benchmark.
  - Digest: In contrast to prior robustness work that introduces novel factors of variation during test time, such as blur or other (un)structured noise, we here recompose, interpolate, or extrapolate only existing factors of variation from the training data set (e.g., small and medium-sized objects during training and large objects during testing). Models that learn the correct mechanism should be able to generalize to this benchmark. In total, we train and test 2000+ models and observe that all of them struggle to learn the underlying mechanism regardless of supervision signal and architectural bias. Moreover, the generalization capabilities of all tested models drop significantly as we move from artificial datasets towards more realistic real-world datasets.

- Global Filter Networks for Image Classification. [[paper]](https://arxiv.org/abs/2107.00645) [[code]](https://github.com/raoyongming/GFNet)
  - Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, Jie Zhou.
  - Key Word: Corruption Robustness; Frequency.
  - Digest: We present the Global Filter Network (GFNet), a conceptually simple yet computationally efficient architecture, that learns long-term spatial dependencies in the frequency domain with log-linear complexity.

- Test-Time Adaptation to Distribution Shift by Confidence Maximization and Input Transformation. [[paper]](https://arxiv.org/abs/2106.14999)
  - Chaithanya Kumar Mummadi, Robin Hutmacher, Kilian Rambach, Evgeny Levinkov, Thomas Brox, Jan Hendrik Metzen.
  - Key Word: Corruption Robustness; Test-Time Adaptation.
  - Digest: We propose non-saturating losses based on the negative log likelihood ratio, such that gradients from high confidence predictions still contribute to test-time adaptation.

- Exploring Corruption Robustness: Inductive Biases in Vision Transformers and MLP-Mixers. [[paper]](https://arxiv.org/abs/2106.13122) [[code]](https://github.com/katelyn98/CorruptionRobustness)
  - Katelyn Morrison, Benjamin Gilby, Colton Lipchak, Adam Mattioli, Adriana Kovashka.
  - Key Word: Corruption Robustness; Transformers.
  - Digest: We find that vision transformer architectures are inherently more robust to corruptions than the ResNet and MLP-Mixers.

- Dangers of Bayesian Model Averaging under Covariate Shift. [[paper]](https://arxiv.org/abs/2106.11905)
  - Pavel Izmailov, Patrick Nicholson, Sanae Lotfi, Andrew Gordon Wilson.
  - Key Word: Corruption Robustness; Bayesian Neural Networks.
  - Digest: Bayesian neural networks (BNNs) with high-fidelity approximate inference via full-batch Hamiltonian Monte Carlo achieve poor generalization under covariate shift, even underperforming classical estimation. We explain this surprising result, showing how a Bayesian model average can in fact be problematic under covariate shift, particularly in cases where linear dependencies in the input features cause a lack of posterior contraction.

- A Winning Hand: Compressing Deep Networks Can Improve Out-Of-Distribution Robustness. [[paper]](https://arxiv.org/abs/2106.09129)
  - James Diffenderfer, Brian R. Bartoldson, Shreya Chaganti, Jize Zhang, Bhavya Kailkhura.
  - Key Word: Corruption Robustness; Lottery Ticket Hypothesis.
  - Digest: We present the first positive result on simultaneously achieving high accuracy and OoD robustness at extreme levels of model compression.

- Delving Deep into the Generalization of Vision Transformers under Distribution Shifts. [[paper]](https://arxiv.org/abs/2106.07617) [[code]](https://github.com/Phoenix1153/ViT_OOD_generalization)
  - Chongzhi Zhang, Mingyuan Zhang, Shanghang Zhang, Daisheng Jin, Qiang Zhou, Zhongang Cai, Haiyu Zhao, Shuai Yi, Xianglong Liu, Ziwei Liu.
  - Key Word: Corruption Robustness; Transformers.
  - Digest: We first present a taxonomy of distribution shifts by categorizing them into five conceptual groups: corruption shift, background shift, texture shift, destruction shift, and style shift. Then we perform extensive evaluations of ViT variants under different groups of distribution shifts and compare their generalization ability with Convolutional Neural Network (CNN) models.

- Towards a Theoretical Framework of Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2106.04496)
  - Haotian Ye, Chuanlong Xie, Tianle Cai, Ruichen Li, Zhenguo Li, Liwei Wang.
  - Key Word: Theoretical Framework; Out-of-Distribution Generalization.
  - Digest: We take the first step towards rigorous and quantitative definitions of 1) what is OOD; and 2) what does it mean by saying an OOD problem is learnable. We also introduce a new concept of expansion function, which characterizes to what extent the variance is amplified in the test domains over the training domains, and therefore give a quantitative meaning of invariant features.

- An Information-theoretic Approach to Distribution Shifts. [[paper]](https://arxiv.org/abs/2106.03783) [[code]](https://github.com/mfederici/dsit)
  - Marco Federici, Ryota Tomioka, Patrick Forré. *NeurIPS 2021*
  - Key Word: Information Theory; Distribution Shift.
  - Digest: We describe the problem of data shift from a novel information-theoretic perspective by (i) identifying and describing the different sources of error, (ii) comparing some of the most promising objectives explored in the recent domain generalization, and fair classification literature.

- OoD-Bench: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2106.03721) [[code]](https://github.com/ynysjtu/ood_bench)
  - Nanyang Ye, Kaican Li, Haoyue Bai, Runpeng Yu, Lanqing Hong, Fengwei Zhou, Zhenguo Li, Jun Zhu. *CVPR 2022*
  - Key Word: Out-of-Distribution Generalization; Benchmark.
  - Digest: We first identify and measure two distinct kinds of distribution shifts that are ubiquitous in various datasets. Next, through extensive experiments, we compare OoD generalization algorithms across two groups of benchmarks, each dominated by one of the distribution shifts, revealing their strengths on one shift as well as limitations on the other shift.

- Predict then Interpolate: A Simple Algorithm to Learn Stable Classifiers. [[paper]](https://arxiv.org/abs/2105.12628) [[code]](https://github.com/YujiaBao/Predict-then-Interpolate)
  - Yujia Bao, Shiyu Chang, Regina Barzilay. *ICML 2022*
  - Key Word: Invariant Learning; Fairness.
  - Digest: We prove that by interpolating the distributions of the correct predictions and the wrong predictions, we can uncover an oracle distribution where the unstable correlation vanishes. Since the oracle interpolation coefficients are not accessible, we use group distributionally robust optimization to minimize the worst-case risk across all such interpolations.

- Using the Overlapping Score to Improve Corruption Benchmarks. [[paper]](https://arxiv.org/abs/2105.12357)
  - Alfred Laugros, Alice Caplier, Matthieu Ospici.
  - Key Word: Corruption Robustness; Benchmark.
  - Digest: We propose a metric called corruption overlapping score, which can be used to reveal flaws in corruption benchmarks. Two corruptions overlap when the robustnesses of neural networks to these corruptions are correlated.

- Improved OOD Generalization via Adversarial Training and Pre-training. [[paper]](https://arxiv.org/abs/2105.11144)
  - Mingyang Yi, Lu Hou, Jiacheng Sun, Lifeng Shang, Xin Jiang, Qun Liu, Zhi-Ming Ma. *ICML 2021*
  - Key Word: Corruption Robustness; Adversarial Training; Pre-Trainig.
  - Digest: In this paper, after defining OOD generalization via Wasserstein distance, we theoretically show that a model robust to input perturbation generalizes well on OOD data. Inspired by previous findings that adversarial training helps improve input-robustness, we theoretically show that adversarially trained models have converged excess risk on OOD data, and empirically verify it on both image classification and natural language understanding tasks.

- Balancing Robustness and Sensitivity using Feature Contrastive Learning. [[paper]](https://arxiv.org/abs/2105.09394)
  - Seungyeon Kim, Daniel Glasner, Srikumar Ramalingam, Cho-Jui Hsieh, Kishore Papineni, Sanjiv Kumar.
  - Key Word: Corruption Robustness.
  - Digest: We discuss this trade-off between sensitivity and robustness to natural (non-adversarial) perturbations by introducing two notions: contextual feature utility and contextual feature sensitivity. We propose Feature Contrastive Learning (FCL) that encourages a model to be more sensitive to the features that have higher contextual utility.

- Towards Robust Vision Transformer. [[paper]](https://arxiv.org/abs/2105.07926) [[code]](https://github.com/vtddggg/Robust-Vision-Transformer)
  - Xiaofeng Mao, Gege Qi, Yuefeng Chen, Xiaodan Li, Ranjie Duan, Shaokai Ye, Yuan He, Hui Xue.
  - Key Word: Corruption Robustness; Transformers.
  - Digest: Wwe propose a Robust Vision Transformer (RVT), by using and combining robust components as building blocks.To further improve the RVT, we propose two new plug-and-play techniques called position-aware attention scaling and patch-wise augmentation.

- Vision Transformers are Robust Learners. [[paper]](https://arxiv.org/abs/2105.07581) [[code]](https://github.com/sayakpaul/robustness-vit)
  - Sayak Paul, Pin-Yu Chen.
  - Key Word: Corruption Robustness; Transformers.
  - Digest: We study the robustness of the Vision Transformer (ViT) against common corruptions and perturbations, distribution shifts, and natural adversarial examples.

- When Human Pose Estimation Meets Robustness: Adversarial Algorithms and Benchmarks. [[paper]](https://arxiv.org/abs/2105.06152) [[code]](https://github.com/AIprogrammer/AdvMix)
  - Jiahang Wang, Sheng Jin, Wentao Liu, Weizhong Liu, Chen Qian, Ping Luo. *CVPR 2021*
  - Key Word: Corruption Robustness; Benchmark; Data Augmentation; Pose Estimation.
  - Digest: This work comprehensively studies and addresses this problem by building rigorous robust benchmarks, termed COCO-C, MPII-C, and OCHuman-C, to evaluate the weaknesses of current advanced pose estimators, and a new algorithm termed AdvMix is proposed to improve their robustness in different corruptions.

- Heterogeneous Risk Minimization. [[paper]](https://arxiv.org/abs/2105.03818)
  - Jiashuo Liu, Zheyuan Hu, Peng Cui, Bo Li, Zheyan Shen. *ICML 2021*
  - Key Word: Invariant Learning; Causality; Robust Optimization.
  - Digest: We propose Heterogeneous Risk Minimization (HRM) framework to achieve joint learning of latent heterogeneity among the data and invariant relationship, which leads to stable prediction despite distributional shifts. We theoretically characterize the roles of the environment labels in invariant learning and justify our newly proposed HRM framework.

- What Are Bayesian Neural Network Posteriors Really Like? [[paper]](https://arxiv.org/abs/2104.14421) [[code]](https://github.com/google-research/google-research/tree/master/bnn_hmc)
  - Pavel Izmailov, Sharad Vikram, Matthew D. Hoffman, Andrew Gordon Wilson.
  - Key Word: Corruption Robustness; Bayesian Neural Networks.
  - Digest: Bayesian neural networks show surprisingly poor generalization under domain shift; while cheaper alternatives such as deep ensembles and SGMCMC methods can provide good generalization, they provide distinct predictive distributions from HMC.

- Adapting ImageNet-scale models to complex distribution shifts with self-learning. [[paper]](https://arxiv.org/abs/2104.12928) [[code]](https://domainadaptation.org/selflearning/)
  - Evgenia Rusak, Steffen Schneider, Peter Gehler, Oliver Bringmann, Wieland Brendel, Matthias Bethge.
  - Key Word: Corruption Robustness; Domain Adaptation.
  - Digest: We find that three components are crucial for increasing performance with self-learning: (i) using short update times between the teacher and the student network, (ii) fine-tuning only few affine parameters distributed across the network, and (iii) leveraging methods from robust classification to counteract the effect of label noise. We therefore re-purpose the dataset from the Visual Domain Adaptation Challenge 2019 and use a subset of it as a new robustness benchmark (ImageNet-D) which proves to be a more challenging dataset.

- Towards Corruption-Agnostic Robust Domain Adaptation. [[paper]](https://arxiv.org/abs/2104.10376) [[code]](https://github.com/Mike9674/CRDA)
  - Yifan Xu, Kekai Sheng, Weiming Dong, Baoyuan Wu, Changsheng Xu, Bao-Gang Hu.
  - Key Word: Corruption Robustness; Domain Adaptation.
  - Digest: We investigate a new scenario called corruption-agnostic robust domain adaptation (CRDA) to equip domain adaptation models with corruption robustness. We take use of information of domain discrepancy to propose a novel module Domain Discrepancy Generator (DDG) for corruption robustness that mimic unpredictable corruptions.

- Gradient Matching for Domain Generalization. [[paper]](https://arxiv.org/abs/2104.09937) [[code]](https://github.com/YugeTen/fish)
  - Yuge Shi, Jeffrey Seely, Philip H.S. Torr, N. Siddharth, Awni Hannun, Nicolas Usunier, Gabriel Synnaeve. *ICLR 2022*
  - Key Word: Domain Generalization, Multi-Source Domain Adaptation.
  - Digest: Here, we propose an inter-domain gradient matching objective that targets domain generalization by maximizing the inner product between gradients from different domains. Since direct optimization of the gradient inner product can be computationally prohibitive -- it requires computation of second-order derivatives -- we derive a simpler first-order algorithm named Fish that approximates its optimization.

- Does enhanced shape bias improve neural network robustness to common corruptions? [[paper]](https://arxiv.org/abs/2104.09789)
  - Chaithanya Kumar Mummadi, Ranjitha Subramaniam, Robin Hutmacher, Julien Vitay, Volker Fischer, Jan Hendrik Metzen. *ICLR 2021*
  - Key Word: Corruption Robustness; Shape-Texture.
  - Digest: While pre-training on stylized images increases both shape bias and corruption robustness, these two quantities are not necessarily correlated: pre-training on edge maps increases the shape bias without consistently helping in terms of corruption robustness.

- Misclassification-Aware Gaussian Smoothing improves Robustness against Domain Shifts. [[paper]](https://arxiv.org/abs/2104.01231)
  - Athanasios Tsiligkaridis, Theodoros Tsiligkaridis.
  - Key Word: Corruption Robustness; Regularization.
  - Digest: We introduce a misclassification-aware consistency loss coupled with Gaussian noise regularization and a corresponding training algorithm. Additionally, we present some theoretical properties of our new regularization approach that reveals its connection to local loss geometry.

- Defending Against Image Corruptions Through Adversarial Augmentations. [[paper]](https://arxiv.org/abs/2104.01086)
  - Dan A. Calian, Florian Stimberg, Olivia Wiles, Sylvestre-Alvise Rebuffi, Andras Gyorgy, Timothy Mann, Sven Gowal.
  - Key Word: Corruption Robustness; Data Augmentation.
  - Digest: We propose AdversarialAugment, a technique which optimizes the parameters of image-to-image models to generate adversarially corrupted augmented images.

- Improving robustness against common corruptions with frequency biased models. [[paper]](https://arxiv.org/abs/2103.16241)
  - Tonmoy Saikia, Cordelia Schmid, Thomas Brox.
  - Key Word: Corruption Robustness; Frequency.
  - Digest: We propose a new regularization scheme that enforces convolutional feature maps to have a low total variation. We introduce the idea of mixing two experts that specialize in high-frequency and low-frequency robustness.

- Improving Model Robustness by Adaptively Correcting Perturbation Levels with Active Queries. [[paper]](https://arxiv.org/abs/2103.14824)
  - Kun-Peng Ning, Lue Tao, Songcan Chen, Sheng-Jun Huang. *AAAI 2021*
  - Key Word: Corruption Robustness; Active Learning.
  - Digest: We propose to adaptively adjust the perturbation levels for each example in the training process. Specifically, a novel active learning framework is proposed to allow the model to interactively query the correct perturbation level from human experts.

- Understanding Robustness of Transformers for Image Classification. [[paper]](https://arxiv.org/abs/2103.14586)
  - Srinadh Bhojanapalli, Ayan Chakrabarti, Daniel Glasner, Daliang Li, Thomas Unterthiner, Andreas Veit.
  - Key Word: Corruption Robustness; Transformers.
  - Digest: When the training set is small, the ViTs are less robust compared to ResNets of comparable sizes, and increasing the size of the ViTs does not lead to better robustness.

- StyleLess layer: Improving robustness for real-world driving. [[paper]](https://arxiv.org/abs/2103.13905)
  - Julien Rebut, Andrei Bursuc, Patrick Pérez.
  - Key Word: Corruption Robustness; Style Information.
  - Digest: We design a new layer, StyleLess, that enables the model to ignore noisy style information and learn from the content instead.

- Robust and Accurate Object Detection via Adversarial Learning. [[paper]](https://arxiv.org/abs/2103.13886) [[code]](https://github.com/google/automl/blob/master/efficientdet/Det-AdvProp.md)
  - Xiangning Chen, Cihang Xie, Mingxing Tan, Li Zhang, Cho-Jui Hsieh, Boqing Gong. *CVPR 2021*
  - Key Word: Corruption Robustness; Detection.
  - Digest: Our method dynamically selects the stronger adversarial images sourced from a detector’s classification and localization branches and evolves with the detector to ensure the augmentation policy stays current and relevant.

- Generalizing to Unseen Domains: A Survey on Domain Generalization. [[paper]](https://arxiv.org/abs/2103.03097) [[code]](https://github.com/jindongwang/transferlearning)
  - Jindong Wang, Cuiling Lan, Chang Liu, Yidong Ouyang, Tao Qin, Wang Lu, Yiqiang Chen, Wenjun Zeng, Philip S. Yu. *IJCAI 2021*
  - Key Word: Survey; Domain Generalization.
  - Digest: This paper presents the first review of recent advances in this area. First, we provide a formal definition of domain generalization and discuss several related fields. We then thoroughly review the theories related to domain generalization and carefully analyze the theory behind generalization. We categorize recent algorithms into three classes: data manipulation, representation learning, and learning strategy, and present several popular algorithms in detail for each category. Third, we introduce the commonly used datasets, applications, and our open-sourced codebase for fair evaluation. Finally, we summarize existing literature and present some potential research topics for the future.

- Lost in Pruning: The Effects of Pruning Neural Networks beyond Test Accuracy. [[paper]](https://arxiv.org/abs/2103.03014) [[code]](https://github.com/lucaslie/torchprune)
  - Lucas Liebenwein, Cenk Baykal, Brandon Carter, David Gifford, Daniela Rus. *MLSys 2021*
  - Key Word: Corruption Robustness; Pruning.
  - Digest: Here, we reassess and evaluate whether the use of test accuracy alone in the terminating condition is sufficient to ensure that the resulting model performs well across a wide spectrum of "harder" metrics such as generalization to out-of-distribution data and resilience to noise.

- Domain Generalization: A Survey. [[paper]](https://arxiv.org/abs/2103.02503) [[code]](https://github.com/KaiyangZhou/Dassl.pytorch)
  - Kaiyang Zhou, Ziwei Liu, Yu Qiao, Tao Xiang, Chen Change Loy.
  - Key Word: Survey; Domain Generalization.
  - Digest: In this paper, for the first time a comprehensive literature review in DG is provided to summarize the developments over the past decade. Specifically, we first cover the background by formally defining DG and relating it to other relevant fields like domain adaptation and transfer learning. Then, we conduct a thorough review into existing methods and theories. Finally, we conclude this survey with insights and discussions on future research directions.

- Regularizing towards Causal Invariance: Linear Models with Proxies. [[paper]](https://arxiv.org/abs/2103.02477) [[code]](https://github.com/clinicalml/proxy-anchor-regression)
  - Michael Oberst, Nikolaj Thams, Jonas Peters, David Sontag. *ICML 2021*
  - Key Word: Causal Invariance; Distribution Shift.
  - Digest: We propose a method for learning linear models whose predictive performance is robust to causal interventions on unobserved variables, when noisy proxies of those variables are available. Our approach takes the form of a regularization term that trades off between in-distribution performance and robustness to interventions. Under the assumption of a linear structural causal model, we show that a single proxy can be used to create estimators that are prediction optimal under interventions of bounded strength.

- On the effectiveness of adversarial training against common corruptions. [[paper]](https://arxiv.org/abs/2103.02325v1) [[code]](https://github.com/tml-epfl/adv-training-corruptions)
  - Klim Kireev, Maksym Andriushchenko, Nicolas Flammarion.
  - Key Word: Corruption Robustness; Adversarial Robustness.
  - Digest: Adversarial training with proper perturbation radius serves as a strong baseline against corruption robustness. Further, adversarial training with LPIPS still works and is significantly faster.

- Nonlinear Invariant Risk Minimization: A Causal Approach. [[paper]](https://arxiv.org/abs/2102.12353)
  - Chaochao Lu, Yuhuai Wu, Jośe Miguel Hernández-Lobato, Bernhard Schölkopf.
  - Key Word: Invariant Learning; Causality.
  - Digest: We propose invariant Causal Representation Learning (iCaRL), an approach that enables out-of-distribution (OOD) generalization in the nonlinear setting (i.e., nonlinear representations and nonlinear classifiers). It builds upon a practical and general assumption: the prior over the data representation (i.e., a set of latent variables encoding the data) given the target and the environment belongs to general exponential family distributions.

- Explainers in the Wild: Making Surrogate Explainers Robust to Distortions through Perception. [[paper]](https://arxiv.org/abs/2102.10951v1)
  - Alexander Hepburn, Raul Santos-Rodriguez.
  - Key Word: Corruption Robustness.
  - Digest: We propose a methodology to evaluate the effect of distortions in explanations by embedding perceptual distances that tailor the neighbourhoods used to training surrogate explainers. We also show that by operating in this way, we can make the explanations more robust to distortions.

- Linear unit-tests for invariance discovery. [[paper]](https://arxiv.org/abs/2102.10867) [[code]](https://github.com/facebookresearch/InvarianceUnitTests)
  - Benjamin Aubin, Agnieszka Słowik, Martin Arjovsky, Leon Bottou, David Lopez-Paz.
  - Key Word: Invariant Learning; Causality; Empirical Study.
  - Digest: There is an increasing interest in algorithms to learn invariant correlations across training environments. A big share of the current proposals find theoretical support in the causality literature but, how useful are they in practice? The purpose of this note is to propose six linear low-dimensional problems -- unit tests -- to evaluate different types of out-of-distribution generalization in a precise manner. Following initial experiments, none of the three recently proposed alternatives passes all tests.

- SelfNorm and CrossNorm for Out-of-Distribution Robustness. [[paper]](https://arxiv.org/abs/2102.02811) [[code]](https://github.com/amazon-research/crossnorm-selfnorm)
  - Zhiqiang Tang, Yunhe Gao, Yi Zhu, Zhi Zhang, Mu Li, Dimitris Metaxas. *ICCV 2021*
  - Key Word: Corruption Robustness; Normalization.
  - Digest: Unlike most previous works, this paper presents two normalization methods, SelfNorm and CrossNorm, to promote OOD generalization. SelfNorm uses attention to recalibrate statistics (channel-wise mean and variance), while CrossNorm exchanges the statistics between feature maps.

- Does Invariant Risk Minimization Capture Invariance? [[paper]](https://arxiv.org/abs/2101.01134)
  - Pritish Kamath, Akilesh Tangella, Danica J. Sutherland, Nathan Srebro. *AISTATS 2021*
  - Key Word: Invariant Learning; Causality.
  - Digest: We show that the Invariant Risk Minimization (IRM) formulation of Arjovsky et al. (2019) can fail to capture "natural" invariances, at least when used in its practical "linear" form, and even on very simple problems which directly follow the motivating examples for IRM. This can lead to worse generalization on new environments, even when compared to unconstrained ERM. The issue stems from a significant gap between the linear variant (as in their concrete method IRMv1) and the full non-linear IRM formulation.

### Out-of-Distribution Generalization: 2020

- Auxiliary Training: Towards Accurate and Robust Models. [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Auxiliary_Training_Towards_Accurate_and_Robust_Models_CVPR_2020_paper.html)
  - Linfeng Zhang, Muzhou Yu, Tong Chen, Zuoqiang Shi, Chenglong Bao, Kaisheng Ma. *CVPR 2020*
  - Key Word: Auxiliary Learning; Corruption Robustness.
  - Digest: In this paper, we propose a novel training method via introducing the auxiliary classifiers for training on corrupted samples, while the clean samples are normally trained with the primary classifier.  

- Generative Interventions for Causal Learning. [[paper]](https://arxiv.org/abs/2012.12265) [[code]](https://github.com/cvlab-columbia/GenInt)
  - Chengzhi Mao, Augustine Cha, Amogh Gupta, Hao Wang, Junfeng Yang, Carl Vondrick. *CVPR 2021*
  - Key Word: Corruption Robustness; Causal Learning.
  - Digest: We introduce a framework for learning robust visual representations that generalize to new viewpoints, backgrounds, and scene contexts. Discriminative models often learn naturally occurring spurious correlations, which cause them to fail on images outside of the training distribution. In this paper, we show that we can steer generative models to manufacture interventions on features caused by confounding factors.

- Simulating a Primary Visual Cortex at the Front of CNNs Improves Robustness to Image Perturbations. [[paper]](https://nips.cc/virtual/2020/public/poster_98b17f068d5d9b7668e19fb8ae470841.html) [[code]](https://github.com/dicarlolab/vonenet)
  - Joel Dapello, Tiago Marques, Martin Schrimpf, Franziska Geiger, David Cox, James J DiCarlo. *NeurIPS 2020*
  - Key Word: Corruption Robustness; Neuroscience.
  - Digest: We developed VOneNets, a new class of hybrid CNN vision models. The VOneBlock is based on a classical neuroscientific model of V1: the linear-nonlinear-Poisson model, consisting of a biologically-constrained Gabor filter bank, simple and complex cell nonlinearities, and a V1 neuronal stochasticity generator. We show that all components of the VOneBlock work in synergy to improve robustness (white box adversarial attack and corruption).

- What Can Style Transfer and Paintings Do For Model Robustness? [[paper]](https://arxiv.org/abs/2011.14477) [[code]](https://github.com/hubertsgithub/style_painting_robustness)
  - Hubert Lin, Mitchell van Zuijlen, Sylvia C. Pont, Maarten W.A. Wijntjes, Kavita Bala.
  - Key Word: Corruption Robustness; Style Transfer.
  - Digest: We argue that paintings can be considered a form of perceptual data augmentation, and demonstrate that it can improve model robustness.

- Improved Handling of Motion Blur in Online Object Detection. [[paper]](https://arxiv.org/abs/2011.14448)
  - Mohamed Sayed, Gabriel Brostow. *CVPR 2021*
  - Key Word: Corruption Robustness; Detection.
  - Digest: We explore five classes of remedies in blurring object detection: deblurring, squint, augmentation, using minibatch statistics and label modification.

- StackMix: A complementary Mix algorithm. [[paper]](https://arxiv.org/abs/2011.12618v2)
  - John Chen, Samarth Sinha, Anastasios Kyrillidis.
  - Key Word: Corruption Robustness; Data Augmentation.
  - Digest: We present StackMix: Each input is presented as a concatenation of two images, and the label is the mean of the two one-hot labels. We further show that gains hold for robustness to common input corruptions and perturbations at varying severities with a 0.7% improvement on CIFAR-100-C, by combining StackMix with AugMix over AugMix.

- An Effective Anti-Aliasing Approach for Residual Networks. [[paper]](https://arxiv.org/abs/2011.10675v1)
  - Cristina Vasconcelos, Hugo Larochelle, Vincent Dumoulin, Nicolas Le Roux, Ross Goroshin.
  - Key Word: Corruption Robustness; Anti-Aliasing.
  - Digest: We show that we can mitigate this effect by placing non-trainable blur filters and using smooth activation functions at key locations, particularly where networks lack the capacity to learn them. These simple architectural changes lead to substantial improvements in out-of-distribution generalization on both image classification under natural corruptions on ImageNetC and few-shot learning on Meta-Dataset.

- Empirical or Invariant Risk Minimization? A Sample Complexity Perspective. [[paper]](https://arxiv.org/abs/2010.16412) [[code]](https://github.com/IBM/IRM-games)
  - Kartik Ahuja, Jun Wang, Amit Dhurandhar, Karthikeyan Shanmugam, Kush R. Varshney. *ICLR 2021*
  - Key Word: Invariant Learning; Causality.
  - Digest: Recently, invariant risk minimization (IRM) was proposed as a promising solution to address out-of-distribution (OOD) generalization. However, it is unclear when IRM should be preferred over the widely-employed empirical risk minimization (ERM) framework. In this work, we analyze both these frameworks from the perspective of sample complexity, thus taking a firm step towards answering this important question. We find that depending on the type of data generation mechanism, the two approaches might have very different finite sample and asymptotic behavior.

- Understanding the Failure Modes of Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2010.15775) [[code]](https://github.com/google-research/OOD-failures)
  - Vaishnavh Nagarajan, Anders Andreassen, Behnam Neyshabur. *ICLR 2021*
  - Key Word: Out-of-Distribution Generalization.
  - Digest: We identify the fundamental factors that give rise to this behavior, by explaining why models fail this way even in easy-to-learn tasks where one would expect these models to succeed. In particular, through a theoretical study of gradient-descent-trained linear classifiers on some easy-to-learn tasks, we uncover two complementary failure modes. These modes arise from how spurious correlations induce two kinds of skews in the data: one geometric in nature and another, statistical.

- Improving Transformation Invariance in Contrastive Representation Learning. [[paper]](https://arxiv.org/abs/2010.09515) [[code]](https://github.com/ae-foster/invclr)
  - Adam Foster, Rattana Pukdee, Tom Rainforth. *ICLR 2021*
  - Key Word: Transformation Invariance; Contrastive Learning.
  - Digest: We first introduce a training objective for contrastive learning that uses a novel regularizer to control how the representation changes under transformation. We show that representations trained with this objective perform better on downstream tasks and are more robust to the introduction of nuisance transformations at test time. Second, we propose a change to how test time representations are generated by introducing a feature averaging approach that combines encodings from multiple transformations of the original input, finding that this leads to across the board performance gains.

- Maximum-Entropy Adversarial Data Augmentation for Improved Generalization and Robustness. [[paper]](https://arxiv.org/abs/2010.08001) [[code]](https://github.com/garyzhao/ME-ADA)
  - Long Zhao, Ting Liu, Xi Peng, Dimitris Metaxas. *NeurIPS 2020*
  - Key Word: Corruption Robustness; Data Augmentation.
  - Digest: We develop an efficient maximum-entropy regularizer in the maximization phase of adversarial data argumentation, which results in a notable improvement over corruption (adv-corrupt) benchmarks.

- Environment Inference for Invariant Learning. [[paper]](https://arxiv.org/abs/2010.07249) [[code]](https://github.com/ecreager/eiil)
  - Elliot Creager, Jörn-Henrik Jacobsen, Richard Zemel. *ICML 2021*
  - Key Word: Invariant Learning; Causality; Fairness.
  - Digest: We propose EIIL, a general framework for domain-invariant learning that incorporates Environment Inference to directly infer partitions that are maximally informative for downstream Invariant Learning. We show that EIIL outperforms invariant learning methods on the CMNIST benchmark without using environment labels, and significantly outperforms ERM on worst-group performance in the Waterbirds and CivilComments datasets. Finally, we establish connections between EIIL and algorithmic fairness, which enables EIIL to improve accuracy and calibration in a fair prediction problem.

- Shape-Texture Debiased Neural Network Training. [[paper]](https://arxiv.org/abs/2010.05981v2) [[code]](https://github.com/LiYingwei/ShapeTextureDebiasedTraining)
  - Yingwei Li, Qihang Yu, Mingxing Tan, Jieru Mei, Peng Tang, Wei Shen, Alan Yuille, Cihang Xie. *ICLR 2021*
  - Key Word: Corruption Robustness; Shape-Texture.
  - Digest: We develop a shape-texture debiased neural network training framework to guide CNNs for learning better representations. Our method is a data-driven approach, which let CNNs automatically figure out how to avoid being biased towards either shape or texture from their training samples.

- Permuted AdaIN: Reducing the Bias Towards Global Statistics in Image Classification. [[paper]](https://arxiv.org/abs/2010.05785v3) [[code]](https://github.com/onuriel/PermutedAdaIN)
  - Oren Nuriel, Sagie Benaim, Lior Wolf. *CVPR 2021*
  - Key Word: Corruption Robustness; Normalization.
  - Digest: Our method, called Permuted Adaptive Instance Normalization (pAdaIN), reduces the representation of global statistics in the hidden layers of image classifiers. pAdaIN samples a random permutation that rearranges the samples in a given batch.

- The Risks of Invariant Risk Minimization. [[paper]](https://arxiv.org/abs/2010.05761)
  - Elan Rosenfeld, Pradeep Ravikumar, Andrej Risteski. *ICLR 2021*
  - Key Word: Invariant Learning; Causality.
  - Digest: We present the first analysis of classification under the IRM objective--as well as these recently proposed alternatives--under a fairly natural and general model. In the linear case, we show simple conditions under which the optimal solution succeeds or, more often, fails to recover the optimal invariant predictor.

- Increasing the Robustness of Semantic Segmentation Models with Painting-by-Numbers. [[paper]](https://arxiv.org/abs/2010.05495)
  - Christoph Kamann, Burkhard Güssefeld, Robin Hutmacher, Jan Hendrik Metzen, Carsten Rother. *ECCV 2020*
  - Key Word: Corruption Robustness; Data Augmentation; Segmentation.
  - Digest: Our basic idea is to alpha-blend a portion of the RGB training images with faked images, where each class-label is given a fixed, randomly chosen color that is not likely to appear in real imagery. This forces the network to rely more strongly on shape cues. We call this data augmentation technique “Painting-by-Numbers”.

- Revisiting Batch Normalization for Improving Corruption Robustness. [[paper]](https://arxiv.org/abs/2010.03630v4)
  - Philipp Benz, Chaoning Zhang, Adil Karjauv, In So Kweon. *WACV 2021*
  - Key Word: Corruption Robustness; Normalization.
  - Digest: We interpret corruption robustness as a domain shift and propose to rectify batch normalization (BN) statistics for improving model robustness.

- Batch Normalization Increases Adversarial Vulnerability: Disentangling Usefulness and Robustness of Model Features. [[paper]](https://arxiv.org/abs/2010.03316)
  - Philipp Benz, Chaoning Zhang, In So Kweon.
  - Key Word: Corruption Robustness; Normalization.
  - Digest: We conjecture that the increased adversarial vulnerability is caused by BN shifting the model to rely more on non-robust features (NRFs). Our exploration finds that other normalization techniques also increase adversarial vulnerability and our conjecture is also supported by analyzing the model corruption robustness and feature transferability.

- Adversarial and Natural Perturbations for General Robustness. [[paper]](https://arxiv.org/abs/2010.01401)
  - Sadaf Gulshad, Jan Hendrik Metzen, Arnold Smeulders.
  - Key Word: Common Robustness; Adversarial Robustness.
  - Digest: We demonstrate that although adversarial training improves the performance of the networks against adversarial perturbations, it leads to drop in the performance for naturally perturbed samples besides clean samples. In contrast, natural perturbations like elastic deformations, occlusions and wave does not only improve the performance against natural perturbations, but also lead to improvement in the performance for the adversarial perturbations.

- Prepare for the Worst: Generalizing across Domain Shifts with Adversarial Batch Normalization. [[paper]](https://arxiv.org/abs/2009.08965v2)
  - Manli Shu, Zuxuan Wu, Micah Goldblum, Tom Goldstein.
  - Key Word: Corruption Robustness; Normalization.
  - Digest: Our method, Adversarial Batch Normalization (AdvBN, adversarially perturbing these feature statistics), significantly improves robustness.

- Stochastic-YOLO: Efficient Probabilistic Object Detection under Dataset Shifts. [[paper]](https://arxiv.org/abs/2009.02967) [[code]](https://github.com/tjiagom/stochastic-yolo)
  - Tiago Azevedo, René de Jong, Matthew Mattina, Partha Maji.
  - Key Word: Corruption Robustness; Detection.
  - Digest: We adapt the well-established YOLOv3 architecture to generate uncertainty estimations by introducing stochasticity in the form of Monte Carlo Dropout (MC-Drop), and evaluate it across different levels of dataset shift.

- Learning explanations that are hard to vary. [[paper]](https://arxiv.org/abs/2009.00329) [[code]](https://github.com/gibipara92/learning-explanations-hard-to-vary)
  - Giambattista Parascandolo, Alexander Neitz, Antonio Orvieto, Luigi Gresele, Bernhard Schölkopf. *ICLR 2021*
  - Key Word: Invariant Learning; Gradient Alignment.
  - Digest: In this paper, we investigate the principle that good explanations are hard to vary in the context of deep learning. We show that averaging gradients across examples -- akin to a logical OR of patterns -- can favor memorization and 'patchwork' solutions that sew together different strategies, instead of identifying invariances. To inspect this, we first formalize a notion of consistency for minima of the loss surface, which measures to what extent a minimum appears only when examples are pooled.

- Addressing Neural Network Robustness with Mixup and Targeted Labeling Adversarial Training. [[paper]](https://arxiv.org/abs/2008.08384)
  - Alfred Laugros, Alice Caplier, Matthieu Ospici.
  - Key Word: Corruption Robustness; Adversarial Robustness.
  - Digest: Our approach combines the Mixup augmentation and a new adversarial training algorithm called Targeted Labeling Adversarial Training (TLAT). The idea of TLAT is to interpolate the target labels of adversarial examples with the groundtruth labels.

- What Should Not Be Contrastive in Contrastive Learning. [[paper]](https://arxiv.org/abs/2008.05659)
  - Tete Xiao, Xiaolong Wang, Alexei A. Efros, Trevor Darrell. *ICLR 2021*
  - Key Word: Corruption Robustness; Contrastive Learning.
  - Digest: Current methods introduce inductive bias by encouraging neural networks to be less sensitive to information w.r.t. augmentation, which may help or may hurt. We use a multi-head network with a shared backbone which captures information across each augmentation and alone outperforms all baselines on downstream tasks.

- Informative Dropout for Robust Representation Learning: A Shape-bias Perspective. [[paper]](https://arxiv.org/abs/2008.04254) [[code]](https://github.com/bfshi/InfoDrop)
  - Baifeng Shi, Dinghuai Zhang, Qi Dai, Jingdong Wang, Zhanxing Zhu, Yadong Mu. *ICML 2020*
  - Key Word: Dropout; Shape-Texture.
  - Digest: In this work, we attempt at improving various kinds of robustness universally by alleviating CNN's texture bias. Specifically, with inspiration from human visual system, we propose a light-weight model-agnostic method, namely Informative Dropout (InfoDrop), to improve interpretability and reduce texture bias.  

- Robust and Generalizable Visual Representation Learning via Random Convolutions. [[paper]](https://arxiv.org/abs/2007.13003) [[code]](https://github.com/wildphoton/RandConv)
  - Zhenlin Xu, Deyi Liu, Junlin Yang, Colin Raffel, Marc Niethammer. *ICLR 2021*
  - Key Word: Corruption Robustness; Data Augmentation.
  - Digest: We develop RandConv, a data augmentation technique using multi-scale random-convolutions to generate images with random texture while maintaining global shapes. We explore using the RandConv output as training images or mixing it with the original images. We show that a consistency loss can further enforce invariance under texture changes.

- Robust Image Classification Using A Low-Pass Activation Function and DCT Augmentation. [[paper]](https://arxiv.org/abs/2007.09453v1) [[code]](https://github.com/tahmid0007/Low_Pass_ReLU)
  - Md Tahmid Hossain, Shyh Wei Teng, Ferdous Sohel, Guojun Lu.
  - Key Word: Corruption Robustness; Activation Function.
  - Digest: We propose a family of novel AFs with low-pass filtering to improve robustness against HFc (we call it Low-Pass ReLU or LP-ReLU). To deal with LFc, we further enhance the AFs with Discrete Cosine Transform (DCT) based augmentation. LPReLU coupled with DCT augmentation, enables a deep network to tackle a variety of corruptions.

- Learning perturbation sets for robust machine learning. [[paper]](https://arxiv.org/abs/2007.08450) [[code]](https://github.com/locuslab/perturbation_learning)
  - Eric Wong, J. Zico Kolter. *ICLR 2021*
  - Key Word: Corruption Robustness; Adversarial Robustness.
  - Digest: We use a conditional generator that defines the perturbation set over a constrained region of the latent space. We formulate desirable properties that measure the quality of a learned perturbation set, and theoretically prove that a conditional variational autoencoder naturally satisfies these criteria.

- On the relationship between class selectivity, dimensionality, and robustness. [[paper]](https://arxiv.org/abs/2007.04440)
  - Matthew L. Leavitt, Ari S. Morcos.
  - Key Word: Corruption Robustness; Class Selectivity; Interpretability.
  - Digest: We found that mean class selectivity predicts vulnerability to naturalistic corruptions; networks regularized to have lower levels of class selectivity are more robust to corruption, while networks with higher class selectivity are more vulnerable to corruption, as measured using Tiny ImageNetC and CIFAR10C. In contrast, we found that class selectivity increases robustness to multiple types of gradient-based adversarial attacks.

- In Search of Lost Domain Generalization. [[paper]](https://arxiv.org/abs/2007.01434) [[code]](https://github.com/facebookresearch/DomainBed)
  - Ishaan Gulrajani, David Lopez-Paz. *ICLR 2021*
  - Key Word: Domain Generalization; Empirical Study.
  - Digest: As a first step, we realize that model selection is non-trivial for domain generalization tasks, and we argue that algorithms without a model selection criterion remain incomplete. Next we implement DomainBed, a testbed for domain generalization including seven benchmarks, fourteen algorithms, and three model selection criteria. When conducting extensive experiments using DomainBed we find that when carefully implemented and tuned, ERM outperforms the state-of-the-art in terms of average performance.

- Improving robustness against common corruptions by covariate shift adaptation. [[paper]](https://arxiv.org/abs/2006.16971) [[code]](https://github.com/bethgelab/robustness)
  - Steffen Schneider, Evgenia Rusak, Luisa Eck, Oliver Bringmann, Wieland Brendel, Matthias Bethge. *NeurIPS 2020*
  - Key Word: Corruption Robustness; Unsupervised Domain Adaptation.
  - Digest: We suggest to augment current benchmarks for common corruptions with two additional performance metrics that measure robustness after partial and full unsupervised adaptation to the corrupted images.

- The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2006.16241) [[dataset]](https://github.com/hendrycks/imagenet-r)
  - Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, Justin Gilmer.
  - Key Word: Dataset; Benchmark; Data Augmentation; Corruption Robustness.
  - Digest: We introduce three new robustness benchmarks consisting of naturally occurring distribution changes in image style, geographic location, camera operation, and more. Using our benchmarks, we take stock of previously proposed hypotheses for out-of-distribution robustness and put them to the test.  

- Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift. [[paper]](https://arxiv.org/abs/2006.10963v3)
  - Zachary Nado, Shreyas Padhy, D. Sculley, Alexander D'Amour, Balaji Lakshminarayanan, Jasper Snoek.
  - Key Word: Corruption Robustness; Normalization.
  - Digest: We formalize a perhaps underappreciated prediction setting, which we call the prediction-time batch setting. We then propose a simple method, prediction-time BN, for using this information to effectively correct for covariate shift.

- Tent: Fully Test-time Adaptation by Entropy Minimization. [[paper]](https://arxiv.org/abs/2006.10726v3) [[code]](https://github.com/DequanWang/tent)
  - Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, Trevor Darrell. *ICLR 2021*
  - Key Word: Corruption Robustness; Test-Time Adaptation.
  - Digest: We examine entropy as an adaptation objective and propose tent: a test-time entropy minimization scheme to reduce generalization error by reducing the entropy of model predictions on test data.

- Risk Variance Penalization. [[paper]](https://arxiv.org/abs/2006.07544)
  - Chuanlong Xie, Haotian Ye, Fei Chen, Yue Liu, Rui Sun, Zhenguo Li.
  - Key Word: Invariant Learning; Causality; Distributional Robustness.
  - Digest: The key of the out-of-distribution (OOD) generalization is to generalize invariance from training domains to target domains. The variance risk extrapolation (V-REx) is a practical OOD method, which depends on a domain-level regularization but lacks theoretical verifications about its motivation and utility. This article provides theoretical insights into V-REx by studying a variance-based regularizer. We propose Risk Variance Penalization (RVP), which slightly changes the regularization of V-REx but addresses the theory concerns about V-REx.

- An Unsupervised Information-Theoretic Perceptual Quality Metric. [[paper]](https://arxiv.org/abs/2006.06752v3) [[code]](https://github.com/google-research/perceptual-quality)
  - Sangnie Bhardwaj, Ian Fischer, Johannes Ballé, Troy Chinen. *NeurIPS 2020*
  - Key Word: Corruption Robustness; Information Theory.
  - Digest: We combine information-theoretic objective functions with a computational architecture informed by the physiology of the human visual system and unsupervised training on pairs of video frames, yielding our Perceptual Information Metric (PIM). We show that PIM is competitive with supervised metrics on the recent and challenging BAPPS image quality assessment dataset and perform qualitative experiments using the ImageNet-C dataset, and establish that PIM is robust with respect to architectural details.

- Wavelet Integrated CNNs for Noise-Robust Image Classification. [[paper]](https://arxiv.org/abs/2005.03337v2) [[code]](https://github.com/LiQiufu/WaveCNet)
  - Qiufu Li, Linlin Shen, Sheng Guo, Zhihui Lai. *CVPR 2020*
  - Key Word: Corruption Robustness; Pooling.
  - Digest: We enhance CNNs by replacing max-pooling, strided-convolution, and average-pooling with Discrete Wavelet Transform (DWT). We present general DWT and Inverse DWT (IDWT) layers applicable to various wavelets like Haar, Daubechies, and Cohen, etc., and design wavelet integrated CNNs (WaveCNets) using these layers for image classification

- Shortcut Learning in Deep Neural Networks. [[paper]](https://arxiv.org/abs/2004.07780) [[code]](https://github.com/rgeirhos/shortcut-perspective)
  - Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, Felix A. Wichmann. *Nature Machine Intelligence*
  - Key Word: Survey; Out-of-Distribution Generalization; Neuroscience.
  - Digest: We seek to distil how many of deep learning's problem can be seen as different symptoms of the same underlying problem: shortcut learning. Shortcuts are decision rules that perform well on standard benchmarks but fail to transfer to more challenging testing conditions, such as real-world scenarios. Related issues are known in Comparative Psychology, Education and Linguistics, suggesting that shortcut learning may be a common characteristic of learning systems, biological and artificial alike.

- ObjectNet Dataset: Reanalysis and Correction. [[paper]](https://arxiv.org/abs/2004.02042) [[code]](https://github.com/aliborji/ObjectNetReanalysis)
  - Ali Borji.
  - Key Word: Dataset; Detection.
  - Digest: We highlight a major problem with their work which is applying object recognizers to the scenes containing multiple objects rather than isolated objects. The latter results in around 20-30% performance gain using our code. Compared with the results reported in the ObjectNet paper, we observe that around 10-15 % of the performance loss can be recovered, without any test time data augmentation.  

- Invariant Rationalization. [[paper]](https://arxiv.org/abs/2003.09772) [[code]](https://github.com/code-terminator/invariant_rationalization)
  - Shiyu Chang, Yang Zhang, Mo Yu, Tommi S. Jaakkola.
  - Key Word: Invariant Learning; Causality; Mutual Information.
  - Digest: Selective rationalization improves neural network interpretability by identifying a small subset of input features -- the rationale -- that best explains or supports the prediction. A typical rationalization criterion, i.e. maximum mutual information (MMI), finds the rationale that maximizes the prediction performance based only on the rationale. However, MMI can be problematic because it picks up spurious correlations between the input features and the output. Instead, we introduce a game-theoretic invariant rationalization criterion where the rationales are constrained to enable the same predictor to be optimal across different environments.

- Out-of-Distribution Generalization via Risk Extrapolation (REx). [[paper]](https://arxiv.org/abs/2003.00688) [[code]](https://github.com/capybaralet/REx_code_release)
  - David Krueger, Ethan Caballero, Joern-Henrik Jacobsen, Amy Zhang, Jonathan Binas, Dinghuai Zhang, Remi Le Priol, Aaron Courville. *ICML 2021*
  - Key Word: Invariant Learning; Causality; Robust Optimization.
  - Digest: We show that reducing differences in risk across training domains can reduce a model's sensitivity to a wide range of extreme distributional shifts, including the challenging setting where the input contains both causal and anti-causal elements. We motivate this approach, Risk Extrapolation (REx), as a form of robust optimization over a perturbation set of extrapolated domains (MM-REx), and propose a penalty on the variance of training risks (V-REx) as a simpler variant.

- CEB Improves Model Robustness. [[paper]](https://arxiv.org/abs/2002.05380)
  - Ian Fischer, Alexander A. Alemi.
  - Key Word: Information Bottleneck; Corruption Robustness; Adversarial Robustness.
  - Digest: We demonstrate that the Conditional Entropy Bottleneck (CEB) can improve model robustness. CEB is an easy strategy to implement and works in tandem with data augmentation procedures. We report results of a large scale adversarial robustness study on CIFAR-10, as well as the ImageNet-C Common Corruptions Benchmark, ImageNet-A, and PGD attacks.  

- Invariant Risk Minimization Games. [[paper]](https://arxiv.org/abs/2002.04692) [[code]](https://github.com/IBM/OoD/tree/master/IRM_games)
  - Kartik Ahuja, Karthikeyan Shanmugam, Kush R. Varshney, Amit Dhurandhar. *ICML 2020*
  - Key Word: Invariant Learning; Causality; Game Theory.
  - Digest: We pose such invariant risk minimization as finding the Nash equilibrium of an ensemble game among several environments. By doing so, we develop a simple training algorithm that uses best response dynamics and, in our experiments, yields similar or better empirical accuracy with much lower variance than the challenging bi-level optimization problem of Arjovsky et al. (2019). One key theoretical contribution is showing that the set of Nash equilibria for the proposed game are equivalent to the set of invariant predictors for any finite number of environments, even with nonlinear classifiers and transformations.

- A simple way to make neural networks robust against diverse image corruptions. [[paper]](https://arxiv.org/abs/2001.06057) [[code]](https://github.com/bethgelab/game-of-noise)
  - Evgenia Rusak, Lukas Schott, Roland S. Zimmermann, Julian Bitterwolf, Oliver Bringmann, Matthias Bethge, Wieland Brendel. *ECCV 2020*
  - Key Word: Data Augmentation; Corruption Robustness.
  - Digest: We demonstrate that a simple but properly tuned training with additive Gaussian and Speckle noise generalizes surprisingly well to unseen corruptions, easily reaching the previous state of the art on the corruption benchmark ImageNet-C (with ResNet50) and on MNIST-C.  

### Out-of-Distribution Generalization: 2019

- ObjectNet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. [[paper]](https://papers.nips.cc/paper/9142-objectnet-a-large-scale-bias-controlled-dataset-for-pushing-the-limits-of-object-recognition-models.pdf) [[dataset]](https://objectnet.dev/)
  - Barbu, A, Mayo, D, Alverio, J, Luo, W, Wang, C, Gutfreund, D, Tenenabum, JB, Katz, B. *NeurIPS 2019*
  - Key Word: Dataset.
  - Digest: A new method, which we refer to as convex layerwise adversarial training (COLT), that can train provably robust neural networks and conceptually bridges the gap between adversarial training and existing provable defense methods.  

- AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty. [[paper]](https://arxiv.org/abs/1912.02781) [[code]](https://github.com/google-research/augmix)
  - Dan Hendrycks, Norman Mu, Ekin D. Cubuk, Barret Zoph, Justin Gilmer, Balaji Lakshminarayanan. *ICLR 2020*
  - Key Word: Dataset; Corruption Robustness.
  - Digest: In this work, we propose a technique to improve the robustness and uncertainty estimates of image classifiers. We propose AugMix, a data processing technique that is simple to implement, adds limited computational overhead, and helps models withstand unforeseen corruptions.  

- Adversarial Examples Improve Image Recognition. [[paper]](https://arxiv.org/abs/1911.09665v2) [[code]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
  - Cihang Xie, Mingxing Tan, Boqing Gong, Jiang Wang, Alan Yuille, Quoc V. Le. *CVPR 2020*
  - Key Word: Corruption Robustness; Adversarial Robustness.
  - Digest: Adversarial examples can be used to improve image recognition models. Key to our method is the usage of a separate auxiliary batch norm for adversarial examples, as they have different underlying distributions to normal examples.

- Self-training with Noisy Student improves ImageNet classification. [[paper]](https://arxiv.org/abs/1911.04252v4) [[code]](https://github.com/google-research/noisystudent)
  - Qizhe Xie, Minh-Thang Luong, Eduard Hovy, Quoc V. Le. *CVPR 2020*
  - Key Word: Corruption Robustness; Self-Training.
  - Digest: Noisy Student Training extends the idea of self-training and distillation with the use of equal-or-larger student models and noise added to the student during learning.

- Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. [[paper]](https://arxiv.org/abs/1909.13231) [[code]](https://github.com/yueatsprograms/ttt_cifar_release)
  - Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, Moritz Hardt. *ICML 2020*
  - Key Word: Corruption Robustness; Test-Time Training.
  - Digest: We turn a single unlabeled test sample into a self-supervised learning problem, on which we update the model parameters before making a prediction. Our simple approach leads to improvements on diverse image classification benchmarks aimed at evaluating robustness to distribution shifts.

- Pretraining boosts out-of-domain robustness for pose estimation. [[paper]](https://arxiv.org/abs/1909.11229) [[code]](https://github.com/DeepLabCut/DeepLabCut)
  - Alexander Mathis, Thomas Biasi, Steffen Schneider, Mert Yüksekgönül, Byron Rogers, Matthias Bethge, Mackenzie W. Mathis. *WACV 2021*
  - Key Word: Corruption Robustness; Pre-Trainig; Pose Estimation.
  - Digest: We show that better ImageNet-performing architectures perform better on both within- and out-of-domain data if they are first pretrained on ImageNet. We additionally show that better ImageNet models generalize better across animal species. Furthermore, we introduce Horse-C, a new benchmark for common corruptions for pose estimation.

- Training Robust Deep Neural Networks via Adversarial Noise Propagation. [[paper]](https://arxiv.org/abs/1909.09034) [[code]](https://github.com/AnonymousCodeRepo/ANP)
  - Aishan Liu, Xianglong Liu, Chongzhi Zhang, Hang Yu, Qiang Liu, Dacheng Tao.
  - Key Word: Corruption Robustness; Data Augmentation.
  - Digest: This paper proposes a simple yet powerful training algorithm, named Adversarial Noise Propagation (ANP), which injects noise into the hidden layers in a layer-wise manner.

- PDA: Progressive Data Augmentation for General Robustness of Deep Neural Networks. [[paper]](https://arxiv.org/abs/1909.04839v3)
  - Hang Yu, Aishan Liu, Xianglong Liu, Gengchao Li, Ping Luo, Ran Cheng, Jichen Yang, Chongzhi Zhang.
  - Key Word: Corruption Robustness; Data Augmentation.
  - Digest: We propose a simple yet effective method, named Progressive Data Augmentation (PDA), which enables general robustness of DNNs by progressively injecting diverse adversarial noises during training.

- Are Adversarial Robustness and Common Perturbation Robustness Independent Attributes? [[paper]](https://arxiv.org/abs/1909.02436v2)
  - Alfred Laugros, Alice Caplier, Matthieu Ospici.
  - Key Word: Corruption Robustness; Adversarial Robustness.
  - Digest: We show that increasing the robustness to carefully selected common perturbations, can make neural networks more robust to unseen common perturbations. We also prove that adversarial robustness and robustness to common perturbations are independent.

- Testing Robustness Against Unforeseen Adversaries. [[paper]](https://arxiv.org/abs/1908.08016) [[code]](https://github.com/ddkang/advex-uar)
  - Daniel Kang, Yi Sun, Dan Hendrycks, Tom Brown, Jacob Steinhardt.
  - Key Word: Corruption Robustness; Adversarial Robustness.
  - Digest: We propose unforeseen attacks. The JPEG, Fog, Snow, and Gabor adversarial attacks are visually distinct from previous attacks and serve as unforeseen attacks in the ImageNet-UA attack suite.

- Distributionally Robust Optimization: A Review. [[paper]](https://arxiv.org/abs/1908.05659)
  - Hamed Rahimian, Sanjay Mehrotra.
  - Key Word: Survey; Robust Optimization.
  - Digest: The concepts of risk-aversion, chance-constrained optimization, and robust optimization have developed significantly over the last decade. Statistical learning community has also witnessed a rapid theoretical and applied growth by relying on these concepts. A modeling framework, called distributionally robust optimization (DRO), has recently received significant attention in both the operations research and statistical learning communities. This paper surveys main concepts and contributions to DRO, and its relationships with robust optimization, risk-aversion, chance-constrained optimization, and function regularization.

- Benchmarking the Robustness of Semantic Segmentation Models. [[paper]](https://arxiv.org/abs/1908.05005)
  - Christoph Kamann, Carsten Rother. *CVPR 2020*
  - Key Word: Corruption Robustness; Benchmark; Segmentation.
  - Digest: While there are recent robustness studies for full-image classification, we are the first to present an exhaustive study for semantic segmentation, based on the state-of-the-art model DeepLabv3+. To increase the realism of our study, we utilize almost 400,000 images generated from Cityscapes, PASCAL VOC 2012, and ADE20K.

- Robustness properties of Facebook's ResNeXt WSL models. [[paper]](https://arxiv.org/abs/1907.07640) [[code]](https://github.com/eminorhan/resnext-wsl)
  - A. Emin Orhan.
  - Key Word: Corruption Robustness; Emprical Study.
  - Digest: We show that although the ResNeXt WSL models are more shape-biased than comparable ImageNet-trained models in a shape-texture cue conflict experiment, they still remain much more texture-biased than humans, suggesting that they share some of the underlying characteristics of ImageNet-trained models that make this benchmark challenging.

- Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming. [[paper]](https://arxiv.org/abs/1907.07484)
  - Claudio Michaelis, Benjamin Mitzkus, Robert Geirhos, Evgenia Rusak, Oliver Bringmann, Alexander S. Ecker, Matthias Bethge, Wieland Brendel.
  - Key Word: Corruption Robustness; Benchmark; Detection.
  - Digest: We here provide an easy-to-use benchmark to assess how object detection models perform when image quality degrades. The three resulting benchmark datasets, termed Pascal-C, Coco-C and Cityscapes-C, contain a large variety of image corruptions.

- Natural Adversarial Examples. [[paper]](https://arxiv.org/abs/1907.07174) [[dataset]](https://github.com/hendrycks/natural-adv-examples)
  - Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, Dawn Song.
  - Key Word: Dataset; Out-of-Distribution Detection; Natural Adversarial Examples.
  - Digest: We introduce natural adversarial examples–real-world, unmodified, and naturally occurring examples that cause machine learning model performance to substantially degrade.  

- Invariant Risk Minimization. [[paper]](https://arxiv.org/abs/1907.02893) [[code]](https://github.com/facebookresearch/InvariantRiskMinimization)
  - Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, David Lopez-Paz.
  - Key Word: Invariant Learning; Causality.
  - Digest: We introduce Invariant Risk Minimization (IRM), a learning paradigm to estimate invariant correlations across multiple training distributions. To achieve this goal, IRM learns a data representation such that the optimal classifier, on top of that data representation, matches for all training distributions. Through theory and experiments, we show how the invariances learned by IRM relate to the causal structures governing the data and enable out-of-distribution generalization.

- A Fourier Perspective on Model Robustness in Computer Vision. [[paper]](https://arxiv.org/abs/1906.08988v3) [[code]](https://github.com/google-research/google-research/tree/master/frequency_analysis)
  - Dong Yin, Raphael Gontijo Lopes, Jonathon Shlens, Ekin D. Cubuk, Justin Gilmer. *NeurIPS 2019*
  - Key Word: Corruption Robustness; Frequency.
  - Digest: We investigate recently observed tradeoffs caused by Gaussian data augmentation and adversarial training. We find that both methods improve robustness to corruptions that are concentrated in the high frequency domain while reducing robustness to corruptions that are concentrated in the low frequency domain.

- A Closed-Form Learned Pooling for Deep Classification Networks. [[paper]](https://arxiv.org/abs/1906.03808v1)
  - Vighnesh Birodkar, Hossein Mobahi, Dilip Krishnan, Samy Bengio.
  - Key Word: Corruption Robustness; Pooling.
  - Digest: We propose a way to enable CNNs to learn different pooling weights for each pixel location. We do so by introducing an extended definition of a pooling operator. This operator can learn a strict super-set of what can be learned by average pooling or convolutions.

- Towards Non-I.I.D. Image Classification: A Dataset and Baselines. [[paper]](https://arxiv.org/abs/1906.02899)
  - Yue He, Zheyan Shen, Peng Cui. *Pattern Recognition*
  - Key Word: Dataset; Distribution Shift.
  - Digest: We construct and release a Non-I.I.D. image dataset called NICO, which uses contexts to create Non-IIDness consciously. Compared to other datasets, extended analyses prove NICO can support various Non-I.I.D. situations with sufficient flexibility. Meanwhile, we propose a baseline model with ConvNet structure for General Non-I.I.D. image classification, where distribution of testing data is unknown but different from training data.

- MNIST-C: A Robustness Benchmark for Computer Vision. [[paper]](https://arxiv.org/abs/1906.02337v1) [[code]](https://github.com/google-research/mnist-c)
  - Norman Mu, Justin Gilmer.
  - Key Word: Corruption Robustness; Benchmark.
  - Digest: We introduce the MNIST-C dataset, a comprehensive suite of 15 corruptions applied to the MNIST test set.

- Improving Robustness Without Sacrificing Accuracy with Patch Gaussian Augmentation. [[paper]](https://arxiv.org/abs/1906.02611)
  - Raphael Gontijo Lopes, Dong Yin, Ben Poole, Justin Gilmer, Ekin D. Cubuk.
  - Key Word: Data Augmentation; Corruption Robustness.
  - Digest: We introduce Patch Gaussian, a simple augmentation scheme that adds noise to randomly selected patches in an input image. Models trained with Patch Gaussian achieve state of the art on the CIFAR-10 and ImageNetCommon Corruptions benchmarks while also improving accuracy on clean data.  

- Making Convolutional Networks Shift-Invariant Again. [[paper]](https://arxiv.org/abs/1904.11486) [[code]](https://richzhang.github.io/antialiased-cnns/)
  - Richard Zhang. *ICML 2019*
  - Key Word: Anti-aliasing.
  - Digest: Small shifts -- even by a single pixel -- can drastically change the output of a deep network (bars on left). We identify the cause: aliasing during downsampling. We anti-alias modern deep networks with classic signal processing, stabilizing output classifications (bars on right). We observe "free", unexpected improvements as well: accuracy increases and improved robustness.  

- Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. [[paper]](https://arxiv.org/abs/1903.12261) [[dataset]](https://github.com/hendrycks/robustness)
  - Dan Hendrycks, Thomas Dietterich. *ICLR 2019*
  - Key Word: Dataset; Benchmark; Corruption Robustness.
  - Digest: In this paper we establish rigorous benchmarks for image classifier robustness. Our first benchmark, ImageNet-C, standardizes and expands the corruption robustness topic, while showing which classifiers are preferable in safety-critical applications. Then we propose a new dataset called ImageNet-P which enables researchers to benchmark a classifier's robustness to common perturbations.  

- Adversarial Examples Are a Natural Consequence of Test Error in Noise. [[paper]](https://arxiv.org/abs/1901.10513v1) [[code]](https://github.com/nicf/corruption-robustness)
  - Nic Ford, Justin Gilmer, Nicolas Carlini, Dogus Cubuk. *ICML 2019*
  - Keyword: Corruption Robustness; Adversarial Robustness.
  - Digest: The existence of adversarial examples follows naturally from the fact that our models have nonzero test error in certain corrupted image distributions (connecting adversarial and corruption robustness).

### Out-of-Distribution Generalization: 2018

- ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. [[paper]](https://arxiv.org/abs/1811.12231) [[code]](https://github.com/rgeirhos/texture-vs-shape)
  - Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A. Wichmann, Wieland Brendel. *ICLR 2019*
  - Key Word: Shape-Texture; Style Transfer; Data Augmentation.
  - Digest: We show that ImageNet-trained CNNs are strongly biased towards recognising textures rather than shapes, which is in stark contrast to human behavioural evidence and reveals fundamentally different classification strategies.  

- Strike (with) a Pose: Neural Networks Are Easily Fooled by Strange Poses of Familiar Objects. [[paper]](https://arxiv.org/abs/1811.11553) [[code]](https://github.com/airalcorn2/strike-with-a-pose)
  - Michael A. Alcorn, Qi Li, Zhitao Gong, Chengfei Wang, Long Mai, Wei-Shinn Ku, Anh Nguyen. *CVPR 2019*
  - Key Word: Adversarial Poses Transfer.
  - Digest: In this paper, we present a framework for discovering DNN failures that harnesses 3D renderers and 3D models. That is, we estimate the parameters of a 3D renderer that cause a target DNN to misbehave in response to the rendered image.  

- Generalisation in humans and deep neural networks. [[paper]](https://arxiv.org/abs/1808.08750) [[code]](https://github.com/rgeirhos/generalisation-humans-DNNs)
  - Robert Geirhos, Carlos R. Medina Temme, Jonas Rauber, Heiko H. Schütt, Matthias Bethge, Felix A. Wichmann. *NeurIPS 2018*
  - Key Word: Corruption Robustness.
  - Digest: We compare the robustness of humans and current convolutional deep neural networks (DNNs) on object recognition under twelve different types of image degradations.  

- Why do deep convolutional networks generalize so poorly to small image transformations? [[paper]](https://arxiv.org/abs/1805.12177) [[code]](https://github.com/AzulEye/CNN-Failures)
  - Aharon Azulay, Yair Weiss. *JMLR 2019*
  - Key Word: Transformation Invaraince.
  - Digest: In this paper, we quantify this phenomena and ask why neither the convolutional architecture nor data augmentation are sufficient to achieve the desired invariance. Specifically, we show that the convolutional architecture does not give invariance since architectures ignore the classical sampling theorem, and data augmentation does not give invariance because the CNNs learn to be invariant to transformations only for images that are very similar to typical images from the training set.  

## Evasion Attacks and Defenses

### Evasion Attacks and Defenses: 2022

- Implicit Bias of Adversarial Training for Deep Neural Networks. [[paper]](https://openreview.net/forum?id=l8It-0lE5e7)
  - Bochen Lv, Zhanxing Zhu. *ICLR 2022*
  - Key Word: Adversarial Training.
  - Digest: We provide theoretical understandings of the implicit bias imposed by adversarial training for homogeneous deep neural networks without any explicit regularization. In particular, for deep linear networks adversarially trained by gradient descent on a linearly separable dataset, we prove that the direction of the product of weight matrices converges to the direction of the max-margin solution of the original dataset.

- Reducing Excessive Margin to Achieve a Better Accuracy vs. Robustness Trade-off. [[paper]](https://openreview.net/forum?id=Azh9QBQ4tR7) [[code]](https://github.com/imrahulr/hat)
  - Rahul Rade, Seyed-Mohsen Moosavi-Dezfooli. *ICLR 2022*
  - Key Word: Adversarial Training.
  - Digest: We closely examine the changes induced in the decision boundary of a deep network during adversarial training. We find that adversarial training leads to unwarranted increase in the margin along certain adversarial directions, thereby hurting accuracy. Motivated by this observation, we present a novel algorithm, called Helper-based Adversarial Training (HAT), to reduce this effect by incorporating additional wrongly labelled examples during training.

- Diffusion Models for Adversarial Purification. [[paper]](https://arxiv.org/abs/2205.07460) [[code]](https://diffpure.github.io/)
  - Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, Anima Anandkumar. *ICML 2022*
  - Key Word: Adversarial Purification; Diffusion Models.
  - Digest: We propose DiffPure that uses diffusion models for adversarial purification: Given an adversarial example, we first diffuse it with a small amount of noise following a forward diffusion process, and then recover the clean image through a reverse generative process. To evaluate our method against strong adaptive attacks in an efficient and scalable way, we propose to use the adjoint method to compute full gradients of the reverse generative process.

- Self-Ensemble Adversarial Training for Improved Robustness. [[paper]](https://arxiv.org/abs/2203.09678) [[code]](https://github.com/whj363636/self-ensemble-adversarial-training)
  - Hongjun Wang, Yisen Wang. *ICLR 2022*
  - Key Word: Adversarial Robustness.
  - Digest: We are dedicated to the weight states of models through the training process and devise a simple but powerful Self-Ensemble Adversarial Training (SEAT) method for yielding a robust classifier by averaging weights of history models. This considerably improves the robustness of the target model against several well known adversarial attacks, even merely utilizing the naive cross-entropy loss to supervise.

- A Unified Wasserstein Distributional Robustness Framework for Adversarial Training. [[paper]](https://arxiv.org/abs/2202.13437) [[code]](https://github.com/tuananhbui89/unified-distributional-robustness)
  - Tuan Anh Bui, Trung Le, Quan Tran, He Zhao, Dinh Phung. *ICLR 2022*
  - Key Word: Adversarial Robustness; Distribution Shift.
  - Digest: This paper presents a unified framework that connects Wasserstein distributional robustness with current state-of-the-art AT methods. We introduce a new Wasserstein cost function and a new series of risk functions, with which we show that standard AT methods are special cases of their counterparts in our framework.

### Evasion Attacks and Defenses: 2021

- Adversarial Robustness through the Lens of Causality. [[paper]](https://arxiv.org/abs/2106.06196)
  - Yonggang Zhang, Mingming Gong, Tongliang Liu, Gang Niu, Xinmei Tian, Bo Han, Bernhard Schölkopf, Kun Zhang. *ICLR 2022*
  - Key Word: Adversarial Robustness; Causality.
  - Digest: We construct a causal graph to model the generation process of adversarial examples and define the adversarial distribution to formalize the intuition of adversarial attacks. From a causal perspective, we find that the label is spuriously correlated with the style (content-independent) information when an instance is given. The spurious correlation implies that the adversarial distribution is constructed via making the statistical conditional association between style information and labels drastically different from that in natural distribution.

- Exploring Memorization in Adversarial Training. [[paper]](https://arxiv.org/abs/2106.01606) [[code]](https://github.com/dongyp13/memorization-AT)
  - Yinpeng Dong, Ke Xu, Xiao Yang, Tianyu Pang, Zhijie Deng, Hang Su, Jun Zhu. *ICLR 2022*
  - Key Word: Adversarial Training; Memorization.
  - Digest: We explore the memorization effect in adversarial training (AT) for promoting a deeper understanding of model capacity, convergence, generalization, and especially robust overfitting of the adversarially trained models. We first demonstrate that deep networks have sufficient capacity to memorize adversarial examples of training data with completely random labels, but not all AT algorithms can converge under the extreme circumstance. Our study of AT with random labels motivates further analyses on the convergence and generalization of AT. We find that some AT approaches suffer from a gradient instability issue and most recently suggested complexity measures cannot explain robust generalization by considering models trained on random labels.

- Robust Learning Meets Generative Models: Can Proxy Distributions Improve Adversarial Robustness? [[paper]](https://arxiv.org/abs/2104.09425) [[code]](https://github.com/inspire-group/proxy-distributions)
  - Vikash Sehwag, Saeed Mahloujifar, Tinashe Handina, Sihui Dai, Chong Xiang, Mung Chiang, Prateek Mittal. *ICLR 2022*
  - Key Word: Adversarial Robustness; Proxy Distribution.
  - Digest: We first seek to formally understand the transfer of robustness from classifiers trained on proxy distributions to the real data distribution. We prove that the difference between the robustness of a classifier on the two distributions is upper bounded by the conditional Wasserstein distance between them. Next we use proxy distributions to significantly improve the performance of adversarial training on five different datasets.

### Evasion Attacks and Defenses: 2020

- Adversarial Training Reduces Information and Improves Transferability. [[paper]](https://arxiv.org/abs/2007.11259)
  - Matteo Terzi, Alessandro Achille, Marco Maggipinto, Gian Antonio Susto.
  - Key Word: Adversarial Training; Inforamtion Bottleneck.
  - Digest: We investigate the dual relationship between Adversarial Training and Information Theory. We show that the Adversarial Training can improve linear transferability to new tasks, from which arises a new trade-off between transferability of representations and accuracy on the source task.  

- Do Adversarially Robust ImageNet Models Transfer Better? [[paper]](https://arxiv.org/abs/2007.08489) [[code]](https://github.com/Microsoft/robust-models-transfer)
  - Hadi Salman, Andrew Ilyas, Logan Engstrom, Ashish Kapoor, Aleksander Madry.
  - Key Word: Adversarial Robustness; Transfer Learning.
  - Digest: In this work, we identify another such aspect: we find that adversarially robust models, while less accurate, often perform better than their standard-trained counterparts when used for transfer learning. Specifically, we focus on adversarially robust ImageNet classifiers, and show that they yield improved accuracy on a standard suite of downstream classification tasks.  

- Learning perturbation sets for robust machine learning. [[paper]](https://arxiv.org/abs/2007.08450) [[code]](https://github.com/locuslab/perturbation_learning)
  - Eric Wong, J. Zico Kolter.
  - Key Word: Data Augmentation; Generative Model; Adversarial Robustness.
  - Digest: Although much progress has been made towards robust deep learning, a significant gap in robustness remains between real-world perturbations and more narrowly defined sets typically studied in adversarial defenses. In this paper, we aim to bridge this gap by learning perturbation sets from data, in order to characterize real-world effects for robust training and evaluation.  

- Understanding Adversarial Examples from the Mutual Influence of Images and Perturbations. [[paper]](https://arxiv.org/abs/2007.06189)
  - Chaoning Zhang, Philipp Benz, Tooba Imtiaz, In-So Kweon. *CVPR 2020*
  - Key Word: Universal Adversarial Perturbations.
  - Digest: Our results suggest a new perspective towards the relationship between images and universal perturbations: Universal perturbations contain dominant features, and images behave like noise to them. We are the first to achieve the challenging task of a targeted universal attack without utilizing original training data. Our approach using a proxy dataset achieves comparable performance to the state-of-the-art baselines which utilize the original training dataset.  

- Understanding and Improving Fast Adversarial Training. [[paper]](https://arxiv.org/abs/2007.02617) [[code]](https://github.com/tml-epfl/understanding-fast-adv-training)
  - Maksym Andriushchenko, Nicolas Flammarion.
  - Key Word: Adversarial Training.
  - Digest: We explored the questions of when and why FGSM adversarial training works, and how to improve it by increasing the gradient alignment, and thus the quality of the solution of the inner maximization problem.  

- On Connections between Regularizations for Improving DNN Robustness. [[paper]](https://arxiv.org/abs/2007.02209)
  - Yiwen Guo, Long Chen, Yurong Chen, Changshui Zhang. *TPAMI*
  - Key Word: Adversarial Robustness; Regularizations.
  - Digest: This paper analyzes regularization terms proposed recently for improving the adversarial robustness of deep neural networks (DNNs), from a theoretical point of view.  

- Biologically Inspired Mechanisms for Adversarial Robustness. [[paper]](https://arxiv.org/abs/2006.16427)
  - Manish V. Reddy, Andrzej Banburski, Nishka Pant, Tomaso Poggio.
  - Key Word: Biologically inspired mechanisms; Adversarial Training.
  - Digest: In this work, we investigate the role of two biologically plausible mechanisms in adversarial robustness. We demonstrate that the non-uniform sampling performed by the primate retina and the presence of multiple receptive fields with a range of receptive field sizes at each eccentricity improve the robustness of neural networks to small adversarial perturbations.  

- Proper Network Interpretability Helps Adversarial Robustness in Classification. [[paper]](https://arxiv.org/abs/2006.14748) [[code]](https://github.com/AkhilanB/Proper-Interpretability)
  - Akhilan Boopathy, Sijia Liu, Gaoyuan Zhang, Cynthia Liu, Pin-Yu Chen, Shiyu Chang, Luca Daniel. *ICML 2020*
  - Key Word: Interpretability; Adversarial Robustness.
  - Digest: In this paper, we theoretically show that with a proper measurement of interpretation, it is actually difficult to prevent prediction-evasion adversarial attacks from causing interpretation discrepancy, as confirmed by experiments on MNIST, CIFAR-10 and Restricted ImageNet.  

- Smooth Adversarial Training. [[paper]](https://arxiv.org/abs/2006.14536)
  - Cihang Xie, Mingxing Tan, Boqing Gong, Alan Yuille, Quoc V. Le.
  - Key Word: ReLU; Adversarial Training.
  - Digest: Hence we propose smooth adversarial training (SAT), in which we replace ReLU with its smooth approximations to strengthen adversarial training. The purpose of smooth activation functions in SAT is to allow it to find harder adversarial examples and compute better gradient updates during adversarial training. Compared to standard adversarial training, SAT improves adversarial robustness for "free", i.e., no drop in accuracy and no increase in computational cost.  

- A Causal View on Robustness of Neural Networks. [[paper]](https://arxiv.org/abs/2005.01095)
  - Cheng Zhang, Kun Zhang, Yingzhen Li. *NeurIPS 2020*
  - Key Word: Adversarial Robustness; Causal Learning; Disentangled Representations.
  - Digest: We present a causal view on the robustness of neural networks against input manipulations, which applies not only to traditional classification tasks but also to general measurement data. Based on this view, we design a deep causal manipulation augmented model (deep CAMA) which explicitly models possible manipulations on certain causes leading to changes in the observed effect.  

- Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. [[paper]](https://arxiv.org/abs/2003.01690) [[code]](https://github.com/fra31/auto-attack)
  - Francesco Croce, Matthias Hein. *ICML 2020*
  - Key Word: Adversarial Attacks.
  - Digest: In this paper we first propose two extensions of the PGD-attack overcoming failures due to suboptimal step size and problems of the objective function. We then combine our novel attacks with two complementary existing ones to form a parameter-free, computationally affordable and user-independent ensemble of attacks to test adversarial robustness.  

- Overfitting in adversarially robust deep learning. [[paper]](https://arxiv.org/abs/2002.11569) [[code]](https://github.com/locuslab/robust_overfitting)
  - Leslie Rice, Eric Wong, J. Zico Kolter. *ICML 2020*
  - Key Word: Adversarial Training; Overfitting.
  - Digest: In this paper, we empirically study this phenomenon in the setting of adversarially trained deep networks, which are trained to minimize the loss under worst-case adversarial perturbations. We find that overfitting to the training set does in fact harm robust performance to a very large degree in adversarially robust training across multiple datasets (SVHN, CIFAR-10, CIFAR-100, and ImageNet) and perturbation models.  

- Attacks Which Do Not Kill Training Make Adversarial Learning Stronger. [[paper]](https://arxiv.org/abs/2002.11242)
  - Jingfeng Zhang, Xilie Xu, Bo Han, Gang Niu, Lizhen Cui, Masashi Sugiyama, Mohan Kankanhalli. *ICML 2020*
  - Key Word: Adversarial Training.
  - Digest: In this paper, we raise a fundamental question---do we have to trade off natural generalization for adversarial robustness? We argue that adversarial training is to employ confident adversarial data for updating the current model. We propose a novel approach of friendly adversarial training (FAT): rather than employing most adversarial data maximizing the loss, we search for least adversarial (i.e., friendly adversarial) data minimizing the loss, among the adversarial data that are confidently misclassified.  

- The Curious Case of Adversarially Robust Models: More Data Can Help, Double Descend, or Hurt Generalization. [[paper]](https://arxiv.org/abs/2002.11080)
  - Yifei Min, Lin Chen, Amin Karbasi.
  - Key Word: Adversarial Robustness; Deep Double Descend.
  - Digest: In the weak adversary regime, more data improves the generalization of adversarially robust models. In the medium adversary regime, with more training data, the generalization loss exhibits a double descent curve, which implies the existence of an intermediate stage where more training data hurts the generalization. In the strong adversary regime, more data almost immediately causes the generalization error to increase.  

- Understanding and Mitigating the Tradeoff Between Robustness and Accuracy. [[paper]](https://arxiv.org/abs/2002.10716)
  - Aditi Raghunathan, Sang Michael Xie, Fanny Yang, John Duchi, Percy Liang. *ICML 2020*
  - Key Word: Adversarial Robustness.
  - Digest: In this work, we precisely characterize the effect of augmentation on the standard error in linear regression when the optimal linear predictor has zero standard and robust error.  

- A Bayes-Optimal View on Adversarial Examples. [[paper]](https://arxiv.org/abs/2002.08859)
  - Eitan Richardson, Yair Weiss.
  - Key Word: Adversarial Robustness; Bayes-Optimal.
  - Digest: In this paper, we argue for examining adversarial examples from the perspective of Bayes-Optimal classification. We construct realistic image datasets for which the Bayes-Optimal classifier can be efficiently computed and derive analytic conditions on the distributions so that the optimal classifier is either robust or vulnerable.  

- More Data Can Expand the Generalization Gap Between Adversarially Robust and Standard Models. [[paper]](https://arxiv.org/abs/2002.04725)
  - Lin Chen, Yifei Min, Mingrui Zhang, Amin Karbasi. *ICML 2020*
  - Key Word: Adversarial Robustness; Deep Double Descent.
  - Digest: The conventional wisdom is that more training data should shrink the generalization gap between adversarially-trained models and standard models. However, we study the training of robust classifiers for both Gaussian and Bernoulli models under l-inf attacks, and we prove that more data may actually increase this gap.  

- Fundamental Tradeoffs between Invariance and Sensitivity to Adversarial Perturbations. [[paper]](https://arxiv.org/abs/2002.04599) [[code]](https://github.com/ftramer/Excessive-Invariance)
  - Florian Tramèr, Jens Behrmann, Nicholas Carlini, Nicolas Papernot, Jörn-Henrik Jacobsen. *ICML 2020*
  - Key Word: Adversarial Robustness; Invariance; Sensitivity.
  - Digest:  We demonstrate fundamental tradeoffs between these two types of adversarial examples.
    We show that defenses against sensitivity-based attacks actively harm a model's accuracy on invariance-based attacks, and that new approaches are needed to resist both attack types.  

### Evasion Attacks and Defenses: 2019

- Adversarial Training and Provable Defenses: Bridging the Gap. [[paper]](https://openreview.net/forum?id=SJxSDxrKDr) [[code]](https://github.com/eth-sri/colt)
  - Mislav Balunovic, Martin Vechev. *ICLR 2020*
  - Key Word: Adversarial Training; Provable Defenses.
  - Digest: The key idea is to model neural network training as a procedure which includes both, the verifier and the adversary. In every iteration, the verifier aims to certify the network using convex relaxation while the adversary tries to find inputs inside that convex relaxation which cause verification to fail.  

- When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks. [[paper]](https://arxiv.org/abs/1911.10695) [[code]](https://github.com/gmh14/RobNets)
  - Minghao Guo, Yuzhe Yang, Rui Xu, Ziwei Liu, Dahua Lin. *CVPR 2020*
  - Key Word: Neural Achitecture Search; Adversarial Robustness.
  - Digest: In this work, we take an architectural perspective and investigate the patterns of network architectures that are resilient to adversarial attacks. We discover a family of robust architectures (RobNets).  

- Confidence-Calibrated Adversarial Training: Generalizing to Unseen Attacks. [[paper]](https://arxiv.org/abs/1910.06259)
  - David Stutz, Matthias Hein, Bernt Schiele. *ICML 2020*
  - Key Word: Adversarial Training; Out-of-Distribution Detection.
  - Digest: Typically robustness does not generalize to previously unseen threat models, e.g., other Lp norms, or larger perturbations. Our confidence-calibrated adversarial training (CCAT) tackles this problem by biasing the model towards low confidence predictions on adversarial examples. By allowing to reject examples with low confidence, robustness generalizes beyond the threat model employed during training.  

- Lower Bounds on Adversarial Robustness from Optimal Transport. [[paper]](https://arxiv.org/abs/1909.12272) [[code]](https://github.com/inspire-group/robustness-via-transport)
  - Arjun Nitin Bhagoji, Daniel Cullina, Prateek Mittal. *NeurIPS 2019*
  - Key Word: Adversarial Robustness; Optimal Transport.
  - Digest: In this paper, we use optimal transport to characterize the minimum possible loss in an adversarial classification scenario. In this setting, an adversary receives a random labeled example from one of two classes, perturbs the example subject to a neighborhood constraint, and presents the modified example to the classifier.  

- Defending Against Physically Realizable Attacks on Image Classification. [[paper]](https://arxiv.org/abs/1909.09552) [[code]](https://github.com/tongwu2020/phattacks)
  - Tong Wu, Liang Tong, Yevgeniy Vorobeychik. *ICLR 2020*
  - Key Word: Physical Attacks.
  - Digest: First, we demonstrate that the two most scalable and effective methods for learning robust models, adversarial training with PGD attacks and randomized smoothing, exhibit very limited effectiveness against three of the highest profile physical attacks. Next, we propose a new abstract adversarial model, rectangular occlusion attacks, in which an adversary places a small adversarially crafted rectangle in an image, and develop two approaches for efficiently computing the resulting adversarial examples.  

- Biologically inspired sleep algorithm for artificial neural networks. [[paper]](https://arxiv.org/abs/1908.02240)
  - Giri P Krishnan, Timothy Tadros, Ramyaa Ramyaa, Maxim Bazhenov. *ICLR 2020*
  - Key Word: Biologically Inspired Sleep Algorithm.
  - Digest: We provide a theoretical basis for the beneficial role of the brain-inspired sleep-like phase for the ANNs and present an algorithmic way for future implementations of the various features of sleep in deep learning ANNs.  

- Defense Against Adversarial Attacks Using Feature Scattering-based Adversarial Training. [[paper]](https://arxiv.org/abs/1907.10764) [[code]](https://github.com/Haichao-Zhang/FeatureScatter)
  - Haichao Zhang, Jianyu Wang. *NeurIPS 2019*
  - Key Word: Feature Scattering.
  - Digest: We introduce a feature scattering-based adversarial training approach for improving model robustness against adversarial attacks. Conventional adversarial training approaches leverage a supervised scheme (either targeted or non-targeted) in generating attacks for training, which typically suffer from issues such as label leaking as noted in recent works.  

- Image Synthesis with a Single (Robust) Classifier. [[paper]](https://arxiv.org/abs/1906.09453) [[code]](https://github.com/MadryLab/robustness_applications?)
  - Shibani Santurkar, Dimitris Tsipras, Brandon Tran, Andrew Ilyas, Logan Engstrom, Aleksander Madry. *NeurIPS 2019*
  - Key Word: Adversarial Robustness; Image Synthesis.
  - Digest: The crux of our approach is that we train this classifier to be adversarially robust. It turns out that adversarial robustness is precisely what we need to directly manipulate salient features of the input. Overall, our findings demonstrate the utility of robustness in the broader machine learning context.  

- Convergence of Adversarial Training in Overparametrized Neural Networks. [[paper]](https://arxiv.org/abs/1906.07916)
  - Ruiqi Gao, Tianle Cai, Haochuan Li, Liwei Wang, Cho-Jui Hsieh, Jason D. Lee. *NeurIPS 2019*
  - Key Word: Adversarial Training.
  - Digest:  This paper provides a partial answer to the success of adversarial training, by showing that it converges to a network where the surrogate loss with respect to the the attack algorithm is within theta of the optimal robust loss.  

- Lower Bounds for Adversarially Robust PAC Learning. [[paper]](https://arxiv.org/abs/1906.05815)
  - Dimitrios I. Diochnos, Saeed Mahloujifar, Mohammad Mahmoody. *ISAIM 2020*
  - Key Word: Adversarial Attacks; Poisoning Attacks.
  - Digest: We formalize hybrid attacks in which the evasion attack is preceded by a poisoning attack. This is perhaps reminiscent of "trapdoor attacks" in which a poisoning phase is involved as well, but the evasion phase here uses the error-region definition of risk that aims at misclassifying the perturbed instances.

- Adversarial Robustness as a Prior for Learned Representations. [[paper]](https://arxiv.org/abs/1906.00945) [[code]](https://github.com/MadryLab/robust_representations)
  - Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Brandon Tran, Aleksander Madry.
  - Key Word: Adversarial Robustness; Interpretability; Feature Visualization.
  - Digest: In this work, we show that robust optimization can be re-cast as a tool for enforcing priors on the features learned by deep neural networks. It turns out that representations learned by robust models address the aforementioned shortcomings and make significant progress towards learning a high-level encoding of inputs.  

- Unlabeled Data Improves Adversarial Robustness. [[paper]](https://arxiv.org/abs/1905.13736) [[code]](https://github.com/yaircarmon/semisup-adv)
  - Yair Carmon, Aditi Raghunathan, Ludwig Schmidt, Percy Liang, John C. Duchi. *NeurIPS 2019*
  - Key Word: Adversarial Robustness; Semi-Supervision.
  - Digest: Theoretically, we revisit the simple Gaussian model of [Schmidt et al](https://arxiv.org/abs/1804.11285). that shows a sample complexity gap between standard and robust classification. We prove that unlabeled data bridges this gap: a simple semi-supervised learning procedure (self-training) achieves high robust accuracy using the same number of labels required for achieving high standard accuracy.  

- Are Labels Required for Improving Adversarial Robustness? [[paper]](https://arxiv.org/abs/1905.13725) [[code]](https://github.com/deepmind/deepmind-research/tree/master/unsupervised_adversarial_training)
  - Jonathan Uesato, Jean-Baptiste Alayrac, Po-Sen Huang, Robert Stanforth, Alhussein Fawzi, Pushmeet Kohli. *NeurIPS 2019*
  - Key Word: Unsupervised Adversarial Training.
  - Digest: Our main insight is that unlabeled data can be a competitive alternative to labeled data for training adversarially robust models. Theoretically, we show that in a simple statistical setting, the sample complexity for learning an adversarially robust model from unlabeled data matches the fully supervised case up to constant factors.  

- High Frequency Component Helps Explain the Generalization of Convolutional Neural Networks. [[paper]](https://arxiv.org/abs/1905.13545) [[code]](https://github.com/HaohanWang/HFC)
  - Haohan Wang, Xindi Wu, Zeyi Huang, Eric P. Xing. *CVPR 2020*
  - Key Word: Adversarial Robustness; High-Frequency.
  - Digest: We investigate the relationship between the frequency spectrum of image data and the generalization behavior of convolutional neural networks (CNN). We first notice CNN's ability in capturing the high-frequency components of images. These high-frequency components are almost imperceptible to a human.  

- Interpreting Adversarially Trained Convolutional Neural Networks. [[paper]](https://arxiv.org/abs/1905.09797) [[code]](https://github.com/PKUAI26/AT-CNN)
  - Tianyuan Zhang, Zhanxing Zhu. *ICML 2019*
  - Key Word: Adversarial Robustness; Shape-Texture; Interpretability.
  - Digest: Surprisingly, we find that adversarial training alleviates the texture bias of standard CNNs when trained on object recognition tasks, and helps CNNs learn a more shape-biased representation.  

- Adversarial Examples Are Not Bugs, They Are Features. [[paper]](https://arxiv.org/abs/1905.02175) [[dataset]](https://github.com/MadryLab/constructed-datasets)
  - Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Logan Engstrom, Brandon Tran, Aleksander Madry. *NeurIPS 2019*
  - Key Word: Adversarial Robustness; Interpretability.
  - Digest: We demonstrate that adversarial examples can be directly attributed to the presence of non-robust features: features derived from patterns in the data distribution that are highly predictive, yet brittle and incomprehensible to humans.  

- You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle. [[paper]](https://arxiv.org/abs/1905.00877) [[code]](https://github.com/a1600012888/YOPO-You-Only-Propagate-Once)
  - Dinghuai Zhang, Tianyuan Zhang, Yiping Lu, Zhanxing Zhu, Bin Dong. *NeurIPS 2019*
  - Key Word: Adversarial Training.
  - Digest: In this paper, we show that adversarial training can be cast as a discrete time differential game. Through analyzing the Pontryagin's Maximal Principle (PMP) of the problem, we observe that the adversary update is only coupled with the parameters of the first layer of the network. This inspires us to restrict most of the forward and back propagation within the first layer of the network during adversary updates.  

- NATTACK: Learning the Distributions of Adversarial Examples for an Improved Black-Box Attack on Deep Neural Networks. [[paper]](https://arxiv.org/abs/1905.00441) [[code]](https://github.com/Cold-Winter/Nattack)
  - Yandong Li, Lijun Li, Liqiang Wang, Tong Zhang, Boqing Gong. *ICML 2019*
  - Key Word: Adversarial Attacks.
  - Digest: Instead of searching for an "optimal" adversarial example for a benign input to a targeted DNN, our algorithm finds a probability density distribution over a small region centered around the input, such that a sample drawn from this distribution is likely an adversarial example, without the need of accessing the DNN's internal layers or weights.  

- Unrestricted Adversarial Examples via Semantic Manipulation. [[paper]](https://arxiv.org/abs/1904.06347) [[code]](https://github.com/AI-secure/Big-but-Invisible-Adversarial-Attack)
  - Anand Bhattad, Min Jin Chong, Kaizhao Liang, Bo Li, D. A. Forsyth. *ICLR 2020*
  - Key Word: Adversarial Attacks; Semantic Adversarial Examples.
  - Digest: In this paper, we instead introduce "unrestricted" perturbations that manipulate semantically meaningful image-based visual descriptors - color and texture - in order to generate effective and photorealistic adversarial examples. We show that these semantically aware perturbations are effective against JPEG compression, feature squeezing and adversarially trained model.

- VC Classes are Adversarially Robustly Learnable, but Only Improperly. [[paper]](https://arxiv.org/abs/1902.04217)
  - Omar Montasser, Steve Hanneke, Nathan Srebro. *COLT 2019*
  - Key Word: Adversarial Robustness; PAC Learning.
  - Digest: We study the question of learning an adversarially robust predictor. We show that any hypothesis class H with finite VC dimension is robustly PAC learnable with an improper learning rule. The requirement of being improper is necessary as we exhibit examples of hypothesis classes H with finite VC dimension that are not robustly PAC learnable with any proper learning rule.

- Certified Adversarial Robustness via Randomized Smoothing. [[paper]](https://arxiv.org/abs/1902.02918) [[code]](https://github.com/locuslab/smoothing)
  - Jeremy M Cohen, Elan Rosenfeld, J. Zico Kolter. *ICML 2019*
  - Key Word: Certified Adversarial Robustness.
  - Digest: We show how to turn any classifier that classifies well under Gaussian noise into a new classifier that is certifiably robust to adversarial perturbations under the L-2 norm. This "randomized smoothing" technique has been proposed recently in the literature, but existing guarantees are loose. We prove a tight robustness guarantee in L-2 norm for smoothing with Gaussian noise.  

- Theoretically Principled Trade-off between Robustness and Accuracy. [[paper]](https://arxiv.org/abs/1901.08573) [[code]](https://github.com/yaodongyu/TRADES)
  - Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, Michael I. Jordan. *ICML 2019*
  - Key Word: Adversarial Training.
  - Digest: In this work, we decompose the prediction error for adversarial examples (robust error) as the sum of the natural (classification) error and boundary error, and provide a differentiable upper bound using the theory of classification-calibrated loss, which is shown to be the tightest possible upper bound uniform over all probability distributions and measurable predictors. Inspired by our theoretical analysis, we also design a new defense method, TRADES, to trade adversarial robustness off against accuracy.  

### Evasion Attacks and Defenses: 2018

- Feature Denoising for Improving Adversarial Robustness. [[paper]](https://arxiv.org/abs/1812.03411) [[code]](https://github.com/facebookresearch/ImageNet-Adversarial-Training)
  - Cihang Xie, Yuxin Wu, Laurens van der Maaten, Alan Yuille, Kaiming He. *CVPR 2019*
  - Key Word: Adversarial Robustness; Pixel Denoising.
  - Digest: This study suggests that adversarial perturbations on images lead to noise in the features constructed by these networks. Motivated by this observation, we develop new network architectures that increase adversarial robustness by performing feature denoising. Specifically, our networks contain blocks that denoise the features using non-local means or other filters; the entire networks are trained end-to-end.  

- Excessive Invariance Causes Adversarial Vulnerability. [[paper]](https://arxiv.org/abs/1811.00401)
  - Jörn-Henrik Jacobsen, Jens Behrmann, Richard Zemel, Matthias Bethge. *ICLR 2019*
  - Key Word: Adversarial Robustness; Information Theory.
  - Digest: We show deep networks are not only too sensitive to task-irrelevant changes of their input, as is well-known from epsilon-adversarial examples, but are also too invariant to a wide range of task-relevant changes, thus making vast regions in input space vulnerable to adversarial attacks. We show such excessive invariance occurs across various tasks and architecture types.  

- Rademacher Complexity for Adversarially Robust Generalization. [[paper]](https://arxiv.org/abs/1810.11914)
  - Dong Yin, Kannan Ramchandran, Peter Bartlett. *ICML 2019*
  - Key Word: Adversarial Robustness; Rademacher Complexity.
  - Digest: In this paper, we focus on L-inf attacks, and study the adversarially robust generalization problem through the lens of Rademacher complexity. For binary linear classifiers, we prove tight bounds for the adversarial Rademacher complexity, and show that the adversarial Rademacher complexity is never smaller than its natural counterpart, and it has an unavoidable dimension dependence, unless the weight vector has bounded L-1 norm.  

- Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples. [[paper]](https://arxiv.org/abs/1810.11580)
  - Guanhong Tao, Shiqing Ma, Yingqi Liu, Xiangyu Zhang. *NeurIPS 2018*
  - Key Word: Adversarial Examples Detection; Interpretability.
  - Digest: We argue that adversarial sample attacks are deeply entangled with interpretability of DNN models: while classification results on benign inputs can be reasoned based on the human perceptible features/attributes, results on adversarial samples can hardly be explained. Therefore, we propose a novel adversarial sample detection technique for face recognition models, based on interpretability.  

- Sparse DNNs with Improved Adversarial Robustness. [[paper]](https://arxiv.org/abs/1810.09619)
  - Yiwen Guo, Chao Zhang, Changshui Zhang, Yurong Chen.
  - Key Word: Adversarial Robustness; Sparsity.
  - Digest: Our analyses reveal, both theoretically and empirically, that nonlinear DNN-based classifiers behave differently under l2 attacks from some linear ones. We further demonstrate that an appropriately higher model sparsity implies better robustness of nonlinear DNNs, whereas over-sparsified models can be more difficult to resist adversarial examples.  

- Adversarial Reprogramming of Neural Networks. [[paper]](https://arxiv.org/abs/1806.11146)
  - Gamaleldin F. Elsayed, Ian Goodfellow, Jascha Sohl-Dickstein. *ICLR 2019*
  - Key Word: Adversarial Examples; Transfer Learning.
  - Digest:  We introduce attacks that instead reprogram the target model to perform a task chosen by the attacker without the attacker needing to specify or compute the desired output for each test-time input. This attack finds a single adversarial perturbation, that can be added to all test-time inputs to a machine learning model in order to cause the model to perform a task chosen by the adversary—even if the model was not trained to do this task.  

- PAC-learning in the presence of evasion adversaries. [[paper]](https://arxiv.org/abs/1806.01471)
  - Daniel Cullina, Arjun Nitin Bhagoji, Prateek Mittal. *NeurIPS 2018*
  - Key Word: Adversarial Robustness; PAC Learning.
  - Digest: In this paper, we step away from the attack-defense arms race and seek to understand the limits of what can be learned in the presence of an evasion adversary. In particular, we extend the Probably Approximately Correct (PAC)-learning framework to account for the presence of an adversary.  

- Robustness May Be at Odds with Accuracy. [[paper]](https://arxiv.org/abs/1805.12152)
  - Dimitris Tsipras, Shibani Santurkar, Logan Engstrom, Alexander Turner, Aleksander Madry. *ICLR 2019*
  - Key Word: Adversarial Robustness.
  - Digest: We show that there may exist an inherent tension between the goal of adversarial robustness and that of standard generalization. Specifically, training robust models may not only be more resource-consuming, but also lead to a reduction of standard accuracy.  

- Adversarial examples from computational constraints. [[paper]](https://arxiv.org/abs/1805.10204)
  - Sébastien Bubeck, Eric Price, Ilya Razenshteyn. *ICML 2019*
  - Key Word: Adversarial Robustness.
  - Digest: First we prove that, for a broad set of classification tasks, the mere existence of a robust classifier implies that it can be found by a possibly exponential-time algorithm with relatively few training examples. Then we give a particular classification task where learning a robust classifier is computationally intractable.  

- Towards the first adversarially robust neural network model on MNIST. [[paper]](https://arxiv.org/abs/1805.09190)
  - Lukas Schott, Jonas Rauber, Matthias Bethge, Wieland Brendel. *ICLR 2019*
  - Key Word: Adversarial Robustness.
  - Digest: We present a novel robust classification model that performs analysis by synthesis using learned class-conditional data distributions. We demonstrate that most adversarial examples are strongly perturbed towards the perceptual boundary between the original and the adversarial class.

- Adversarially Robust Generalization Requires More Data. [[paper]](https://arxiv.org/abs/1804.11285)
  - Ludwig Schmidt, Shibani Santurkar, Dimitris Tsipras, Kunal Talwar, Aleksander Mądry. *NeurIPS 2018*
  - Key Word: Adversarial Robustness.
  - Digest: We show that already in a simple natural data model, the sample complexity of robust learning can be significantly larger than that of "standard" learning. This gap is information theoretic and holds irrespective of the training algorithm or the model family.  

- Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples. [[paper]](https://arxiv.org/abs/1802.00420) [[code]](https://github.com/anishathalye/obfuscated-gradients)
  - Anish Athalye, Nicholas Carlini, David Wagner. *ICML 2018*
  - Key Word: Adversarial Robustness.
  - Digest: We identify obfuscated gradients, a kind of gradient masking, as a phenomenon that leads to a false sense of security in defenses against adversarial examples. While defenses that cause obfuscated gradients appear to defeat iterative optimization-based attacks, we find defenses relying on this effect can be circumvented.  

- Stochastic Activation Pruning for Robust Adversarial Defense. [[paper]](https://arxiv.org/abs/1803.01442) [[code]](https://github.com/Guneet-Dhillon/Stochastic-Activation-Pruning)
  - Guneet S. Dhillon, Kamyar Azizzadenesheli, Zachary C. Lipton, Jeremy Bernstein, Jean Kossaifi, Aran Khanna, Anima Anandkumar. *ICLR 2018*
  - Key Word: Adversarial Robustness; Activation Function.
  - Digest: We propose Stochastic Activation Pruning (SAP), a mixed strategy for adversarial defense. SAP prunes a random subset of activations (preferentially pruning those with smaller magnitude) and scales up the survivors to compensate.  

- Adversarial vulnerability for any classifier. [[paper]](https://arxiv.org/abs/1802.08686)
  - Alhussein Fawzi, Hamza Fawzi, Omar Fawzi. *NeurIPS 2018*
  - Key Word: Adversarial Examples.
  - Digest: In this paper, we study the phenomenon of adversarial perturbations under the assumption that the data is generated with a smooth generative model. We derive fundamental upper bounds on the robustness to perturbations of any classification function, and prove the existence of adversarial perturbations that transfer well across different classifiers with small risk.  

### Evasion Attacks and Defenses: 2017

- Certifying Some Distributional Robustness with Principled Adversarial Training. [[paper]](https://arxiv.org/abs/1710.10571)
  - Aman Sinha, Hongseok Namkoong, Riccardo Volpi, John Duchi. *ICLR 2018*
  - Key Word: Certificated Adversarial Robustness.
  - Digest: By considering a Lagrangian penalty formulation of perturbing the underlying data distribution in a Wasserstein ball, we provide a training procedure that augments model parameter updates with worst-case perturbations of training data. For smooth losses, our procedure provably achieves moderate levels of robustness with little computational or statistical cost relative to empirical risk minimization.  

- One pixel attack for fooling deep neural networks. [[paper]](https://arxiv.org/abs/1710.08864)
  - Jiawei Su, Danilo Vasconcellos Vargas, Sakurai Kouichi. *IEEE Transactions on Evolutionary Computation*
  - Key Word: Adversarial Attacks.
  - Digest: In this paper, we analyze an attack in an extremely limited scenario where only one pixel can be modified. For that we propose a novel method for generating one-pixel adversarial perturbations based on differential evolution (DE). It requires less adversarial information (a black-box attack) and can fool more types of networks due to the inherent features of DE.  

### Evasion Attacks and Defenses: 2016

- Practical Black-Box Attacks against Machine Learning. [[paper]](https://arxiv.org/abs/1602.02697)
  - Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z. Berkay Celik, Ananthram Swami. *AsiaCCS 2017*
  - Key Word: Adversarial Attacks.
  - Digest: We introduce the first practical demonstration of an attacker controlling a remotely hosted DNN with no such knowledge. Our attack evades a category of defenses, which we call gradient masking, previously proposed to increase resilience to adversarial examples.  

## Poisoning Attacks and Defenses

### Poisoning Attacks and Defenses: 2020

- Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks. [[paper]](https://arxiv.org/abs/2007.02343)
  - Yunfei Liu, Xingjun Ma, James Bailey, Feng Lu. *ECCV 2020*
  - Key Word: Backdoor Attacks; Natural Reflection.
  - Digest: In this paper, we present a new type of backdoor attack inspired by an important natural phenomenon: reflection. Using mathematical modeling of physical reflection models, we propose reflection backdoor (Refool) to plant reflections as backdoor into a victim model.  

### Poisoning Attacks and Defenses: 2017

- BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain. [[paper]](https://arxiv.org/abs/1708.06733)
  - Tianyu Gu, Brendan Dolan-Gavitt, Siddharth Garg.
  - Key Word: Backdoor Attacks; Neuron Activation.
  - Digest:  In this paper we show that outsourced training introduces new security risks: an adversary can create a maliciously trained network (a backdoored neural network, or a BadNet that has state-of-the-art performance on the user's training and validation samples, but behaves badly on specific attacker-chosen inputs.  

## Interpretability

### Interpretability: 2020

- Understanding and Diagnosing Vulnerability under Adversarial Attacks. [[paper]](https://arxiv.org/abs/2007.08716)
  - Haizhong Zheng, Ziqi Zhang, Honglak Lee, Atul Prakash.
  - Key Word: Generative Adversarial Nets; Adversarial Attacks.
  - Digest: In this work, we propose a novel interpretability method, InterpretGAN, to generate explanations for features used for classification in latent variables. Interpreting the classification process of adversarial examples exposes how adversarial perturbations influence features layer by layer as well as which features are modified by perturbations.  

### Interpretability: 2019

- Attributional Robustness Training using Input-Gradient Spatial Alignment. [[paper]](https://arxiv.org/abs/1911.13073) [[code]](https://github.com/nupurkmr9/Attributional-Robustness)
  - Mayank Singh, Nupur Kumari, Puneet Mangla, Abhishek Sinha, Vineeth N Balasubramanian, Balaji Krishnamurthy. *ECCV 2020*
  - Key Word: Attributional Robustness.
  - Digest: In this work, we study the problem of attributional robustness (i.e. models having robust explanations) by showing an upper bound for attributional vulnerability in terms of spatial correlation between the input image and its explanation map. We propose a training methodology that learns robust features by minimizing this upper bound using soft-margin triplet loss.  

## Causality

### Causality: 2021

- Towards Causal Representation Learning. [[paper]](https://arxiv.org/abs/2102.11107)
  - Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, Yoshua Bengio. *Proceedings of the IEEE*
  - Key Word: Causal Representation Learning.
  - Digest: The two fields of machine learning and graphical causality arose and developed separately. However, there is now cross-pollination and increasing interest in both fields to benefit from the advances of the other. In the present paper, we review fundamental concepts of causal inference and relate them to crucial open problems of machine learning, including transfer and generalization, thereby assaying how causality can contribute to modern machine learning research.

### Causality: 2020

- Representation Learning via Invariant Causal Mechanisms. [[paper]](https://arxiv.org/abs/2010.07922)
  - Jovana Mitrovic, Brian McWilliams, Jacob Walker, Lars Buesing, Charles Blundell. *ICLR 2021*
  - Key Word: Self-Supervision; Causality.
  - Digest: We propose a novel self-supervised objective, Representation Learning via Invariant Causal Mechanisms (ReLIC), that enforces invariant prediction of proxy targets across augmentations through an invariance regularizer which yields improved generalization guarantees.

- Disentangled Generative Causal Representation Learning. [[paper]](https://arxiv.org/abs/2010.02637) [[code]](https://github.com/xwshen51/DEAR)
  - Xinwei Shen, Furui Liu, Hanze Dong, Qing Lian, Zhitang Chen, Tong Zhang.
  - Key Word: Disentanglement; Generative Model.
  - Digest: This paper proposes a Disentangled gEnerative cAusal Representation (DEAR) learning method. Unlike existing disentanglement methods that enforce independence of the latent variables, we consider the general case where the underlying factors of interests can be causally correlated. We show that previous methods with independent priors fail to disentangle causally correlated factors. Motivated by this finding, we propose a new disentangled learning method called DEAR that enables causal controllable generation and causal representation learning. The key ingredient of this new formulation is to use a structural causal model (SCM) as the prior for a bidirectional generative model.

- Active Invariant Causal Prediction: Experiment Selection through Stability. [[paper]](https://arxiv.org/abs/2006.05690) [[code]](https://github.com/juangamella/aicp)
  - Juan L. Gamella, Christina Heinze-Deml. *NeurIPS 2020*
  - Key Word: Invariant Causal Prediction.
  - Digest: In this work we propose a new active learning (i.e. experiment selection) framework (A-ICP) based on Invariant Causal Prediction (ICP). For general structural causal models, we characterize the effect of interventions on so-called stable sets. We leverage these results to propose several intervention selection policies for A-ICP which quickly reveal the direct causes of a response variable in the causal graph while maintaining the error control inherent in ICP.

- CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models. [[paper]](https://arxiv.org/abs/2004.08697)
  - Mengyue Yang, Furui Liu, Zhitang Chen, Xinwei Shen, Jianye Hao, Jun Wang. *CVPR 2021*
  - Key Word: Disentanlged Representation Learning.
  - Digest: The framework of variational autoencoder (VAE) is commonly used to disentangle independent factors from observations. However, in real scenarios, factors with semantics are not necessarily independent. Instead, there might be an underlying causal structure which renders these factors dependent. We thus propose a new VAE based framework named CausalVAE, which includes a Causal Layer to transform independent exogenous factors into causal endogenous ones that correspond to causally related concepts in data.

### Causality: 2019

- Variational Autoencoders and Nonlinear ICA: A Unifying Framework. [[paper]](https://arxiv.org/abs/1907.04809)
  - Ilyes Khemakhem, Diederik P. Kingma, Ricardo Pio Monti, Aapo Hyvärinen. *AISTATS 2022*
  - Key Word: Causal Representation Learning.
  - Digest: The framework of variational autoencoders allows us to efficiently learn deep latent-variable models, such that the model's marginal distribution over observed variables fits the data. Often, we're interested in going a step further, and want to approximate the true joint distribution over observed and latent variables, including the true prior and posterior distributions over latent variables. This is known to be generally impossible due to unidentifiability of the model. We address this issue by showing that for a broad family of deep latent-variable models, identification of the true joint distribution over observed and latent variables is actually possible up to very simple transformations, thus achieving a principled and powerful form of disentanglement.

## Privacy

### Privacy: 2022

- FedILC: Weighted Geometric Mean and Invariant Gradient Covariance for Federated Learning on Non-IID Data. [[paper]](https://arxiv.org/abs/2205.09305) [[code]](https://github.com/mikemikezhu/FedILC)
  - Mike He Zhu, Léna Néhale Ezzine, Dianbo Liu, Yoshua Bengio.
  - Key Word: Regularization; Federated Learning.
  - Digest: We propose the Federated Invariant Learning Consistency (FedILC) approach, which leverages the gradient covariance and the geometric mean of Hessians to capture both inter-silo and intra-silo consistencies of environments and unravel the domain shift problems in federated networks.

- Robust Unlearnable Examples: Protecting Data Against Adversarial Learning. [[paper]](https://arxiv.org/abs/2203.14533) [[code]](https://github.com/fshp971/robust-unlearnable-examples)
  - Shaopeng Fu, Fengxiang He, Yang Liu, Li Shen, Dacheng Tao. *ICLR 2022*
  - Key Word: Privacy; Adversarial Training.
  - Digest: We first find that the vanilla error-minimizing noise, which suppresses the informative knowledge of data via minimizing the corresponding training loss, could not effectively minimize the adversarial training loss. This explains the vulnerability of error-minimizing noise in adversarial training. Based on the observation, robust error-minimizing noise is then introduced to reduce the adversarial training loss.

### Privacy: 2021

- What Do We Mean by Generalization in Federated Learning? [[paper]](https://arxiv.org/abs/2110.14216) [[code]](https://github.com/google-research/federated/tree/master/generalization)
  - Honglin Yuan, Warren Morningstar, Lin Ning, Karan Singhal. *ICLR 2022*
  - Key Word: Federated Learning.
  - Digest: We propose a framework for disentangling these performance gaps. Using this framework, we observe and explain differences in behavior across natural and synthetic federated datasets, indicating that dataset synthesis strategy can be important for realistic simulations of generalization in federated learning. We propose a semantic synthesis strategy that enables realistic simulation without naturally partitioned data.

### Privacy: 2020

- Descent-to-Delete: Gradient-Based Methods for Machine Unlearning. [[paper]](https://arxiv.org/abs/2007.02923)
  - Seth Neel, Aaron Roth, Saeed Sharifi-Malvajerdi.
  - Key Word: Machine Unlearning.
  - Digest: We study the data deletion problem for convex models. By leveraging techniques from convex optimization and reservoir sampling, we give the first data deletion algorithms that are able to handle an arbitrarily long sequence of adversarial updates while promising both per-deletion run-time and steady-state error that do not grow with the length of the update sequence.  

- When Machine Unlearning Jeopardizes Privacy. [[paper]](https://arxiv.org/abs/2005.02205)
  - Min Chen, Zhikun Zhang, Tianhao Wang, Michael Backes, Mathias Humbert, Yang Zhang.
  - Key Word: Machine Unlearning.
  - Digest: In this paper, we perform the first study on investigating the unintended information leakage caused by machine unlearning. We propose a novel membership inference attack which leverages the different outputs of an ML model's two versions to infer whether the deleted sample is part of the training set.  

- A Framework for Evaluating Gradient Leakage Attacks in Federated Learning. [[paper]](https://arxiv.org/abs/2004.10397)
  - Wenqi Wei, Ling Liu, Margaret Loper, Ka-Ho Chow, Mehmet Emre Gursoy, Stacey Truex, Yanzhao Wu.
  - Key Word: Gradient Inversion Attacks.
  - Digest: In this paper, we present a principled framework for evaluating and comparing different forms of client privacy leakage attacks. We first provide formal and experimental analysis to show how adversaries can reconstruct the private local training data by simply analyzing the shared parameter update from local training (e.g., local gradient or weight update vector).  

- Inverting Gradients -- How easy is it to break privacy in federated learning? [[paper]](https://arxiv.org/abs/2003.14053) [[code]](https://github.com/JonasGeiping/invertinggradients)
  - Jonas Geiping, Hartmut Bauermeister, Hannah Dröge, Michael Moeller.
  - Key Word: Gradient Inversion Attacks.
  - Digest: In this paper we show that sharing parameter gradients is by no means secure: By exploiting a cosine similarity loss along with optimization methods from adversarial attacks, we are able to faithfully reconstruct images at high resolution from the knowledge of their parameter gradients, and demonstrate that such a break of privacy is possible even for trained deep networks.  

- iDLG: Improved Deep Leakage from Gradients. [[paper]](https://arxiv.org/abs/2001.02610) [[code]](https://github.com/PatrickZH/Improved-Deep-Leakage-from-Gradients)
  - Bo Zhao, Konda Reddy Mopuri, Hakan Bilen.
  - Key Word: Gradient Inversion Attacks.
  - Digest: DLG has difficulty in convergence and discovering the ground-truth labels consistently. In this paper, we find that sharing gradients definitely leaks the ground-truth labels. We propose a simple but reliable approach to extract accurate data from the gradients.  

### Privacy: 2019

- Machine Unlearning. [[paper]](https://arxiv.org/abs/1912.03817) [[code]](https://github.com/cleverhans-lab/machine-unlearning)
  - Lucas Bourtoule, Varun Chandrasekaran, Christopher A. Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, Nicolas Papernot.
  - Key Word: Machine Unlearning.
  - Digest:  We introduce SISA training, a framework that expedites the unlearning process by strategically limiting the influence of a data point in the training procedure. While our framework is applicable to any learning algorithm, it is designed to achieve the largest improvements for stateful algorithms like stochastic gradient descent for deep neural networks.  

- Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks. [[paper]](https://arxiv.org/abs/1911.04933)
  - Aditya Golatkar, Alessandro Achille, Stefano Soatto. *CVPR 2020*
  - Key Word: Machine Unlearning.
  - Digest: We propose a method for "scrubbing'" the weights clean of information about a particular set of training data. The method does not require retraining from scratch, nor access to the data originally used for training. Instead, the weights are modified so that any probing function of the weights is indistinguishable from the same function applied to the weights of a network trained without the data to be forgotten.  

- Alleviating Privacy Attacks via Causal Learning. [[paper]](https://arxiv.org/abs/1909.12732) [[code]](https://github.com/microsoft/robustdg)
  - Shruti Tople, Amit Sharma, Aditya Nori. *ICML 2020*
  - Key Word: Membership Inversion Attacks.
  - Digest: To alleviate privacy attacks, we demonstrate the benefit of predictive models that are based on the causal relationships between input features and the outcome. We first show that models learnt using causal structure generalize better to unseen data, especially on data from different distributions than the train distribution.  

- Deep Leakage from Gradients. [[paper]](https://arxiv.org/abs/1906.08935) [[code]](https://github.com/mit-han-lab/dlg)
  - Ligeng Zhu, Zhijian Liu, Song Han. *NeurIPS 2019*
  - Key Word: Gradient Inversion Attacks.
  - Digest: We show that it is possible to obtain the private training data from the publicly shared gradients. We name this leakage as Deep Leakage from Gradient and empirically validate the effectiveness on both computer vision and natural language processing tasks.  

## Fairness

### Fairness: 2022

- Fairness via Explanation Quality: Evaluating Disparities in the Quality of Post hoc Explanations. [[paper]](https://arxiv.org/abs/2205.07277)
  - Jessica Dai, Sohini Upadhyay, Ulrich Aivodji, Stephen H. Bach, Himabindu Lakkaraju. *AIES 2022*
  - Key Word: Fairness; Interpretability.
  - Digest: We first outline the key properties which constitute explanation quality and where disparities can be particularly problematic. We then leverage these properties to propose a novel evaluation framework which can quantitatively measure disparities in the quality of explanations output by state-of-the-art methods. Using this framework, we carry out a rigorous empirical analysis to understand if and when group-based disparities in explanation quality arise. Our results indicate that such disparities are more likely to occur when the models being explained are complex and highly non-linear. In addition, we also observe that certain post hoc explanation methods (e.g., Integrated Gradients, SHAP) are more likely to exhibit the aforementioned disparities.

### Fairness: 2020

- Fairness in the Eyes of the Data: Certifying Machine-Learning Models. [[paper]](https://arxiv.org/abs/2009.01534)
  - Shahar Segal, Yossi Adi, Benny Pinkas, Carsten Baum, Chaya Ganesh, Joseph Keshet.
  - Key Word: Fairness; Privacy.
  - Digest: We present a framework that allows to certify the fairness degree of a model based on an interactive and privacy-preserving test. The framework verifies any trained model, regardless of its training process and architecture. Thus, it allows us to evaluate any deep learning model on multiple fairness definitions empirically.  

### Fairness: 2019

- Training individually fair ML models with Sensitive Subspace Robustness. [[paper]](https://arxiv.org/abs/1907.00020) [[code]](https://github.com/IBM/sensitive-subspace-robustness)
  - Mikhail Yurochkin, Amanda Bower, Yuekai Sun. *ICLR 2020*
  - Key Word: Distributionally Robust Optimization.
  - Digest: We consider training machine learning models that are fair in the sense that their performance is invariant under certain sensitive perturbations to the inputs. For example, the performance of a resume screening system should be invariant under changes to the gender and/or ethnicity of the applicant. We formalize this notion of algorithmic fairness as a variant of individual fairness and develop a distributionally robust optimization approach to enforce it during training.  

## Uncertainty

### Uncertainty: 2019

- Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One. [[paper]](https://arxiv.org/abs/1912.03263) [[code]](https://github.com/wgrathwohl/JEM)
  - Will Grathwohl, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, Kevin Swersky. *ICLR 2020*
  - Key Word: Energy-Based Model; Calibration; Adversarial Robustness; Out-of-Distribution Detection.
  - Digest: Within this framework, standard discriminative architectures may beused and the model can also be trained on unlabeled data. We demonstrate that energy based training of the joint distribution improves calibration, robustness, andout-of-distribution detection while also enabling our models to generate samplesrivaling the quality of recent GAN approaches.  

- Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. [[paper]](https://arxiv.org/abs/1906.02530) [[code]](https://github.com/google-research/google-research/tree/master/uq_benchmark_2019)
  - Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, Jasper Snoek. *NeurIPS 2019*
  - Key Word: Uncerntainty Estimation; Distribution Shift.
  - Digest: Quantifying uncertainty is especially critical in real-world settings, which often involve input distributions that are shifted from the training distribution due to a variety of factors including sample bias and non-stationarity. In such settings, well calibrated uncertainty estimates convey information about when a model's output should (or should not) be trusted.

- A Simple Baseline for Bayesian Uncertainty in Deep Learning. [[paper]](https://arxiv.org/abs/1902.02476) [[code]](https://github.com/wjmaddox/swa_gaussian)
  - Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, Andrew Gordon Wilson. *NeurIPS 2019*
  - Key Word: Calibration; Bayesian Deep Learning.
  - Digest: We propose SWA-Gaussian (SWAG), a simple, scalable, and general purpose approach for uncertainty representation and calibration in deep learning. Stochastic Weight Averaging (SWA), which computes the first moment of stochastic gradient descent (SGD) iterates with a modified learning rate schedule, has recently been shown to improve generalization in deep learning.  

## Ownership

### Ownership: 2019

- Rethinking Deep Neural Network Ownership Verification: Embedding Passports to Defeat Ambiguity Attacks. [[paper]](https://arxiv.org/abs/1909.07830) [[code]](https://github.com/kamwoh/DeepIPR)
  - Lixin Fan, Kam Woh Ng, Chee Seng Chan. *NeurIPS 2019*
  - Key Work: Neural Network Watermarking.
  - Digest: This work proposes novel passport-based DNN ownership verification schemes which are both robust to network modifications and resilient to ambiguity attacks. The gist of embedding digital passports is to design and train DNN models in a way such that, the DNN model performance of an original task will be significantly deteriorated due to forged passports.  

## Environmental Well-being

### Environmental Well-being: 2019

- Green AI. [[paper]](https://arxiv.org/abs/1907.10597)
  - Roy Schwartz, Jesse Dodge, Noah A. Smith, Oren Etzioni.
  - Key Word: Data Efficiency.
  - Digest: This position paper advocates a practical solution by making efficiency an evaluation criterion for research alongside accuracy and related measures. In addition, we propose reporting the financial cost or "price tag" of developing, training, and running models to provide baselines for the investigation of increasingly efficient methods. Our goal is to make AI both greener and more inclusive---enabling any inspired undergraduate with a laptop to write high-quality research papers. Green AI is an emerging focus at the Allen Institute for AI.

## Interactions with Blockchain

### Interactions with Blockchain: 2021

- Blockchain-based Federated Learning: A Comprehensive Survey. [[paper]](https://arxiv.org/abs/2110.02182)
  - Zhilin Wang, Qin Hu.
  - Key Word: Federated Learning; Blockchain.
  - Digest: We conduct a comprehensive survey of the literature on blockchained FL (BCFL). First, we investigate how blockchain can be applied to federal learning from the perspective of system composition. Then, we analyze the concrete functions of BCFL from the perspective of mechanism design and illustrate what problems blockchain addresses specifically for FL. We also survey the applications of BCFL in reality. Finally, we discuss some challenges and future research directions.

### Interactions with Blockchain: 2020

- When Federated Learning Meets Blockchain: A New Distributed Learning Paradigm. [[paper]](https://arxiv.org/abs/2009.09338)
  - Chuan Ma, Jun Li, Ming Ding, Long Shi, Taotao Wang, Zhu Han, H. Vincent Poor.
  - Key Word: Federated Leraning; Blockchain.
  - Digest: This work investigates a blockchain assisted decentralized FL (BLADE-FL) framework, which can well prevent the malicious clients from poisoning the learning process, and further provides a self-motivated and reliable learning environment for clients. In detail, the model aggregation process is fully decentralized and the tasks of training for FL and mining for blockchain are integrated into each participant.

## Others

### Others: 2018

- Towards Reverse-Engineering Black-Box Neural Networks. [[paper]](https://arxiv.org/abs/1711.01768) [[code]](https://github.com/coallaoh/WhitenBlackBox)
  - Seong Joon Oh, Max Augustin, Bernt Schiele, Mario Fritz. *ICLR 2018*
  - Key Word: Model Extraction Attacks; Membership inference Attacks.
  - Digest: This work shows that such attributes of neural networks can be exposed from a sequence of queries. This has multiple implications. On the one hand, our work exposes the vulnerability of black-box neural networks to different types of attacks -- we show that the revealed internal information helps generate more effective adversarial examples against the black box model.  
