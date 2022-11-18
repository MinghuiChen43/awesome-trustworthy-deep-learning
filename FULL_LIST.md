[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/graphs/commit-activity)
![ ](https://img.shields.io/github/last-commit/MinghuiChen43/awesome-trustworthy-deep-learning)
[![GitHub stars](https://img.shields.io/github/stars/MinghuiChen43/awesome-trustworthy-deep-learning?color=blue&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/MinghuiChen43/awesome-trustworthy-deep-learning?color=yellow&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning)
[![GitHub forks](https://img.shields.io/github/forks/MinghuiChen43/awesome-trustworthy-deep-learning?color=red&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/watchers)
[![GitHub Contributors](https://img.shields.io/github/contributors/MinghuiChen43/awesome-trustworthy-deep-learning?color=green&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/network/members)

# Awesome Trustworthy Deep Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning) ![if Useful](https://camo.githubusercontent.com/1ef04f27611ff643eb57eb87cc0f1204d7a6a14d/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d254630253946253843253946266d6573736167653d496625323055736566756c267374796c653d7374796c653d666c617426636f6c6f723d424334453939)

The deployment of deep learning in real-world systems calls for a set of complementary technologies that will ensure that deep learning is trustworthy [(Nicolas Papernot)](https://www.papernot.fr/teaching/f19-trustworthy-ml). The list covers different topics in emerging research areas including but not limited to out-of-distribution generalization, adversarial examples, backdoor attack, model inversion attack, machine unlearning, etc.

## Table of Contents

- [Awesome Trustworthy Deep Learning Paper List:page_with_curl:](#awesome-trustworthy--deep-learning)
  - [Survey](#survey)
  - [Out-of-Distribution Generalization](#out-of-distribution-generalization)
  - [Evasion Attacks and Defenses](#evasion-attacks-and-defenses)
  - [Poisoning Attacks and Defenses](#poisoning-attacks-and-defenses)
  - [Privacy](#privacy)
  - [Fairness](#fairness)
  - [Interpretability](#interpretability)
  - [Open-World Learning](#open-world-learning)
  - [Environmental Well-being](#environmental-well-being)
  - [Interactions with Blockchain](#interactions-with-blockchain)
  - [Others](#others)
- [Related Awesome Lists:astonished:](#related-awesome-lists)
- [Toolboxes:toolbox:](#toolboxes)
- [Workshops:fire:](#workshops)
- [Tutorials:woman_teacher:](#tutorials)
- [Talks:microphone:](#talks)
- [Blogs:writing_hand:](#blogs)
- [Other Resources:sparkles:](#other-resources)
- [Contributing:wink:](#contributing)

## Survey

### Survey: 2021

- A Survey on AI Assurance. [[paper]](https://arxiv.org/abs/2111.07505)
  - Feras A. Batarseh, Laura Freeman. *Journal of Big Data*
  - Key Word: Survey; Validation; Verification; Testing; Assurance.
  - <details><summary>Digest</summary> Artificial Intelligence (AI) algorithms are increasingly providing decision making and operational support across multiple domains. AI includes a wide library of algorithms for different problems. One important notion for the adoption of AI algorithms into operational decision process is the concept of assurance. The literature on assurance, unfortunately, conceals its outcomes within a tangled landscape of conflicting approaches, driven by contradicting motivations, assumptions, and intuitions. Accordingly, albeit a rising and novel area, this manuscript provides a systematic review of research works that are relevant to AI assurance, between years 1985 - 2021, and aims to provide a structured alternative to the landscape.

- Trustworthy AI: From Principles to Practices. [[paper]](https://arxiv.org/abs/2110.01167)
  - Bo Li, Peng Qi, Bo Liu, Shuai Di, Jingen Liu, Jiquan Pei, Jinfeng Yi, Bowen Zhou.
  - Key Word: Survey.
  - <details><summary>Digest</summary> In this review, we strive to provide AI practitioners a comprehensive guide towards building trustworthy AI systems. We first introduce the theoretical framework of important aspects of AI trustworthiness, including robustness, generalization, explainability, transparency, reproducibility, fairness, privacy preservation, alignment with human values, and accountability. We then survey leading approaches in these aspects in the industry. To unify the current fragmented approaches towards trustworthy AI, we propose a systematic approach that considers the entire lifecycle of AI systems, ranging from data acquisition to model development, to development and deployment, finally to continuous monitoring and governance.

- Trustworthy AI: A Computational Perspective. [[paper]](https://arxiv.org/abs/2107.06641)
  - Haochen Liu, Yiqi Wang, Wenqi Fan, Xiaorui Liu, Yaxin Li, Shaili Jain, Yunhao Liu, Anil K. Jain, Jiliang Tang.
  - Key Word: Survey.
  - <details><summary>Digest</summary> We present a comprehensive survey of trustworthy AI from a computational perspective, to help readers understand the latest technologies for achieving trustworthy AI. Trustworthy AI is a large and complex area, involving various dimensions. In this work, we focus on six of the most crucial dimensions in achieving trustworthy AI: (i) Safety & Robustness, (ii) Non-discrimination & Fairness, (iii) Explainability, (iv) Privacy, (v) Accountability & Auditability, and (vi) Environmental Well-Being.

- Causal Learning for Socially Responsible AI. [[paper]](https://arxiv.org/abs/2104.12278)
  - Lu Cheng, Ahmadreza Mosallanezhad, Paras Sheth, Huan Liu. *IJCAI 2021*
  - Key Word: Survey; Bias Mitigation; Transparency.
  - <details><summary>Digest</summary> To make AI address ethical challenges and shun undesirable outcomes, researchers proposed to develop socially responsible AI (SRAI). One of these approaches is causal learning (CL). We survey state-of-the-art methods of CL for SRAI. We begin by examining the seven CL tools to enhance the social responsibility of AI, then review how existing works have succeeded using these tools to tackle issues in developing SRAI such as fairness. The goal of this survey is to bring forefront the potentials and promises of CL for SRAI.

### Survey: 2020

- Technologies for Trustworthy Machine Learning: A Survey in a Socio-Technical Context. [[paper]](https://arxiv.org/abs/2007.08911)
  - Ehsan Toreini, Mhairi Aitken, Kovila P. L. Coopamootoo, Karen Elliott, Vladimiro Gonzalez Zelaya, Paolo Missier, Magdalene Ng, Aad van Moorsel.
  - Key Word: Survey.
  - <details><summary>Digest</summary>  In this paper we provide an overview of technologies that support building trustworthy machine learning systems, i.e., systems whose properties justify that people place trust in them. We argue that four categories of system properties are instrumental in achieving the policy objectives, namely fairness, explainability, auditability and safety & security (FEAS). We discuss how these properties need to be considered across all stages of the machine learning life cycle, from data collection through run-time model inference.

- Trust in Data Science: Collaboration, Translation, and Accountability in Corporate Data Science Projects. [[paper]](https://arxiv.org/abs/2002.03389)
  - Samir Passi, Steven J. Jackson.
  - Key Word: Survey; Data Science.
  - <details><summary>Digest</summary> The trustworthiness of data science systems in applied and real-world settings emerges from the resolution of specific tensions through situated, pragmatic, and ongoing forms of work. Drawing on research in CSCW, critical data studies, and history and sociology of science, and six months of immersive ethnographic fieldwork with a corporate data science team, we describe four common tensions in applied data science work: (un)equivocal numbers, (counter)intuitive knowledge, (in)credible data, and (in)scrutable models. We show how organizational actors establish and re-negotiate trust under messy and uncertain analytic conditions through practices of skepticism, assessment, and credibility.

- Artificial Intelligence for Social Good: A Survey. [[paper]](https://arxiv.org/abs/2001.01818)
  - Zheyuan Ryan Shi, Claire Wang, Fei Fang.
  - Key Word: Survey; Social Good.
  - <details><summary>Digest</summary> Artificial intelligence for social good (AI4SG) is a research theme that aims to use and advance artificial intelligence to address societal issues and improve the well-being of the world. AI4SG has received lots of attention from the research community in the past decade with several successful applications.

### Survey: 2019

- The relationship between trust in AI and trustworthy machine learning technologies. [[paper]](https://arxiv.org/abs/1912.00782)
  - Ehsan Toreini, Mhairi Aitken, Kovila Coopamootoo, Karen Elliott, Carlos Gonzalez Zelaya, Aad van Moorsel. *FAT 2020*
  - Key Word: Survey; Social Science.
  - <details><summary>Digest</summary> To build AI-based systems that users and the public can justifiably trust one needs to understand how machine learning technologies impact trust put in these services. To guide technology developments, this paper provides a systematic approach to relate social science concepts of trust with the technologies used in AI-based services and products. We conceive trust as discussed in the ABI (Ability, Benevolence, Integrity) framework and use a recently proposed mapping of ABI on qualities of technologies. We consider four categories of machine learning technologies, namely these for Fairness, Explainability, Auditability and Safety (FEAS) and discuss if and how these possess the required qualities.

### Survey: 2018

- A Survey of Safety and Trustworthiness of Deep Neural Networks: Verification, Testing, Adversarial Attack and Defence, and Interpretability. [[paper]](https://arxiv.org/abs/1812.08342)
  - Xiaowei Huang, Daniel Kroening, Wenjie Ruan, James Sharp, Youcheng Sun, Emese Thamo, Min Wu, Xinping Yi. *Computer Science Review*
  - Key Word: Survey.
  - <details><summary>Digest</summary> This survey paper conducts a review of the current research effort into making DNNs safe and trustworthy, by focusing on four aspects: verification, testing, adversarial attack and defence, and interpretability. In total, we survey 202 papers, most of which were published after 2017.

## Out-of-Distribution Generalization

### Out-of-Distribution Generalization: 2021

- Margin Calibration for Long-Tailed Visual Recognition. [[paper]](https://arxiv.org/abs/2112.07225)
  - Yidong Wang, Bowen Zhang, Wenxin Hou, Zhen Wu, Jindong Wang, and Takahiro Shinozaki. *ACML 2022*
  - Key Word: long-tailed recognition; imbalance learning
  - <details><summary>Digest</summary> We study the relationship between the margins and logits (classification scores) and empirically observe the biased margins and the biased logits are positively correlated. We propose MARC, a simple yet effective MARgin Calibration function to dynamically calibrate the biased margins for unbiased logits. MARC is extremely easy: just three lines of code.

- PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures. [[paper]](https://arxiv.org/abs/2112.05135) [[code]](https://github.com/andyzoujm/pixmix)
  - Dan Hendrycks, Andy Zou, Mantas Mazeika, Leonard Tang, Bo Li, Dawn Song, Jacob Steinhardt. *CVPR 2022*
  - Key Word: Corruption Robustness; Data Augmentation; Calibration; Anomaly Detection.
  - <details><summary>Digest</summary> In real-world applications of machine learning, reliable and safe systems must consider measures of performance beyond standard test set accuracy. These other goals include out-of-distribution (OOD) robustness, prediction consistency, resilience to adversaries, calibrated uncertainty estimates, and the ability to detect anomalous inputs. However, improving performance towards these goals is often a balancing act that today's methods cannot achieve without sacrificing performance on other safety axes. For instance, adversarial training improves adversarial robustness but sharply degrades other classifier performance metrics. Similarly, strong data augmentation and regularization techniques often improve OOD robustness but harm anomaly detection, raising the question of whether a Pareto improvement on all existing safety measures is possible. To meet this challenge, we design a new data augmentation strategy utilizing the natural structural complexity of pictures such as fractals, which outperforms numerous baselines, is near Pareto-optimal, and roundly improves safety measures.

- Certified Adversarial Defenses Meet Out-of-Distribution Corruptions: Benchmarking Robustness and Simple Baselines. [[paper]](https://arxiv.org/abs/2112.00659)
  - Jiachen Sun, Akshay Mehra, Bhavya Kailkhura, Pin-Yu Chen, Dan Hendrycks, Jihun Hamm, Z. Morley Mao. *ECCV 2022*
  - Key Word: Corruption Robustness; Certified Adversarial Robustness.
  - <details><summary>Digest</summary> Certified robustness guarantee gauges a model's robustness to test-time attacks and can assess the model's readiness for deployment in the real world. In this work, we critically examine how the adversarial robustness guarantees from randomized smoothing-based certification methods change when state-of-the-art certifiably robust models encounter out-of-distribution (OOD) data. Our analysis demonstrates a previously unknown vulnerability of these models to low-frequency OOD data such as weather-related corruptions, rendering these models unfit for deployment in the wild.

- Failure Modes of Domain Generalization Algorithms. [[paper]](https://arxiv.org/abs/2111.13733) [[code]](https://github.com/YerevaNN/dom-gen-failure-modes)  
  - Tigran Galstyan, Hrayr Harutyunyan, Hrant Khachatrian, Greg Ver Steeg, Aram Galstyan. *CVPR 2022*
  - Key Word: Domain Generalization; Invariant Learning.
  - <details><summary>Digest</summary> We propose an evaluation framework for domain generalization algorithms that allows decomposition of the error into components capturing distinct aspects of generalization. Inspired by the prevalence of algorithms based on the idea of domain-invariant representation learning, we extend the evaluation framework to capture various types of failures in achieving invariance. We show that the largest contributor to the generalization error varies across methods, datasets, regularization strengths and even training lengths.

- Why Stable Learning Works? A Theory of Covariate Shift Generalization. [[paper]](https://arxiv.org/abs/2111.02355)
  - Renzhe Xu, Peng Cui, Zheyan Shen, Xingxuan Zhang, Tong Zhang.
  - Key Word: Stable Learning.
  - <details><summary>Digest</summary> Covariate shift generalization, a typical case in out-of-distribution (OOD) generalization, requires a good performance on the unknown testing distribution, which varies from the accessible training distribution in the form of covariate shift. Recently, stable learning algorithms have shown empirical effectiveness to deal with covariate shift generalization on several learning models involving regression algorithms and deep neural networks. However, the theoretical explanations for such effectiveness are still missing. In this paper, we take a step further towards the theoretical analysis of stable learning algorithms by explaining them as feature selection processes.

- Benchmarks for Corruption Invariant Person Re-identification. [[paper]](https://arxiv.org/abs/2111.00880) [[code]](https://github.com/MinghuiChen43/CIL-ReID)
  - Minghui Chen, Zhiqiang Wang, Feng Zheng. *NeurIPS 2021*
  - Key Word: Corruption Robustness; Benchmark; Person Re-Identificaiton.
  - <details><summary>Digest</summary> We comprehensively establish six ReID benchmarks for learning corruption invariant representation. In the field of ReID, we are the first to conduct an exhaustive study on corruption invariant learning in single- and cross-modality datasets, including Market-1501, CUHK03, MSMT17, RegDB, SYSU-MM01. After reproducing and examining the robustness performance of 21 recent ReID methods, we have some observations: 1) transformer-based models are more robust towards corrupted images, compared with CNN-based models, 2) increasing the probability of random erasing (a commonly used augmentation method) hurts model corruption robustness, 3) cross-dataset generalization improves with corruption robustness increases. By analyzing the above observations, we propose a strong baseline on both single- and cross-modality ReID datasets which achieves improved robustness against diverse corruptions.

- Kernelized Heterogeneous Risk Minimization. [[paper]](https://arxiv.org/abs/2110.12425) [[code]](https://github.com/LJSthu/Kernelized-HRM)
  - Jiashuo Liu, Zheyuan Hu, Peng Cui, Bo Li, Zheyan Shen. *NeurIPS 2021*
  - Key Word: Invariant Learning; Neural Tangent Kernel.
  - <details><summary>Digest</summary> We propose Kernelized Heterogeneous Risk Minimization (KerHRM) algorithm, which achieves both the latent heterogeneity exploration and invariant learning in kernel space, and then gives feedback to the original neural network by appointing invariant gradient direction. We theoretically justify our algorithm and empirically validate the effectiveness of our algorithm with extensive experiments.

- A Fine-Grained Analysis on Distribution Shift. [[paper]](https://arxiv.org/abs/2110.11328)  
  - Olivia Wiles, Sven Gowal, Florian Stimberg, Sylvestre Alvise-Rebuffi, Ira Ktena, Krishnamurthy Dvijotham, Taylan Cemgil. *ICLR 2022*
  - Key Word: Distribution Shifts; Data Augmentation.
  - <details><summary>Digest</summary> Robustness to distribution shifts is critical for deploying machine learning models in the real world. Despite this necessity, there has been little work in defining the underlying mechanisms that cause these shifts and evaluating the robustness of algorithms across multiple, different distribution shifts. To this end, we introduce a framework that enables fine-grained analysis of various distribution shifts. We provide a holistic analysis of current state-of-the-art methods by evaluating 19 distinct methods grouped into five categories across both synthetic and real-world datasets. Overall, we train more than 85K models. Our experimental framework can be easily extended to include new methods, shifts, and datasets. We find, unlike previous work, that progress has been made over a standard ERM baseline; in particular, pretraining and augmentations (learned or heuristic) offer large gains in many cases. However, the best methods are not consistent over different datasets and shifts.

- Benchmarking the Robustness of Spatial-Temporal Models Against Corruptions. [[paper]](https://arxiv.org/abs/2110.06513) [[code]](https://github.com/newbeeyoung/video-corruption-robustness)
  - Chenyu Yi, SIYUAN YANG, Haoliang Li, Yap-peng Tan, Alex Kot. *NeurIPS 2021*
  - Key Word: Video Understanding; Corruption Robustness.
  - <details><summary>Digest</summary> We establish a corruption robustness benchmark, Mini Kinetics-C and Mini SSV2-C, which considers temporal corruptions beyond spatial corruptions in images. We make the first attempt to conduct an exhaustive study on the corruption robustness of established CNN-based and Transformer-based spatial-temporal models.

- Towards Out-Of-Distribution Generalization: A Survey. [[paper]](https://arxiv.org/abs/2108.13624)
  - Zheyan Shen, Jiashuo Liu, Yue He, Xingxuan Zhang, Renzhe Xu, Han Yu, Peng Cui.
  - Key Word: Survey; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> Out-of-Distribution (OOD) generalization problem addresses the challenging setting where the testing distribution is unknown and different from the training. This paper serves as the first effort to systematically and comprehensively discuss the OOD generalization problem, from the definition, methodology, evaluation to the implications and future directions.

- Learning to Diversify for Single Domain Generalization. [[paper]](https://arxiv.org/abs/2108.11726) [[code]](https://github.com/busername/learning_to_diversify)
  - Zijian Wang, Yadan Luo, Ruihong Qiu, Zi Huang, Mahsa Baktashmotlagh. *ICCV 2021*
  - Key Word: Corruption Robustness; Single Domain Generalization.
  - <details><summary>Digest</summary> To tackle this problem, we propose a style-complement module to enhance the generalization power of the model by synthesizing images from diverse distributions that are complementary to the source ones. More specifically, we adopt a tractable upper bound of mutual information (MI) between the generated and source samples and perform a two-step optimization iteratively.

- StyleAugment: Learning Texture De-biased Representations by Style Augmentation without Pre-defined Textures. [[paper]](https://arxiv.org/abs/2108.10549)
  - Sanghyuk Chun, Song Park.
  - Key Word: Corruption Robustness; Shape-Texture; Style Transfer; Data Augmentation.
  - <details><summary>Digest</summary> Stylized ImageNet approach has two drawbacks in fidelity and diversity. First, the generated images show low image quality due to the significant semantic gap betweeen natural images and artistic paintings. Also, Stylized ImageNet training samples are pre-computed before training, resulting in showing the lack of diversity for each sample. We propose a StyleAugment by augmenting styles from the mini-batch.

- Impact of Aliasing on Generalization in Deep Convolutional Networks. [[paper]](https://arxiv.org/abs/2108.03489)
  - Cristina Vasconcelos, Hugo Larochelle, Vincent Dumoulin, Rob Romijnders, Nicolas Le Roux, Ross Goroshin. *ICCV 2021*
  - Key Word: Corruption Robustness; Anti-Aliasing.
  - <details><summary>Digest</summary> Drawing insights from frequency analysis theory, we take a closer look at ResNet and EfficientNet architectures and review the trade-off between aliasing and information loss in each of their major components. We show how to mitigate aliasing by inserting non-trainable low-pass filters at key locations, particularly where networks lack the capacity to learn them.

- Using Synthetic Corruptions to Measure Robustness to Natural Distribution Shifts. [[paper]](https://arxiv.org/abs/2107.12052)
  - Alfred Laugros, Alice Caplier, Matthieu Ospici.
  - Key Word: Corruption Robustness; Benchmark.
  - <details><summary>Digest</summary> We propose a methodology to build synthetic corruption benchmarks that make robustness estimations more correlated with robustness to real-world distribution shifts. Using the overlapping criterion, we split synthetic corruptions into categories that help to better understand neural network robustness.

- Just Train Twice: Improving Group Robustness without Training Group Information. [[paper]](https://arxiv.org/abs/2107.09044) [[code]](https://github.com/anniesch/jtt)
  - Evan Zheran Liu, Behzad Haghgoo, Annie S. Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, Chelsea Finn. *ICML 2021*
  - Key Word: Invariant Learning; Robust Optimization; Fairness.
  - <details><summary>Digest</summary> We propose a simple two-stage approach, JTT, that first trains a standard ERM model for several epochs, and then trains a second model that upweights the training examples that the first model misclassified. Intuitively, this upweights examples from groups on which standard ERM models perform poorly, leading to improved worst-group performance.

- Visual Representation Learning Does Not Generalize Strongly Within the Same Domain. [[paper]](https://arxiv.org/abs/2107.08221) [[code]](https://github.com/bethgelab/InDomainGeneralizationBenchmark)
  - Lukas Schott, Julius von Kügelgen, Frederik Träuble, Peter Gehler, Chris Russell, Matthias Bethge, Bernhard Schölkopf, Francesco Locatello, Wieland Brendel. *ICLR 2022*
  - Key Word: Out-of-Distribution Generalization; Disentanglement; Benchmark.
  - <details><summary>Digest</summary> In contrast to prior robustness work that introduces novel factors of variation during test time, such as blur or other (un)structured noise, we here recompose, interpolate, or extrapolate only existing factors of variation from the training data set (e.g., small and medium-sized objects during training and large objects during testing). Models that learn the correct mechanism should be able to generalize to this benchmark. In total, we train and test 2000+ models and observe that all of them struggle to learn the underlying mechanism regardless of supervision signal and architectural bias. Moreover, the generalization capabilities of all tested models drop significantly as we move from artificial datasets towards more realistic real-world datasets.

- Global Filter Networks for Image Classification. [[paper]](https://arxiv.org/abs/2107.00645) [[code]](https://github.com/raoyongming/GFNet)
  - Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, Jie Zhou.
  - Key Word: Corruption Robustness; Frequency.
  - <details><summary>Digest</summary> We present the Global Filter Network (GFNet), a conceptually simple yet computationally efficient architecture, that learns long-term spatial dependencies in the frequency domain with log-linear complexity.

- The Evolution of Out-of-Distribution Robustness Throughout Fine-Tuning. [[paper]](https://arxiv.org/abs/2106.15831)
  - Anders Andreassen, Yasaman Bahri, Behnam Neyshabur, Rebecca Roelofs.
  - Key Word: Out-of-Distribution Generalization; Fine-Tuning.
  - <details><summary>Digest</summary> Although machine learning models typically experience a drop in performance on out-of-distribution data, accuracies on in- versus out-of-distribution data are widely observed to follow a single linear trend when evaluated across a testbed of models. Models that are more accurate on the out-of-distribution data relative to this baseline exhibit "effective robustness" and are exceedingly rare. Identifying such models, and understanding their properties, is key to improving out-of-distribution performance. We conduct a thorough empirical investigation of effective robustness during fine-tuning and surprisingly find that models pre-trained on larger datasets exhibit effective robustness during training that vanishes at convergence. We study how properties of the data influence effective robustness, and we show that it increases with the larger size, more diversity, and higher example difficulty of the dataset. 

- Test-Time Adaptation to Distribution Shift by Confidence Maximization and Input Transformation. [[paper]](https://arxiv.org/abs/2106.14999)
  - Chaithanya Kumar Mummadi, Robin Hutmacher, Kilian Rambach, Evgeny Levinkov, Thomas Brox, Jan Hendrik Metzen.
  - Key Word: Corruption Robustness; Test-Time Adaptation.
  - <details><summary>Digest</summary> We propose non-saturating losses based on the negative log likelihood ratio, such that gradients from high confidence predictions still contribute to test-time adaptation.

- Exploring Corruption Robustness: Inductive Biases in Vision Transformers and MLP-Mixers. [[paper]](https://arxiv.org/abs/2106.13122) [[code]](https://github.com/katelyn98/CorruptionRobustness)
  - Katelyn Morrison, Benjamin Gilby, Colton Lipchak, Adam Mattioli, Adriana Kovashka.
  - Key Word: Corruption Robustness; Transformers.
  - <details><summary>Digest</summary> We find that vision transformer architectures are inherently more robust to corruptions than the ResNet and MLP-Mixers.

- Dangers of Bayesian Model Averaging under Covariate Shift. [[paper]](https://arxiv.org/abs/2106.11905)
  - Pavel Izmailov, Patrick Nicholson, Sanae Lotfi, Andrew Gordon Wilson.
  - Key Word: Corruption Robustness; Bayesian Neural Networks.
  - <details><summary>Digest</summary> Bayesian neural networks (BNNs) with high-fidelity approximate inference via full-batch Hamiltonian Monte Carlo achieve poor generalization under covariate shift, even underperforming classical estimation. We explain this surprising result, showing how a Bayesian model average can in fact be problematic under covariate shift, particularly in cases where linear dependencies in the input features cause a lack of posterior contraction.

- A Winning Hand: Compressing Deep Networks Can Improve Out-Of-Distribution Robustness. [[paper]](https://arxiv.org/abs/2106.09129)
  - James Diffenderfer, Brian R. Bartoldson, Shreya Chaganti, Jize Zhang, Bhavya Kailkhura.
  - Key Word: Corruption Robustness; Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> We present the first positive result on simultaneously achieving high accuracy and OoD robustness at extreme levels of model compression.

- Delving Deep into the Generalization of Vision Transformers under Distribution Shifts. [[paper]](https://arxiv.org/abs/2106.07617) [[code]](https://github.com/Phoenix1153/ViT_OOD_generalization)
  - Chongzhi Zhang, Mingyuan Zhang, Shanghang Zhang, Daisheng Jin, Qiang Zhou, Zhongang Cai, Haiyu Zhao, Shuai Yi, Xianglong Liu, Ziwei Liu.
  - Key Word: Corruption Robustness; Transformers.
  - <details><summary>Digest</summary> We first present a taxonomy of distribution shifts by categorizing them into five conceptual groups: corruption shift, background shift, texture shift, destruction shift, and style shift. Then we perform extensive evaluations of ViT variants under different groups of distribution shifts and compare their generalization ability with Convolutional Neural Network (CNN) models.

- RobustNav: Towards Benchmarking Robustness in Embodied Navigation. [[paper]](https://arxiv.org/abs/2106.04531) [[code]](https://github.com/allenai/robustnav)
  - Prithvijit Chattopadhyay, Judy Hoffman, Roozbeh Mottaghi, Aniruddha Kembhavi. *ICCV 2021*
  - Key Word: Corruption Robustness; Embodied Navigation.
  - <details><summary>Digest</summary> As an attempt towards assessing the robustness of embodied navigation agents, we propose RobustNav, a framework to quantify the performance of embodied navigation agents when exposed to a wide variety of visual - affecting RGB inputs - and dynamics - affecting transition dynamics - corruptions. Most recent efforts in visual navigation have typically focused on generalizing to novel target environments with similar appearance and dynamics characteristics. With RobustNav, we find that some standard embodied navigation agents significantly underperform (or fail) in the presence of visual or dynamics corruptions.

- Towards a Theoretical Framework of Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2106.04496)
  - Haotian Ye, Chuanlong Xie, Tianle Cai, Ruichen Li, Zhenguo Li, Liwei Wang.
  - Key Word: Theoretical Framework; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> We take the first step towards rigorous and quantitative definitions of 1) what is OOD; and 2) what does it mean by saying an OOD problem is learnable. We also introduce a new concept of expansion function, which characterizes to what extent the variance is amplified in the test domains over the training domains, and therefore give a quantitative meaning of invariant features.

- An Information-theoretic Approach to Distribution Shifts. [[paper]](https://arxiv.org/abs/2106.03783) [[code]](https://github.com/mfederici/dsit)
  - Marco Federici, Ryota Tomioka, Patrick Forré. *NeurIPS 2021*
  - Key Word: Information Theory; Distribution Shift.
  - <details><summary>Digest</summary> We describe the problem of data shift from a novel information-theoretic perspective by (i) identifying and describing the different sources of error, (ii) comparing some of the most promising objectives explored in the recent domain generalization, and fair classification literature.

- OoD-Bench: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2106.03721) [[code]](https://github.com/ynysjtu/ood_bench)
  - Nanyang Ye, Kaican Li, Haoyue Bai, Runpeng Yu, Lanqing Hong, Fengwei Zhou, Zhenguo Li, Jun Zhu. *CVPR 2022*
  - Key Word: Out-of-Distribution Generalization; Benchmark.
  - <details><summary>Digest</summary> We first identify and measure two distinct kinds of distribution shifts that are ubiquitous in various datasets. Next, through extensive experiments, we compare OoD generalization algorithms across two groups of benchmarks, each dominated by one of the distribution shifts, revealing their strengths on one shift as well as limitations on the other shift.

- Predict then Interpolate: A Simple Algorithm to Learn Stable Classifiers. [[paper]](https://arxiv.org/abs/2105.12628) [[code]](https://github.com/YujiaBao/Predict-then-Interpolate)
  - Yujia Bao, Shiyu Chang, Regina Barzilay. *ICML 2022*
  - Key Word: Invariant Learning; Fairness.
  - <details><summary>Digest</summary> We prove that by interpolating the distributions of the correct predictions and the wrong predictions, we can uncover an oracle distribution where the unstable correlation vanishes. Since the oracle interpolation coefficients are not accessible, we use group distributionally robust optimization to minimize the worst-case risk across all such interpolations.

- Using the Overlapping Score to Improve Corruption Benchmarks. [[paper]](https://arxiv.org/abs/2105.12357)
  - Alfred Laugros, Alice Caplier, Matthieu Ospici.
  - Key Word: Corruption Robustness; Benchmark.
  - <details><summary>Digest</summary> We propose a metric called corruption overlapping score, which can be used to reveal flaws in corruption benchmarks. Two corruptions overlap when the robustnesses of neural networks to these corruptions are correlated.

- Improved OOD Generalization via Adversarial Training and Pre-training. [[paper]](https://arxiv.org/abs/2105.11144)
  - Mingyang Yi, Lu Hou, Jiacheng Sun, Lifeng Shang, Xin Jiang, Qun Liu, Zhi-Ming Ma. *ICML 2021*
  - Key Word: Corruption Robustness; Adversarial Training; Pre-Trainig.
  - <details><summary>Digest</summary> In this paper, after defining OOD generalization via Wasserstein distance, we theoretically show that a model robust to input perturbation generalizes well on OOD data. Inspired by previous findings that adversarial training helps improve input-robustness, we theoretically show that adversarially trained models have converged excess risk on OOD data, and empirically verify it on both image classification and natural language understanding tasks.

- Balancing Robustness and Sensitivity using Feature Contrastive Learning. [[paper]](https://arxiv.org/abs/2105.09394)
  - Seungyeon Kim, Daniel Glasner, Srikumar Ramalingam, Cho-Jui Hsieh, Kishore Papineni, Sanjiv Kumar.
  - Key Word: Corruption Robustness.
  - <details><summary>Digest</summary> We discuss this trade-off between sensitivity and robustness to natural (non-adversarial) perturbations by introducing two notions: contextual feature utility and contextual feature sensitivity. We propose Feature Contrastive Learning (FCL) that encourages a model to be more sensitive to the features that have higher contextual utility.

- Towards Robust Vision Transformer. [[paper]](https://arxiv.org/abs/2105.07926) [[code]](https://github.com/vtddggg/Robust-Vision-Transformer)
  - Xiaofeng Mao, Gege Qi, Yuefeng Chen, Xiaodan Li, Ranjie Duan, Shaokai Ye, Yuan He, Hui Xue.
  - Key Word: Corruption Robustness; Transformers.
  - <details><summary>Digest</summary> Wwe propose a Robust Vision Transformer (RVT), by using and combining robust components as building blocks.To further improve the RVT, we propose two new plug-and-play techniques called position-aware attention scaling and patch-wise augmentation.

- Vision Transformers are Robust Learners. [[paper]](https://arxiv.org/abs/2105.07581) [[code]](https://github.com/sayakpaul/robustness-vit)
  - Sayak Paul, Pin-Yu Chen.
  - Key Word: Corruption Robustness; Transformers.
  - <details><summary>Digest</summary> We study the robustness of the Vision Transformer (ViT) against common corruptions and perturbations, distribution shifts, and natural adversarial examples.

- When Human Pose Estimation Meets Robustness: Adversarial Algorithms and Benchmarks. [[paper]](https://arxiv.org/abs/2105.06152) [[code]](https://github.com/AIprogrammer/AdvMix)
  - Jiahang Wang, Sheng Jin, Wentao Liu, Weizhong Liu, Chen Qian, Ping Luo. *CVPR 2021*
  - Key Word: Corruption Robustness; Benchmark; Data Augmentation; Pose Estimation.
  - <details><summary>Digest</summary> This work comprehensively studies and addresses this problem by building rigorous robust benchmarks, termed COCO-C, MPII-C, and OCHuman-C, to evaluate the weaknesses of current advanced pose estimators, and a new algorithm termed AdvMix is proposed to improve their robustness in different corruptions.

- FL-NTK: A Neural Tangent Kernel-based Framework for Federated Learning Convergence Analysis. [[paper]](https://arxiv.org/abs/2105.05001)
  - Baihe Huang, Xiaoxiao Li, Zhao Song, Xin Yang. *ICML 2021*
  - Keyword: Federated Learning; Neural Tangent Kernel.
  - <details><summary>Digest</summary> This paper presents a new class of convergence analysis for FL, Federated Learning Neural Tangent Kernel (FL-NTK), which corresponds to overparamterized ReLU neural networks trained by gradient descent in FL and is inspired by the analysis in Neural Tangent Kernel (NTK). Theoretically, FL-NTK converges to a global-optimal solution at a linear rate with properly tuned learning parameters.

- Heterogeneous Risk Minimization. [[paper]](https://arxiv.org/abs/2105.03818)
  - Jiashuo Liu, Zheyuan Hu, Peng Cui, Bo Li, Zheyan Shen. *ICML 2021*
  - Key Word: Invariant Learning; Causality; Robust Optimization.
  - <details><summary>Digest</summary> We propose Heterogeneous Risk Minimization (HRM) framework to achieve joint learning of latent heterogeneity among the data and invariant relationship, which leads to stable prediction despite distributional shifts. We theoretically characterize the roles of the environment labels in invariant learning and justify our newly proposed HRM framework.

- What Are Bayesian Neural Network Posteriors Really Like? [[paper]](https://arxiv.org/abs/2104.14421) [[code]](https://github.com/google-research/google-research/tree/master/bnn_hmc)
  - Pavel Izmailov, Sharad Vikram, Matthew D. Hoffman, Andrew Gordon Wilson.
  - Key Word: Corruption Robustness; Bayesian Neural Networks.
  - <details><summary>Digest</summary> Bayesian neural networks show surprisingly poor generalization under domain shift; while cheaper alternatives such as deep ensembles and SGMCMC methods can provide good generalization, they provide distinct predictive distributions from HMC.

- Adapting ImageNet-scale models to complex distribution shifts with self-learning. [[paper]](https://arxiv.org/abs/2104.12928) [[code]](https://domainadaptation.org/selflearning/)
  - Evgenia Rusak, Steffen Schneider, Peter Gehler, Oliver Bringmann, Wieland Brendel, Matthias Bethge.
  - Key Word: Corruption Robustness; Domain Adaptation.
  - <details><summary>Digest</summary> We find that three components are crucial for increasing performance with self-learning: (i) using short update times between the teacher and the student network, (ii) fine-tuning only few affine parameters distributed across the network, and (iii) leveraging methods from robust classification to counteract the effect of label noise. We therefore re-purpose the dataset from the Visual Domain Adaptation Challenge 2019 and use a subset of it as a new robustness benchmark (ImageNet-D) which proves to be a more challenging dataset.

- Towards Corruption-Agnostic Robust Domain Adaptation. [[paper]](https://arxiv.org/abs/2104.10376) [[code]](https://github.com/Mike9674/CRDA)
  - Yifan Xu, Kekai Sheng, Weiming Dong, Baoyuan Wu, Changsheng Xu, Bao-Gang Hu.
  - Key Word: Corruption Robustness; Domain Adaptation.
  - <details><summary>Digest</summary> We investigate a new scenario called corruption-agnostic robust domain adaptation (CRDA) to equip domain adaptation models with corruption robustness. We take use of information of domain discrepancy to propose a novel module Domain Discrepancy Generator (DDG) for corruption robustness that mimic unpredictable corruptions.

- Gradient Matching for Domain Generalization. [[paper]](https://arxiv.org/abs/2104.09937) [[code]](https://github.com/YugeTen/fish)
  - Yuge Shi, Jeffrey Seely, Philip H.S. Torr, N. Siddharth, Awni Hannun, Nicolas Usunier, Gabriel Synnaeve. *ICLR 2022*
  - Key Word: Domain Generalization, Multi-Source Domain Adaptation.
  - <details><summary>Digest</summary> Here, we propose an inter-domain gradient matching objective that targets domain generalization by maximizing the inner product between gradients from different domains. Since direct optimization of the gradient inner product can be computationally prohibitive -- it requires computation of second-order derivatives -- we derive a simpler first-order algorithm named Fish that approximates its optimization.

- Does enhanced shape bias improve neural network robustness to common corruptions? [[paper]](https://arxiv.org/abs/2104.09789)
  - Chaithanya Kumar Mummadi, Ranjitha Subramaniam, Robin Hutmacher, Julien Vitay, Volker Fischer, Jan Hendrik Metzen. *ICLR 2021*
  - Key Word: Corruption Robustness; Shape-Texture.
  - <details><summary>Digest</summary> While pre-training on stylized images increases both shape bias and corruption robustness, these two quantities are not necessarily correlated: pre-training on edge maps increases the shape bias without consistently helping in terms of corruption robustness.

- Deep Stable Learning for Out-Of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2104.07876) [[code]](https://github.com/xxgege/StableNet)
  - Xingxuan Zhang, Peng Cui, Renzhe Xu, Linjun Zhou, Yue He, Zheyan Shen. *CVPR 2021*
  - Key Word: Stable Learning; Feature Decorrelation.
  - <details><summary>Digest</summary> Conventional methods assume either the known heterogeneity of training data (e.g. domain labels) or the approximately equal capacities of different domains. In this paper, we consider a more challenging case where neither of the above assumptions holds. We propose to address this problem by removing the dependencies between features via learning weights for training samples, which helps deep models get rid of spurious correlations and, in turn, concentrate more on the true connection between discriminative features and labels.

- Misclassification-Aware Gaussian Smoothing improves Robustness against Domain Shifts. [[paper]](https://arxiv.org/abs/2104.01231)
  - Athanasios Tsiligkaridis, Theodoros Tsiligkaridis.
  - Key Word: Corruption Robustness; Regularization.
  - <details><summary>Digest</summary> We introduce a misclassification-aware consistency loss coupled with Gaussian noise regularization and a corresponding training algorithm. Additionally, we present some theoretical properties of our new regularization approach that reveals its connection to local loss geometry.

- Defending Against Image Corruptions Through Adversarial Augmentations. [[paper]](https://arxiv.org/abs/2104.01086)
  - Dan A. Calian, Florian Stimberg, Olivia Wiles, Sylvestre-Alvise Rebuffi, Andras Gyorgy, Timothy Mann, Sven Gowal.
  - Key Word: Corruption Robustness; Data Augmentation.
  - <details><summary>Digest</summary> We propose AdversarialAugment, a technique which optimizes the parameters of image-to-image models to generate adversarially corrupted augmented images.

- Improving robustness against common corruptions with frequency biased models. [[paper]](https://arxiv.org/abs/2103.16241)
  - Tonmoy Saikia, Cordelia Schmid, Thomas Brox.
  - Key Word: Corruption Robustness; Frequency.
  - <details><summary>Digest</summary> We propose a new regularization scheme that enforces convolutional feature maps to have a low total variation. We introduce the idea of mixing two experts that specialize in high-frequency and low-frequency robustness.

- Improving Model Robustness by Adaptively Correcting Perturbation Levels with Active Queries. [[paper]](https://arxiv.org/abs/2103.14824)
  - Kun-Peng Ning, Lue Tao, Songcan Chen, Sheng-Jun Huang. *AAAI 2021*
  - Key Word: Corruption Robustness; Active Learning.
  - <details><summary>Digest</summary> We propose to adaptively adjust the perturbation levels for each example in the training process. Specifically, a novel active learning framework is proposed to allow the model to interactively query the correct perturbation level from human experts.

- Understanding Robustness of Transformers for Image Classification. [[paper]](https://arxiv.org/abs/2103.14586)
  - Srinadh Bhojanapalli, Ayan Chakrabarti, Daniel Glasner, Daliang Li, Thomas Unterthiner, Andreas Veit.
  - Key Word: Corruption Robustness; Transformers.
  - <details><summary>Digest</summary> When the training set is small, the ViTs are less robust compared to ResNets of comparable sizes, and increasing the size of the ViTs does not lead to better robustness.

- StyleLess layer: Improving robustness for real-world driving. [[paper]](https://arxiv.org/abs/2103.13905)
  - Julien Rebut, Andrei Bursuc, Patrick Pérez.
  - Key Word: Corruption Robustness; Style Information.
  - <details><summary>Digest</summary> We design a new layer, StyleLess, that enables the model to ignore noisy style information and learn from the content instead.

- Robust and Accurate Object Detection via Adversarial Learning. [[paper]](https://arxiv.org/abs/2103.13886) [[code]](https://github.com/google/automl/blob/master/efficientdet/Det-AdvProp.md)
  - Xiangning Chen, Cihang Xie, Mingxing Tan, Li Zhang, Cho-Jui Hsieh, Boqing Gong. *CVPR 2021*
  - Key Word: Corruption Robustness; Detection.
  - <details><summary>Digest</summary> Our method dynamically selects the stronger adversarial images sourced from a detector’s classification and localization branches and evolves with the detector to ensure the augmentation policy stays current and relevant.

- Generalizing to Unseen Domains: A Survey on Domain Generalization. [[paper]](https://arxiv.org/abs/2103.03097) [[code]](https://github.com/jindongwang/transferlearning)
  - Jindong Wang, Cuiling Lan, Chang Liu, Yidong Ouyang, Tao Qin, Wang Lu, Yiqiang Chen, Wenjun Zeng, Philip S. Yu. *IJCAI 2021*
  - Key Word: Survey; Domain Generalization.
  - <details><summary>Digest</summary> This paper presents the first review of recent advances in this area. First, we provide a formal definition of domain generalization and discuss several related fields. We then thoroughly review the theories related to domain generalization and carefully analyze the theory behind generalization. We categorize recent algorithms into three classes: data manipulation, representation learning, and learning strategy, and present several popular algorithms in detail for each category. Third, we introduce the commonly used datasets, applications, and our open-sourced codebase for fair evaluation. Finally, we summarize existing literature and present some potential research topics for the future.

- Lost in Pruning: The Effects of Pruning Neural Networks beyond Test Accuracy. [[paper]](https://arxiv.org/abs/2103.03014) [[code]](https://github.com/lucaslie/torchprune)
  - Lucas Liebenwein, Cenk Baykal, Brandon Carter, David Gifford, Daniela Rus. *MLSys 2021*
  - Key Word: Corruption Robustness; Pruning.
  - <details><summary>Digest</summary> Here, we reassess and evaluate whether the use of test accuracy alone in the terminating condition is sufficient to ensure that the resulting model performs well across a wide spectrum of "harder" metrics such as generalization to out-of-distribution data and resilience to noise.

- Domain Generalization: A Survey. [[paper]](https://arxiv.org/abs/2103.02503) [[code]](https://github.com/KaiyangZhou/Dassl.pytorch)
  - Kaiyang Zhou, Ziwei Liu, Yu Qiao, Tao Xiang, Chen Change Loy. *TPAMI 2022*
  - Key Word: Survey; Domain Generalization.
  - <details><summary>Digest</summary> In this paper, for the first time a comprehensive literature review in DG is provided to summarize the developments over the past decade. Specifically, we first cover the background by formally defining DG and relating it to other relevant fields like domain adaptation and transfer learning. Then, we conduct a thorough review into existing methods and theories. Finally, we conclude this survey with insights and discussions on future research directions.

- Regularizing towards Causal Invariance: Linear Models with Proxies. [[paper]](https://arxiv.org/abs/2103.02477) [[code]](https://github.com/clinicalml/proxy-anchor-regression)
  - Michael Oberst, Nikolaj Thams, Jonas Peters, David Sontag. *ICML 2021*
  - Key Word: Causal Invariance; Distribution Shift.
  - <details><summary>Digest</summary> We propose a method for learning linear models whose predictive performance is robust to causal interventions on unobserved variables, when noisy proxies of those variables are available. Our approach takes the form of a regularization term that trades off between in-distribution performance and robustness to interventions. Under the assumption of a linear structural causal model, we show that a single proxy can be used to create estimators that are prediction optimal under interventions of bounded strength.

- On the effectiveness of adversarial training against common corruptions. [[paper]](https://arxiv.org/abs/2103.02325v1) [[code]](https://github.com/tml-epfl/adv-training-corruptions)
  - Klim Kireev, Maksym Andriushchenko, Nicolas Flammarion.
  - Key Word: Corruption Robustness; Adversarial Robustness.
  - <details><summary>Digest</summary> Adversarial training with proper perturbation radius serves as a strong baseline against corruption robustness. Further, adversarial training with LPIPS still works and is significantly faster.

- Nonlinear Invariant Risk Minimization: A Causal Approach. [[paper]](https://arxiv.org/abs/2102.12353)
  - Chaochao Lu, Yuhuai Wu, Jośe Miguel Hernández-Lobato, Bernhard Schölkopf.
  - Key Word: Invariant Learning; Causality.
  - <details><summary>Digest</summary> We propose invariant Causal Representation Learning (iCaRL), an approach that enables out-of-distribution (OOD) generalization in the nonlinear setting (i.e., nonlinear representations and nonlinear classifiers). It builds upon a practical and general assumption: the prior over the data representation (i.e., a set of latent variables encoding the data) given the target and the environment belongs to general exponential family distributions.

- Explainers in the Wild: Making Surrogate Explainers Robust to Distortions through Perception. [[paper]](https://arxiv.org/abs/2102.10951v1)
  - Alexander Hepburn, Raul Santos-Rodriguez.
  - Key Word: Corruption Robustness.
  - <details><summary>Digest</summary> We propose a methodology to evaluate the effect of distortions in explanations by embedding perceptual distances that tailor the neighbourhoods used to training surrogate explainers. We also show that by operating in this way, we can make the explanations more robust to distortions.

- Linear unit-tests for invariance discovery. [[paper]](https://arxiv.org/abs/2102.10867) [[code]](https://github.com/facebookresearch/InvarianceUnitTests)
  - Benjamin Aubin, Agnieszka Słowik, Martin Arjovsky, Leon Bottou, David Lopez-Paz.
  - Key Word: Invariant Learning; Causality; Empirical Study.
  - <details><summary>Digest</summary> There is an increasing interest in algorithms to learn invariant correlations across training environments. A big share of the current proposals find theoretical support in the causality literature but, how useful are they in practice? The purpose of this note is to propose six linear low-dimensional problems -- unit tests -- to evaluate different types of out-of-distribution generalization in a precise manner. Following initial experiments, none of the three recently proposed alternatives passes all tests.

- SWAD: Domain Generalization by Seeking Flat Minima. [[paper]](https://arxiv.org/abs/2102.08604) [[code]](https://github.com/khanrc/swad)
  - Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, Sungrae Park. *NeurIPS 2021*
  - Key Word: Domain Generalization; Weight Averaging; Flat Minima.
  - <details><summary>Digest</summary> We theoretically show that finding flat minima results in a smaller domain generalization gap. We also propose a simple yet effective method, named Stochastic Weight Averaging Densely (SWAD), to find flat minima. SWAD finds flatter minima and suffers less from overfitting than does the vanilla SWA by a dense and overfit-aware stochastic weight sampling strategy. SWAD shows state-of-the-art performances on five DG benchmarks, namely PACS, VLCS, OfficeHome, TerraIncognita, and DomainNet, with consistent and large margins of +1.6% averagely on out-of-domain accuracy. 

- FedBN: Federated Learning on Non-IID Features via Local Batch Normalization. [[paper]](https://arxiv.org/abs/2102.07623) [[code]](https://github.com/med-air/FedBN)
  - Xiaoxiao Li, Meirui Jiang, Xiaofei Zhang, Michael Kamp, Qi Dou. *ICLR 2021*
  - Key Word: Federated Learning; Normalization.
  - <details><summary>Digest</summary> We propose an effective method that uses local batch normalization to alleviate the feature shift before averaging models. The resulting scheme, called FedBN, outperforms both classical FedAvg, as well as the state-of-the-art for non-iid data (FedProx) on our extensive experiments. These empirical results are supported by a convergence analysis that shows in a simplified setting that FedBN has a faster convergence rate than FedAvg.

- FLOP: Federated Learning on Medical Datasets using Partial Networks. [[paper]](https://arxiv.org/abs/2102.05218)
  - Qian Yang, Jianyi Zhang, Weituo Hao, Gregory Spell, Lawrence Carin. *KDD 2021*
  - Key Word: Federated Learning; Disease Diagnosis.
  - <details><summary>Digest</summary> We investigate this challenging problem by proposing a simple yet effective algorithm, named Federated Learning on Medical Datasets using Partial Networks (FLOP), that shares only a partial model between the server and clients.

- SelfNorm and CrossNorm for Out-of-Distribution Robustness. [[paper]](https://arxiv.org/abs/2102.02811) [[code]](https://github.com/amazon-research/crossnorm-selfnorm)
  - Zhiqiang Tang, Yunhe Gao, Yi Zhu, Zhi Zhang, Mu Li, Dimitris Metaxas. *ICCV 2021*
  - Key Word: Corruption Robustness; Normalization.
  - <details><summary>Digest</summary> Unlike most previous works, this paper presents two normalization methods, SelfNorm and CrossNorm, to promote OOD generalization. SelfNorm uses attention to recalibrate statistics (channel-wise mean and variance), while CrossNorm exchanges the statistics between feature maps.

- Does Invariant Risk Minimization Capture Invariance? [[paper]](https://arxiv.org/abs/2101.01134)
  - Pritish Kamath, Akilesh Tangella, Danica J. Sutherland, Nathan Srebro. *AISTATS 2021*
  - Key Word: Invariant Learning; Causality.
  - <details><summary>Digest</summary> We show that the Invariant Risk Minimization (IRM) formulation of Arjovsky et al. (2019) can fail to capture "natural" invariances, at least when used in its practical "linear" form, and even on very simple problems which directly follow the motivating examples for IRM. This can lead to worse generalization on new environments, even when compared to unconstrained ERM. The issue stems from a significant gap between the linear variant (as in their concrete method IRMv1) and the full non-linear IRM formulation.

### Out-of-Distribution Generalization: 2020

- Auxiliary Training: Towards Accurate and Robust Models. [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Auxiliary_Training_Towards_Accurate_and_Robust_Models_CVPR_2020_paper.html)
  - Linfeng Zhang, Muzhou Yu, Tong Chen, Zuoqiang Shi, Chenglong Bao, Kaisheng Ma. *CVPR 2020*
  - Key Word: Auxiliary Learning; Corruption Robustness.
  - <details><summary>Digest</summary> In this paper, we propose a novel training method via introducing the auxiliary classifiers for training on corrupted samples, while the clean samples are normally trained with the primary classifier.  

- Generative Interventions for Causal Learning. [[paper]](https://arxiv.org/abs/2012.12265) [[code]](https://github.com/cvlab-columbia/GenInt)
  - Chengzhi Mao, Augustine Cha, Amogh Gupta, Hao Wang, Junfeng Yang, Carl Vondrick. *CVPR 2021*
  - Key Word: Corruption Robustness; Causal Learning.
  - <details><summary>Digest</summary> We introduce a framework for learning robust visual representations that generalize to new viewpoints, backgrounds, and scene contexts. Discriminative models often learn naturally occurring spurious correlations, which cause them to fail on images outside of the training distribution. In this paper, we show that we can steer generative models to manufacture interventions on features caused by confounding factors.

- Simulating a Primary Visual Cortex at the Front of CNNs Improves Robustness to Image Perturbations. [[paper]](https://nips.cc/virtual/2020/public/poster_98b17f068d5d9b7668e19fb8ae470841.html) [[code]](https://github.com/dicarlolab/vonenet)
  - Joel Dapello, Tiago Marques, Martin Schrimpf, Franziska Geiger, David Cox, James J DiCarlo. *NeurIPS 2020*
  - Key Word: Corruption Robustness; Neuroscience.
  - <details><summary>Digest</summary> We developed VOneNets, a new class of hybrid CNN vision models. The VOneBlock is based on a classical neuroscientific model of V1: the linear-nonlinear-Poisson model, consisting of a biologically-constrained Gabor filter bank, simple and complex cell nonlinearities, and a V1 neuronal stochasticity generator. We show that all components of the VOneBlock work in synergy to improve robustness (white box adversarial attack and corruption).

- What Can Style Transfer and Paintings Do For Model Robustness? [[paper]](https://arxiv.org/abs/2011.14477) [[code]](https://github.com/hubertsgithub/style_painting_robustness)
  - Hubert Lin, Mitchell van Zuijlen, Sylvia C. Pont, Maarten W.A. Wijntjes, Kavita Bala.
  - Key Word: Corruption Robustness; Style Transfer.
  - <details><summary>Digest</summary> We argue that paintings can be considered a form of perceptual data augmentation, and demonstrate that it can improve model robustness.

- Improved Handling of Motion Blur in Online Object Detection. [[paper]](https://arxiv.org/abs/2011.14448)
  - Mohamed Sayed, Gabriel Brostow. *CVPR 2021*
  - Key Word: Corruption Robustness; Detection.
  - <details><summary>Digest</summary> We explore five classes of remedies in blurring object detection: deblurring, squint, augmentation, using minibatch statistics and label modification.

- StackMix: A complementary Mix algorithm. [[paper]](https://arxiv.org/abs/2011.12618v2)
  - John Chen, Samarth Sinha, Anastasios Kyrillidis.
  - Key Word: Corruption Robustness; Data Augmentation.
  - <details><summary>Digest</summary> We present StackMix: Each input is presented as a concatenation of two images, and the label is the mean of the two one-hot labels. We further show that gains hold for robustness to common input corruptions and perturbations at varying severities with a 0.7% improvement on CIFAR-100-C, by combining StackMix with AugMix over AugMix.

- An Effective Anti-Aliasing Approach for Residual Networks. [[paper]](https://arxiv.org/abs/2011.10675v1)
  - Cristina Vasconcelos, Hugo Larochelle, Vincent Dumoulin, Nicolas Le Roux, Ross Goroshin.
  - Key Word: Corruption Robustness; Anti-Aliasing.
  - <details><summary>Digest</summary> We show that we can mitigate this effect by placing non-trainable blur filters and using smooth activation functions at key locations, particularly where networks lack the capacity to learn them. These simple architectural changes lead to substantial improvements in out-of-distribution generalization on both image classification under natural corruptions on ImageNetC and few-shot learning on Meta-Dataset.

- Empirical or Invariant Risk Minimization? A Sample Complexity Perspective. [[paper]](https://arxiv.org/abs/2010.16412) [[code]](https://github.com/IBM/IRM-games)
  - Kartik Ahuja, Jun Wang, Amit Dhurandhar, Karthikeyan Shanmugam, Kush R. Varshney. *ICLR 2021*
  - Key Word: Invariant Learning; Causality.
  - <details><summary>Digest</summary> Recently, invariant risk minimization (IRM) was proposed as a promising solution to address out-of-distribution (OOD) generalization. However, it is unclear when IRM should be preferred over the widely-employed empirical risk minimization (ERM) framework. In this work, we analyze both these frameworks from the perspective of sample complexity, thus taking a firm step towards answering this important question. We find that depending on the type of data generation mechanism, the two approaches might have very different finite sample and asymptotic behavior.

- Understanding the Failure Modes of Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2010.15775) [[code]](https://github.com/google-research/OOD-failures)
  - Vaishnavh Nagarajan, Anders Andreassen, Behnam Neyshabur. *ICLR 2021*
  - Key Word: Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> We identify the fundamental factors that give rise to this behavior, by explaining why models fail this way even in easy-to-learn tasks where one would expect these models to succeed. In particular, through a theoretical study of gradient-descent-trained linear classifiers on some easy-to-learn tasks, we uncover two complementary failure modes. These modes arise from how spurious correlations induce two kinds of skews in the data: one geometric in nature and another, statistical.

- Improving Transformation Invariance in Contrastive Representation Learning. [[paper]](https://arxiv.org/abs/2010.09515) [[code]](https://github.com/ae-foster/invclr)
  - Adam Foster, Rattana Pukdee, Tom Rainforth. *ICLR 2021*
  - Key Word: Transformation Invariance; Contrastive Learning.
  - <details><summary>Digest</summary> We first introduce a training objective for contrastive learning that uses a novel regularizer to control how the representation changes under transformation. We show that representations trained with this objective perform better on downstream tasks and are more robust to the introduction of nuisance transformations at test time. Second, we propose a change to how test time representations are generated by introducing a feature averaging approach that combines encodings from multiple transformations of the original input, finding that this leads to across the board performance gains.

- Maximum-Entropy Adversarial Data Augmentation for Improved Generalization and Robustness. [[paper]](https://arxiv.org/abs/2010.08001) [[code]](https://github.com/garyzhao/ME-ADA)
  - Long Zhao, Ting Liu, Xi Peng, Dimitris Metaxas. *NeurIPS 2020*
  - Key Word: Corruption Robustness; Data Augmentation.
  - <details><summary>Digest</summary> We develop an efficient maximum-entropy regularizer in the maximization phase of adversarial data argumentation, which results in a notable improvement over corruption (adv-corrupt) benchmarks.

- Environment Inference for Invariant Learning. [[paper]](https://arxiv.org/abs/2010.07249) [[code]](https://github.com/ecreager/eiil)
  - Elliot Creager, Jörn-Henrik Jacobsen, Richard Zemel. *ICML 2021*
  - Key Word: Invariant Learning; Causality; Fairness.
  - <details><summary>Digest</summary> We propose EIIL, a general framework for domain-invariant learning that incorporates Environment Inference to directly infer partitions that are maximally informative for downstream Invariant Learning. We show that EIIL outperforms invariant learning methods on the CMNIST benchmark without using environment labels, and significantly outperforms ERM on worst-group performance in the Waterbirds and CivilComments datasets. Finally, we establish connections between EIIL and algorithmic fairness, which enables EIIL to improve accuracy and calibration in a fair prediction problem.

- Shape-Texture Debiased Neural Network Training. [[paper]](https://arxiv.org/abs/2010.05981v2) [[code]](https://github.com/LiYingwei/ShapeTextureDebiasedTraining)
  - Yingwei Li, Qihang Yu, Mingxing Tan, Jieru Mei, Peng Tang, Wei Shen, Alan Yuille, Cihang Xie. *ICLR 2021*
  - Key Word: Corruption Robustness; Shape-Texture.
  - <details><summary>Digest</summary> We develop a shape-texture debiased neural network training framework to guide CNNs for learning better representations. Our method is a data-driven approach, which let CNNs automatically figure out how to avoid being biased towards either shape or texture from their training samples.

- Permuted AdaIN: Reducing the Bias Towards Global Statistics in Image Classification. [[paper]](https://arxiv.org/abs/2010.05785v3) [[code]](https://github.com/onuriel/PermutedAdaIN)
  - Oren Nuriel, Sagie Benaim, Lior Wolf. *CVPR 2021*
  - Key Word: Corruption Robustness; Normalization.
  - <details><summary>Digest</summary> Our method, called Permuted Adaptive Instance Normalization (pAdaIN), reduces the representation of global statistics in the hidden layers of image classifiers. pAdaIN samples a random permutation that rearranges the samples in a given batch.

- The Risks of Invariant Risk Minimization. [[paper]](https://arxiv.org/abs/2010.05761)
  - Elan Rosenfeld, Pradeep Ravikumar, Andrej Risteski. *ICLR 2021*
  - Key Word: Invariant Learning; Causality.
  - <details><summary>Digest</summary> We present the first analysis of classification under the IRM objective--as well as these recently proposed alternatives--under a fairly natural and general model. In the linear case, we show simple conditions under which the optimal solution succeeds or, more often, fails to recover the optimal invariant predictor.

- Increasing the Robustness of Semantic Segmentation Models with Painting-by-Numbers. [[paper]](https://arxiv.org/abs/2010.05495)
  - Christoph Kamann, Burkhard Güssefeld, Robin Hutmacher, Jan Hendrik Metzen, Carsten Rother. *ECCV 2020*
  - Key Word: Corruption Robustness; Data Augmentation; Segmentation.
  - <details><summary>Digest</summary> Our basic idea is to alpha-blend a portion of the RGB training images with faked images, where each class-label is given a fixed, randomly chosen color that is not likely to appear in real imagery. This forces the network to rely more strongly on shape cues. We call this data augmentation technique “Painting-by-Numbers”.

- Revisiting Batch Normalization for Improving Corruption Robustness. [[paper]](https://arxiv.org/abs/2010.03630v4)
  - Philipp Benz, Chaoning Zhang, Adil Karjauv, In So Kweon. *WACV 2021*
  - Key Word: Corruption Robustness; Normalization.
  - <details><summary>Digest</summary> We interpret corruption robustness as a domain shift and propose to rectify batch normalization (BN) statistics for improving model robustness.

- Batch Normalization Increases Adversarial Vulnerability: Disentangling Usefulness and Robustness of Model Features. [[paper]](https://arxiv.org/abs/2010.03316)
  - Philipp Benz, Chaoning Zhang, In So Kweon.
  - Key Word: Corruption Robustness; Normalization.
  - <details><summary>Digest</summary> We conjecture that the increased adversarial vulnerability is caused by BN shifting the model to rely more on non-robust features (NRFs). Our exploration finds that other normalization techniques also increase adversarial vulnerability and our conjecture is also supported by analyzing the model corruption robustness and feature transferability.

- Adversarial and Natural Perturbations for General Robustness. [[paper]](https://arxiv.org/abs/2010.01401)
  - Sadaf Gulshad, Jan Hendrik Metzen, Arnold Smeulders.
  - Key Word: Common Robustness; Adversarial Robustness.
  - <details><summary>Digest</summary> We demonstrate that although adversarial training improves the performance of the networks against adversarial perturbations, it leads to drop in the performance for naturally perturbed samples besides clean samples. In contrast, natural perturbations like elastic deformations, occlusions and wave does not only improve the performance against natural perturbations, but also lead to improvement in the performance for the adversarial perturbations.

- Encoding Robustness to Image Style via Adversarial Feature Perturbations. [[paper]](https://arxiv.org/abs/2009.08965v2) [[code]](https://github.com/azshue/AdvBN)
  - Manli Shu, Zuxuan Wu, Micah Goldblum, Tom Goldstein. *NeurIPS 2021*
  - Key Word: Corruption Robustness; Normalization.
  - <details><summary>Digest</summary> Our method, Adversarial Batch Normalization (AdvBN, adversarially perturbing these feature statistics), significantly improves robustness.

- Stochastic-YOLO: Efficient Probabilistic Object Detection under Dataset Shifts. [[paper]](https://arxiv.org/abs/2009.02967) [[code]](https://github.com/tjiagom/stochastic-yolo)
  - Tiago Azevedo, René de Jong, Matthew Mattina, Partha Maji.
  - Key Word: Corruption Robustness; Detection.
  - <details><summary>Digest</summary> We adapt the well-established YOLOv3 architecture to generate uncertainty estimations by introducing stochasticity in the form of Monte Carlo Dropout (MC-Drop), and evaluate it across different levels of dataset shift.

- Learning explanations that are hard to vary. [[paper]](https://arxiv.org/abs/2009.00329) [[code]](https://github.com/gibipara92/learning-explanations-hard-to-vary)
  - Giambattista Parascandolo, Alexander Neitz, Antonio Orvieto, Luigi Gresele, Bernhard Schölkopf. *ICLR 2021*
  - Key Word: Invariant Learning; Gradient Alignment.
  - <details><summary>Digest</summary> In this paper, we investigate the principle that good explanations are hard to vary in the context of deep learning. We show that averaging gradients across examples -- akin to a logical OR of patterns -- can favor memorization and 'patchwork' solutions that sew together different strategies, instead of identifying invariances. To inspect this, we first formalize a notion of consistency for minima of the loss surface, which measures to what extent a minimum appears only when examples are pooled.

- Addressing Neural Network Robustness with Mixup and Targeted Labeling Adversarial Training. [[paper]](https://arxiv.org/abs/2008.08384)
  - Alfred Laugros, Alice Caplier, Matthieu Ospici.
  - Key Word: Corruption Robustness; Adversarial Robustness.
  - <details><summary>Digest</summary> Our approach combines the Mixup augmentation and a new adversarial training algorithm called Targeted Labeling Adversarial Training (TLAT). The idea of TLAT is to interpolate the target labels of adversarial examples with the groundtruth labels.

- What Should Not Be Contrastive in Contrastive Learning. [[paper]](https://arxiv.org/abs/2008.05659)
  - Tete Xiao, Xiaolong Wang, Alexei A. Efros, Trevor Darrell. *ICLR 2021*
  - Key Word: Corruption Robustness; Contrastive Learning.
  - <details><summary>Digest</summary> Current methods introduce inductive bias by encouraging neural networks to be less sensitive to information w.r.t. augmentation, which may help or may hurt. We use a multi-head network with a shared backbone which captures information across each augmentation and alone outperforms all baselines on downstream tasks.

- Informative Dropout for Robust Representation Learning: A Shape-bias Perspective. [[paper]](https://arxiv.org/abs/2008.04254) [[code]](https://github.com/bfshi/InfoDrop)
  - Baifeng Shi, Dinghuai Zhang, Qi Dai, Jingdong Wang, Zhanxing Zhu, Yadong Mu. *ICML 2020*
  - Key Word: Dropout; Shape-Texture.
  - <details><summary>Digest</summary> In this work, we attempt at improving various kinds of robustness universally by alleviating CNN's texture bias. Specifically, with inspiration from human visual system, we propose a light-weight model-agnostic method, namely Informative Dropout (InfoDrop), to improve interpretability and reduce texture bias.  

- Robust and Generalizable Visual Representation Learning via Random Convolutions. [[paper]](https://arxiv.org/abs/2007.13003) [[code]](https://github.com/wildphoton/RandConv)
  - Zhenlin Xu, Deyi Liu, Junlin Yang, Colin Raffel, Marc Niethammer. *ICLR 2021*
  - Key Word: Corruption Robustness; Data Augmentation.
  - <details><summary>Digest</summary> We develop RandConv, a data augmentation technique using multi-scale random-convolutions to generate images with random texture while maintaining global shapes. We explore using the RandConv output as training images or mixing it with the original images. We show that a consistency loss can further enforce invariance under texture changes.

- Robust Image Classification Using A Low-Pass Activation Function and DCT Augmentation. [[paper]](https://arxiv.org/abs/2007.09453v1) [[code]](https://github.com/tahmid0007/Low_Pass_ReLU)
  - Md Tahmid Hossain, Shyh Wei Teng, Ferdous Sohel, Guojun Lu.
  - Key Word: Corruption Robustness; Activation Function.
  - <details><summary>Digest</summary> We propose a family of novel AFs with low-pass filtering to improve robustness against HFc (we call it Low-Pass ReLU or LP-ReLU). To deal with LFc, we further enhance the AFs with Discrete Cosine Transform (DCT) based augmentation. LPReLU coupled with DCT augmentation, enables a deep network to tackle a variety of corruptions.

- Learning perturbation sets for robust machine learning. [[paper]](https://arxiv.org/abs/2007.08450) [[code]](https://github.com/locuslab/perturbation_learning)
  - Eric Wong, J. Zico Kolter. *ICLR 2021*
  - Key Word: Corruption Robustness; Adversarial Robustness.
  - <details><summary>Digest</summary> We use a conditional generator that defines the perturbation set over a constrained region of the latent space. We formulate desirable properties that measure the quality of a learned perturbation set, and theoretically prove that a conditional variational autoencoder naturally satisfies these criteria.

- On the relationship between class selectivity, dimensionality, and robustness. [[paper]](https://arxiv.org/abs/2007.04440)
  - Matthew L. Leavitt, Ari S. Morcos.
  - Key Word: Corruption Robustness; Class Selectivity; Interpretability.
  - <details><summary>Digest</summary> We found that mean class selectivity predicts vulnerability to naturalistic corruptions; networks regularized to have lower levels of class selectivity are more robust to corruption, while networks with higher class selectivity are more vulnerable to corruption, as measured using Tiny ImageNetC and CIFAR10C. In contrast, we found that class selectivity increases robustness to multiple types of gradient-based adversarial attacks.

- In Search of Lost Domain Generalization. [[paper]](https://arxiv.org/abs/2007.01434) [[code]](https://github.com/facebookresearch/DomainBed)
  - Ishaan Gulrajani, David Lopez-Paz. *ICLR 2021*
  - Key Word: Domain Generalization; Empirical Study.
  - <details><summary>Digest</summary> As a first step, we realize that model selection is non-trivial for domain generalization tasks, and we argue that algorithms without a model selection criterion remain incomplete. Next we implement DomainBed, a testbed for domain generalization including seven benchmarks, fourteen algorithms, and three model selection criteria. When conducting extensive experiments using DomainBed we find that when carefully implemented and tuned, ERM outperforms the state-of-the-art in terms of average performance.

- Improving robustness against common corruptions by covariate shift adaptation. [[paper]](https://arxiv.org/abs/2006.16971) [[code]](https://github.com/bethgelab/robustness)
  - Steffen Schneider, Evgenia Rusak, Luisa Eck, Oliver Bringmann, Wieland Brendel, Matthias Bethge. *NeurIPS 2020*
  - Key Word: Corruption Robustness; Unsupervised Domain Adaptation.
  - <details><summary>Digest</summary> We suggest to augment current benchmarks for common corruptions with two additional performance metrics that measure robustness after partial and full unsupervised adaptation to the corrupted images.

- The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2006.16241) [[dataset]](https://github.com/hendrycks/imagenet-r)
  - Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, Justin Gilmer.
  - Key Word: Dataset; Benchmark; Data Augmentation; Corruption Robustness.
  - <details><summary>Digest</summary> We introduce three new robustness benchmarks consisting of naturally occurring distribution changes in image style, geographic location, camera operation, and more. Using our benchmarks, we take stock of previously proposed hypotheses for out-of-distribution robustness and put them to the test.  

- Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift. [[paper]](https://arxiv.org/abs/2006.10963v3)
  - Zachary Nado, Shreyas Padhy, D. Sculley, Alexander D'Amour, Balaji Lakshminarayanan, Jasper Snoek.
  - Key Word: Corruption Robustness; Normalization.
  - <details><summary>Digest</summary> We formalize a perhaps underappreciated prediction setting, which we call the prediction-time batch setting. We then propose a simple method, prediction-time BN, for using this information to effectively correct for covariate shift.

- Tent: Fully Test-time Adaptation by Entropy Minimization. [[paper]](https://arxiv.org/abs/2006.10726v3) [[code]](https://github.com/DequanWang/tent)
  - Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, Trevor Darrell. *ICLR 2021*
  - Key Word: Corruption Robustness; Test-Time Adaptation.
  - <details><summary>Digest</summary> We examine entropy as an adaptation objective and propose tent: a test-time entropy minimization scheme to reduce generalization error by reducing the entropy of model predictions on test data.

- Risk Variance Penalization. [[paper]](https://arxiv.org/abs/2006.07544)
  - Chuanlong Xie, Haotian Ye, Fei Chen, Yue Liu, Rui Sun, Zhenguo Li.
  - Key Word: Invariant Learning; Causality; Distributional Robustness.
  - <details><summary>Digest</summary> The key of the out-of-distribution (OOD) generalization is to generalize invariance from training domains to target domains. The variance risk extrapolation (V-REx) is a practical OOD method, which depends on a domain-level regularization but lacks theoretical verifications about its motivation and utility. This article provides theoretical insights into V-REx by studying a variance-based regularizer. We propose Risk Variance Penalization (RVP), which slightly changes the regularization of V-REx but addresses the theory concerns about V-REx.

- An Unsupervised Information-Theoretic Perceptual Quality Metric. [[paper]](https://arxiv.org/abs/2006.06752v3) [[code]](https://github.com/google-research/perceptual-quality)
  - Sangnie Bhardwaj, Ian Fischer, Johannes Ballé, Troy Chinen. *NeurIPS 2020*
  - Key Word: Corruption Robustness; Information Theory.
  - <details><summary>Digest</summary> We combine information-theoretic objective functions with a computational architecture informed by the physiology of the human visual system and unsupervised training on pairs of video frames, yielding our Perceptual Information Metric (PIM). We show that PIM is competitive with supervised metrics on the recent and challenging BAPPS image quality assessment dataset and perform qualitative experiments using the ImageNet-C dataset, and establish that PIM is robust with respect to architectural details.

- Wavelet Integrated CNNs for Noise-Robust Image Classification. [[paper]](https://arxiv.org/abs/2005.03337v2) [[code]](https://github.com/LiQiufu/WaveCNet)
  - Qiufu Li, Linlin Shen, Sheng Guo, Zhihui Lai. *CVPR 2020*
  - Key Word: Corruption Robustness; Pooling.
  - <details><summary>Digest</summary> We enhance CNNs by replacing max-pooling, strided-convolution, and average-pooling with Discrete Wavelet Transform (DWT). We present general DWT and Inverse DWT (IDWT) layers applicable to various wavelets like Haar, Daubechies, and Cohen, etc., and design wavelet integrated CNNs (WaveCNets) using these layers for image classification

- Shortcut Learning in Deep Neural Networks. [[paper]](https://arxiv.org/abs/2004.07780) [[code]](https://github.com/rgeirhos/shortcut-perspective)
  - Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, Felix A. Wichmann. *Nature Machine Intelligence*
  - Key Word: Survey; Out-of-Distribution Generalization; Neuroscience.
  - <details><summary>Digest</summary> We seek to distil how many of deep learning's problem can be seen as different symptoms of the same underlying problem: shortcut learning. Shortcuts are decision rules that perform well on standard benchmarks but fail to transfer to more challenging testing conditions, such as real-world scenarios. Related issues are known in Comparative Psychology, Education and Linguistics, suggesting that shortcut learning may be a common characteristic of learning systems, biological and artificial alike.

- ObjectNet Dataset: Reanalysis and Correction. [[paper]](https://arxiv.org/abs/2004.02042) [[code]](https://github.com/aliborji/ObjectNetReanalysis)
  - Ali Borji.
  - Key Word: Dataset; Detection.
  - <details><summary>Digest</summary> We highlight a major problem with their work which is applying object recognizers to the scenes containing multiple objects rather than isolated objects. The latter results in around 20-30% performance gain using our code. Compared with the results reported in the ObjectNet paper, we observe that around 10-15 % of the performance loss can be recovered, without any test time data augmentation.  

- Invariant Rationalization. [[paper]](https://arxiv.org/abs/2003.09772) [[code]](https://github.com/code-terminator/invariant_rationalization)
  - Shiyu Chang, Yang Zhang, Mo Yu, Tommi S. Jaakkola.
  - Key Word: Invariant Learning; Causality; Mutual Information.
  - <details><summary>Digest</summary> Selective rationalization improves neural network interpretability by identifying a small subset of input features -- the rationale -- that best explains or supports the prediction. A typical rationalization criterion, i.e. maximum mutual information (MMI), finds the rationale that maximizes the prediction performance based only on the rationale. However, MMI can be problematic because it picks up spurious correlations between the input features and the output. Instead, we introduce a game-theoretic invariant rationalization criterion where the rationales are constrained to enable the same predictor to be optimal across different environments.

- Out-of-Distribution Generalization via Risk Extrapolation (REx). [[paper]](https://arxiv.org/abs/2003.00688) [[code]](https://github.com/capybaralet/REx_code_release)
  - David Krueger, Ethan Caballero, Joern-Henrik Jacobsen, Amy Zhang, Jonathan Binas, Dinghuai Zhang, Remi Le Priol, Aaron Courville. *ICML 2021*
  - Key Word: Invariant Learning; Causality; Robust Optimization.
  - <details><summary>Digest</summary> We show that reducing differences in risk across training domains can reduce a model's sensitivity to a wide range of extreme distributional shifts, including the challenging setting where the input contains both causal and anti-causal elements. We motivate this approach, Risk Extrapolation (REx), as a form of robust optimization over a perturbation set of extrapolated domains (MM-REx), and propose a penalty on the variance of training risks (V-REx) as a simpler variant.

- CEB Improves Model Robustness. [[paper]](https://arxiv.org/abs/2002.05380)
  - Ian Fischer, Alexander A. Alemi.
  - Key Word: Information Bottleneck; Corruption Robustness; Adversarial Robustness.
  - <details><summary>Digest</summary> We demonstrate that the Conditional Entropy Bottleneck (CEB) can improve model robustness. CEB is an easy strategy to implement and works in tandem with data augmentation procedures. We report results of a large scale adversarial robustness study on CIFAR-10, as well as the ImageNet-C Common Corruptions Benchmark, ImageNet-A, and PGD attacks.  

- Invariant Risk Minimization Games. [[paper]](https://arxiv.org/abs/2002.04692) [[code]](https://github.com/IBM/OoD/tree/master/IRM_games)
  - Kartik Ahuja, Karthikeyan Shanmugam, Kush R. Varshney, Amit Dhurandhar. *ICML 2020*
  - Key Word: Invariant Learning; Causality; Game Theory.
  - <details><summary>Digest</summary> We pose such invariant risk minimization as finding the Nash equilibrium of an ensemble game among several environments. By doing so, we develop a simple training algorithm that uses best response dynamics and, in our experiments, yields similar or better empirical accuracy with much lower variance than the challenging bi-level optimization problem of Arjovsky et al. (2019). One key theoretical contribution is showing that the set of Nash equilibria for the proposed game are equivalent to the set of invariant predictors for any finite number of environments, even with nonlinear classifiers and transformations.

- A simple way to make neural networks robust against diverse image corruptions. [[paper]](https://arxiv.org/abs/2001.06057) [[code]](https://github.com/bethgelab/game-of-noise)
  - Evgenia Rusak, Lukas Schott, Roland S. Zimmermann, Julian Bitterwolf, Oliver Bringmann, Matthias Bethge, Wieland Brendel. *ECCV 2020*
  - Key Word: Data Augmentation; Corruption Robustness.
  - <details><summary>Digest</summary> We demonstrate that a simple but properly tuned training with additive Gaussian and Speckle noise generalizes surprisingly well to unseen corruptions, easily reaching the previous state of the art on the corruption benchmark ImageNet-C (with ResNet50) and on MNIST-C.  

### Out-of-Distribution Generalization: 2019

- ObjectNet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. [[paper]](https://papers.nips.cc/paper/9142-objectnet-a-large-scale-bias-controlled-dataset-for-pushing-the-limits-of-object-recognition-models.pdf) [[dataset]](https://objectnet.dev/)
  - Barbu, A, Mayo, D, Alverio, J, Luo, W, Wang, C, Gutfreund, D, Tenenabum, JB, Katz, B. *NeurIPS 2019*
  - Key Word: Dataset.
  - <details><summary>Digest</summary> A new method, which we refer to as convex layerwise adversarial training (COLT), that can train provably robust neural networks and conceptually bridges the gap between adversarial training and existing provable defense methods.  

- AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty. [[paper]](https://arxiv.org/abs/1912.02781) [[code]](https://github.com/google-research/augmix)
  - Dan Hendrycks, Norman Mu, Ekin D. Cubuk, Barret Zoph, Justin Gilmer, Balaji Lakshminarayanan. *ICLR 2020*
  - Key Word: Dataset; Corruption Robustness.
  - <details><summary>Digest</summary> In this work, we propose a technique to improve the robustness and uncertainty estimates of image classifiers. We propose AugMix, a data processing technique that is simple to implement, adds limited computational overhead, and helps models withstand unforeseen corruptions.  

- Adversarial Examples Improve Image Recognition. [[paper]](https://arxiv.org/abs/1911.09665v2) [[code]](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
  - Cihang Xie, Mingxing Tan, Boqing Gong, Jiang Wang, Alan Yuille, Quoc V. Le. *CVPR 2020*
  - Key Word: Corruption Robustness; Adversarial Robustness.
  - <details><summary>Digest</summary> Adversarial examples can be used to improve image recognition models. Key to our method is the usage of a separate auxiliary batch norm for adversarial examples, as they have different underlying distributions to normal examples.

- Self-training with Noisy Student improves ImageNet classification. [[paper]](https://arxiv.org/abs/1911.04252v4) [[code]](https://github.com/google-research/noisystudent)
  - Qizhe Xie, Minh-Thang Luong, Eduard Hovy, Quoc V. Le. *CVPR 2020*
  - Key Word: Corruption Robustness; Self-Training.
  - <details><summary>Digest</summary> Noisy Student Training extends the idea of self-training and distillation with the use of equal-or-larger student models and noise added to the student during learning.

- Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. [[paper]](https://arxiv.org/abs/1909.13231) [[code]](https://github.com/yueatsprograms/ttt_cifar_release)
  - Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, Moritz Hardt. *ICML 2020*
  - Key Word: Corruption Robustness; Test-Time Training.
  - <details><summary>Digest</summary> We turn a single unlabeled test sample into a self-supervised learning problem, on which we update the model parameters before making a prediction. Our simple approach leads to improvements on diverse image classification benchmarks aimed at evaluating robustness to distribution shifts.

- Pretraining boosts out-of-domain robustness for pose estimation. [[paper]](https://arxiv.org/abs/1909.11229) [[code]](https://github.com/DeepLabCut/DeepLabCut)
  - Alexander Mathis, Thomas Biasi, Steffen Schneider, Mert Yüksekgönül, Byron Rogers, Matthias Bethge, Mackenzie W. Mathis. *WACV 2021*
  - Key Word: Corruption Robustness; Pre-Trainig; Pose Estimation.
  - <details><summary>Digest</summary> We show that better ImageNet-performing architectures perform better on both within- and out-of-domain data if they are first pretrained on ImageNet. We additionally show that better ImageNet models generalize better across animal species. Furthermore, we introduce Horse-C, a new benchmark for common corruptions for pose estimation.

- Training Robust Deep Neural Networks via Adversarial Noise Propagation. [[paper]](https://arxiv.org/abs/1909.09034) [[code]](https://github.com/AnonymousCodeRepo/ANP)
  - Aishan Liu, Xianglong Liu, Chongzhi Zhang, Hang Yu, Qiang Liu, Dacheng Tao.
  - Key Word: Corruption Robustness; Data Augmentation.
  - <details><summary>Digest</summary> This paper proposes a simple yet powerful training algorithm, named Adversarial Noise Propagation (ANP), which injects noise into the hidden layers in a layer-wise manner.

- PDA: Progressive Data Augmentation for General Robustness of Deep Neural Networks. [[paper]](https://arxiv.org/abs/1909.04839v3)
  - Hang Yu, Aishan Liu, Xianglong Liu, Gengchao Li, Ping Luo, Ran Cheng, Jichen Yang, Chongzhi Zhang.
  - Key Word: Corruption Robustness; Data Augmentation.
  - <details><summary>Digest</summary> We propose a simple yet effective method, named Progressive Data Augmentation (PDA), which enables general robustness of DNNs by progressively injecting diverse adversarial noises during training.

- Are Adversarial Robustness and Common Perturbation Robustness Independent Attributes? [[paper]](https://arxiv.org/abs/1909.02436v2)
  - Alfred Laugros, Alice Caplier, Matthieu Ospici.
  - Key Word: Corruption Robustness; Adversarial Robustness.
  - <details><summary>Digest</summary> We show that increasing the robustness to carefully selected common perturbations, can make neural networks more robust to unseen common perturbations. We also prove that adversarial robustness and robustness to common perturbations are independent.

- Testing Robustness Against Unforeseen Adversaries. [[paper]](https://arxiv.org/abs/1908.08016) [[code]](https://github.com/ddkang/advex-uar)
  - Daniel Kang, Yi Sun, Dan Hendrycks, Tom Brown, Jacob Steinhardt.
  - Key Word: Corruption Robustness; Adversarial Robustness.
  - <details><summary>Digest</summary> We propose unforeseen attacks. The JPEG, Fog, Snow, and Gabor adversarial attacks are visually distinct from previous attacks and serve as unforeseen attacks in the ImageNet-UA attack suite.

- Distributionally Robust Optimization: A Review. [[paper]](https://arxiv.org/abs/1908.05659)
  - Hamed Rahimian, Sanjay Mehrotra.
  - Key Word: Survey; Robust Optimization.
  - <details><summary>Digest</summary> The concepts of risk-aversion, chance-constrained optimization, and robust optimization have developed significantly over the last decade. Statistical learning community has also witnessed a rapid theoretical and applied growth by relying on these concepts. A modeling framework, called distributionally robust optimization (DRO), has recently received significant attention in both the operations research and statistical learning communities. This paper surveys main concepts and contributions to DRO, and its relationships with robust optimization, risk-aversion, chance-constrained optimization, and function regularization.

- Benchmarking the Robustness of Semantic Segmentation Models. [[paper]](https://arxiv.org/abs/1908.05005)
  - Christoph Kamann, Carsten Rother. *CVPR 2020*
  - Key Word: Corruption Robustness; Benchmark; Segmentation.
  - <details><summary>Digest</summary> While there are recent robustness studies for full-image classification, we are the first to present an exhaustive study for semantic segmentation, based on the state-of-the-art model DeepLabv3+. To increase the realism of our study, we utilize almost 400,000 images generated from Cityscapes, PASCAL VOC 2012, and ADE20K.

- Robustness properties of Facebook's ResNeXt WSL models. [[paper]](https://arxiv.org/abs/1907.07640) [[code]](https://github.com/eminorhan/resnext-wsl)
  - A. Emin Orhan.
  - Key Word: Corruption Robustness; Emprical Study.
  - <details><summary>Digest</summary> We show that although the ResNeXt WSL models are more shape-biased than comparable ImageNet-trained models in a shape-texture cue conflict experiment, they still remain much more texture-biased than humans, suggesting that they share some of the underlying characteristics of ImageNet-trained models that make this benchmark challenging.

- Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming. [[paper]](https://arxiv.org/abs/1907.07484)
  - Claudio Michaelis, Benjamin Mitzkus, Robert Geirhos, Evgenia Rusak, Oliver Bringmann, Alexander S. Ecker, Matthias Bethge, Wieland Brendel.
  - Key Word: Corruption Robustness; Benchmark; Detection.
  - <details><summary>Digest</summary> We here provide an easy-to-use benchmark to assess how object detection models perform when image quality degrades. The three resulting benchmark datasets, termed Pascal-C, Coco-C and Cityscapes-C, contain a large variety of image corruptions.

- Natural Adversarial Examples. [[paper]](https://arxiv.org/abs/1907.07174) [[dataset]](https://github.com/hendrycks/natural-adv-examples)
  - Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, Dawn Song.
  - Key Word: Dataset; Out-of-Distribution Detection; Natural Adversarial Examples.
  - <details><summary>Digest</summary> We introduce natural adversarial examples–real-world, unmodified, and naturally occurring examples that cause machine learning model performance to substantially degrade.  

- Invariant Risk Minimization. [[paper]](https://arxiv.org/abs/1907.02893) [[code]](https://github.com/facebookresearch/InvariantRiskMinimization)
  - Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, David Lopez-Paz.
  - Key Word: Invariant Learning; Causality.
  - <details><summary>Digest</summary> We introduce Invariant Risk Minimization (IRM), a learning paradigm to estimate invariant correlations across multiple training distributions. To achieve this goal, IRM learns a data representation such that the optimal classifier, on top of that data representation, matches for all training distributions. Through theory and experiments, we show how the invariances learned by IRM relate to the causal structures governing the data and enable out-of-distribution generalization.

- A Fourier Perspective on Model Robustness in Computer Vision. [[paper]](https://arxiv.org/abs/1906.08988v3) [[code]](https://github.com/google-research/google-research/tree/master/frequency_analysis)
  - Dong Yin, Raphael Gontijo Lopes, Jonathon Shlens, Ekin D. Cubuk, Justin Gilmer. *NeurIPS 2019*
  - Key Word: Corruption Robustness; Frequency.
  - <details><summary>Digest</summary> We investigate recently observed tradeoffs caused by Gaussian data augmentation and adversarial training. We find that both methods improve robustness to corruptions that are concentrated in the high frequency domain while reducing robustness to corruptions that are concentrated in the low frequency domain.

- A Closed-Form Learned Pooling for Deep Classification Networks. [[paper]](https://arxiv.org/abs/1906.03808v1)
  - Vighnesh Birodkar, Hossein Mobahi, Dilip Krishnan, Samy Bengio.
  - Key Word: Corruption Robustness; Pooling.
  - <details><summary>Digest</summary> We propose a way to enable CNNs to learn different pooling weights for each pixel location. We do so by introducing an extended definition of a pooling operator. This operator can learn a strict super-set of what can be learned by average pooling or convolutions.

- Towards Non-I.I.D. Image Classification: A Dataset and Baselines. [[paper]](https://arxiv.org/abs/1906.02899)
  - Yue He, Zheyan Shen, Peng Cui. *Pattern Recognition*
  - Key Word: Dataset; Distribution Shift.
  - <details><summary>Digest</summary> We construct and release a Non-I.I.D. image dataset called NICO, which uses contexts to create Non-IIDness consciously. Compared to other datasets, extended analyses prove NICO can support various Non-I.I.D. situations with sufficient flexibility. Meanwhile, we propose a baseline model with ConvNet structure for General Non-I.I.D. image classification, where distribution of testing data is unknown but different from training data.

- MNIST-C: A Robustness Benchmark for Computer Vision. [[paper]](https://arxiv.org/abs/1906.02337v1) [[code]](https://github.com/google-research/mnist-c)
  - Norman Mu, Justin Gilmer.
  - Key Word: Corruption Robustness; Benchmark.
  - <details><summary>Digest</summary> We introduce the MNIST-C dataset, a comprehensive suite of 15 corruptions applied to the MNIST test set.

- Improving Robustness Without Sacrificing Accuracy with Patch Gaussian Augmentation. [[paper]](https://arxiv.org/abs/1906.02611)
  - Raphael Gontijo Lopes, Dong Yin, Ben Poole, Justin Gilmer, Ekin D. Cubuk.
  - Key Word: Data Augmentation; Corruption Robustness.
  - <details><summary>Digest</summary> We introduce Patch Gaussian, a simple augmentation scheme that adds noise to randomly selected patches in an input image. Models trained with Patch Gaussian achieve state of the art on the CIFAR-10 and ImageNetCommon Corruptions benchmarks while also improving accuracy on clean data.  

- Making Convolutional Networks Shift-Invariant Again. [[paper]](https://arxiv.org/abs/1904.11486) [[code]](https://richzhang.github.io/antialiased-cnns/)
  - Richard Zhang. *ICML 2019*
  - Key Word: Anti-aliasing.
  - <details><summary>Digest</summary> Small shifts -- even by a single pixel -- can drastically change the output of a deep network (bars on left). We identify the cause: aliasing during downsampling. We anti-alias modern deep networks with classic signal processing, stabilizing output classifications (bars on right). We observe "free", unexpected improvements as well: accuracy increases and improved robustness.  

- Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. [[paper]](https://arxiv.org/abs/1903.12261) [[dataset]](https://github.com/hendrycks/robustness)
  - Dan Hendrycks, Thomas Dietterich. *ICLR 2019*
  - Key Word: Dataset; Benchmark; Corruption Robustness.
  - <details><summary>Digest</summary> In this paper we establish rigorous benchmarks for image classifier robustness. Our first benchmark, ImageNet-C, standardizes and expands the corruption robustness topic, while showing which classifiers are preferable in safety-critical applications. Then we propose a new dataset called ImageNet-P which enables researchers to benchmark a classifier's robustness to common perturbations.  

- Adversarial Examples Are a Natural Consequence of Test Error in Noise. [[paper]](https://arxiv.org/abs/1901.10513v1) [[code]](https://github.com/nicf/corruption-robustness)
  - Nic Ford, Justin Gilmer, Nicolas Carlini, Dogus Cubuk. *ICML 2019*
  - Keyword: Corruption Robustness; Adversarial Robustness.
  - <details><summary>Digest</summary> The existence of adversarial examples follows naturally from the fact that our models have nonzero test error in certain corrupted image distributions (connecting adversarial and corruption robustness).

### Out-of-Distribution Generalization: 2018

- ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. [[paper]](https://arxiv.org/abs/1811.12231) [[code]](https://github.com/rgeirhos/texture-vs-shape)
  - Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A. Wichmann, Wieland Brendel. *ICLR 2019*
  - Key Word: Shape-Texture; Style Transfer; Data Augmentation.
  - <details><summary>Digest</summary> We show that ImageNet-trained CNNs are strongly biased towards recognising textures rather than shapes, which is in stark contrast to human behavioural evidence and reveals fundamentally different classification strategies.  

- Strike (with) a Pose: Neural Networks Are Easily Fooled by Strange Poses of Familiar Objects. [[paper]](https://arxiv.org/abs/1811.11553) [[code]](https://github.com/airalcorn2/strike-with-a-pose)
  - Michael A. Alcorn, Qi Li, Zhitao Gong, Chengfei Wang, Long Mai, Wei-Shinn Ku, Anh Nguyen. *CVPR 2019*
  - Key Word: Adversarial Poses Transfer.
  - <details><summary>Digest</summary> In this paper, we present a framework for discovering DNN failures that harnesses 3D renderers and 3D models. That is, we estimate the parameters of a 3D renderer that cause a target DNN to misbehave in response to the rendered image.  

- Generalisation in humans and deep neural networks. [[paper]](https://arxiv.org/abs/1808.08750) [[code]](https://github.com/rgeirhos/generalisation-humans-DNNs)
  - Robert Geirhos, Carlos R. Medina Temme, Jonas Rauber, Heiko H. Schütt, Matthias Bethge, Felix A. Wichmann. *NeurIPS 2018*
  - Key Word: Corruption Robustness.
  - <details><summary>Digest</summary> We compare the robustness of humans and current convolutional deep neural networks (DNNs) on object recognition under twelve different types of image degradations.  

- Why do deep convolutional networks generalize so poorly to small image transformations? [[paper]](https://arxiv.org/abs/1805.12177) [[code]](https://github.com/AzulEye/CNN-Failures)
  - Aharon Azulay, Yair Weiss. *JMLR 2019*
  - Key Word: Transformation Invaraince.
  - <details><summary>Digest</summary> In this paper, we quantify this phenomena and ask why neither the convolutional architecture nor data augmentation are sufficient to achieve the desired invariance. Specifically, we show that the convolutional architecture does not give invariance since architectures ignore the classical sampling theorem, and data augmentation does not give invariance because the CNNs learn to be invariant to transformations only for images that are very similar to typical images from the training set.  

## Evasion Attacks and Defenses

### Evasion Attacks and Defenses: 2021

- MedRDF: A Robust and Retrain-Less Diagnostic Framework for Medical Pretrained Models Against Adversarial Attack. [[paper]](https://arxiv.org/abs/2111.14564)
  - Mengting Xu, Tao Zhang, Daoqiang Zhang. *TMI*
  - Key Word: Adversarial Robustness; Medical Image; Healthcare.
  - <details><summary>Digest</summary> We propose a Robust and Retrain-Less Diagnostic Framework for Medical pretrained models against adversarial attack (i.e., MedRDF). It acts on the inference time of the pertained medical model. Specifically, for each test image, MedRDF firstly creates a large number of noisy copies of it, and obtains the output labels of these copies from the pretrained medical diagnostic model. Then, based on the labels of these copies, MedRDF outputs the final robust diagnostic result by majority voting.

- Moiré Attack (MA): A New Potential Risk of Screen Photos. [[paper]](https://arxiv.org/abs/2110.10444) [[code]](https://github.com/Dantong88/Moire_Attack)
  - Dantong Niu, Ruohao Guo, Yisen Wang. *NeurIPS 2021*
  - Key Word: Adversarial Attacks; Digital Attacks; Physical Attacks; Moiré Effect.
  - <details><summary>Digest</summary> We find a special phenomenon in digital image processing, the moiré effect, that could cause unnoticed security threats to DNNs. Based on it, we propose a Moiré Attack (MA) that generates the physical-world moiré pattern adding to the images by mimicking the shooting process of digital devices.

- Towards Robust General Medical Image Segmentation. [[paper]](https://arxiv.org/abs/2107.04263) [[code]](https://github.com/BCV-Uniandes/ROG)
  - Laura Daza, Juan C. Pérez, Pablo Arbeláez. *MICCAI 2021*
  - Key Word: Adversarial Attack; Medical Image Segmentation; Healthcare.
  - <details><summary>Digest</summary> We propose a new framework to assess the robustness of general medical image segmentation systems. Our contributions are two-fold: (i) we propose a new benchmark to evaluate robustness in the context of the Medical Segmentation Decathlon (MSD) by extending the recent AutoAttack natural image classification framework to the domain of volumetric data segmentation, and (ii) we present a novel lattice architecture for RObust Generic medical image segmentation (ROG).

- The Dimpled Manifold Model of Adversarial Examples in Machine Learning. [[paper]](https://arxiv.org/abs/2106.10151)
  - Adi Shamir, Odelia Melamed, Oriel BenShmuel.
  - Key Word: Adversarial Robustness.
  - <details><summary>Digest</summary> We introduce a new conceptual framework for how the decision boundary between classes evolves during training, which we call the Dimpled Manifold Model. In particular, we demonstrate that training is divided into two distinct phases. The first phase is a (typically fast) clinging process in which the initially randomly oriented decision boundary gets very close to the low dimensional image manifold, which contains all the training examples. Next, there is a (typically slow) dimpling phase which creates shallow bulges in the decision boundary that move it to the correct side of the training examples. This framework provides a simple explanation for why adversarial examples exist, why their perturbations have such tiny norms, and why they look like random noise rather than like the target class.

- Adversarial Robustness through the Lens of Causality. [[paper]](https://arxiv.org/abs/2106.06196)
  - Yonggang Zhang, Mingming Gong, Tongliang Liu, Gang Niu, Xinmei Tian, Bo Han, Bernhard Schölkopf, Kun Zhang. *ICLR 2022*
  - Key Word: Adversarial Robustness; Causality.
  - <details><summary>Digest</summary> We construct a causal graph to model the generation process of adversarial examples and define the adversarial distribution to formalize the intuition of adversarial attacks. From a causal perspective, we find that the label is spuriously correlated with the style (content-independent) information when an instance is given. The spurious correlation implies that the adversarial distribution is constructed via making the statistical conditional association between style information and labels drastically different from that in natural distribution.

- Exploring Memorization in Adversarial Training. [[paper]](https://arxiv.org/abs/2106.01606) [[code]](https://github.com/dongyp13/memorization-AT)
  - Yinpeng Dong, Ke Xu, Xiao Yang, Tianyu Pang, Zhijie Deng, Hang Su, Jun Zhu. *ICLR 2022*
  - Key Word: Adversarial Training; Memorization.
  - <details><summary>Digest</summary> We explore the memorization effect in adversarial training (AT) for promoting a deeper understanding of model capacity, convergence, generalization, and especially robust overfitting of the adversarially trained models. We first demonstrate that deep networks have sufficient capacity to memorize adversarial examples of training data with completely random labels, but not all AT algorithms can converge under the extreme circumstance. Our study of AT with random labels motivates further analyses on the convergence and generalization of AT. We find that some AT approaches suffer from a gradient instability issue and most recently suggested complexity measures cannot explain robust generalization by considering models trained on random labels.

- BBAEG: Towards BERT-based Biomedical Adversarial Example Generation for Text Classification. [[paper]](https://arxiv.org/abs/2104.01782) [[code]](https://github.com/Ishani-Mondal/BBAEG)
  - Ishani Mondal. *NAACL 2021*
  - Key Word: Adversarial Attack; Biomedical Text Classification; Healthcare.
  - <details><summary>Digest</summary> Recent efforts of generating adversaries using rule-based synonyms and BERT-MLMs have been witnessed in general domain, but the ever increasing biomedical literature poses unique challenges. We propose BBAEG (Biomedical BERT-based Adversarial Example Generation), a black-box attack algorithm for biomedical text classification, leveraging the strengths of both domain-specific synonym replacement for biomedical named entities and BERTMLM predictions, spelling variation and number replacement.

- Robust Learning Meets Generative Models: Can Proxy Distributions Improve Adversarial Robustness? [[paper]](https://arxiv.org/abs/2104.09425) [[code]](https://github.com/inspire-group/proxy-distributions)
  - Vikash Sehwag, Saeed Mahloujifar, Tinashe Handina, Sihui Dai, Chong Xiang, Mung Chiang, Prateek Mittal. *ICLR 2022*
  - Key Word: Adversarial Robustness; Proxy Distribution.
  - <details><summary>Digest</summary> We first seek to formally understand the transfer of robustness from classifiers trained on proxy distributions to the real data distribution. We prove that the difference between the robustness of a classifier on the two distributions is upper bounded by the conditional Wasserstein distance between them. Next we use proxy distributions to significantly improve the performance of adversarial training on five different datasets.

- Rethinking Image-Scaling Attacks: The Interplay Between Vulnerabilities in Machine Learning Systems. [[paper]](https://arxiv.org/abs/2104.08690) [[code]](https://github.com/wi-pi/rethinking-image-scaling-attacks)  
  - Yue Gao, Ilia Shumailov, Kassem Fawaz. *ICML 2022*
  - Key Word: Adversarial Attacks.
  - <details><summary>Digest</summary> As real-world images come in varying sizes, the machine learning model is part of a larger system that includes an upstream image scaling algorithm. In this paper, we investigate the interplay between vulnerabilities of the image scaling procedure and machine learning models in the decision-based black-box setting. We propose a novel sampling strategy to make a black-box attack exploit vulnerabilities in scaling algorithms, scaling defenses, and the final machine learning model in an end-to-end manner.

- Stabilized Medical Image Attacks. [[paper]](https://arxiv.org/abs/2103.05232) [[code]](https://github.com/imogenqi/SMA)
  - Gege Qi, Lijun Gong, Yibing Song, Kai Ma, Yefeng Zheng. *ICLR 2021*
  - Key Word: Adversarial Attack; Medical Image; Healthcare.
  - <details><summary>Digest</summary> We propose an image-based medical adversarial attack method to consistently produce adversarial perturbations on medical images. The objective function of our method consists of a loss deviation term and a loss stabilization term. The loss deviation term increases the divergence between the CNN prediction of an adversarial example and its ground truth label. Meanwhile, the loss stabilization term ensures similar CNN predictions of this example and its smoothed input.

- Towards Evaluating the Robustness of Deep Diagnostic Models by Adversarial Attack. [[paper]](https://arxiv.org/abs/2103.03438) [[code]](https://github.com/MengtingXu1203/EvaluatingRobustness)
  - Mengting Xu, Tao Zhang, Zhongnian Li, Mingxia Liu, Daoqiang Zhang. *MedIA*
  - Key Word: Adversarial Attack; Medical Image; Healthcare.
  - <details><summary>Digest</summary> We evaluate the robustness of deep diagnostic models by adversarial attack. Specifically, we have performed two types of adversarial attacks to three deep diagnostic models in both single-label and multi-label classification tasks, and found that these models are not reliable when attacked by adversarial example.

### Evasion Attacks and Defenses: 2020

- A Hierarchical Feature Constraint to Camouflage Medical Adversarial Attacks. [[paper]](https://arxiv.org/abs/2012.09501) [[code]](https://github.com/ICT-MIRACLE-lab/Hierarchical_Feature_Constraint)
  - Qingsong Yao, Zecheng He, Yi Lin, Kai Ma, Yefeng Zheng, S. Kevin Zhou. *MICCAI 2021*
  - Key Word: Adversarial Attack; Medical Image; Healthcare.
  - <details><summary>Digest</summary> To better understand this phenomenon, we thoroughly investigate the intrinsic characteristic of medical AEs in feature space, providing both empirical evidence and theoretical explanations for the question: why are medical adversarial attacks easy to detect? We first perform a stress test to reveal the vulnerability of deep representations of medical images, in contrast to natural images. We then theoretically prove that typical adversarial attacks to binary disease diagnosis network manipulate the prediction by continuously optimizing the vulnerable representations in a fixed direction, resulting in outlier features that make medical AEs easy to detect.

- Advancing diagnostic performance and clinical usability of neural networks via adversarial training and dual batch normalization. [[paper]](https://arxiv.org/abs/2011.13011) [[code]](https://github.com/peterhan91/Medical-Robust-Training)
  - Tianyu Han, Sven Nebelung, Federico Pedersoli, Markus Zimmermann, Maximilian Schulze-Hagen, Michael Ho, Christoph Haarburger, Fabian Kiessling, Christiane Kuhl, Volkmar Schulz, Daniel Truhn. *Nature Communications*
  - Key Word: Adversarail Training; Batch Normalization; Healtcare.
  - <details><summary>Digest</summary> Unmasking the decision-making process of machine learning models is essential for implementing diagnostic support systems in clinical practice. Here, we demonstrate that adversarially trained models can significantly enhance the usability of pathology detection as compared to their standard counterparts. We let six experienced radiologists rate the interpretability of saliency maps in datasets of X-rays, computed tomography, and magnetic resonance imaging scans. Significant improvements were found for our adversarial models, which could be further improved by the application of dual batch normalization.

- Adversarial Training Reduces Information and Improves Transferability. [[paper]](https://arxiv.org/abs/2007.11259)
  - Matteo Terzi, Alessandro Achille, Marco Maggipinto, Gian Antonio Susto.
  - Key Word: Adversarial Training; Inforamtion Bottleneck.
  - <details><summary>Digest</summary> We investigate the dual relationship between Adversarial Training and Information Theory. We show that the Adversarial Training can improve linear transferability to new tasks, from which arises a new trade-off between transferability of representations and accuracy on the source task.  

- Do Adversarially Robust ImageNet Models Transfer Better? [[paper]](https://arxiv.org/abs/2007.08489) [[code]](https://github.com/Microsoft/robust-models-transfer)
  - Hadi Salman, Andrew Ilyas, Logan Engstrom, Ashish Kapoor, Aleksander Madry.
  - Key Word: Adversarial Robustness; Transfer Learning.
  - <details><summary>Digest</summary> In this work, we identify another such aspect: we find that adversarially robust models, while less accurate, often perform better than their standard-trained counterparts when used for transfer learning. Specifically, we focus on adversarially robust ImageNet classifiers, and show that they yield improved accuracy on a standard suite of downstream classification tasks.  

- Learning perturbation sets for robust machine learning. [[paper]](https://arxiv.org/abs/2007.08450) [[code]](https://github.com/locuslab/perturbation_learning)
  - Eric Wong, J. Zico Kolter.
  - Key Word: Data Augmentation; Generative Model; Adversarial Robustness.
  - <details><summary>Digest</summary> Although much progress has been made towards robust deep learning, a significant gap in robustness remains between real-world perturbations and more narrowly defined sets typically studied in adversarial defenses. In this paper, we aim to bridge this gap by learning perturbation sets from data, in order to characterize real-world effects for robust training and evaluation.  

- Understanding Adversarial Examples from the Mutual Influence of Images and Perturbations. [[paper]](https://arxiv.org/abs/2007.06189)
  - Chaoning Zhang, Philipp Benz, Tooba Imtiaz, In-So Kweon. *CVPR 2020*
  - Key Word: Universal Adversarial Perturbations.
  - <details><summary>Digest</summary> Our results suggest a new perspective towards the relationship between images and universal perturbations: Universal perturbations contain dominant features, and images behave like noise to them. We are the first to achieve the challenging task of a targeted universal attack without utilizing original training data. Our approach using a proxy dataset achieves comparable performance to the state-of-the-art baselines which utilize the original training dataset.  

- Miss the Point: Targeted Adversarial Attack on Multiple Landmark Detection. [[paper]](https://arxiv.org/abs/2007.05225) [[code]](https://github.com/qsyao/attack_landmark_detection)
  - Qingsong Yao, Zecheng He, Hu Han, S. Kevin Zhou. *MICCAI 2020*
  - Key Word: Adversarial Attack; Medical Image; Landmark Detection; Healthcare.
  - <details><summary>Digest</summary> This paper is the first to study how fragile a CNN-based model on multiple landmark detection to adversarial perturbations. Specifically, we propose a novel Adaptive Targeted Iterative FGSM (ATI-FGSM) attack against the state-of-the-art models in multiple landmark detection. The attacker can use ATI-FGSM to precisely control the model predictions of arbitrarily selected landmarks, while keeping other stationary landmarks still, by adding imperceptible perturbations to the original image.

- Understanding and Improving Fast Adversarial Training. [[paper]](https://arxiv.org/abs/2007.02617) [[code]](https://github.com/tml-epfl/understanding-fast-adv-training)
  - Maksym Andriushchenko, Nicolas Flammarion.
  - Key Word: Adversarial Training.
  - <details><summary>Digest</summary> We explored the questions of when and why FGSM adversarial training works, and how to improve it by increasing the gradient alignment, and thus the quality of the solution of the inner maximization problem.  

- On Connections between Regularizations for Improving DNN Robustness. [[paper]](https://arxiv.org/abs/2007.02209)
  - Yiwen Guo, Long Chen, Yurong Chen, Changshui Zhang. *TPAMI*
  - Key Word: Adversarial Robustness; Regularizations.
  - <details><summary>Digest</summary> This paper analyzes regularization terms proposed recently for improving the adversarial robustness of deep neural networks (DNNs), from a theoretical point of view.  

- Biologically Inspired Mechanisms for Adversarial Robustness. [[paper]](https://arxiv.org/abs/2006.16427)
  - Manish V. Reddy, Andrzej Banburski, Nishka Pant, Tomaso Poggio.
  - Key Word: Biologically inspired mechanisms; Adversarial Training.
  - <details><summary>Digest</summary> In this work, we investigate the role of two biologically plausible mechanisms in adversarial robustness. We demonstrate that the non-uniform sampling performed by the primate retina and the presence of multiple receptive fields with a range of receptive field sizes at each eccentricity improve the robustness of neural networks to small adversarial perturbations.  

- Proper Network Interpretability Helps Adversarial Robustness in Classification. [[paper]](https://arxiv.org/abs/2006.14748) [[code]](https://github.com/AkhilanB/Proper-Interpretability)
  - Akhilan Boopathy, Sijia Liu, Gaoyuan Zhang, Cynthia Liu, Pin-Yu Chen, Shiyu Chang, Luca Daniel. *ICML 2020*
  - Key Word: Interpretability; Adversarial Robustness.
  - <details><summary>Digest</summary> In this paper, we theoretically show that with a proper measurement of interpretation, it is actually difficult to prevent prediction-evasion adversarial attacks from causing interpretation discrepancy, as confirmed by experiments on MNIST, CIFAR-10 and Restricted ImageNet.  

- Smooth Adversarial Training. [[paper]](https://arxiv.org/abs/2006.14536)
  - Cihang Xie, Mingxing Tan, Boqing Gong, Alan Yuille, Quoc V. Le.
  - Key Word: ReLU; Adversarial Training.
  - <details><summary>Digest</summary> Hence we propose smooth adversarial training (SAT), in which we replace ReLU with its smooth approximations to strengthen adversarial training. The purpose of smooth activation functions in SAT is to allow it to find harder adversarial examples and compute better gradient updates during adversarial training. Compared to standard adversarial training, SAT improves adversarial robustness for "free", i.e., no drop in accuracy and no increase in computational cost.  

- Adversarial Attack Vulnerability of Medical Image Analysis Systems: Unexplored Factors. [[paper]](https://arxiv.org/abs/2006.06356) [[code]](https://github.com/Gerda92/adversarial_transfer_factors)
  - Gerda Bortsova, Cristina González-Gonzalo, Suzanne C. Wetstein, Florian Dubost, Ioannis Katramados, Laurens Hogeweg, Bart Liefers, Bram van Ginneken, Josien P.W. Pluim, Mitko Veta, Clara I. Sánchez, Marleen de Bruijne. *MedIA*
  - Key Word: Adversarial Attack; Medical Image; Healthcare.
  - <details><summary>Digest</summary> We study previously unexplored factors affecting adversarial attack vulnerability of deep learning MedIA systems in three medical domains: ophthalmology, radiology, and pathology. We focus on adversarial black-box settings, in which the attacker does not have full access to the target model and usually uses another model, commonly referred to as surrogate model, to craft adversarial examples. We consider this to be the most realistic scenario for MedIA systems.

- Feature Purification: How Adversarial Training Performs Robust Deep Learning. [[paper]](https://arxiv.org/abs/2005.10190)
  - Zeyuan Allen-Zhu, Yuanzhi Li. *FOCS 2021*
  - Key Word: Adversarial Training; Sparse Coding; Feature Purification.
  - <details><summary>Digest</summary> In this paper, we present a principle that we call Feature Purification, where we show one of the causes of the existence of adversarial examples is the accumulation of certain small dense mixtures in the hidden weights during the training process of a neural network; and more importantly, one of the goals of adversarial training is to remove such mixtures to purify hidden weights. We present both experiments on the CIFAR-10 dataset to illustrate this principle, and a theoretical result proving that for certain natural classification tasks, training a two-layer neural network with ReLU activation using randomly initialized gradient descent indeed satisfies this principle.

- A Causal View on Robustness of Neural Networks. [[paper]](https://arxiv.org/abs/2005.01095)
  - Cheng Zhang, Kun Zhang, Yingzhen Li. *NeurIPS 2020*
  - Key Word: Adversarial Robustness; Causal Learning; Disentangled Representations.
  - <details><summary>Digest</summary> We present a causal view on the robustness of neural networks against input manipulations, which applies not only to traditional classification tasks but also to general measurement data. Based on this view, we design a deep causal manipulation augmented model (deep CAMA) which explicitly models possible manipulations on certain causes leading to changes in the observed effect.  

- No Surprises: Training Robust Lung Nodule Detection for Low-Dose CT Scans by Augmenting with Adversarial Attacks. [[paper]](https://arxiv.org/abs/2003.03824)
  - Siqi Liu, Arnaud Arindra Adiyoso Setio, Florin C. Ghesu, Eli Gibson, Sasa Grbic, Bogdan Georgescu, Dorin Comaniciu. *TMI*
  - Key Word: Adversarial Defense; Medical Image; Healthcare.
  - <details><summary>Digest</summary> We propose to add adversarial synthetic nodules and adversarial attack samples to the training data to improve the generalization and the robustness of the lung nodule detection systems. To generate hard examples of nodules from a differentiable nodule synthesizer, we use projected gradient descent (PGD) to search the latent code within a bounded neighbourhood that would generate nodules to decrease the detector response. To make the network more robust to unanticipated noise perturbations, we use PGD to search for noise patterns that can trigger the network to give over-confident mistakes.

- Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. [[paper]](https://arxiv.org/abs/2003.01690) [[code]](https://github.com/fra31/auto-attack)
  - Francesco Croce, Matthias Hein. *ICML 2020*
  - Key Word: Adversarial Attacks.
  - <details><summary>Digest</summary> In this paper we first propose two extensions of the PGD-attack overcoming failures due to suboptimal step size and problems of the objective function. We then combine our novel attacks with two complementary existing ones to form a parameter-free, computationally affordable and user-independent ensemble of attacks to test adversarial robustness.  

- Overfitting in adversarially robust deep learning. [[paper]](https://arxiv.org/abs/2002.11569) [[code]](https://github.com/locuslab/robust_overfitting)
  - Leslie Rice, Eric Wong, J. Zico Kolter. *ICML 2020*
  - Key Word: Adversarial Training; Overfitting.
  - <details><summary>Digest</summary> In this paper, we empirically study this phenomenon in the setting of adversarially trained deep networks, which are trained to minimize the loss under worst-case adversarial perturbations. We find that overfitting to the training set does in fact harm robust performance to a very large degree in adversarially robust training across multiple datasets (SVHN, CIFAR-10, CIFAR-100, and ImageNet) and perturbation models.  

- Attacks Which Do Not Kill Training Make Adversarial Learning Stronger. [[paper]](https://arxiv.org/abs/2002.11242)
  - Jingfeng Zhang, Xilie Xu, Bo Han, Gang Niu, Lizhen Cui, Masashi Sugiyama, Mohan Kankanhalli. *ICML 2020*
  - Key Word: Adversarial Training.
  - <details><summary>Digest</summary> In this paper, we raise a fundamental question---do we have to trade off natural generalization for adversarial robustness? We argue that adversarial training is to employ confident adversarial data for updating the current model. We propose a novel approach of friendly adversarial training (FAT): rather than employing most adversarial data maximizing the loss, we search for least adversarial (i.e., friendly adversarial) data minimizing the loss, among the adversarial data that are confidently misclassified.  

- The Curious Case of Adversarially Robust Models: More Data Can Help, Double Descend, or Hurt Generalization. [[paper]](https://arxiv.org/abs/2002.11080)
  - Yifei Min, Lin Chen, Amin Karbasi.
  - Key Word: Adversarial Robustness; Deep Double Descend.
  - <details><summary>Digest</summary> In the weak adversary regime, more data improves the generalization of adversarially robust models. In the medium adversary regime, with more training data, the generalization loss exhibits a double descent curve, which implies the existence of an intermediate stage where more training data hurts the generalization. In the strong adversary regime, more data almost immediately causes the generalization error to increase.  

- Understanding and Mitigating the Tradeoff Between Robustness and Accuracy. [[paper]](https://arxiv.org/abs/2002.10716)
  - Aditi Raghunathan, Sang Michael Xie, Fanny Yang, John Duchi, Percy Liang. *ICML 2020*
  - Key Word: Adversarial Robustness.
  - <details><summary>Digest</summary> In this work, we precisely characterize the effect of augmentation on the standard error in linear regression when the optimal linear predictor has zero standard and robust error.  

- A Bayes-Optimal View on Adversarial Examples. [[paper]](https://arxiv.org/abs/2002.08859)
  - Eitan Richardson, Yair Weiss.
  - Key Word: Adversarial Robustness; Bayes-Optimal.
  - <details><summary>Digest</summary> In this paper, we argue for examining adversarial examples from the perspective of Bayes-Optimal classification. We construct realistic image datasets for which the Bayes-Optimal classifier can be efficiently computed and derive analytic conditions on the distributions so that the optimal classifier is either robust or vulnerable.  

- More Data Can Expand the Generalization Gap Between Adversarially Robust and Standard Models. [[paper]](https://arxiv.org/abs/2002.04725)
  - Lin Chen, Yifei Min, Mingrui Zhang, Amin Karbasi. *ICML 2020*
  - Key Word: Adversarial Robustness; Deep Double Descent.
  - <details><summary>Digest</summary> The conventional wisdom is that more training data should shrink the generalization gap between adversarially-trained models and standard models. However, we study the training of robust classifiers for both Gaussian and Bernoulli models under l-inf attacks, and we prove that more data may actually increase this gap.  

- Fundamental Tradeoffs between Invariance and Sensitivity to Adversarial Perturbations. [[paper]](https://arxiv.org/abs/2002.04599) [[code]](https://github.com/ftramer/Excessive-Invariance)
  - Florian Tramèr, Jens Behrmann, Nicholas Carlini, Nicolas Papernot, Jörn-Henrik Jacobsen. *ICML 2020*
  - Key Word: Adversarial Robustness; Invariance; Sensitivity.
  - <details><summary>Digest</summary>  We demonstrate fundamental tradeoffs between these two types of adversarial examples.
    We show that defenses against sensitivity-based attacks actively harm a model's accuracy on invariance-based attacks, and that new approaches are needed to resist both attack types.  

### Evasion Attacks and Defenses: 2019

- Adversarial Training and Provable Defenses: Bridging the Gap. [[paper]](https://openreview.net/forum?id=SJxSDxrKDr) [[code]](https://github.com/eth-sri/colt)
  - Mislav Balunovic, Martin Vechev. *ICLR 2020*
  - Key Word: Adversarial Training; Provable Defenses.
  - <details><summary>Digest</summary> The key idea is to model neural network training as a procedure which includes both, the verifier and the adversary. In every iteration, the verifier aims to certify the network using convex relaxation while the adversary tries to find inputs inside that convex relaxation which cause verification to fail.  

- When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks. [[paper]](https://arxiv.org/abs/1911.10695) [[code]](https://github.com/gmh14/RobNets)
  - Minghao Guo, Yuzhe Yang, Rui Xu, Ziwei Liu, Dahua Lin. *CVPR 2020*
  - Key Word: Neural Achitecture Search; Adversarial Robustness.
  - <details><summary>Digest</summary> In this work, we take an architectural perspective and investigate the patterns of network architectures that are resilient to adversarial attacks. We discover a family of robust architectures (RobNets).  

- Adversarial Machine Learning Phases of Matter. [[paper]](https://arxiv.org/abs/1910.13453)
  - Si Jiang, Sirui Lu, Dong-Ling Deng.
  - Key Word: Adversarial Robustness; Condensed Matter Physics.
  - <details><summary>Digest</summary> We find that some important underlying physical principles and symmetries remain to be adequately captured for classifiers with even near-perfect performance. This explains why adversarial perturbations exist for fooling these classifiers. In addition, we find that, after adversarial training, the classifiers will become more consistent with physical laws and consequently more robust to certain kinds of adversarial perturbations. Our results provide valuable guidance for both theoretical and experimental future studies on applying machine learning techniques to condensed matter physics.

- Confidence-Calibrated Adversarial Training: Generalizing to Unseen Attacks. [[paper]](https://arxiv.org/abs/1910.06259)
  - David Stutz, Matthias Hein, Bernt Schiele. *ICML 2020*
  - Key Word: Adversarial Training; Out-of-Distribution Detection.
  - <details><summary>Digest</summary> Typically robustness does not generalize to previously unseen threat models, e.g., other Lp norms, or larger perturbations. Our confidence-calibrated adversarial training (CCAT) tackles this problem by biasing the model towards low confidence predictions on adversarial examples. By allowing to reject examples with low confidence, robustness generalizes beyond the threat model employed during training.  

- Lower Bounds on Adversarial Robustness from Optimal Transport. [[paper]](https://arxiv.org/abs/1909.12272) [[code]](https://github.com/inspire-group/robustness-via-transport)
  - Arjun Nitin Bhagoji, Daniel Cullina, Prateek Mittal. *NeurIPS 2019*
  - Key Word: Adversarial Robustness; Optimal Transport.
  - <details><summary>Digest</summary> In this paper, we use optimal transport to characterize the minimum possible loss in an adversarial classification scenario. In this setting, an adversary receives a random labeled example from one of two classes, perturbs the example subject to a neighborhood constraint, and presents the modified example to the classifier.  

- Defending Against Physically Realizable Attacks on Image Classification. [[paper]](https://arxiv.org/abs/1909.09552) [[code]](https://github.com/tongwu2020/phattacks)
  - Tong Wu, Liang Tong, Yevgeniy Vorobeychik. *ICLR 2020*
  - Key Word: Physical Attacks.
  - <details><summary>Digest</summary> First, we demonstrate that the two most scalable and effective methods for learning robust models, adversarial training with PGD attacks and randomized smoothing, exhibit very limited effectiveness against three of the highest profile physical attacks. Next, we propose a new abstract adversarial model, rectangular occlusion attacks, in which an adversary places a small adversarially crafted rectangle in an image, and develop two approaches for efficiently computing the resulting adversarial examples.  

- Biologically inspired sleep algorithm for artificial neural networks. [[paper]](https://arxiv.org/abs/1908.02240)
  - Giri P Krishnan, Timothy Tadros, Ramyaa Ramyaa, Maxim Bazhenov. *ICLR 2020*
  - Key Word: Biologically Inspired Sleep Algorithm.
  - <details><summary>Digest</summary> We provide a theoretical basis for the beneficial role of the brain-inspired sleep-like phase for the ANNs and present an algorithmic way for future implementations of the various features of sleep in deep learning ANNs.  

- Impact of Adversarial Examples on Deep Learning Models for Biomedical Image Segmentation. [[paper]](https://arxiv.org/abs/1907.13124) [[code]](https://github.com/utkuozbulak/adaptive-segmentation-mask-attack)
  - Utku Ozbulak, Arnout Van Messem, Wesley De Neve. *MICCAI 2019*
  - Key Word: Adversarial Attack; Biomedical Image Segmentation; Healthcare.
  - <details><summary>Digest</summary> Given that a large portion of medical imaging problems are effectively segmentation problems, we analyze the impact of adversarial examples on deep learning-based image segmentation models. Specifically, we expose the vulnerability of these models to adversarial examples by proposing the Adaptive Segmentation Mask Attack (ASMA). This novel algorithm makes it possible to craft targeted adversarial examples that come with (1) high intersection-over-union rates between the target adversarial mask and the prediction and (2) with perturbation that is, for the most part, invisible to the bare eye.

- Defense Against Adversarial Attacks Using Feature Scattering-based Adversarial Training. [[paper]](https://arxiv.org/abs/1907.10764) [[code]](https://github.com/Haichao-Zhang/FeatureScatter)
  - Haichao Zhang, Jianyu Wang. *NeurIPS 2019*
  - Key Word: Feature Scattering.
  - <details><summary>Digest</summary> We introduce a feature scattering-based adversarial training approach for improving model robustness against adversarial attacks. Conventional adversarial training approaches leverage a supervised scheme (either targeted or non-targeted) in generating attacks for training, which typically suffer from issues such as label leaking as noted in recent works.  

- Image Synthesis with a Single (Robust) Classifier. [[paper]](https://arxiv.org/abs/1906.09453) [[code]](https://github.com/MadryLab/robustness_applications?)
  - Shibani Santurkar, Dimitris Tsipras, Brandon Tran, Andrew Ilyas, Logan Engstrom, Aleksander Madry. *NeurIPS 2019*
  - Key Word: Adversarial Robustness; Image Synthesis.
  - <details><summary>Digest</summary> The crux of our approach is that we train this classifier to be adversarially robust. It turns out that adversarial robustness is precisely what we need to directly manipulate salient features of the input. Overall, our findings demonstrate the utility of robustness in the broader machine learning context.  

- Convergence of Adversarial Training in Overparametrized Neural Networks. [[paper]](https://arxiv.org/abs/1906.07916)
  - Ruiqi Gao, Tianle Cai, Haochuan Li, Liwei Wang, Cho-Jui Hsieh, Jason D. Lee. *NeurIPS 2019*
  - Key Word: Adversarial Training.
  - <details><summary>Digest</summary>  This paper provides a partial answer to the success of adversarial training, by showing that it converges to a network where the surrogate loss with respect to the the attack algorithm is within theta of the optimal robust loss.  

- Lower Bounds for Adversarially Robust PAC Learning. [[paper]](https://arxiv.org/abs/1906.05815)
  - Dimitrios I. Diochnos, Saeed Mahloujifar, Mohammad Mahmoody. *ISAIM 2020*
  - Key Word: Adversarial Attacks; Poisoning Attacks.
  - <details><summary>Digest</summary> We formalize hybrid attacks in which the evasion attack is preceded by a poisoning attack. This is perhaps reminiscent of "trapdoor attacks" in which a poisoning phase is involved as well, but the evasion phase here uses the error-region definition of risk that aims at misclassifying the perturbed instances.

- Adversarial Robustness as a Prior for Learned Representations. [[paper]](https://arxiv.org/abs/1906.00945) [[code]](https://github.com/MadryLab/robust_representations)
  - Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Brandon Tran, Aleksander Madry.
  - Key Word: Adversarial Robustness; Interpretability; Feature Visualization.
  - <details><summary>Digest</summary> In this work, we show that robust optimization can be re-cast as a tool for enforcing priors on the features learned by deep neural networks. It turns out that representations learned by robust models address the aforementioned shortcomings and make significant progress towards learning a high-level encoding of inputs.  

- Unlabeled Data Improves Adversarial Robustness. [[paper]](https://arxiv.org/abs/1905.13736) [[code]](https://github.com/yaircarmon/semisup-adv)
  - Yair Carmon, Aditi Raghunathan, Ludwig Schmidt, Percy Liang, John C. Duchi. *NeurIPS 2019*
  - Key Word: Adversarial Robustness; Semi-Supervision.
  - <details><summary>Digest</summary> Theoretically, we revisit the simple Gaussian model of [Schmidt et al](https://arxiv.org/abs/1804.11285). that shows a sample complexity gap between standard and robust classification. We prove that unlabeled data bridges this gap: a simple semi-supervised learning procedure (self-training) achieves high robust accuracy using the same number of labels required for achieving high standard accuracy.  

- Are Labels Required for Improving Adversarial Robustness? [[paper]](https://arxiv.org/abs/1905.13725) [[code]](https://github.com/deepmind/deepmind-research/tree/master/unsupervised_adversarial_training)
  - Jonathan Uesato, Jean-Baptiste Alayrac, Po-Sen Huang, Robert Stanforth, Alhussein Fawzi, Pushmeet Kohli. *NeurIPS 2019*
  - Key Word: Unsupervised Adversarial Training.
  - <details><summary>Digest</summary> Our main insight is that unlabeled data can be a competitive alternative to labeled data for training adversarially robust models. Theoretically, we show that in a simple statistical setting, the sample complexity for learning an adversarially robust model from unlabeled data matches the fully supervised case up to constant factors.  

- High Frequency Component Helps Explain the Generalization of Convolutional Neural Networks. [[paper]](https://arxiv.org/abs/1905.13545) [[code]](https://github.com/HaohanWang/HFC)
  - Haohan Wang, Xindi Wu, Zeyi Huang, Eric P. Xing. *CVPR 2020*
  - Key Word: Adversarial Robustness; High-Frequency.
  - <details><summary>Digest</summary> We investigate the relationship between the frequency spectrum of image data and the generalization behavior of convolutional neural networks (CNN). We first notice CNN's ability in capturing the high-frequency components of images. These high-frequency components are almost imperceptible to a human.  

- Interpreting Adversarially Trained Convolutional Neural Networks. [[paper]](https://arxiv.org/abs/1905.09797) [[code]](https://github.com/PKUAI26/AT-CNN)
  - Tianyuan Zhang, Zhanxing Zhu. *ICML 2019*
  - Key Word: Adversarial Robustness; Shape-Texture; Interpretability.
  - <details><summary>Digest</summary> Surprisingly, we find that adversarial training alleviates the texture bias of standard CNNs when trained on object recognition tasks, and helps CNNs learn a more shape-biased representation.  

- Adversarial Examples Are Not Bugs, They Are Features. [[paper]](https://arxiv.org/abs/1905.02175) [[dataset]](https://github.com/MadryLab/constructed-datasets)
  - Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Logan Engstrom, Brandon Tran, Aleksander Madry. *NeurIPS 2019*
  - Key Word: Adversarial Robustness; Interpretability.
  - <details><summary>Digest</summary> We demonstrate that adversarial examples can be directly attributed to the presence of non-robust features: features derived from patterns in the data distribution that are highly predictive, yet brittle and incomprehensible to humans.  

- You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle. [[paper]](https://arxiv.org/abs/1905.00877) [[code]](https://github.com/a1600012888/YOPO-You-Only-Propagate-Once)
  - Dinghuai Zhang, Tianyuan Zhang, Yiping Lu, Zhanxing Zhu, Bin Dong. *NeurIPS 2019*
  - Key Word: Adversarial Training.
  - <details><summary>Digest</summary> In this paper, we show that adversarial training can be cast as a discrete time differential game. Through analyzing the Pontryagin's Maximal Principle (PMP) of the problem, we observe that the adversary update is only coupled with the parameters of the first layer of the network. This inspires us to restrict most of the forward and back propagation within the first layer of the network during adversary updates.  

- NATTACK: Learning the Distributions of Adversarial Examples for an Improved Black-Box Attack on Deep Neural Networks. [[paper]](https://arxiv.org/abs/1905.00441) [[code]](https://github.com/Cold-Winter/Nattack)
  - Yandong Li, Lijun Li, Liqiang Wang, Tong Zhang, Boqing Gong. *ICML 2019*
  - Key Word: Adversarial Attacks.
  - <details><summary>Digest</summary> Instead of searching for an "optimal" adversarial example for a benign input to a targeted DNN, our algorithm finds a probability density distribution over a small region centered around the input, such that a sample drawn from this distribution is likely an adversarial example, without the need of accessing the DNN's internal layers or weights.  

- Non-Local Context Encoder: Robust Biomedical Image Segmentation against Adversarial Attacks. [[paper]](https://arxiv.org/abs/1904.12181)
  - Xiang He, Sibei Yang, Guanbin Li?, Haofeng Li, Huiyou Chang, Yizhou Yu. *AAAI 2019*
  - Key Word: Adversarial Attack; Biomedical Image Segmentation; Healthcare.
  - <details><summary>Digest</summary> We discover that global spatial dependencies and global contextual information in a biomedical image can be exploited to defend against adversarial attacks. To this end, non-local context encoder (NLCE) is proposed to model short- and long range spatial dependencies and encode global contexts for strengthening feature activations by channel-wise attention. The NLCE modules enhance the robustness and accuracy of the non-local context encoding network (NLCEN), which learns robust enhanced pyramid feature representations with NLCE modules, and then integrates the information across different levels.

- Unrestricted Adversarial Examples via Semantic Manipulation. [[paper]](https://arxiv.org/abs/1904.06347) [[code]](https://github.com/AI-secure/Big-but-Invisible-Adversarial-Attack)
  - Anand Bhattad, Min Jin Chong, Kaizhao Liang, Bo Li, D. A. Forsyth. *ICLR 2020*
  - Key Word: Adversarial Attacks; Semantic Adversarial Examples.
  - <details><summary>Digest</summary> In this paper, we instead introduce "unrestricted" perturbations that manipulate semantically meaningful image-based visual descriptors - color and texture - in order to generate effective and photorealistic adversarial examples. We show that these semantically aware perturbations are effective against JPEG compression, feature squeezing and adversarially trained model.

- VC Classes are Adversarially Robustly Learnable, but Only Improperly. [[paper]](https://arxiv.org/abs/1902.04217)
  - Omar Montasser, Steve Hanneke, Nathan Srebro. *COLT 2019*
  - Key Word: Adversarial Robustness; PAC Learning.
  - <details><summary>Digest</summary> We study the question of learning an adversarially robust predictor. We show that any hypothesis class H with finite VC dimension is robustly PAC learnable with an improper learning rule. The requirement of being improper is necessary as we exhibit examples of hypothesis classes H with finite VC dimension that are not robustly PAC learnable with any proper learning rule.

- Certified Adversarial Robustness via Randomized Smoothing. [[paper]](https://arxiv.org/abs/1902.02918) [[code]](https://github.com/locuslab/smoothing)
  - Jeremy M Cohen, Elan Rosenfeld, J. Zico Kolter. *ICML 2019*
  - Key Word: Certified Adversarial Robustness.
  - <details><summary>Digest</summary> We show how to turn any classifier that classifies well under Gaussian noise into a new classifier that is certifiably robust to adversarial perturbations under the L-2 norm. This "randomized smoothing" technique has been proposed recently in the literature, but existing guarantees are loose. We prove a tight robustness guarantee in L-2 norm for smoothing with Gaussian noise.  

- Theoretically Principled Trade-off between Robustness and Accuracy. [[paper]](https://arxiv.org/abs/1901.08573) [[code]](https://github.com/yaodongyu/TRADES)
  - Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, Michael I. Jordan. *ICML 2019*
  - Key Word: Adversarial Training.
  - <details><summary>Digest</summary> In this work, we decompose the prediction error for adversarial examples (robust error) as the sum of the natural (classification) error and boundary error, and provide a differentiable upper bound using the theory of classification-calibrated loss, which is shown to be the tightest possible upper bound uniform over all probability distributions and measurable predictors. Inspired by our theoretical analysis, we also design a new defense method, TRADES, to trade adversarial robustness off against accuracy.  

### Evasion Attacks and Defenses: 2018

- Feature Denoising for Improving Adversarial Robustness. [[paper]](https://arxiv.org/abs/1812.03411) [[code]](https://github.com/facebookresearch/ImageNet-Adversarial-Training)
  - Cihang Xie, Yuxin Wu, Laurens van der Maaten, Alan Yuille, Kaiming He. *CVPR 2019*
  - Key Word: Adversarial Robustness; Pixel Denoising.
  - <details><summary>Digest</summary> This study suggests that adversarial perturbations on images lead to noise in the features constructed by these networks. Motivated by this observation, we develop new network architectures that increase adversarial robustness by performing feature denoising. Specifically, our networks contain blocks that denoise the features using non-local means or other filters; the entire networks are trained end-to-end.  

- Excessive Invariance Causes Adversarial Vulnerability. [[paper]](https://arxiv.org/abs/1811.00401)
  - Jörn-Henrik Jacobsen, Jens Behrmann, Richard Zemel, Matthias Bethge. *ICLR 2019*
  - Key Word: Adversarial Robustness; Information Theory.
  - <details><summary>Digest</summary> We show deep networks are not only too sensitive to task-irrelevant changes of their input, as is well-known from epsilon-adversarial examples, but are also too invariant to a wide range of task-relevant changes, thus making vast regions in input space vulnerable to adversarial attacks. We show such excessive invariance occurs across various tasks and architecture types.  

- Rademacher Complexity for Adversarially Robust Generalization. [[paper]](https://arxiv.org/abs/1810.11914)
  - Dong Yin, Kannan Ramchandran, Peter Bartlett. *ICML 2019*
  - Key Word: Adversarial Robustness; Rademacher Complexity.
  - <details><summary>Digest</summary> In this paper, we focus on L-inf attacks, and study the adversarially robust generalization problem through the lens of Rademacher complexity. For binary linear classifiers, we prove tight bounds for the adversarial Rademacher complexity, and show that the adversarial Rademacher complexity is never smaller than its natural counterpart, and it has an unavoidable dimension dependence, unless the weight vector has bounded L-1 norm.  

- Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples. [[paper]](https://arxiv.org/abs/1810.11580)
  - Guanhong Tao, Shiqing Ma, Yingqi Liu, Xiangyu Zhang. *NeurIPS 2018*
  - Key Word: Adversarial Examples Detection; Interpretability.
  - <details><summary>Digest</summary> We argue that adversarial sample attacks are deeply entangled with interpretability of DNN models: while classification results on benign inputs can be reasoned based on the human perceptible features/attributes, results on adversarial samples can hardly be explained. Therefore, we propose a novel adversarial sample detection technique for face recognition models, based on interpretability.  

- Sparse DNNs with Improved Adversarial Robustness. [[paper]](https://arxiv.org/abs/1810.09619)
  - Yiwen Guo, Chao Zhang, Changshui Zhang, Yurong Chen.
  - Key Word: Adversarial Robustness; Sparsity.
  - <details><summary>Digest</summary> Our analyses reveal, both theoretically and empirically, that nonlinear DNN-based classifiers behave differently under l2 attacks from some linear ones. We further demonstrate that an appropriately higher model sparsity implies better robustness of nonlinear DNNs, whereas over-sparsified models can be more difficult to resist adversarial examples.  

- Adversarial Reprogramming of Neural Networks. [[paper]](https://arxiv.org/abs/1806.11146)
  - Gamaleldin F. Elsayed, Ian Goodfellow, Jascha Sohl-Dickstein. *ICLR 2019*
  - Key Word: Adversarial Examples; Transfer Learning.
  - <details><summary>Digest</summary>  We introduce attacks that instead reprogram the target model to perform a task chosen by the attacker without the attacker needing to specify or compute the desired output for each test-time input. This attack finds a single adversarial perturbation, that can be added to all test-time inputs to a machine learning model in order to cause the model to perform a task chosen by the adversary—even if the model was not trained to do this task.  

- PAC-learning in the presence of evasion adversaries. [[paper]](https://arxiv.org/abs/1806.01471)
  - Daniel Cullina, Arjun Nitin Bhagoji, Prateek Mittal. *NeurIPS 2018*
  - Key Word: Adversarial Robustness; PAC Learning.
  - <details><summary>Digest</summary> In this paper, we step away from the attack-defense arms race and seek to understand the limits of what can be learned in the presence of an evasion adversary. In particular, we extend the Probably Approximately Correct (PAC)-learning framework to account for the presence of an adversary.  

- Robustness May Be at Odds with Accuracy. [[paper]](https://arxiv.org/abs/1805.12152)
  - Dimitris Tsipras, Shibani Santurkar, Logan Engstrom, Alexander Turner, Aleksander Madry. *ICLR 2019*
  - Key Word: Adversarial Robustness.
  - <details><summary>Digest</summary> We show that there may exist an inherent tension between the goal of adversarial robustness and that of standard generalization. Specifically, training robust models may not only be more resource-consuming, but also lead to a reduction of standard accuracy.  

- Adversarial examples from computational constraints. [[paper]](https://arxiv.org/abs/1805.10204)
  - Sébastien Bubeck, Eric Price, Ilya Razenshteyn. *ICML 2019*
  - Key Word: Adversarial Robustness.
  - <details><summary>Digest</summary> First we prove that, for a broad set of classification tasks, the mere existence of a robust classifier implies that it can be found by a possibly exponential-time algorithm with relatively few training examples. Then we give a particular classification task where learning a robust classifier is computationally intractable.  

- Towards the first adversarially robust neural network model on MNIST. [[paper]](https://arxiv.org/abs/1805.09190)
  - Lukas Schott, Jonas Rauber, Matthias Bethge, Wieland Brendel. *ICLR 2019*
  - Key Word: Adversarial Robustness.
  - <details><summary>Digest</summary> We present a novel robust classification model that performs analysis by synthesis using learned class-conditional data distributions. We demonstrate that most adversarial examples are strongly perturbed towards the perceptual boundary between the original and the adversarial class.

- Adversarially Robust Generalization Requires More Data. [[paper]](https://arxiv.org/abs/1804.11285)
  - Ludwig Schmidt, Shibani Santurkar, Dimitris Tsipras, Kunal Talwar, Aleksander Mądry. *NeurIPS 2018*
  - Key Word: Adversarial Robustness.
  - <details><summary>Digest</summary> We show that already in a simple natural data model, the sample complexity of robust learning can be significantly larger than that of "standard" learning. This gap is information theoretic and holds irrespective of the training algorithm or the model family.  

- Generalizability vs. Robustness: Adversarial Examples for Medical Imaging. [[paper]](https://arxiv.org/abs/1804.00504)
  - Magdalini Paschali, Sailesh Conjeti, Fernando Navarro, Nassir Navab. *MICCAI 2018*
  - Key Word: Adversarial Attack; Medical Image; Healthcare.
  - <details><summary>Digest</summary> In this paper, for the first time, we propose an evaluation method for deep learning models that assesses the performance of a model not only in an unseen test scenario, but also in extreme cases of noise, outliers and ambiguous input data. To this end, we utilize adversarial examples, images that fool machine learning models, while looking imperceptibly different from original data, as a measure to evaluate the robustness of a variety of medical imaging models.

- Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples. [[paper]](https://arxiv.org/abs/1802.00420) [[code]](https://github.com/anishathalye/obfuscated-gradients)
  - Anish Athalye, Nicholas Carlini, David Wagner. *ICML 2018*
  - Key Word: Adversarial Robustness.
  - <details><summary>Digest</summary> We identify obfuscated gradients, a kind of gradient masking, as a phenomenon that leads to a false sense of security in defenses against adversarial examples. While defenses that cause obfuscated gradients appear to defeat iterative optimization-based attacks, we find defenses relying on this effect can be circumvented.  

- Stochastic Activation Pruning for Robust Adversarial Defense. [[paper]](https://arxiv.org/abs/1803.01442) [[code]](https://github.com/Guneet-Dhillon/Stochastic-Activation-Pruning)
  - Guneet S. Dhillon, Kamyar Azizzadenesheli, Zachary C. Lipton, Jeremy Bernstein, Jean Kossaifi, Aran Khanna, Anima Anandkumar. *ICLR 2018*
  - Key Word: Adversarial Robustness; Activation Function.
  - <details><summary>Digest</summary> We propose Stochastic Activation Pruning (SAP), a mixed strategy for adversarial defense. SAP prunes a random subset of activations (preferentially pruning those with smaller magnitude) and scales up the survivors to compensate.  

- Adversarial vulnerability for any classifier. [[paper]](https://arxiv.org/abs/1802.08686)
  - Alhussein Fawzi, Hamza Fawzi, Omar Fawzi. *NeurIPS 2018*
  - Key Word: Adversarial Examples.
  - <details><summary>Digest</summary> In this paper, we study the phenomenon of adversarial perturbations under the assumption that the data is generated with a smooth generative model. We derive fundamental upper bounds on the robustness to perturbations of any classification function, and prove the existence of adversarial perturbations that transfer well across different classifiers with small risk.  

- Identify Susceptible Locations in Medical Records via Adversarial Attacks on Deep Predictive Models. [[paper]](https://arxiv.org/abs/1802.04822)
  - Mengying Sun, Fengyi Tang, Jinfeng Yi, Fei Wang, Jiayu Zhou. *KDD 2018*
  - Key Word: Adversarial Attack; Medical Record; Healthcare.
  - <details><summary>Digest</summary> We propose an efficient and effective framework that learns a time-preferential minimum attack targeting the LSTM model with EHR inputs, and we leverage this attack strategy to screen medical records of patients and identify susceptible events and measurements.

### Evasion Attacks and Defenses: 2017

- Certifying Some Distributional Robustness with Principled Adversarial Training. [[paper]](https://arxiv.org/abs/1710.10571)
  - Aman Sinha, Hongseok Namkoong, Riccardo Volpi, John Duchi. *ICLR 2018*
  - Key Word: Certificated Adversarial Robustness.
  - <details><summary>Digest</summary> By considering a Lagrangian penalty formulation of perturbing the underlying data distribution in a Wasserstein ball, we provide a training procedure that augments model parameter updates with worst-case perturbations of training data. For smooth losses, our procedure provably achieves moderate levels of robustness with little computational or statistical cost relative to empirical risk minimization.  

- One pixel attack for fooling deep neural networks. [[paper]](https://arxiv.org/abs/1710.08864)
  - Jiawei Su, Danilo Vasconcellos Vargas, Sakurai Kouichi. *IEEE Transactions on Evolutionary Computation*
  - Key Word: Adversarial Attacks.
  - <details><summary>Digest</summary> In this paper, we analyze an attack in an extremely limited scenario where only one pixel can be modified. For that we propose a novel method for generating one-pixel adversarial perturbations based on differential evolution (DE). It requires less adversarial information (a black-box attack) and can fool more types of networks due to the inherent features of DE.  

### Evasion Attacks and Defenses: 2016

- Practical Black-Box Attacks against Machine Learning. [[paper]](https://arxiv.org/abs/1602.02697)
  - Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z. Berkay Celik, Ananthram Swami. *AsiaCCS 2017*
  - Key Word: Adversarial Attacks.
  - <details><summary>Digest</summary> We introduce the first practical demonstration of an attacker controlling a remotely hosted DNN with no such knowledge. Our attack evades a category of defenses, which we call gradient masking, previously proposed to increase resilience to adversarial examples.  

## Poisoning Attacks and Defenses

### Poisoning Attacks and Defenses: 2021

- FIBA: Frequency-Injection based Backdoor Attack in Medical Image Analysis. [[paper]](https://arxiv.org/abs/2112.01148) [[code]](https://github.com/hazardfy/fiba)
  - Yu Feng, Benteng Ma, Jing Zhang, Shanshan Zhao, Yong Xia, Dacheng Tao. *CVPR 2022*
  - Key Word: Backdoor Attack; Medical Image; Healthcare.
  - <details><summary>Digest</summary> Most existing BA methods are designed to attack natural image classification models, which apply spatial triggers to training images and inevitably corrupt the semantics of poisoned pixels, leading to the failures of attacking dense prediction models. To address this issue, we propose a novel Frequency-Injection based Backdoor Attack method (FIBA) that is capable of delivering attacks in various MIA tasks.

- AEVA: Black-box Backdoor Detection Using Adversarial Extreme Value Analysis. [[paper]](https://arxiv.org/abs/2110.14880) [[code]](https://github.com/junfenggo/aeva-blackbox-backdoor-detection-main)
  - Junfeng Guo, Ang Li, Cong Liu. *ICLR 2022*
  - Key Word: Backdoor Detection.
  - <details><summary>Digest</summary> We address the black-box hard-label backdoor detection problem where the DNN is fully black-box and only its final output label is accessible. We approach this problem from the optimization perspective and show that the objective of backdoor detection is bounded by an adversarial objective.

- Adversarial Unlearning of Backdoors via Implicit Hypergradient. [[paper]](https://arxiv.org/abs/2110.03735) [[code]](https://github.com/yizeng623/i-bau_adversarial_unlearning_of-backdoors_via_implicit_hypergradient)
  - Yi Zeng, Si Chen, Won Park, Z. Morley Mao, Ming Jin, Ruoxi Jia. *ICLR 2022*
  - Key Word: Backdoor Defenses; Backdoor Removal; Adversarial Unlearning.
  - <details><summary>Digest</summary> We propose a minimax formulation for removing backdoors from a given poisoned model based on a small set of clean data. This formulation encompasses much of prior work on backdoor removal. We propose the Implicit Backdoor Adversarial Unlearning (I-BAU) algorithm to solve the minimax. Unlike previous work, which breaks down the minimax into separate inner and outer problems, our algorithm utilizes the implicit hypergradient to account for the interdependence between inner and outer optimization.

- BadPre: Task-agnostic Backdoor Attacks to Pre-trained NLP Foundation Models. [[paper]](https://arxiv.org/abs/2110.02467)
  - Kangjie Chen, Yuxian Meng, Xiaofei Sun, Shangwei Guo, Tianwei Zhang, Jiwei Li, Chun Fan. *ICLR 2022*
  - Key Word: Pre-trained Natural Language Processing Models; Backdoor Attacks.
  - <details><summary>Digest</summary> In this work, we propose BadPre, the first backdoor attack against various downstream models built based on pre-trained NLP models. BadPre can launch trojan attacks against different language tasks with the same trigger. The key insight of our approach is that downstream models can inherit the security characteristics from the pre-trained models.

- How to Inject Backdoors with Better Consistency: Logit Anchoring on Clean Data. [[paper]](https://arxiv.org/abs/2109.01300)
  - Zhiyuan Zhang, Lingjuan Lyu, Weiqiang Wang, Lichao Sun, Xu Sun. *ICLR 2022*
  - Key Word: Backdoor Attacks; Weight Perturbations.
  - <details><summary>Digest</summary> Previous work finds that backdoors can be injected into a trained clean model with Adversarial Weight Perturbation (AWP), which means the variation of parameters are small in backdoor learning. In this work, we observe an interesting phenomenon that the variations of parameters are always AWPs when tuning the trained clean model to inject backdoors. We further provide theoretical analysis to explain this phenomenon. We are the first to formulate the behavior of maintaining accuracy on clean data as the consistency of backdoored models, which includes both global consistency and instance-wise consistency.

- Data Poisoning Won't Save You From Facial Recognition. [[paper]](https://arxiv.org/abs/2106.14851) [[code]](https://github.com/ftramer/facecure)
  - Evani Radiya-Dixit, Sanghyun Hong, Nicholas Carlini, Florian Tramèr. *ICLR 2022*
  - Key Word: Poisoning Attacks; Adversarial Examples; Face Recognition.
  - <details><summary>Digest</summary> We demonstrate that this strategy provides a false sense of security, as it ignores an inherent asymmetry between the parties: users' pictures are perturbed once and for all before being published (at which point they are scraped) and must thereafter fool all future models---including models trained adaptively against the users' past attacks, or models that use new technologies discovered after the attack.

- Poisoning and Backdooring Contrastive Learning. [[paper]](https://arxiv.org/abs/2106.09667)
  - Nicholas Carlini, Andreas Terzis. *ICLR 2022*
  - Key Word: Contrastive Learning; Poisoning Attacks; Backdoor Attacks.
  - <details><summary>Digest</summary> Multimodal contrastive learning methods like CLIP train on noisy and uncurated training datasets. This is cheaper than labeling datasets manually, and even improves out-of-distribution robustness. We show that this practice makes backdoor and poisoning attacks a significant threat. By poisoning just 0.01% of a dataset (e.g., just 300 images of the 3 million-example Conceptual Captions dataset), we can cause the model to misclassify test images by overlaying a small patch.

- Backdoor Attacks on Self-Supervised Learning. [[paper]](https://arxiv.org/abs/2105.10123) [[code]](https://github.com/UMBCvision/SSL-Backdoor)
  - Aniruddha Saha, Ajinkya Tejankar, Soroush Abbasi Koohpayegani, Hamed Pirsiavash.  *CVPR 2022*
  - Key Word: Self-Supervised Learning; Backdoor Attacks.
  - <details><summary>Digest</summary> State-of-the-art self-supervised methods for learning representations from images (e.g., MoCo, BYOL, MSF) use an inductive bias that random augmentations (e.g., random crops) of an image should produce similar embeddings. We show that such methods are vulnerable to backdoor attacks - where an attacker poisons a small part of the unlabeled data by adding a trigger (image patch chosen by the attacker) to the images. The model performance is good on clean test images, but the attacker can manipulate the decision of the model by showing the trigger at test time

- A Backdoor Attack against 3D Point Cloud Classifiers. [[paper]](https://arxiv.org/abs/2104.05808)
  - Zhen Xiang, David J. Miller, Siheng Chen, Xi Li, George Kesidis. *ICCV 2021*
  - Key Word:Backdoor Attack; 3D Point Cloud.
  - <details><summary>Digest</summary> We propose the first backdoor attack (BA) against PC classifiers.  Different from image BAs, we propose to insert a cluster of points into a PC as a robust backdoor pattern customized for 3D PCs. Such clusters are also consistent with a physical attack (i.e., with a captured object in a scene).

- Rethinking the Backdoor Attacks' Triggers: A Frequency Perspective. [[paper]](https://arxiv.org/abs/2104.03413) [[code]](https://github.com/YiZeng623/frequency-backdoor)
  - Yi Zeng, Won Park, Z. Morley Mao, Ruoxi Jia. *ICCV 2021*
  - Key Word: Backdoor; Explanation.
  - <details><summary>Digest</summary> This paper first revisits existing backdoor triggers from a frequency perspective and performs a comprehensive analysis. Our results show that many current backdoor attacks exhibit severe high-frequency artifacts, which persist across different datasets and resolutions. In short, our work emphasizes the importance of considering frequency analysis when designing both backdoor attacks and defenses in deep learning.

- Black-Box Detection of Backdoor Attacks With Limited Information and Data. [[paper]](https://arxiv.org/abs/2103.13127)
  - Yinpeng Dong, Xiao Yang, Zhijie Deng, Tianyu Pang, Zihao Xiao, Hang Su, Jun Zhu. *ICCV 2021*
  - Key Word: Black-Box Backdoor Detection; Backdoor Defense.
  - <details><summary>Digest</summary>  In this paper, we propose a black-box backdoor detection (B3D) method to identify backdoor attacks with only query access to the model. We introduce a gradient-free optimization algorithm to reverse-engineer the potential trigger for each class, which helps to reveal the existence of backdoor attacks. In addition to backdoor detection, we also propose a simple strategy for reliable predictions using the identified backdoored models.

- PointBA: Towards Backdoor Attacks in 3D Point Cloud. [[paper]](https://arxiv.org/abs/2103.16074)
  - Xinke Li, Zhirui Chen, Yue Zhao, Zekun Tong, Yabang Zhao, Andrew Lim, Joey Tianyi Zhou. *ICCV 2021*
  - Key Word: Backdorr Attack; 3D Point Cloud.
  - <details><summary>Digest</summary> We present the backdoor attacks in 3D point cloud with a unified framework that exploits the unique properties of 3D data and networks.

- LIRA: Learnable, Imperceptible and Robust Backdoor Attacks. [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Doan_LIRA_Learnable_Imperceptible_and_Robust_Backdoor_Attacks_ICCV_2021_paper.pdf) [[code]](https://github.com/khoadoan106/backdoor_attacks)
  - Khoa Doan, Yingjie Lao, Weijie Zhao, Ping Li. *ICCV 2021*
  - Key Word: Backdoor Attack.
  - <details><summary>Digest</summary> The trigger injection function is manually defined in most existing backdoor attack methods, e.g., placing a small patch of pixels on an image or slightly deforming the image before poisoning the model. This results in a two-stage approach with a sub-optimal attack success rate and a lack of complete stealthiness under human inspection. In this paper, we propose a novel and stealthy backdoor attack framework, LIRA, which jointly learns the optimal, stealthy trigger injection function and poisons the model. We formulate such an objective as a non-convex, constrained optimization problem. Under this optimization framework, the trigger generator function will learn to manipulate the input with imperceptible noise to preserve the model performance on the clean data and maximize the attack success rate on the poisoned data.

- CLEAR: Clean-Up Sample-Targeted Backdoor in Neural Networks. [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhu_CLEAR_Clean-Up_Sample-Targeted_Backdoor_in_Neural_Networks_ICCV_2021_paper.pdf)
  - Liuwan Zhu, Rui Ning, Chunsheng Xin, Chonggang Wang, Hongyi Wu. *ICCV 2021*
  - Key Word: Backdoor Defense; Backdoor Detection.
  - <details><summary>Digest</summary> We propose a novel scheme to detect and mitigate sample-targeted backdoor attacks. We discover and demonstrate a unique property of the sample-targeted backdoor, which forces a boundary change such that small "pockets" are formed around the target sample. Based on this observation, we propose a novel defense mechanism to pinpoint a malicious pocket by "wrapping" them into a tight convex hull in the feature space. We design an effective algorithm to search for such a convex hull and remove the backdoor by fine-tuning the model using the identified malicious samples with the corrected label according to the convex hull.

### Poisoning Attacks and Defenses: 2020

- Invisible Backdoor Attack with Sample-Specific Triggers. [[paper]](https://arxiv.org/abs/2012.03816) [[code]](https://github.com/yuezunli/issba)
  - Yuezun Li, Yiming Li, Baoyuan Wu, Longkang Li, Ran He, Siwei Lyu. *ICCV 2021*
  - Key Word: Backdoor Attacks; Invisible Triggers; Sample-Specific Triggers.
  - <details><summary>Digest</summary>  We explore a novel attack paradigm, where backdoor triggers are sample-specific. In our attack, we only need to modify certain training samples with invisible perturbation, while not need to manipulate other training components (e.g., training loss, and model structure) as required in many existing attacks.

- One-pixel Signature: Characterizing CNN Models for Backdoor Detection. [[paper]](https://arxiv.org/abs/2008.07711)
  - Shanjiaoyang Huang, Weiqi Peng, Zhiwei Jia, Zhuowen Tu. *ECCV 2020*
  - Key Word: Backdoor Detection.
  - <details><summary>Digest</summary> We tackle the convolution neural networks (CNNs) backdoor detection problem by proposing a new representation called one-pixel signature. Our task is to detect/classify if a CNN model has been maliciously inserted with an unknown Trojan trigger or not. Here, each CNN model is associated with a signature that is created by generating, pixel-by-pixel, an adversarial value that is the result of the largest change to the class prediction. The one-pixel signature is agnostic to the design choice of CNN architectures, and how they were trained. It can be computed efficiently for a black-box CNN model without accessing the network parameters.

- Backdoor Learning: A Survey. [[paper]](https://arxiv.org/abs/2007.08745) [[code]](https://github.com/THUYimingLi/backdoor-learning-resources)
  - Yiming Li, Yong Jiang, Zhifeng Li, Shu-Tao Xia.
  - Key Word: Backdoor Learning; Survey.
  - <details><summary>Digest</summary> We present the first comprehensive survey of this realm. We summarize and categorize existing backdoor attacks and defenses based on their characteristics, and provide a unified framework for analyzing poisoning-based backdoor attacks. Besides, we also analyze the relation between backdoor attacks and relevant fields (i.e., adversarial attacks and data poisoning), and summarize widely adopted benchmark datasets. Finally, we briefly outline certain future research directions relying upon reviewed works.

- Backdoor Attacks and Countermeasures on Deep Learning: A Comprehensive Review. [[paper]](https://arxiv.org/abs/2007.10760)
  - Yansong Gao, Bao Gia Doan, Zhi Zhang, Siqi Ma, Jiliang Zhang, Anmin Fu, Surya Nepal, Hyoungshick Kim.
  - Key Word: Backdoor Attacks and Defenses; Survey.
  - <details><summary>Digest</summary> We review countermeasures, and compare and analyze their advantages and disadvantages. We have also reviewed the flip side of backdoor attacks, which are explored for i) protecting intellectual property of deep learning models, ii) acting as a honeypot to catch adversarial example attacks, and iii) verifying data deletion requested by the data contributor.Overall, the research on defense is far behind the attack, and there is no single defense that can prevent all types of backdoor attacks. In some cases, an attacker can intelligently bypass existing defenses with an adaptive attack.

- Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks. [[paper]](https://arxiv.org/abs/2007.02343)
  - Yunfei Liu, Xingjun Ma, James Bailey, Feng Lu. *ECCV 2020*
  - Key Word: Backdoor Attacks; Natural Reflection.
  - <details><summary>Digest</summary> In this paper, we present a new type of backdoor attack inspired by an important natural phenomenon: reflection. Using mathematical modeling of physical reflection models, we propose reflection backdoor (Refool) to plant reflections as backdoor into a victim model.  

- Backdoor Attacks Against Deep Learning Systems in the Physical World. [[paper]](https://arxiv.org/abs/2006.14580)
  - Emily Wenger, Josephine Passananti, Arjun Bhagoji, Yuanshun Yao, Haitao Zheng, Ben Y. Zhao. *CVPR 2021*
  - Key Word: Backdoor Attacks; Physical Attacks; Facial Recognition.
  - <details><summary>Digest</summary> Can backdoor attacks succeed using physical objects as triggers, thus making them a credible threat against deep learning systems in the real world? We conduct a detailed empirical study to explore this question for facial recognition, a critical deep learning task. Using seven physical objects as triggers, we collect a custom dataset of 3205 images of ten volunteers and use it to study the feasibility of physical backdoor attacks under a variety of real-world conditions.

- Blind Backdoors in Deep Learning Models. [[paper]](https://arxiv.org/abs/2005.03823) [[code]](https://github.com/ebagdasa/backdoors101)
  - Eugene Bagdasaryan, Vitaly Shmatikov. *USENIX 2021*
  - Key Word: Backdoor Attacks; Blind Backdoors.
  - <details><summary>Digest</summary> Our attack is blind: the attacker cannot modify the training data, nor observe the execution of his code, nor access the resulting model. The attack code creates poisoned training inputs "on the fly," as the model is training, and uses multi-objective optimization to achieve high accuracy on both the main and backdoor tasks. We show how a blind attack can evade any known defense and propose new ones.

- Clean-Label Backdoor Attacks on Video Recognition Models. [[paper]](https://arxiv.org/abs/2003.03030) [[code](https://github.com/ShihaoZhaoZSH/Video-Backdoor-Attack)]
  - Shihao Zhao, Xingjun Ma, Xiang Zheng, James Bailey, Jingjing Chen, Yu-Gang Jiang. *CVPR 2020*
  - Key Word: Backdoor Attacks; Video Recognition.
  - <details><summary>Digest</summary> We show that existing image backdoor attacks are far less effective on videos, and outline 4 strict conditions where existing attacks are likely to fail: 1) scenarios with more input dimensions (eg. videos), 2) scenarios with high resolution, 3) scenarios with a large number of classes and few examples per class (a "sparse dataset"), and 4) attacks with access to correct labels (eg. clean-label attacks). We propose the use of a universal adversarial trigger as the backdoor trigger to attack video recognition models, a situation where backdoor attacks are likely to be challenged by the above 4 strict conditions.

### Poisoning Attacks and Defenses: 2019

- Universal Litmus Patterns: Revealing Backdoor Attacks in CNNs. [[paper]](https://arxiv.org/abs/1906.10842) [[code]](https://github.com/UMBCvision/Universal-Litmus-Patterns)
  - Soheil Kolouri, Aniruddha Saha, Hamed Pirsiavash, Heiko Hoffmann. *CVPR 2020*
  - Key Word: Backdoor Detection.
  - <details><summary>Digest</summary> We introduce a benchmark technique for detecting backdoor attacks (aka Trojan attacks) on deep convolutional neural networks (CNNs). We introduce the concept of Universal Litmus Patterns (ULPs), which enable one to reveal backdoor attacks by feeding these universal patterns to the network and analyzing the output (i.e., classifying the network as `clean' or `corrupted'). This detection is fast because it requires only a few forward passes through a CNN.

### Poisoning Attacks and Defenses: 2017

- BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain. [[paper]](https://arxiv.org/abs/1708.06733)
  - Tianyu Gu, Brendan Dolan-Gavitt, Siddharth Garg.
  - Key Word: Backdoor Attacks; Neuron Activation.
  - <details><summary>Digest</summary>  In this paper we show that outsourced training introduces new security risks: an adversary can create a maliciously trained network (a backdoored neural network, or a BadNet that has state-of-the-art performance on the user's training and validation samples, but behaves badly on specific attacker-chosen inputs.  

## Privacy

### Privacy: 2021

- Watermarking Deep Neural Networks with Greedy Residuals. [[paper]](http://proceedings.mlr.press/v139/liu21x.html) [[code]](https://github.com/eil/greedy-residuals)
  - Hanwen Liu, Zhenyu Weng, Yuesheng Zhu. *ICML 2021*
  - Key Word: Ownership Indicators.
  - <details><summary>Digest</summary> We propose a novel watermark-based ownership protection method by using the residuals of important parameters. Different from other watermark-based ownership protection methods that rely on some specific neural network architectures and during verification require external data source, namely ownership indicators, our method does not explicitly use ownership indicators for verification to defeat various attacks against DNN watermarks.

- When Does Data Augmentation Help With Membership Inference Attacks? [[paper]](https://proceedings.mlr.press/v139/kaya21a.html) [[code]](https://github.com/yigitcankaya/augmentation_mia)
  - Yigitcan Kaya, Tudor Dumitras. *ICML 2021*
  - Key Word: Membership Inference Attacks; Data Augmentation.
  - <details><summary>Digest</summary> While many mechanisms exist, their effectiveness against MIAs and privacy properties have not been studied systematically. Employing two recent MIAs, we explore the lower bound on the risk in the absence of formal upper bounds. First, we evaluate 7 mechanisms and differential privacy, on three image classification tasks. We find that applying augmentation to increase the model’s utility does not mitigate the risk and protection comes with a utility penalty.

- Counterfactual Memorization in Neural Language Models. [[paper]](https://arxiv.org/abs/2112.12938)
  - Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr, Nicholas Carlini.
  - Key Word: Counterfactual Memorization; Label Memorization; Influence Function.
  - <details><summary>Digest</summary> An open question in previous studies of memorization in language models is how to filter out "common" memorization. In fact, most memorization criteria strongly correlate with the number of occurrences in the training set, capturing "common" memorization such as familiar phrases, public knowledge or templated texts. In this paper, we provide a principled perspective inspired by a taxonomy of human memory in Psychology. From this perspective, we formulate a notion of counterfactual memorization, which characterizes how a model's predictions change if a particular document is omitted during training. We identify and study counterfactually-memorized training examples in standard text datasets. We further estimate the influence of each training example on the validation set and on generated texts, and show that this can provide direct evidence of the source of memorization at test time.

- Membership Inference Attacks From First Principles. [[paper]](https://arxiv.org/abs/2112.03570) [[code]](https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021)
  - Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramer.
  - Key Word: Membership Inference Attacks; Theory of Memorization.
  - <details><summary>Digest</summary> A membership inference attack allows an adversary to query a trained machine learning model to predict whether or not a particular example was contained in the model's training dataset. These attacks are currently evaluated using average-case "accuracy" metrics that fail to characterize whether the attack can confidently identify any members of the training set. We argue that attacks should instead be evaluated by computing their true-positive rate at low (e.g., less than 0.1%) false-positive rates, and find most prior attacks perform poorly when evaluated in this way. To address this we develop a Likelihood Ratio Attack (LiRA) that carefully combines multiple ideas from the literature. Our attack is 10x more powerful at low false-positive rates, and also strictly dominates prior attacks on existing metrics.

- Personalized Federated Learning with Adaptive Batchnorm for Healthcare. [[paper]](https://arxiv.org/abs/2112.00734)
  - Wang Lu, Jindong Wang, Yiqiang Chen, Xin Qin, Renjun Xu, Dimitrios Dimitriadis, and Tao Qin. *IEEE TBD 2022*
  - Key Word: Federated learning; Batch normalization; Personalized FL
  - <details><summary>Digest</summary> We propose FedAP to tackle domain shifts and then obtain personalized models for local clients. FedAP learns the similarity between clients based on the statistics of the batch normalization layers while preserving the specificity of each client with different local batch normalization.

- Evaluating Gradient Inversion Attacks and Defenses in Federated Learning. [[paper]](https://arxiv.org/abs/2112.00059) [[code]](https://github.com/Princeton-SysML/GradAttack)
  - Yangsibo Huang, Samyak Gupta, Zhao Song, Kai Li, Sanjeev Arora. *NeurIPS 2021*
  - Key Word: Gradient Inversion Attacks and Defenses.
  - <details><summary>Digest</summary> Gradient inversion attack (or input recovery from gradient) is an emerging threat to the security and privacy preservation of Federated learning, whereby malicious eavesdroppers or participants in the protocol can recover (partially) the clients' private data. This paper evaluates existing attacks and defenses. We find that some attacks make strong assumptions about the setup. Relaxing such assumptions can substantially weaken these attacks. We then evaluate the benefits of three proposed defense mechanisms against gradient inversion attacks.

- On the Importance of Difficulty Calibration in Membership Inference Attacks. [[paper]](https://arxiv.org/abs/2111.08440) [[code]](https://github.com/facebookresearch/calibration_membership)
  - Lauren Watson, Chuan Guo, Graham Cormode, Alex Sablayrolles. *ICLR 2022*
  - Key Word: Membership Inference Attack; Privacy.
  - <details><summary>Digest</summary> We argue that membership inference attacks can benefit drastically from difficulty calibration, where an attack's predicted membership score is adjusted to the difficulty of correctly classifying the target sample. We show that difficulty calibration can significantly reduce the false positive rate of a variety of existing attacks without a loss in accuracy.

- You are caught stealing my winning lottery ticket! Making a lottery ticket claim its ownership. [[paper]](https://arxiv.org/abs/2111.00162) [[code]](https://github.com/VITA-Group/NO-stealing-LTH)
  - Xuxi Chen, Tianlong Chen, Zhenyu Zhang, Zhangyang Wang. *NeurIPS 2021*
  - Key Word: Ownership Verification; Lottery Ticket Hypothesis.
  - <details><summary>Digest</summary> The main resource bottleneck of LTH is however the extraordinary cost to find the sparse mask of the winning ticket. That makes the found winning ticket become a valuable asset to the owners, highlighting the necessity of protecting its copyright. Our setting adds a new dimension to the recently soaring interest in protecting against the intellectual property (IP) infringement of deep models and verifying their ownerships, since they take owners' massive/unique resources to develop or train. While existing methods explored encrypted weights or predictions, we investigate a unique way to leverage sparse topological information to perform lottery verification, by developing several graph-based signatures that can be embedded as credentials.

- Gradient Inversion with Generative Image Prior. [[paper]](https://arxiv.org/abs/2110.14962) [[code]](https://github.com/ml-postech/gradient-inversion-generative-image-prior)
  - Jinwoo Jeon, Jaechang Kim, Kangwook Lee, Sewoong Oh, Jungseul Ok. *NeurIPS 2021*
  - Key Word: Federated Learning, Privacy Leakage, Gradient Inversion.
  - <details><summary>Digest</summary> By exploiting a generative model pretrained on the data distribution, we demonstrate that data privacy can be easily breached. Further, when such prior knowledge is unavailable, we investigate the possibility of learning the prior from a sequence of gradients seen in the process of FL training.

- What Do We Mean by Generalization in Federated Learning? [[paper]](https://arxiv.org/abs/2110.14216) [[code]](https://github.com/google-research/federated/tree/master/generalization)
  - Honglin Yuan, Warren Morningstar, Lin Ning, Karan Singhal. *ICLR 2022*
  - Key Word: Federated Learning.
  - <details><summary>Digest</summary> We propose a framework for disentangling these performance gaps. Using this framework, we observe and explain differences in behavior across natural and synthetic federated datasets, indicating that dataset synthesis strategy can be important for realistic simulations of generalization in federated learning. We propose a semantic synthesis strategy that enables realistic simulation without naturally partitioned data.

- Large Language Models Can Be Strong Differentially Private Learners. [[paper]](https://arxiv.org/abs/2110.05679) [[code]](https://github.com/lxuechen/private-transformers)
  - Xuechen Li, Florian Tramèr, Percy Liang, Tatsunori Hashimoto. *ICLR 2022*
  - Key Word: Language Model; Differential Privacy.
  - <details><summary>Digest</summary> To address the computational challenge of running DP-SGD with large Transformers, we propose a memory saving technique that allows clipping in DP-SGD to run without instantiating per-example gradients for any linear layer in the model. The technique enables privately training Transformers with almost the same memory cost as non-private training at a modest run-time overhead.

- Designing Counterfactual Generators using Deep Model Inversion. [[paper]](https://arxiv.org/abs/2109.14274)
  - Jayaraman J. Thiagarajan, Vivek Narayanaswamy, Deepta Rajan, Jason Liang, Akshay Chaudhari, Andreas Spanias. *NeurIPS 2021*
  - Key Word: Model Inversion; Counterfactual Generation.
  - <details><summary>Digest</summary> We focus on the case where we have access only to the trained deep classifier and not the actual training data. We propose DISC (Deep Inversion for Synthesizing Counterfactuals) that improves upon deep inversion by utilizing (a) stronger image priors, (b) incorporating a novel manifold consistency objective and (c) adopting a progressive optimization strategy.

- EMA: Auditing Data Removal from Trained Models. [[paper]](https://arxiv.org/abs/2109.03675) [[code]](https://github.com/Hazelsuko07/EMA)
  - Yangsibo Huang, Xiaoxiao Li, Kai Li. *MICCAI 2021*
  - Key Word: Privacy; Data Auditing.
  - <details><summary>Digest</summary> we propose a new method called Ensembled Membership Auditing (EMA) for auditing data removal to overcome these limitations. We compare both methods using benchmark datasets (MNIST and SVHN) and Chest X-ray datasets with multi-layer perceptrons (MLP) and convolutional neural networks (CNN). Our experiments show that EMA is robust under various conditions, including the failure cases of the previously proposed method.

- Dataset Inference: Ownership Resolution in Machine Learning. [[paper]](https://arxiv.org/abs/2104.10706) [[code]](https://github.com/cleverhans-lab/dataset-inference)
  - Pratyush Maini, Mohammad Yaghini, Nicolas Papernot. *ICLR 2021*
  - Key Word: Model Ownership; Model Extraction.
  - <details><summary>Digest</summary> We introduce dataset inference, the process of identifying whether a suspected model copy has private knowledge from the original model's dataset, as a defense against model stealing. We develop an approach for dataset inference that combines statistical testing with the ability to estimate the distance of multiple data points to the decision boundary.

- A Review of Anonymization for Healthcare Data. [[paper]](https://arxiv.org/abs/2104.06523) [[code]](https://github.com/iyempissy/anonymization-reconstruction-attack)
  - Iyiola E. Olatunji, Jens Rauch, Matthias Katzensteiner, Megha Khosla.
  - Key Word: Anonymization Reconstruction Attacks; Healthcare.
  - <details><summary>Digest</summary> We review the existing anonymization techniques and their applicability to various types (relational and graph-based) of health data. Besides, we provide an overview of possible attacks on anonymized data. We illustrate via a reconstruction attack that anonymization though necessary, is not sufficient to address patient privacy and discuss methods for protecting against such attacks. Finally, we discuss tools that can be used to achieve anonymization.

- Membership Inference Attacks on Machine Learning: A Survey. [[paper]](https://arxiv.org/abs/2103.07853)
  - Hongsheng Hu, Zoran Salcic, Lichao Sun, Gillian Dobbie, Philip S. Yu, Xuyun Zhang. *ACM Computing Surveys*
  - Key Word: Survey; Membership Inference Attacks.
  - <details><summary>Digest</summary> We conduct the first comprehensive survey on membership inference attacks and defenses. We provide the taxonomies for both attacks and defenses, based on their characterizations, and discuss their pros and cons. Based on the limitations and gaps identified in this survey, we point out several promising future research directions to inspire the researchers who wish to follow this area.

- Measuring Data Leakage in Machine-Learning Models with Fisher Information. [[paper]](https://arxiv.org/abs/2102.11673) [[code]](https://github.com/facebookresearch/fisher_information_loss)
  - Awni Hannun, Chuan Guo, Laurens van der Maaten. *UAI 2021*
  - Key Word: Attribute Inference Attacks; Fisher Information.
  - <details><summary>Digest</summary> We propose a method to quantify this leakage using the Fisher information of the model about the data. Unlike the worst-case a priori guarantees of differential privacy, Fisher information loss measures leakage with respect to specific examples, attributes, or sub-populations within the dataset. We motivate Fisher information loss through the Cramér-Rao bound and delineate the implied threat model. We provide efficient methods to compute Fisher information loss for output-perturbed generalized linear models. 

- Membership Inference Attacks are Easier on Difficult Problems. [[paper]](https://arxiv.org/abs/2102.07762) [[code]](https://github.com/avitalsh/reconst_based_MIA)
  - Avital Shafran, Shmuel Peleg, Yedid Hoshen. *ICCV 2021*
  - Key Word: Membership Inference Attacks; Semantic Segmentation; Medical Imaging; Heathcare.
  - <details><summary>Digest</summary> Membership inference attacks (MIA) try to detect if data samples were used to train a neural network model, e.g. to detect copyright abuses. We show that models with higher dimensional input and output are more vulnerable to MIA, and address in more detail models for image translation and semantic segmentation, including medical image segmentation. We show that reconstruction-errors can lead to very effective MIA attacks as they are indicative of memorization. Unfortunately, reconstruction error alone is less effective at discriminating between non-predictable images used in training and easy to predict images that were never seen before. To overcome this, we propose using a novel predictability error that can be computed for each sample, and its computation does not require a training set.

- CaPC Learning: Confidential and Private Collaborative Learning. [[paper]](https://arxiv.org/abs/2102.05188) [[code]](https://github.com/cleverhans-lab/capc-iclr)
  - Christopher A. Choquette-Choo, Natalie Dullerud, Adam Dziedzic, Yunxiang Zhang, Somesh Jha, Nicolas Papernot, Xiao Wang. *ICLR 2021*
  - Key Word: Differential Privacy; Secure Multi-Party Computation.
  - <details><summary>Digest</summary> We introduce Confidential and Private Collaborative (CaPC) learning, the first method provably achieving both confidentiality and privacy in a collaborative setting. We leverage secure multi-party computation (MPC), homomorphic encryption (HE), and other techniques in combination with privately aggregated teacher models. We demonstrate how CaPC allows participants to collaborate without having to explicitly join their training sets or train a central model. 

### Privacy: 2020

- Illuminating the dark spaces of healthcare with ambient intelligence. [[paper]](https://www.nature.com/articles/s41586-020-2669-y)
  - Albert Haque, Arnold Milstein, Li Fei-Fei. *Nature*
  - Key Word: Healthcare; Privacy.
  - <details><summary>Digest</summary> Advances in machine learning and contactless sensors have given rise to ambient intelligence—physical spaces that are sensitive and responsive to the presence of humans. Here we review how this technology could improve our understanding of the metaphorically dark, unobserved spaces of healthcare. In hospital spaces, early applications could soon enable more efficient clinical workflows and improved patient safety in intensive care units and operating rooms. In daily living spaces, ambient intelligence could prolong the independence of older individuals and improve the management of individuals with a chronic disease by understanding everyday behaviour. Similar to other technologies, transformation into clinical applications at scale must overcome challenges such as rigorous clinical validation, appropriate data privacy and model transparency.

- UDH: Universal Deep Hiding for Steganography, Watermarking, and Light Field Messaging. [[paper]](https://papers.nips.cc/paper/2020/hash/73d02e4344f71a0b0d51a925246990e7-Abstract.html) [[code]](https://github.com/ChaoningZhang/Universal-Deep-Hiding)
  - Chaoning Zhang, Philipp Benz, Adil Karjauv, Geng Sun, In So Kweon. *NeurIPS 2020*
  - Key Word: Watermarking.
  - <details><summary>Digest</summary> Neural networks have been shown effective in deep steganography for hiding a full image in another. However, the reason for its success remains not fully clear. Under the existing cover (C) dependent deep hiding (DDH) pipeline, it is challenging to analyze how the secret (S) image is encoded since the encoded message cannot be analyzed independently. We propose a novel universal deep hiding (UDH) meta-architecture to disentangle the encoding of S from C. We perform extensive analysis and demonstrate that the success of deep steganography can be attributed to a frequency discrepancy between C and the encoded secret image.

- Extracting Training Data from Large Language Models. [[paper]](https://arxiv.org/abs/2012.07805) [[code]](https://github.com/ftramer/LM_Memorization)
  - Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, Alina Oprea, Colin Raffel. *USENIX Security 2021*
  - Key Word: Reconstruction Attacks; Large Language Models.
  - <details><summary>Digest</summary> We demonstrate our attack on GPT-2, a language model trained on scrapes of the public Internet, and are able to extract hundreds of verbatim text sequences from the model's training data. These extracted examples include (public) personally identifiable information (names, phone numbers, and email addresses), IRC conversations, code, and 128-bit UUIDs. Our attack is possible even though each of the above sequences are included in just one document in the training data.

- Label-Only Membership Inference Attacks. [[paper]](https://arxiv.org/abs/2007.14321) [[code]](https://github.com/cchoquette/membership-inference)
  - Christopher A. Choquette-Choo, Florian Tramer, Nicholas Carlini, Nicolas Papernot. *ICML 2021*
  - Key Word: Membership Inference Attacks; Data Augmentation; Adversarial Examples.
  - <details><summary>Digest</summary> We introduce label-only membership inference attacks. Instead of relying on confidence scores, our attacks evaluate the robustness of a model's predicted labels under perturbations to obtain a fine-grained membership signal. These perturbations include common data augmentations or adversarial examples.

- Anonymizing Machine Learning Models. [[paper]](https://arxiv.org/abs/2007.13086) [[code]](https://github.com/IBM/ai-privacy-toolkit)
  - Abigail Goldsteen, Gilad Ezov, Ron Shmelkin, Micha Moffie, Ariel Farkash.
  - Key Word: Anonymization.
  - <details><summary>Digest</summary> Learning on anonymized data typically results in significant degradation in accuracy. In this work, we propose a method that is able to achieve better model accuracy by using the knowledge encoded within the trained model, and guiding our anonymization process to minimize the impact on the model's accuracy, a process we call accuracy-guided anonymization. We demonstrate that by focusing on the model's accuracy rather than generic information loss measures, our method outperforms state of the art k-anonymity methods in terms of the achieved utility, in particular with high values of k and large numbers of quasi-identifiers.

- Descent-to-Delete: Gradient-Based Methods for Machine Unlearning. [[paper]](https://arxiv.org/abs/2007.02923)
  - Seth Neel, Aaron Roth, Saeed Sharifi-Malvajerdi.
  - Key Word: Machine Unlearning.
  - <details><summary>Digest</summary> We study the data deletion problem for convex models. By leveraging techniques from convex optimization and reservoir sampling, we give the first data deletion algorithms that are able to handle an arbitrarily long sequence of adversarial updates while promising both per-deletion run-time and steady-state error that do not grow with the length of the update sequence.  

- When Machine Unlearning Jeopardizes Privacy. [[paper]](https://arxiv.org/abs/2005.02205) [[code]](https://github.com/MinChen00/UnlearningLeaks)
  - Min Chen, Zhikun Zhang, Tianhao Wang, Michael Backes, Mathias Humbert, Yang Zhang. *CCS 2021*
  - Key Word: Machine Unlearning.
  - <details><summary>Digest</summary> In this paper, we perform the first study on investigating the unintended information leakage caused by machine unlearning. We propose a novel membership inference attack which leverages the different outputs of an ML model's two versions to infer whether the deleted sample is part of the training set.  

- A Framework for Evaluating Gradient Leakage Attacks in Federated Learning. [[paper]](https://arxiv.org/abs/2004.10397)
  - Wenqi Wei, Ling Liu, Margaret Loper, Ka-Ho Chow, Mehmet Emre Gursoy, Stacey Truex, Yanzhao Wu.
  - Key Word: Gradient Inversion Attacks.
  - <details><summary>Digest</summary> In this paper, we present a principled framework for evaluating and comparing different forms of client privacy leakage attacks. We first provide formal and experimental analysis to show how adversaries can reconstruct the private local training data by simply analyzing the shared parameter update from local training (e.g., local gradient or weight update vector).  

- Inverting Gradients -- How easy is it to break privacy in federated learning? [[paper]](https://arxiv.org/abs/2003.14053) [[code]](https://github.com/JonasGeiping/invertinggradients)
  - Jonas Geiping, Hartmut Bauermeister, Hannah Dröge, Michael Moeller.
  - Key Word: Gradient Inversion Attacks.
  - <details><summary>Digest</summary> In this paper we show that sharing parameter gradients is by no means secure: By exploiting a cosine similarity loss along with optimization methods from adversarial attacks, we are able to faithfully reconstruct images at high resolution from the knowledge of their parameter gradients, and demonstrate that such a break of privacy is possible even for trained deep networks.  

- Forgetting Outside the Box: Scrubbing Deep Networks of Information Accessible from Input-Output Observations. [[paper]](https://arxiv.org/abs/2003.02960) [[code]](https://github.com/AdityaGolatkar/SelectiveForgetting)
  - Aditya Golatkar, Alessandro Achille, Stefano Soatto. *ECCV 2020*
  - Key Word: Machine Unlearning; Neural Tangent Kernel; Information Theory.
  - <details><summary>Digest</summary> We introduce a new bound on how much information can be extracted per query about the forgotten cohort from a black-box network for which only the input-output behavior is observed. The proposed forgetting procedure has a deterministic part derived from the differential equations of a linearized version of the model, and a stochastic part that ensures information destruction by adding noise tailored to the geometry of the loss landscape. We exploit the connections between the activation and weight dynamics of a DNN inspired by Neural Tangent Kernels to compute the information in the activations.

- Anonymizing Data for Privacy-Preserving Federated Learning. [[paper]](https://arxiv.org/abs/2002.09096)
  - Olivia Choudhury, Aris Gkoulalas-Divanis, Theodoros Salonidis, Issa Sylla, Yoonyoung Park, Grace Hsu, Amar Das. *ECAI 2020*
  - Key Word: Data Anonymization; Federated Learning.
  - <details><summary>Digest</summary> We propose the first syntactic approach for offering privacy in the context of federated learning. Unlike the state-of-the-art differential privacy-based frameworks, our approach aims to maximize utility or model performance, while supporting a defensible level of privacy, as demanded by GDPR and HIPAA. We perform a comprehensive empirical evaluation on two important problems in the healthcare domain, using real-world electronic health data of 1 million patients. The results demonstrate the effectiveness of our approach in achieving high model performance, while offering the desired level of privacy.

- iDLG: Improved Deep Leakage from Gradients. [[paper]](https://arxiv.org/abs/2001.02610) [[code]](https://github.com/PatrickZH/Improved-Deep-Leakage-from-Gradients)
  - Bo Zhao, Konda Reddy Mopuri, Hakan Bilen.
  - Key Word: Gradient Inversion Attacks.
  - <details><summary>Digest</summary> DLG has difficulty in convergence and discovering the ground-truth labels consistently. In this paper, we find that sharing gradients definitely leaks the ground-truth labels. We propose a simple but reliable approach to extract accurate data from the gradients.  

### Privacy: 2019

- Machine Unlearning. [[paper]](https://arxiv.org/abs/1912.03817) [[code]](https://github.com/cleverhans-lab/machine-unlearning)
  - Lucas Bourtoule, Varun Chandrasekaran, Christopher A. Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, Nicolas Papernot.
  - Key Word: Machine Unlearning.
  - <details><summary>Digest</summary>  We introduce SISA training, a framework that expedites the unlearning process by strategically limiting the influence of a data point in the training procedure. While our framework is applicable to any learning algorithm, it is designed to achieve the largest improvements for stateful algorithms like stochastic gradient descent for deep neural networks.  

- Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks. [[paper]](https://arxiv.org/abs/1911.04933)
  - Aditya Golatkar, Alessandro Achille, Stefano Soatto. *CVPR 2020*
  - Key Word: Machine Unlearning.
  - <details><summary>Digest</summary> We propose a method for "scrubbing'" the weights clean of information about a particular set of training data. The method does not require retraining from scratch, nor access to the data originally used for training. Instead, the weights are modified so that any probing function of the weights is indistinguishable from the same function applied to the weights of a network trained without the data to be forgotten.  

- Alleviating Privacy Attacks via Causal Learning. [[paper]](https://arxiv.org/abs/1909.12732) [[code]](https://github.com/microsoft/robustdg)
  - Shruti Tople, Amit Sharma, Aditya Nori. *ICML 2020*
  - Key Word: Membership Inversion Attacks.
  - <details><summary>Digest</summary> To alleviate privacy attacks, we demonstrate the benefit of predictive models that are based on the causal relationships between input features and the outcome. We first show that models learnt using causal structure generalize better to unseen data, especially on data from different distributions than the train distribution.  

- Rethinking Deep Neural Network Ownership Verification: Embedding Passports to Defeat Ambiguity Attacks. [[paper]](https://arxiv.org/abs/1909.07830) [[code]](https://github.com/kamwoh/DeepIPR)
  - Lixin Fan, Kam Woh Ng, Chee Seng Chan. *NeurIPS 2019*
  - Key Work: Neural Network Watermarking.
  - <details><summary>Digest</summary> This work proposes novel passport-based DNN ownership verification schemes which are both robust to network modifications and resilient to ambiguity attacks. The gist of embedding digital passports is to design and train DNN models in a way such that, the DNN model performance of an original task will be significantly deteriorated due to forged passports.  

- Making AI Forget You: Data Deletion in Machine Learning. [[paper]](https://arxiv.org/abs/1907.05012) [[code]](https://github.com/tginart/deletion-efficient-kmeans)
  - Antonio Ginart, Melody Y. Guan, Gregory Valiant, James Zou. *NeurIPS 2019*
  - Key Word: Data Deletion; Clustering.
  - <details><summary>Digest</summary> We formulate the problem of efficiently deleting individual data points from trained machine learning models. For many standard ML models, the only way to completely remove an individual's data is to retrain the whole model from scratch on the remaining data, which is often not computationally practical. We investigate algorithmic principles that enable efficient data deletion in ML. For the specific setting of k-means clustering, we propose two provably efficient deletion algorithms which achieve an average of over 100X improvement in deletion efficiency across 6 datasets, while producing clusters of comparable statistical quality to a canonical k-means++ baseline.

- Deep Leakage from Gradients. [[paper]](https://arxiv.org/abs/1906.08935) [[code]](https://github.com/mit-han-lab/dlg)
  - Ligeng Zhu, Zhijian Liu, Song Han. *NeurIPS 2019*
  - Key Word: Gradient Inversion Attacks.
  - <details><summary>Digest</summary> We show that it is possible to obtain the private training data from the publicly shared gradients. We name this leakage as Deep Leakage from Gradient and empirically validate the effectiveness on both computer vision and natural language processing tasks.  

- P3SGD: Patient Privacy Preserving SGD for Regularizing Deep CNNs in Pathological Image Classification. [[paper]](https://arxiv.org/abs/1905.12883)  
  - Bingzhe Wu, Shiwan Zhao, Guangyu Sun, Xiaolu Zhang, Zhong Su, Caihong Zeng, Zhihong Liu. *CVPR 2019*
  - Key Word: Privacy; Pathological Image Classification; Healthcare.
  - <details><summary>Digest</summary> We introduce a novel stochastic gradient descent (SGD) scheme, named patient privacy preserving SGD (P3SGD), which performs the model update of the SGD in the patient level via a large-step update built upon each patient's data. Specifically, to protect privacy and regularize the CNN model, we propose to inject the well-designed noise into the updates.

## Fairness

### Fairness: 2021

- Fair Normalizing Flows. [[paper]](https://arxiv.org/abs/2106.05937) [[code]](https://github.com/eth-sri/fnf)
  - Mislav Balunović, Anian Ruoss, Martin Vechev. *ICLR 2022*
  - Key Word: Fairness; Normalizing Flows.
  - <details><summary>Digest</summary> We present Fair Normalizing Flows (FNF), a new approach offering more rigorous fairness guarantees for learned representations. Specifically, we consider a practical setting where we can estimate the probability density for sensitive groups. The key idea is to model the encoder as a normalizing flow trained to minimize the statistical distance between the latent representations of different groups.

### Fairness: 2020

- Fairness in the Eyes of the Data: Certifying Machine-Learning Models. [[paper]](https://arxiv.org/abs/2009.01534)
  - Shahar Segal, Yossi Adi, Benny Pinkas, Carsten Baum, Chaya Ganesh, Joseph Keshet.
  - Key Word: Fairness; Privacy.
  - <details><summary>Digest</summary> We present a framework that allows to certify the fairness degree of a model based on an interactive and privacy-preserving test. The framework verifies any trained model, regardless of its training process and architecture. Thus, it allows us to evaluate any deep learning model on multiple fairness definitions empirically.  

### Fairness: 2019

- Training individually fair ML models with Sensitive Subspace Robustness. [[paper]](https://arxiv.org/abs/1907.00020) [[code]](https://github.com/IBM/sensitive-subspace-robustness)
  - Mikhail Yurochkin, Amanda Bower, Yuekai Sun. *ICLR 2020*
  - Key Word: Distributionally Robust Optimization.
  - <details><summary>Digest</summary> We consider training machine learning models that are fair in the sense that their performance is invariant under certain sensitive perturbations to the inputs. For example, the performance of a resume screening system should be invariant under changes to the gender and/or ethnicity of the applicant. We formalize this notion of algorithmic fairness as a variant of individual fairness and develop a distributionally robust optimization approach to enforce it during training.  

## Interpretability

### Interpretability: 2021

- Understanding the Dynamics of DNNs Using Graph Modularity. [[paper]](https://arxiv.org/abs/2111.12485) [[code]](https://github.com/yaolu-zjut/dynamic-graphs-construction)
  - Yao Lu, Wen Yang, Yunzhe Zhang, Zuohui Chen, Jinyin Chen, Qi Xuan, Zhen Wang, Xiaoniu Yang. *ECCV 2022*
  - Key Word: Interpretability; Modularity; Layer Pruning.
  - <details><summary>Digest</summary> There are good arguments to support the claim that deep neural networks (DNNs) capture better feature representations than the previous hand-crafted feature engineering, which leads to a significant performance improvement. In this paper, we move a tiny step towards understanding the dynamics of feature representations over layers. Specifically, we model the process of class separation of intermediate representations in pre-trained DNNs as the evolution of communities in dynamic graphs. Then, we introduce modularity, a generic metric in graph theory, to quantify the evolution of communities. In the preliminary experiment, we find that modularity roughly tends to increase as the layer goes deeper and the degradation and plateau arise when the model complexity is great relative to the dataset.

- Salient ImageNet: How to discover spurious features in Deep Learning? [[paper]](https://arxiv.org/abs/2110.04301) [[code]](https://github.com/singlasahil14/salient_imagenet)
  - Sahil Singla, Soheil Feizi. *ICLR 2022*
  - Key Word: Interpretability; Robustness; Failure Explanations.
  - <details><summary>Digest</summary> Our methodology is based on this key idea: to identify spurious or core visual features used in model predictions, we identify spurious or core neural features (penultimate layer neurons of a robust model) via limited human supervision (e.g., using top 5 activating images per feature). We then show that these neural feature annotations generalize extremely well to many more images without any human supervision.

- Consistent Counterfactuals for Deep Models. [[paper]](https://arxiv.org/abs/2110.03109)
  - Emily Black, Zifan Wang, Matt Fredrikson, Anupam Datta. *ICLR 2022*
  - Key Word: Explainability; Counterfactual Explanations.
  - <details><summary>Digest</summary> This paper studies the consistency of model prediction on counterfactual examples in deep networks under small changes to initial training conditions, such as weight initialization and leave-one-out variations in data, as often occurs during model deployment. We demonstrate experimentally that counterfactual examples for deep models are often inconsistent across such small changes, and that increasing the cost of the counterfactual, a stability-enhancing mitigation suggested by prior work in the context of simpler models, is not a reliable heuristic in deep networks.

- FastSHAP: Real-Time Shapley Value Estimation. [[paper]](https://arxiv.org/abs/2107.07436) [[code]](https://github.com/neiljethani/fastshap)
  - Neil Jethani, Mukund Sudarshan, Ian Covert, Su-In Lee, Rajesh Ranganath. *ICLR 2022*
  - Key Word: Explainability; Shapley Values; Game Theory.
  - <details><summary>Digest</summary> Although Shapley values are theoretically appealing for explaining black-box models, they are costly to calculate and thus impractical in settings that involve large, high-dimensional models. To remedy this issue, we introduce FastSHAP, a new method for estimating Shapley values in a single forward pass using a learned explainer model. To enable efficient training without requiring ground truth Shapley values, we develop an approach to train FastSHAP via stochastic gradient descent using a weighted least-squares objective function.

- Meaningfully Debugging Model Mistakes using Conceptual Counterfactual Explanations. [[paper]](https://arxiv.org/abs/2106.12723) [[code]](https://github.com/mertyg/debug-mistakes-cce)  
  - Abubakar Abid, Mert Yuksekgonul, James Zou. *ICML 2022*
  - Key Word: Concept Activation Vectors; Counterfactual Explanations.
  - <details><summary>Digest</summary> We propose a systematic approach, conceptual counterfactual explanations (CCE), that explains why a classifier makes a mistake on a particular test sample(s) in terms of human-understandable concepts (e.g. this zebra is misclassified as a dog because of faint stripes). We base CCE on two prior ideas: counterfactual explanations and concept activation vectors, and validate our approach on well-known pretrained models, showing that it explains the models' mistakes meaningfully.

- DISSECT: Disentangled Simultaneous Explanations via Concept Traversals. [[paper]](https://arxiv.org/abs/2105.15164) [[code]](https://github.com/asmadotgh/dissect)
  - Asma Ghandeharioun, Been Kim, Chun-Liang Li, Brendan Jou, Brian Eoff, Rosalind W. Picard. *ICLR 2022*
  - Key Word: Interpretability; Counterfactual Generation.
  - <details><summary>Digest</summary>  We propose a novel approach, DISSECT, that jointly trains a generator, a discriminator, and a concept disentangler to overcome such challenges using little supervision. DISSECT generates Concept Traversals (CTs), defined as a sequence of generated examples with increasing degrees of concepts that influence a classifier's decision. By training a generative model from a classifier's signal, DISSECT offers a way to discover a classifier's inherent "notion" of distinct concepts automatically rather than rely on user-predefined concepts.

- Axiomatic Explanations for Visual Search, Retrieval, and Similarity Learning. [[paper]](https://arxiv.org/abs/2103.00370)
  - Mark Hamilton, Scott Lundberg, Lei Zhang, Stephanie Fu, William T. Freeman. *ICLR 2022*
  - Key Word: Interpretability; Shapley Values; Similarity Learning.
  - <details><summary>Digest</summary> We show that the theory of fair credit assignment provides a unique axiomatic solution that generalizes several existing recommendation- and metric-explainability techniques in the literature. Using this formalism, we show when existing approaches violate "fairness" and derive methods that sidestep these shortcomings and naturally handle counterfactual information.

### Interpretability: 2020

- Transformer Interpretability Beyond Attention Visualization. [[paper]](https://arxiv.org/abs/2012.09838) [[code]](https://github.com/hila-chefer/Transformer-Explainability)
  - Hila Chefer, Shir Gur, Lior Wolf. *CVPR 2021*
  - Key Word: Transformers; Explainability.
  - <details><summary>Digest</summary> We propose a novel way to compute relevancy for Transformer networks. The method assigns local relevance based on the Deep Taylor Decomposition principle and then propagates these relevancy scores through the layers. This propagation involves attention layers and skip connections, which challenge existing methods.

- Understanding and Diagnosing Vulnerability under Adversarial Attacks. [[paper]](https://arxiv.org/abs/2007.08716)
  - Haizhong Zheng, Ziqi Zhang, Honglak Lee, Atul Prakash.
  - Key Word: Generative Adversarial Nets; Adversarial Attacks.
  - <details><summary>Digest</summary> In this work, we propose a novel interpretability method, InterpretGAN, to generate explanations for features used for classification in latent variables. Interpreting the classification process of adversarial examples exposes how adversarial perturbations influence features layer by layer as well as which features are modified by perturbations.  

- Rethinking the Role of Gradient-Based Attribution Methods for Model Interpretability. [[paper]](https://arxiv.org/abs/2006.09128)
  - Suraj Srinivas, Francois Fleuret. *ICLR 2021*
  - Key Word: Saliency Maps.
  - <details><summary>Digest</summary> We show that these input-gradients can be arbitrarily manipulated as a consequence of the shift-invariance of softmax without changing the discriminative function. This leaves an open question: if input-gradients can be arbitrary, why are they highly structured and explanatory in standard models? We investigate this by re-interpreting the logits of standard softmax-based classifiers as unnormalized log-densities of the data distribution

- Black Box Explanation by Learning Image Exemplars in the Latent Feature Space. [[paper]](https://arxiv.org/abs/2002.03746)
  - Riccardo Guidotti, Anna Monreale, Stan Matwin, Dino Pedreschi.
  - Key Word: Counterfactuals.
  - <details><summary>Digest</summary> The proposed method first generates exemplar images in the latent feature space and learns a decision tree classifier. Then, it selects and decodes exemplars respecting local decision rules. Finally, it visualizes them in a manner that shows to the user how the exemplars can be modified to either stay within their class, or to become counter-factuals by "morphing" into another class.

### Interpretability: 2019

- Attributional Robustness Training using Input-Gradient Spatial Alignment. [[paper]](https://arxiv.org/abs/1911.13073) [[code]](https://github.com/nupurkmr9/Attributional-Robustness)
  - Mayank Singh, Nupur Kumari, Puneet Mangla, Abhishek Sinha, Vineeth N Balasubramanian, Balaji Krishnamurthy. *ECCV 2020*
  - Key Word: Attributional Robustness.
  - <details><summary>Digest</summary> In this work, we study the problem of attributional robustness (i.e. models having robust explanations) by showing an upper bound for attributional vulnerability in terms of spatial correlation between the input image and its explanation map. We propose a training methodology that learns robust features by minimizing this upper bound using soft-margin triplet loss.  

- On Completeness-aware Concept-Based Explanations in Deep Neural Networks. [[paper]](https://arxiv.org/abs/1910.07969) [[code]](https://github.com/chihkuanyeh/concept_exp)
  - Chih-Kuan Yeh, Been Kim, Sercan O. Arik, Chun-Liang Li, Tomas Pfister, Pradeep Ravikumar. *NeurIPS 2020*
  - Key Word: Concept Attribution.
  - <details><summary>Digest</summary> We study such concept-based explainability for Deep Neural Networks (DNNs). First, we define the notion of completeness, which quantifies how sufficient a particular set of concepts is in explaining a model's prediction behavior based on the assumption that complete concept scores are sufficient statistics of the model prediction. Next, we propose a concept discovery method that aims to infer a complete set of concepts that are additionally encouraged to be interpretable, which addresses the limitations of existing methods on concept explanations.

- Smooth Grad-CAM++: An Enhanced Inference Level Visualization Technique for Deep Convolutional Neural Network Models. [[paper]](https://arxiv.org/abs/1908.01224) [[code]](https://github.com/yiskw713/SmoothGradCAMplusplus)
  - Daniel Omeiza, Skyler Speakman, Celia Cintas, Komminist Weldermariam. *IntelliSys 2019*
  - Key Word: Saliency Maps.
  - <details><summary>Digest</summary> With the intention to create an enhanced visual explanation in terms of visual sharpness, object localization and explaining multiple occurrences of objects in a single image, we present Smooth Grad-CAM++, a technique that combines methods from two other recent techniques---SMOOTHGRAD and Grad-CAM++.

- Explaining Classifiers with Causal Concept Effect (CaCE). [[paper]](https://arxiv.org/abs/1907.07165)
  - Yash Goyal, Amir Feder, Uri Shalit, Been Kim.
  - Key Word: Concept Attribution.
  - <details><summary>Digest</summary> We define the Causal Concept Effect (CaCE) as the causal effect of (the presence or absence of) a human-interpretable concept on a deep neural net's predictions. We show that the CaCE measure can avoid errors stemming from confounding. Estimating CaCE is difficult in situations where we cannot easily simulate the do-operator.

- Interpretable Counterfactual Explanations Guided by Prototypes. [[paper]](https://arxiv.org/abs/1907.02584) [[code]](https://github.com/SeldonIO/alibi)
  - Arnaud Van Looveren, Janis Klaise.
  - Key Word: Counterfactuals; Prototypes.
  - <details><summary>Digest</summary> We propose a fast, model agnostic method for finding interpretable counterfactual explanations of classifier predictions by using class prototypes. We show that class prototypes, obtained using either an encoder or through class specific k-d trees, significantly speed up the the search for counterfactual instances and result in more interpretable explanations.

- XRAI: Better Attributions Through Regions. [[paper]](https://arxiv.org/abs/1906.02825) [[code]](https://github.com/PAIR-code/saliency)
  - Andrei Kapishnikov, Tolga Bolukbasi, Fernanda Viégas, Michael Terry.
  - Key Word: Perturbation-based Sanity Check; Saliency Maps.
  - <details><summary>Digest</summary> We 1) present a novel region-based attribution method, XRAI, that builds upon integrated gradients (Sundararajan et al. 2017), 2) introduce evaluation methods for empirically assessing the quality of image-based saliency maps (Performance Information Curves (PICs)), and 3) contribute an axiom-based sanity check for attribution methods.

- Towards Automatic Concept-based Explanations. [[paper]](https://arxiv.org/abs/1902.03129) [[code]](https://github.com/amiratag/ACE)
  - Amirata Ghorbani, James Wexler, James Zou, Been Kim. *NeurIPS 2019*
  - Key Word: Concept Attribution.
  - <details><summary>Digest</summary> We propose principles and desiderata for concept based explanation, which goes beyond per-sample features to identify higher-level human-understandable concepts that apply across the entire dataset. We develop a new algorithm, ACE, to automatically extract visual concepts.

### Interpretability: 2018

- This Looks Like That: Deep Learning for Interpretable Image Recognition. [[paper]](https://arxiv.org/abs/1806.10574) [[code]](https://github.com/cfchen-duke/ProtoPNet)
  - Chaofan Chen, Oscar Li, Chaofan Tao, Alina Jade Barnett, Jonathan Su, Cynthia Rudin. *NeurIPS 2019*
  - Key Word: Prototypes.
  - <details><summary>Digest</summary> We introduce a deep network architecture -- prototypical part network (ProtoPNet), that reasons in a similar way: the network dissects the image by finding prototypical parts, and combines evidence from the prototypes to make a final classification. The model thus reasons in a way that is qualitatively similar to the way ornithologists, physicians, and others would explain to people on how to solve challenging image classification tasks.

- RISE: Randomized Input Sampling for Explanation of Black-box Models. [[paper]](https://arxiv.org/abs/1806.07421) [[code]](https://github.com/eclique/RISE)
  - Vitali Petsiuk, Abir Das, Kate Saenko. *BMVC 2018*
  - Key Word: Saliency Maps.
  - <details><summary>Digest</summary> We propose an approach called RISE that generates an importance map indicating how salient each pixel is for the model's prediction. In contrast to white-box approaches that estimate pixel importance using gradients or other internal network state, RISE works on black-box models. It estimates importance empirically by probing the model with randomly masked versions of the input image and obtaining the corresponding outputs.

- Learning to Explain: An Information-Theoretic Perspective on Model Interpretation. [[paper]](https://arxiv.org/abs/1802.07814) [[code]](https://github.com/Jianbo-Lab/L2X)
  - Jianbo Chen, Le Song, Martin J. Wainwright, Michael I. Jordan. *ICML 2018*
  - Key Word: Counterfactuals; Information Theory.
  - <details><summary>Digest</summary> We introduce instancewise feature selection as a methodology for model interpretation. Our method is based on learning a function to extract a subset of features that are most informative for each given example. This feature selector is trained to maximize the mutual information between selected features and the response variable, where the conditional distribution of the response variable given the input is the model to be explained.

- Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives. [[paper]](https://arxiv.org/abs/1802.07623) [[code]](https://github.com/IBM/Contrastive-Explanation-Method)
  - Amit Dhurandhar, Pin-Yu Chen, Ronny Luss, Chun-Chen Tu, Paishun Ting, Karthikeyan Shanmugam, Payel Das. *NeurIPS 2018*
  - Key Word: Counterfactuals.
  - <details><summary>Digest</summary> We propose a novel method that provides contrastive explanations justifying the classification of an input by a black box classifier such as a deep neural network. Given an input we find what should be %necessarily and minimally and sufficiently present (viz. important object pixels in an image) to justify its classification and analogously what should be minimally and necessarily absent (viz. certain background pixels).

### Interpretability: 2017

- Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV). [[paper]](https://arxiv.org/abs/1711.11279) [[code]](https://github.com/tensorflow/tcav)
  - Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, Rory Sayres. *ICML 2018*
  - Key Word: Concept Attribution.
  - <details><summary>Digest</summary> We introduce Concept Activation Vectors (CAVs), which provide an interpretation of a neural net's internal state in terms of human-friendly concepts. The key idea is to view the high-dimensional internal state of a neural net as an aid, not an obstacle.

- Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks. [[paper]](https://arxiv.org/abs/1710.11063) [[code]](https://github.com/adityac94/Grad_CAM_plus_plus)
  - Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader, Vineeth N Balasubramanian. *WACV 2018*
  - Key Word: Saliency Maps.
  - <details><summary>Digest</summary> Building on a recently proposed method called Grad-CAM, we propose a generalized method called Grad-CAM++ that can provide better visual explanations of CNN model predictions, in terms of better object localization as well as explaining occurrences of multiple object instances in a single image, when compared to state-of-the-art.

- A Unified Approach to Interpreting Model Predictions. [[paper]](https://arxiv.org/abs/1705.07874)
  - Scott Lundberg, Su-In Lee. *NeurIPS 2017*
  - Key Word: Additive Feature Attribution.
  - <details><summary>Digest</summary> We present a unified framework for interpreting predictions, SHAP (SHapley Additive exPlanations). SHAP assigns each feature an importance value for a particular prediction. Its novel components include: (1) the identification of a new class of additive feature importance measures, and (2) theoretical results showing there is a unique solution in this class with a set of desirable properties.

- SmoothGrad: removing noise by adding noise. [[paper]](https://arxiv.org/abs/1706.03825) [[code]](https://github.com/PAIR-code/saliency)
  - Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, Martin Wattenberg.
  - Key Word: Sensitivity Maps.
  - <details><summary>Digest</summary> This paper makes two contributions: it introduces SmoothGrad, a simple method that can help visually sharpen gradient-based sensitivity maps, and it discusses lessons in the visualization of these maps. We publish the code for our experiments and a website with our results.

- Learning Important Features Through Propagating Activation Differences. [[paper]](https://arxiv.org/abs/1704.02685) [[code]](https://github.com/kundajelab/deeplift)
  - Avanti Shrikumar, Peyton Greenside, Anshul Kundaje. *ICML 2017*
  - Key Word: Perturbation-based Explanations; Backpropagation-based Explanations.
  - <details><summary>Digest</summary> The purported "black box" nature of neural networks is a barrier to adoption in applications where interpretability is essential. Here we present DeepLIFT (Deep Learning Important FeaTures), a method for decomposing the output prediction of a neural network on a specific input by backpropagating the contributions of all neurons in the network to every feature of the input.

- Understanding Black-box Predictions via Influence Functions. [[paper]](https://arxiv.org/abs/1703.04730) [[code]](https://worksheets.codalab.org/worksheets/0x2b314dc3536b482dbba02783a24719fd)
  - Pang Wei Koh, Percy Liang. *ICML 2017*
  - Key Word: Influence Functions.
  - <details><summary>Digest</summary> To scale up influence functions to modern machine learning settings, we develop a simple, efficient implementation that requires only oracle access to gradients and Hessian-vector products. We show that even on non-convex and non-differentiable models where the theory breaks down, approximations to influence functions can still provide valuable information.

- Axiomatic Attribution for Deep Networks. [[paper]](https://arxiv.org/abs/1703.01365) [[code]](https://github.com/ankurtaly/Integrated-Gradients)
  - Mukund Sundararajan, Ankur Taly, Qiqi Yan. *ICML 2017*
  - Key Word: Feature Importance Explanations.
  - <details><summary>Digest</summary> We study the problem of attributing the prediction of a deep network to its input features, a problem previously studied by several other works. We identify two fundamental axioms---Sensitivity and Implementation Invariance that attribution methods ought to satisfy. We show that they are not satisfied by most known attribution methods, which we consider to be a fundamental weakness of those methods. We use the axioms to guide the design of a new attribution method called Integrated Gradients.

### Interpretability: 2016

- Examples are not enough, learn to criticize! Criticism for Interpretability. [[paper]](https://papers.nips.cc/paper/2016/hash/5680522b8e2bb01943234bce7bf84534-Abstract.html)
  - Been Kim, Rajiv Khanna, Oluwasanmi O. Koyejo. *NeurIPS 2016*
  - Key Word: Prototypes.
  - <details><summary>Digest</summary> In order for users to construct better mental models and understand complex data distributions, we also need criticism to explain what are not captured by prototypes. Motivated by the Bayesian model criticism framework, we develop MMD-critic which efficiently learns prototypes and criticism, designed to aid human interpretability. A human subject pilot study shows that the MMD-critic selects prototypes and criticism that are useful to facilitate human understanding and reasoning.

- Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. [[paper]](https://arxiv.org/abs/1610.02391) [[code]](https://github.com/ramprs/grad-cam)
  - Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. *ICCV 2017*
  - Key Word: Saliency Maps.
  - <details><summary>Digest</summary> We propose a technique for producing "visual explanations" for decisions from a large class of CNN-based models, making them more transparent. Our approach - Gradient-weighted Class Activation Mapping (Grad-CAM), uses the gradients of any target concept, flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in the image for predicting the concept.

- "Why Should I Trust You?": Explaining the Predictions of Any Classifier. [[paper]](https://arxiv.org/abs/1602.04938) [[code]](https://github.com/marcotcr/lime-experiments)
  - Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. *KDD 2016*
  - Key Word: Local Interpretation; Model-Agnostic Explanations.
  - <details><summary>Digest</summary> We propose LIME, a novel explanation technique that explains the predictions of any classifier in an interpretable and faithful manner, by learning an interpretable model locally around the prediction.

### Interpretability: 2015

- Explaining NonLinear Classification Decisions with Deep Taylor Decomposition. [[paper]](https://arxiv.org/abs/1512.02479) [[code]](https://github.com/myc159/Deep-Taylor-Decomposition)
  - Grégoire Montavon, Sebastian Bach, Alexander Binder, Wojciech Samek, Klaus-Robert Müller.
  - Key Word: Saliency Maps; Deep Taylor Decomposition.
  - <details><summary>Digest</summary> We introduce a novel methodology for interpreting generic multilayer neural networks by decomposing the network classification decision into contributions of its input elements. Although our focus is on image classification, the method is applicable to a broad set of input data, learning tasks and network architectures.

- Learning Deep Features for Discriminative Localization. [[paper]](https://arxiv.org/abs/1512.04150) [[code]](https://github.com/frgfm/torch-cam)
  - Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, Antonio Torralba. *CVPR 2016*
  - Key Word: Saliency Maps.
  - <details><summary>Digest</summary> We revisit the global average pooling layer proposed, and shed light on how it explicitly enables the convolutional neural network to have remarkable localization ability despite being trained on image-level labels. While this technique was previously proposed as a means for regularizing training, we find that it actually builds a generic localizable deep representation that can be applied to a variety of tasks.

## Open-World Learning

### Open-World Learning: 2021

- Open-set 3D Object Detection. [[paper]](https://arxiv.org/abs/2111.08644)
  - Jun Cen, Peng Yun, Junhao Cai, Michael Yu Wang, Ming Liu. *3DV 2021*
  - Key Word: 3D Object Detection; Video Anomaly Detection; Discovery of Unseen Instances.
  - <details><summary>Digest</summary> In this paper, we propose an open-set 3D object detector, which aims to (1) identify known objects, like the closed-set detection, and (2) identify unknown objects and give their accurate bounding boxes. Specifically, we divide the open-set 3D object detection problem into two steps: (1) finding out the regions containing the unknown objects with high probability and (2) enclosing the points of these regions with proper bounding boxes. The first step is solved by the finding that unknown objects are often classified as known objects with low confidence, and we show that the Euclidean distance sum based on metric learning is a better confidence score than the naive softmax probability to differentiate unknown objects from known objects. On this basis, unsupervised clustering is used to refine the bounding boxes of unknown objects. The proposed method combining metric learning and unsupervised clustering is called the MLUC network.

- UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection. [[paper]](https://arxiv.org/abs/2111.08644) [[code]](https://github.com/lilygeorgescu/ubnormal)
  - Andra Acsintoae, Andrei Florescu, Mariana-Iuliana Georgescu, Tudor Mare, Paul Sumedrea, Radu Tudor Ionescu, Fahad Shahbaz Khan, Mubarak Shah. *CVPR 2022*
  - Key Word: Open-set Anomaly Detection; Video Anomaly Detection; Discovery of Unseen Instances.
  - <details><summary>Digest</summary> We propose UBnormal, a new supervised open-set benchmark composed of multiple virtual scenes for video anomaly detection. Unlike existing data sets, we introduce abnormal events annotated at the pixel level at training time, for the first time enabling the use of fully-supervised learning methods for abnormal event detection. To preserve the typical open-set formulation, we make sure to include disjoint sets of anomaly types in our training and test collections of videos. To our knowledge, UBnormal is the first video anomaly detection benchmark to allow a fair head-to-head comparison between one-class open-set models and supervised closed-set models, as shown in our experiments.

- Generalized Out-of-Distribution Detection: A Survey. [[paper]](https://arxiv.org/abs/2110.11334) [[code]](https://github.com/Jingkang50/OODSurvey)
  - Jingkang Yang, Kaiyang Zhou, Yixuan Li, Ziwei Liu.
  - Key Word: Anomaly Detection; Novelty Detection; Open Set Recognition; Out-of-Distribution Detection; Outlier Detection.
  - <details><summary>Digest</summary> Several other problems are closely related to OOD detection in terms of motivation and methodology. These include anomaly detection (AD), novelty detection (ND), open set recognition (OSR), and outlier detection (OD). Despite having different definitions and problem settings, these problems often confuse readers and practitioners, and as a result, some existing studies misuse terms. In this survey, we first present a generic framework called generalized OOD detection, which encompasses the five aforementioned problems, i.e., AD, ND, OSR, OOD detection, and OD.

- Open-Set Recognition: a Good Closed-Set Classifier is All You Need? [[paper]](https://arxiv.org/abs/2110.06207) [[code]](https://github.com/sgvaze/osr_closed_set_all_you_need)
  - Sagar Vaze, Kai Han, Andrea Vedaldi, Andrew Zisserman. *ICLR 2022*
  - Key Word: Open-Set Recognition.
  - <details><summary>Digest</summary> We first demonstrate that the ability of a classifier to make the 'none-of-above' decision is highly correlated with its accuracy on the closed-set classes. We find that this relationship holds across loss objectives and architectures, and further demonstrate the trend both on the standard OSR benchmarks as well as on a large-scale ImageNet evaluation. Second, we use this correlation to boost the performance of a maximum logit score OSR 'baseline' by improving its closed-set accuracy, and with this strong baseline achieve state-of-the-art on a number of OSR benchmarks. 

- Evidential Deep Learning for Open Set Action Recognition. [[paper]](https://arxiv.org/abs/2107.10161) [[code]](https://github.com/Cogito2012/DEAR)
  - Wentao Bao, Qi Yu, Yu Kong. *ICCV 2021 oral*
  - Key Word: Open-World Learning; Open-Set Action Recognition; Discovery of Unseen Instances; Evidential Deep Learning.
  - <details><summary>Digest</summary> In a real-world scenario, human actions are typically out of the distribution from training data, which requires a model to both recognize the known actions and reject the unknown. Different from image data, video actions are more challenging to be recognized in an open-set setting due to the uncertain temporal dynamics and static bias of human actions. In this paper, we propose a Deep Evidential Action Recognition (DEAR) method to recognize actions in an open testing set. Specifically, we formulate the action recognition problem from the evidential deep learning (EDL) perspective and propose a novel model calibration method to regularize the EDL training. Besides, to mitigate the static bias of video representation, we propose a plug-and-play module to debias the learned representation through contrastive learning.

- Open-world Machine Learning: Applications, Challenges, and Opportunities. [[paper]](https://arxiv.org/abs/2105.13448)
  - Jitendra Parmar, Satyendra Singh Chouhan, Vaskar Raychoudhury, Santosh Singh Rathore.
  - Key Word: Open-World Learning; Open-Set Recognition; Discovery of Unseen Instances; Incremental Learning.
  - <details><summary>Digest</summary> Traditional machine learning mainly supervised learning, follows the assumptions of closed-world learning, i.e., for each testing class, a training class is available. However, such machine learning models fail to identify the classes which were not available during training time. These classes can be referred to as unseen classes. Whereas open-world machine learning (OWML) deals with unseen classes. In this paper, first, we present an overview of OWML with importance to the real-world context. Next, different dimensions of open-world machine learning are explored and discussed. The area of OWML gained the attention of the research community in the last decade only. We have searched through different online digital libraries and scrutinized the work done in the last decade. This paper presents a systematic review of various techniques for OWML.  

- Opening up Open-World Tracking. [[paper]](https://arxiv.org/abs/2104.11221) [[code]](https://openworldtracking.github.io/)
  - Yang Liu, Idil Esen Zulfikar, Jonathon Luiten, Achal Dave, Deva Ramanan, Bastian Leibe, Aljoša Ošep, Laura Leal-Taixé. *CVPR 2022 oral*
  - Key Word: Open-World Tracking; Multi-object Tracking; Discovery of Unseen Instances.
  - <details><summary>Digest</summary> This paper addresses this evaluation deficit and lays out the landscape and evaluation methodology for detecting and tracking both known and unknown objects in the open-world setting. We propose a new benchmark, TAO-OW: Tracking Any Object in an Open World, analyze existing efforts in multi-object tracking, and construct a baseline for this task while highlighting future challenges.

- Unidentified Video Objects: A Benchmark for Dense, Open-World Segmentation. [[paper]](https://arxiv.org/abs/2104.04691) [[code]](https://sites.google.com/view/unidentified-video-object/home?authuser=0)
  - Weiyao Wang, Matt Feiszli, Heng Wang, Du Tran.
  - Key Word: Open-World Segmentation; Class-agnostic segmentation.
  - <details><summary>Digest</summary> In this paper, we present, UVO (Unidentified Video Objects), a new benchmark for open-world class-agnostic object segmentation in videos. Besides shifting the problem focus to the open-world setup, UVO is significantly larger, providing approximately 8 times more videos compared with DAVIS, and 7 times more mask (instance) annotations per video compared with YouTube-VOS and YouTube-VIS. UVO is also more challenging as it includes many videos with crowded scenes and complex background motions. We demonstrated that UVO can be used for other applications, such as object tracking and super-voxel segmentation, besides open-world object segmentation. We believe that UVo is a versatile testbed for researchers to develop novel approaches for open-world class-agnostic object segmentation, and inspires new research directions towards a more comprehensive video understanding beyond classification and detection.

- OpenGAN: Open-Set Recognition via Open Data Generation. [[paper]](https://arxiv.org/abs/2104.02939) [[code]](https://github.com/aimerykong/OpenGAN)
  - Shu Kong, Deva Ramanan.  *ICCV 2021 Best Paper Honorable Mention*
  - Key Word: Open-set recognition; Data augmentation.
  - <details><summary>Digest</summary> Two conceptually elegant ideas for open-set discrimination are: 1) discriminatively learning an open-vs-closed binary discriminator by exploiting some outlier data as the open-set, and 2) unsupervised learning the closed-set data distribution with a GAN, using its discriminator as the open-set likelihood function. However, the former generalizes poorly to diverse open test data due to overfitting to the training outliers, which are unlikely to exhaustively span the open-world. The latter does not work well, presumably due to the instable training of GANs. Motivated by the above, we propose OpenGAN, which addresses the limitation of each approach by combining them with several technical insights. First, we show that a carefully selected GAN-discriminator on some real outlier data already achieves the state-of-the-art. Second, we augment the available set of real open training examples with adversarially synthesized "fake" data. Third and most importantly, we build the discriminator over the features computed by the closed-world K-way networks. This allows OpenGAN to be implemented via a lightweight discriminator head built on top of an existing K-way network.

- Towards Open World Object Detection. [[paper]](https://arxiv.org/abs/2103.02603) [[code]](https://github.com/JosephKJ/OWOD)
  - K J Joseph, Salman Khan, Fahad Shahbaz Khan, Vineeth N Balasubramanian.  *CVPR 2021 oral*
  - Key Word: Open-world learning; Incremental learning; Discovery of Unseen Instances; Object detection.
  - <details><summary>Digest</summary> The intrinsic curiosity about these unknown instances aids in learning about them, when the corresponding knowledge is eventually available. This motivates us to propose a novel computer vision problem called: `Open World Object Detection', where a model is tasked to: 1) identify objects that have not been introduced to it as `unknown', without explicit supervision to do so, and 2) incrementally learn these identified unknown categories without forgetting previously learned classes, when the corresponding labels are progressively received. We formulate the problem, introduce a strong evaluation protocol and provide a novel solution, which we call ORE: Open World Object Detector, based on contrastive clustering and energy based unknown identification. As an interesting by-product, we find that identifying and characterizing unknown instances helps to reduce confusion in an incremental object detection setting, where we achieve state-of-the-art performance, with no extra methodological effort.

## Environmental Well-being

### Environmental Well-being: 2021

- A Survey on Green Deep Learning. [[paper]](https://arxiv.org/abs/2111.05193)
  - Jingjing Xu, Wangchunshu Zhou, Zhiyi Fu, Hao Zhou, Lei Li.
  - Key Word: Compact Networks; Energy Efficiency.
  - <details><summary>Digest</summary> Green deep learning is an increasingly hot research field that appeals to researchers to pay attention to energy usage and carbon emission during model training and inference. The target is to yield novel results with lightweight and efficient technologies. Many technologies can be used to achieve this goal, like model compression and knowledge distillation. This paper focuses on presenting a systematic review of the development of Green deep learning technologies. We classify these approaches into four categories: (1) compact networks, (2) energy-efficient training strategies, (3) energy-efficient inference approaches, and (4) efficient data usage. For each category, we discuss the progress that has been achieved and the unresolved challenges.

- Sustainable AI: Environmental Implications, Challenges and Opportunities. [[paper]](https://arxiv.org/abs/2111.00364)
  - Carole-Jean Wu, Ramya Raghavendra, Udit Gupta, Bilge Acun, Newsha Ardalani, Kiwan Maeng, Gloria Chang, Fiona Aga Behram, James Huang, Charles Bai, Michael Gschwind, Anurag Gupta, Myle Ott, Anastasia Melnikov, Salvatore Candido, David Brooks, Geeta Chauhan, Benjamin Lee, Hsien-Hsin S. Lee, Bugra Akyildiz, Maximilian Balandat, Joe Spisak, Ravi Jain, Mike Rabbat, Kim Hazelwood.
  - Key Word: Sustainable AI; Carbon Footprint.
  - <details><summary>Digest</summary> This paper explores the environmental impact of the super-linear growth trends for AI from a holistic perspective, spanning Data, Algorithms, and System Hardware. We characterize the carbon footprint of AI computing by examining the model development cycle across industry-scale machine learning use cases and, at the same time, considering the life cycle of system hardware. Taking a step further, we capture the operational and manufacturing carbon footprint of AI computing and present an end-to-end analysis for what and how hardware-software design and at-scale optimization can help reduce the overall carbon footprint of AI. 

- Compute and Energy Consumption Trends in Deep Learning Inference. [[paper]](https://arxiv.org/abs/2109.05472)
  - Radosvet Desislavov, Fernando Martínez-Plumed, José Hernández-Orallo.
  - Key Word: Compute and Energy Consumption.
  - <details><summary>Digest</summary> We focus on inference costs rather than training costs, as the former account for most of the computing effort, solely because of the multiplicative factors. Also, apart from algorithmic innovations, we account for more specific and powerful hardware (leading to higher FLOPS) that is usually accompanied with important energy efficiency optimisations. We also move the focus from the first implementation of a breakthrough paper towards the consolidated version of the techniques one or two year later.

- Energy-Efficient Distributed Machine Learning in Cloud Fog Networks. [[paper]](https://arxiv.org/abs/2105.10048)
  - Mohammed M. Alenazi, Barzan A. Yosuf, Sanaa H. Mohamed, Taisir E.H. El-Gorashi, Jaafar M. H. Elmirghani.
  - Key Word: Energy Efficiency; Internet-of-Things; Cloud Networks.
  - <details><summary>Digest</summary> We propose a distributed ML approach where the processing can take place in intermediary devices such as IoT nodes and fog servers in addition to the cloud. We abstract the ML models into Virtual Service Requests (VSRs) to represent multiple interconnected layers of a Deep Neural Network (DNN). Using Mixed Integer Linear Programming (MILP), we design an optimization model that allocates the layers of a DNN in a Cloud/Fog Network (CFN) in an energy efficient way.

- Carbon Emissions and Large Neural Network Training. [[paper]](https://arxiv.org/abs/2104.10350)
  - David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, Jeff Dean.
  - Key Word: Carbon Emissions.
  - <details><summary>Digest</summary> We highlight the following opportunities to improve energy efficiency and CO2 equivalent emissions (CO2e): Large but sparsely activated DNNs can consume less than 1/10th the energy of large, dense DNNs without sacrificing accuracy despite using as many or even more parameters. Geographic location matters for ML workload scheduling since the fraction of carbon-free energy and resulting CO2e vary ~5X-10X, even within the same country and the same organization.

- A first look into the carbon footprint of federated learning. [[paper]](https://arxiv.org/abs/2102.07627)
  - Xinchi Qiu, Titouan Parcollet, Javier Fernandez-Marques, Pedro Porto Buarque de Gusmao, Daniel J. Beutel, Taner Topal, Akhil Mathur, Nicholas D. Lane.
  - Key Word: Federated Learning; Carbon Footprint; Energy Analysis.
  - <details><summary>Digest</summary> This paper offers the first-ever systematic study of the carbon footprint of FL. First, we propose a rigorous model to quantify the carbon footprint, hence facilitating the investigation of the relationship between FL design and carbon emissions. Then, we compare the carbon footprint of FL to traditional centralized learning. Our findings show that FL, despite being slower to converge in some cases, may result in a comparatively greener impact than a centralized equivalent setup. We performed extensive experiments across different types of datasets, settings, and various deep learning models with FL. Finally, we highlight and connect the reported results to the future challenges and trends in FL to reduce its environmental impact, including algorithms efficiency, hardware capabilities, and stronger industry transparency.

### Environmental Well-being: 2020

- Can Federated Learning Save The Planet? [[paper]](https://arxiv.org/abs/2010.06537)
  - Xinchi Qiu, Titouan Parcollet, Daniel J. Beutel, Taner Topal, Akhil Mathur, Nicholas D. Lane.
  - Key Word: Federated Learning; Carbon Footprint.
  - <details><summary>Digest</summary> This paper offers the first-ever systematic study of the carbon footprint of FL. First, we propose a rigorous model to quantify the carbon footprint, hence facilitating the investigation of the relationship between FL design and carbon emissions. Then, we compare the carbon footprint of FL to traditional centralized learning. Our findings show FL, despite being slower to converge, can be a greener technology than data center GPUs. Finally, we highlight and connect the reported results to the future challenges and trends in FL to reduce its environmental impact, including algorithms efficiency, hardware capabilities, and stronger industry transparency.

- Carbontracker: Tracking and Predicting the Carbon Footprint of Training Deep Learning Models. [[paper]](https://arxiv.org/abs/2007.03051) [[code]](https://github.com/lfwa/carbontracker/)
  - Lasse F. Wolff Anthony, Benjamin Kanding, Raghavendra Selvan.
  - Key Word: Carbon Footprint.
  - <details><summary>Digest</summary> We present Carbontracker, a tool for tracking and predicting the energy and carbon footprint of training DL models. We propose that energy and carbon footprint of model development and training is reported alongside performance metrics using tools like Carbontracker. We hope this will promote responsible computing in ML and encourage research into energy-efficient deep neural networks.

### Environmental Well-being: 2019

- Quantifying the Carbon Emissions of Machine Learning. [[paper]](https://arxiv.org/abs/1910.09700) [[code]](https://mlco2.github.io/impact/)
  - Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, Thomas Dandres.
  - Key Word: Carbon Emissions.
  - <details><summary>Digest</summary> From an environmental standpoint, there are a few crucial aspects of training a neural network that have a major impact on the quantity of carbon that it emits. These factors include: the location of the server used for training and the energy grid that it uses, the length of the training procedure, and even the make and model of hardware on which the training takes place. In order to approximate these emissions, we present our Machine Learning Emissions Calculator, a tool for our community to better understand the environmental impact of training ML models.

- Benchmarking the Performance and Energy Efficiency of AI Accelerators for AI Training. [[paper]](https://arxiv.org/abs/1909.06842)
  - Yuxin Wang, Qiang Wang, Shaohuai Shi, Xin He, Zhenheng Tang, Kaiyong Zhao, Xiaowen Chu.
  - Key Word: Energy Efficiency; AI Accelerator.
  - <details><summary>Digest</summary> To investigate the differences among several popular off-the-shelf processors (i.e., Intel CPU, NVIDIA GPU, AMD GPU, and Google TPU) in training DNNs, we carry out a comprehensive empirical study on the performance and energy efficiency of these processors by benchmarking a representative set of deep learning workloads, including computation-intensive operations, classical convolutional neural networks (CNNs), recurrent neural networks (LSTM), Deep Speech 2, and Transformer.

- Green AI. [[paper]](https://arxiv.org/abs/1907.10597)
  - Roy Schwartz, Jesse Dodge, Noah A. Smith, Oren Etzioni.
  - Key Word: Data Efficiency.
  - <details><summary>Digest</summary> This position paper advocates a practical solution by making efficiency an evaluation criterion for research alongside accuracy and related measures. In addition, we propose reporting the financial cost or "price tag" of developing, training, and running models to provide baselines for the investigation of increasingly efficient methods. Our goal is to make AI both greener and more inclusive---enabling any inspired undergraduate with a laptop to write high-quality research papers. Green AI is an emerging focus at the Allen Institute for AI.

## Interactions with Blockchain

### Interactions with Blockchain: 2021

- Blockchain-based Federated Learning: A Comprehensive Survey. [[paper]](https://arxiv.org/abs/2110.02182)
  - Zhilin Wang, Qin Hu.
  - Key Word: Federated Learning; Blockchain.
  - <details><summary>Digest</summary> We conduct a comprehensive survey of the literature on blockchained FL (BCFL). First, we investigate how blockchain can be applied to federal learning from the perspective of system composition. Then, we analyze the concrete functions of BCFL from the perspective of mechanism design and illustrate what problems blockchain addresses specifically for FL. We also survey the applications of BCFL in reality. Finally, we discuss some challenges and future research directions.

- Smart Contract Vulnerability Detection: From Pure Neural Network to Interpretable Graph Feature and Expert Pattern Fusion. [[paper]](https://arxiv.org/abs/2106.09282)
  - Zhenguang Liu, Peng Qian, Xiang Wang, Lei Zhu, Qinming He, Shouling Ji. *IJCAI 2021*
  - Key Word: Smart Contract Vulnerability Detection.
  - <details><summary>Digest</summary> We explore combining deep learning with expert patterns in an explainable fashion. Specifically, we develop automatic tools to extract expert patterns from the source code. We then cast the code into a semantic graph to extract deep graph features. Thereafter, the global graph feature and local expert patterns are fused to cooperate and approach the final prediction, while yielding their interpretable weights.

- Eth2Vec: Learning Contract-Wide Code Representations for Vulnerability Detection on Ethereum Smart Contracts. [[paper]](https://arxiv.org/abs/2101.02377)
  - Nami Ashizawa, Naoto Yanai, Jason Paul Cruz, Shingo Okamura.
  - Key Word: Ethereum; Natural Language Processing; Security Analysis.
  - <details><summary>Digest</summary> Ethereum smart contracts are programs that run on the Ethereum blockchain, and many smart contract vulnerabilities have been discovered in the past decade. Many security analysis tools have been created to detect such vulnerabilities, but their performance decreases drastically when codes to be analyzed are being rewritten. In this paper, we propose Eth2Vec, a machine-learning-based static analysis tool for vulnerability detection, with robustness against code rewrites in smart contracts.

### Interactions with Blockchain: 2020

- When Federated Learning Meets Blockchain: A New Distributed Learning Paradigm. [[paper]](https://arxiv.org/abs/2009.09338)
  - Chuan Ma, Jun Li, Ming Ding, Long Shi, Taotao Wang, Zhu Han, H. Vincent Poor.
  - Key Word: Federated Leraning; Blockchain.
  - <details><summary>Digest</summary> This work investigates a blockchain assisted decentralized FL (BLADE-FL) framework, which can well prevent the malicious clients from poisoning the learning process, and further provides a self-motivated and reliable learning environment for clients. In detail, the model aggregation process is fully decentralized and the tasks of training for FL and mining for blockchain are integrated into each participant.

- A Blockchain-based Decentralized Federated Learning Framework with Committee Consensus. [[paper]](https://arxiv.org/abs/2004.00773)
  - Yuzheng Li, Chuan Chen, Nan Liu, Huawei Huang, Zibin Zheng, Qiang Yan.
  - Key Word: Blockchain; Smart Contracts; Federated Learning.
  - <details><summary>Digest</summary> Federated learning has been widely studied and applied to various scenarios. In mobile computing scenarios, federated learning protects users from exposing their private data, while cooperatively training the global model for a variety of real-world applications. However, the security of federated learning is increasingly being questioned, due to the malicious clients or central servers' constant attack to the global model or user privacy data. To address these security issues, we proposed a decentralized federated learning framework based on blockchain, i.e., a Blockchain-based Federated Learning framework with Committee consensus (BFLC). The framework uses blockchain for the global model storage and the local model update exchange.

### Interactions with Blockchain: 2019

- A blockchain-orchestrated Federated Learning architecture for healthcare consortia. [[paper]](https://arxiv.org/abs/1910.12603)
  - Jonathan Passerat-Palmbach, Tyler Farnan, Robert Miller, Marielle S. Gross, Heather Leigh Flannery, Bill Gleim.
  - Key Word: Blockchain; Federated Learning; Healthcare.
  - <details><summary>Digest</summary> We propose a novel architecture for federated learning within healthcare consortia. At the heart of the solution is a unique integration of privacy preserving technologies, built upon native enterprise blockchain components available in the Ethereum ecosystem. We show how the specific characteristics and challenges of healthcare consortia informed our design choices, notably the conception of a new Secure Aggregation protocol assembled with a protected hardware component and an encryption toolkit native to Ethereum. Our architecture also brings in a privacy preserving audit trail that logs events in the network without revealing identities.

- BAFFLE : Blockchain Based Aggregator Free Federated Learning. [[paper]](https://arxiv.org/abs/1909.07452)
  - Paritosh Ramanan, Kiyoshi Nakayama.
  - Key Word: Blockchain; Smart Contracts; Federated Learning.
  - <details><summary>Digest</summary> A key aspect of Federated Learning (FL) is the requirement of a centralized aggregator to maintain and update the global model. However, in many cases orchestrating a centralized aggregator might be infeasible due to numerous operational constraints. In this paper, we introduce BAFFLE, an aggregator free, blockchain driven, FL environment that is inherently decentralized. BAFFLE leverages Smart Contracts (SC) to coordinate the round delineation, model aggregation and update tasks in FL. BAFFLE boosts computational performance by decomposing the global parameter space into distinct chunks followed by a score and bid strategy.

- Machine Learning in/for Blockchain: Future and Challenges. [[paper]](https://arxiv.org/abs/1909.06189)
  - Fang Chen, Hong Wan, Hua Cai, Guang Cheng.
  - Key Word: Blockchain; Bitcoin; Deep Learning; Reinforcement Learning.
  - <details><summary>Digest</summary> Machine learning and blockchain are two of the most noticeable technologies in recent years. The first one is the foundation of artificial intelligence and big data, and the second one has significantly disrupted the financial industry. Both technologies are data-driven, and thus there are rapidly growing interests in integrating them for more secure and efficient data sharing and analysis. In this paper, we review the research on combining blockchain and machine learning technologies and demonstrate that they can collaborate efficiently and effectively. In the end, we point out some future directions and expect more researches on deeper integration of the two promising technologies.

- Biometric Template Storage with Blockchain: A First Look into Cost and Performance Tradeoffs. [[paper]](https://arxiv.org/abs/1904.13128)
  - Oscar Delgado-Mohatar, Julian Fierrez, Ruben Tolosana, Ruben Vera-Rodriguez.
  - Key Word: Smart Contracts; Biometric Template Storage.
  - <details><summary>Digest</summary> We explore practical tradeoffs in blockchain-based biometric template storage. We first discuss opportunities and challenges in the integration of blockchain and biometrics, with emphasis in biometric template storage and protection, a key problem in biometrics still largely unsolved. Blockchain technologies provide excellent architectures and practical tools for securing and managing the sensitive and private data stored in biometric templates, but at a cost. We explore experimentally the key tradeoffs involved in that integration, namely: latency, processing time, economic cost, and biometric performance.

- ARCHANGEL: Tamper-proofing Video Archives using Temporal Content Hashes on the Blockchain. [[paper]](https://arxiv.org/abs/1904.12059)
  - Tu Bui, Daniel Cooper, John Collomosse, Mark Bell, Alex Green, John Sheridan, Jez Higgins, Arindra Das, Jared Keller, Olivier Thereaux, Alan Brown.
  - Key Word: Distributed Ledger Technology; Video Content Hashing.
  - <details><summary>Digest</summary> We present ARCHANGEL; a novel distributed ledger based system for assuring the long-term integrity of digital video archives. First, we describe a novel deep network architecture for computing compact temporal content hashes (TCHs) from audio-visual streams with durations of minutes or hours. Our TCHs are sensitive to accidental or malicious content modification (tampering) but invariant to the codec used to encode the video. This is necessary due to the curatorial requirement for archives to format shift video over time to ensure future accessibility. Second, we describe how the TCHs (and the models used to derive them) are secured via a proof-of-authority blockchain distributed across multiple independent archives.  

## Others

### Others: 2021

- Reliable and Trustworthy Machine Learning for Health Using Dataset Shift Detection. [[paper]](https://arxiv.org/abs/2110.14019)
  - Chunjong Park, Anas Awadalla, Tadayoshi Kohno, Shwetak Patel. *NeurIPS 2021*
  - Key Word: Healthcare; Health Screening and Diagnosis; Dataset Shift Detection.
  - <details><summary>Digest</summary> Unpredictable ML model behavior on unseen data, especially in the health domain, raises serious concerns about its safety as repercussions for mistakes can be fatal. In this paper, we explore the feasibility of using state-of-the-art out-of-distribution detectors for reliable and trustworthy diagnostic predictions. We select publicly available deep learning models relating to various health conditions (e.g., skin cancer, lung sound, and Parkinson's disease) using various input data types (e.g., image, audio, and motion data). We demonstrate that these models show unreasonable predictions on out-of-distribution datasets.

- Unsolved Problems in ML Safety. [[paper]](https://arxiv.org/abs/2109.13916)
  - Dan Hendrycks, Nicholas Carlini, John Schulman, Jacob Steinhardt.
  - Key Word: Robustness; Monitoring; Alignment; Systemic Safety.
  - <details><summary>Digest</summary> Machine learning (ML) systems are rapidly increasing in size, are acquiring new capabilities, and are increasingly deployed in high-stakes settings. As with other powerful technologies, safety for ML should be a leading research priority. In response to emerging safety challenges in ML, such as those introduced by recent large-scale models, we provide a new roadmap for ML Safety and refine the technical problems that the field needs to address. We present four problems ready for research, namely withstanding hazards ("Robustness"), identifying hazards ("Monitoring"), reducing inherent model hazards ("Alignment"), and reducing systemic hazards ("Systemic Safety"). Throughout, we clarify each problem's motivation and provide concrete research directions.

- Ethics of AI: A Systematic Literature Review of Principles and Challenges. [[paper]](https://arxiv.org/abs/2109.07906)
  - Arif Ali Khan, Sher Badshah, Peng Liang, Bilal Khan, Muhammad Waseem, Mahmood Niazi, Muhammad Azeem Akbar.
  - Key Word: Survey; Ethics.
  - <details><summary>Digest</summary> We conducted a systematic literature review (SLR) study to investigate the agreement on the significance of AI principles and identify the challenging factors that could negatively impact the adoption of AI ethics principles. The results reveal that the global convergence set consists of 22 ethical principles and 15 challenges. Transparency, privacy, accountability and fairness are identified as the most common AI ethics principles.

- The State of AI Ethics Report (Volume 5). [[paper]](https://arxiv.org/abs/2108.03929)
  - Abhishek Gupta, Connor Wright, Marianna Bergamaschi Ganapini, Masa Sweidan, Renjie Butalid.
  - Key Word: Report; Ethics.
  - <details><summary>Digest</summary> This report from the Montreal AI Ethics Institute covers the most salient progress in research and reporting over the second quarter of 2021 in the field of AI ethics with a special emphasis on "Environment and AI", "Creativity and AI", and "Geopolitics and AI." The report also features an exclusive piece titled "Critical Race Quantum Computer" that applies ideas from quantum physics to explain the complexities of human characteristics and how they can and should shape our interactions with each other. The report also features special contributions on the subject of pedagogy in AI ethics, sociology and AI ethics, and organizational challenges to implementing AI ethics in practice.

- Ethical AI for Social Good. [[paper]](https://arxiv.org/abs/2107.14044)
  - Ramya Akula, Ivan Garibay.
  - Key Word: Survey; Social Good.
  - <details><summary>Digest</summary> The concept of AI for Social Good(AI4SG) is gaining momentum in both information societies and the AI community. Through all the advancement of AI-based solutions, it can solve societal issues effectively. To date, however, there is only a rudimentary grasp of what constitutes AI socially beneficial in principle, what constitutes AI4SG in reality, and what are the policies and regulations needed to ensure it. This paper fills the vacuum by addressing the ethical aspects that are critical for future AI4SG efforts.

- Do Humans Trust Advice More if it Comes from AI? An Analysis of Human-AI Interactions. [[paper]](https://arxiv.org/abs/2107.07015)
  - Kailas Vodrahalli, Roxana Daneshjou, Tobias Gerstenberg, James Zou. *AIES 2022*
  - Key Word: Human–Computer Interaction; Human-Centered Computing; AI Advice.
  - <details><summary>Digest</summary> We recruited over 1100 crowdworkers to characterize how humans use AI suggestions relative to equivalent suggestions from a group of peer humans across several experimental settings. We find that participants' beliefs about how human versus AI performance on a given task affects whether they heed the advice. When participants do heed the advice, they use it similarly for human and AI suggestions. Based on these results, we propose a two-stage, "activation-integration" model for human behavior and use it to characterize the factors that affect human-AI interactions.

- Ethics Sheets for AI Tasks. [[paper]](https://arxiv.org/abs/2107.01183)
  - Saif M. Mohammad. *ACL 2022*
  - Key Word: Ethics; Natural Language Processing.
  - <details><summary>Digest</summary> In this position paper, I make a case for thinking about ethical considerations not just at the level of individual models and datasets, but also at the level of AI tasks. I will present a new form of such an effort, Ethics Sheets for AI Tasks, dedicated to fleshing out the assumptions and ethical considerations hidden in how a task is commonly framed and in the choices we make regarding the data, method, and evaluation.

- AI-Ethics by Design. Evaluating Public Perception on the Importance of Ethical Design Principles of AI. [[paper]](https://arxiv.org/abs/2106.00326)
  - Kimon Kieslich, Birte Keller, Christopher Starke.
  - Key Word: Ethics; Public Perception.
  - <details><summary>Digest</summary> In this study, we investigate how ethical principles (explainability, fairness, security, accountability, accuracy, privacy, machine autonomy) are weighted in comparison to each other. This is especially important, since simultaneously considering ethical principles is not only costly, but sometimes even impossible, as developers must make specific trade-off decisions. In this paper, we give first answers on the relative importance of ethical principles given a specific use case - the use of AI in tax fraud detection.

- The State of AI Ethics Report (Volume 4). [[paper]](https://arxiv.org/abs/2105.09060)
  - Abhishek Gupta, Alexandrine Royer, Connor Wright, Victoria Heath, Muriam Fancy, Marianna Bergamaschi Ganapini, Shannon Egan, Masa Sweidan, Mo Akif, Renjie Butalid.
  - Key Word: Report; Ethics.
  - <details><summary>Digest</summary> The 4th edition of the Montreal AI Ethics Institute's The State of AI Ethics captures the most relevant developments in the field of AI Ethics since January 2021. This report aims to help anyone, from machine learning experts to human rights activists and policymakers, quickly digest and understand the ever-changing developments in the field. Through research and article summaries, as well as expert commentary, this report distills the research and reporting surrounding various domains related to the ethics of AI, with a particular focus on four key themes: Ethical AI, Fairness & Justice, Humans & Tech, and Privacy.

- The State of AI Ethics Report (January 2021). [[paper]](https://arxiv.org/abs/2105.09059)
  - Abhishek Gupta, Alexandrine Royer, Connor Wright, Falaah Arif Khan, Victoria Heath, Erick Galinkin, Ryan Khurana, Marianna Bergamaschi Ganapini, Muriam Fancy, Masa Sweidan, Mo Akif, Renjie Butalid.
  - Key Word: Report; Ethics.
  - <details><summary>Digest</summary> It aims to help anyone, from machine learning experts to human rights activists and policymakers, quickly digest and understand the field's ever-changing developments. Through research and article summaries, as well as expert commentary, this report distills the research and reporting surrounding various domains related to the ethics of AI, including: algorithmic injustice, discrimination, ethical AI, labor impacts, misinformation, privacy, risk and security, social media, and more.

- Ethics and Governance of Artificial Intelligence: Evidence from a Survey of Machine Learning Researchers. [[paper]](https://arxiv.org/abs/2105.02117)
  - Baobao Zhang, Markus Anderljung, Lauren Kahn, Noemi Dreksler, Michael C. Horowitz, Allan Dafoe.
  - Key Word: Survey; Ethics; Governance.
  - <details><summary>Digest</summary> To examine these researchers' views, we conducted a survey of those who published in the top AI/ML conferences (N = 524). We compare these results with those from a 2016 survey of AI/ML researchers and a 2018 survey of the US public. We find that AI/ML researchers place high levels of trust in international organizations and scientific organizations to shape the development and use of AI in the public interest; moderate trust in most Western tech companies; and low trust in national militaries, Chinese tech companies, and Facebook.

- Towards Causal Representation Learning. [[paper]](https://arxiv.org/abs/2102.11107)
  - Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, Yoshua Bengio. *Proceedings of the IEEE*
  - Key Word: Causal Representation Learning.
  - <details><summary>Digest</summary> The two fields of machine learning and graphical causality arose and developed separately. However, there is now cross-pollination and increasing interest in both fields to benefit from the advances of the other. In the present paper, we review fundamental concepts of causal inference and relate them to crucial open problems of machine learning, including transfer and generalization, thereby assaying how causality can contribute to modern machine learning research.

### Others: 2020

- The State of AI Ethics Report (October 2020). [[paper]](https://arxiv.org/abs/2011.02787)
  - Abhishek Gupta, Alexandrine Royer, Victoria Heath, Connor Wright, Camylle Lanteigne, Allison Cohen, Marianna Bergamaschi Ganapini, Muriam Fancy, Erick Galinkin, Ryan Khurana, Mo Akif, Renjie Butalid, Falaah Arif Khan, Masa Sweidan, Audrey Balogh.
  - Key Word: Report; Ethics.
  - <details><summary>Digest</summary> This report aims to help anyone, from machine learning experts to human rights activists and policymakers, quickly digest and understand the ever-changing developments in the field. Through research and article summaries, as well as expert commentary, this report distills the research and reporting surrounding various domains related to the ethics of AI, including: AI and society, bias and algorithmic justice, disinformation, humans and AI, labor impacts, privacy, risk, and future of AI ethics.

- Representation Learning via Invariant Causal Mechanisms. [[paper]](https://arxiv.org/abs/2010.07922)
  - Jovana Mitrovic, Brian McWilliams, Jacob Walker, Lars Buesing, Charles Blundell. *ICLR 2021*
  - Key Word: Self-Supervision; Causality.
  - <details><summary>Digest</summary> We propose a novel self-supervised objective, Representation Learning via Invariant Causal Mechanisms (ReLIC), that enforces invariant prediction of proxy targets across augmentations through an invariance regularizer which yields improved generalization guarantees.

- Disentangled Generative Causal Representation Learning. [[paper]](https://arxiv.org/abs/2010.02637) [[code]](https://github.com/xwshen51/DEAR)
  - Xinwei Shen, Furui Liu, Hanze Dong, Qing Lian, Zhitang Chen, Tong Zhang.
  - Key Word: Disentanglement; Generative Model.
  - <details><summary>Digest</summary> This paper proposes a Disentangled gEnerative cAusal Representation (DEAR) learning method. Unlike existing disentanglement methods that enforce independence of the latent variables, we consider the general case where the underlying factors of interests can be causally correlated. We show that previous methods with independent priors fail to disentangle causally correlated factors. Motivated by this finding, we propose a new disentangled learning method called DEAR that enables causal controllable generation and causal representation learning. The key ingredient of this new formulation is to use a structural causal model (SCM) as the prior for a bidirectional generative model.

- The State of AI Ethics Report (June 2020). [[paper]](https://arxiv.org/abs/2006.14662)
  - Abhishek Gupta, Camylle Lanteigne, Victoria Heath, Marianna Bergamaschi Ganapini, Erick Galinkin, Allison Cohen, Tania De Gasperis, Mo Akif, Renjie Butalid.
  - Key Word: Report; Ethics.
  - <details><summary>Digest</summary> Our goal is to help you navigate this ever-evolving field swiftly and allow you and your organization to make informed decisions. This pulse-check for the state of discourse, research, and development is geared towards researchers and practitioners alike who are making decisions on behalf of their organizations in considering the societal impacts of AI-enabled solutions. We cover a wide set of areas in this report spanning Agency and Responsibility, Security and Risk, Disinformation, Jobs and Labor, the Future of AI Ethics, and more.

- Active Invariant Causal Prediction: Experiment Selection through Stability. [[paper]](https://arxiv.org/abs/2006.05690) [[code]](https://github.com/juangamella/aicp)
  - Juan L. Gamella, Christina Heinze-Deml. *NeurIPS 2020*
  - Key Word: Invariant Causal Prediction.
  - <details><summary>Digest</summary> In this work we propose a new active learning (i.e. experiment selection) framework (A-ICP) based on Invariant Causal Prediction (ICP). For general structural causal models, we characterize the effect of interventions on so-called stable sets. We leverage these results to propose several intervention selection policies for A-ICP which quickly reveal the direct causes of a response variable in the causal graph while maintaining the error control inherent in ICP.

- CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models. [[paper]](https://arxiv.org/abs/2004.08697)
  - Mengyue Yang, Furui Liu, Zhitang Chen, Xinwei Shen, Jianye Hao, Jun Wang. *CVPR 2021*
  - Key Word: Disentanlged Representation Learning.
  - <details><summary>Digest</summary> The framework of variational autoencoder (VAE) is commonly used to disentangle independent factors from observations. However, in real scenarios, factors with semantics are not necessarily independent. Instead, there might be an underlying causal structure which renders these factors dependent. We thus propose a new VAE based framework named CausalVAE, which includes a Causal Layer to transform independent exogenous factors into causal endogenous ones that correspond to causally related concepts in data.

### Others: 2019

- Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One. [[paper]](https://arxiv.org/abs/1912.03263) [[code]](https://github.com/wgrathwohl/JEM)
  - Will Grathwohl, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, Kevin Swersky. *ICLR 2020*
  - Key Word: Energy-Based Model; Calibration; Adversarial Robustness; Out-of-Distribution Detection.
  - <details><summary>Digest</summary> Within this framework, standard discriminative architectures may beused and the model can also be trained on unlabeled data. We demonstrate that energy based training of the joint distribution improves calibration, robustness, andout-of-distribution detection while also enabling our models to generate samplesrivaling the quality of recent GAN approaches.  

- Variational Autoencoders and Nonlinear ICA: A Unifying Framework. [[paper]](https://arxiv.org/abs/1907.04809)
  - Ilyes Khemakhem, Diederik P. Kingma, Ricardo Pio Monti, Aapo Hyvärinen. *AISTATS 2022*
  - Key Word: Causal Representation Learning.
  - <details><summary>Digest</summary> The framework of variational autoencoders allows us to efficiently learn deep latent-variable models, such that the model's marginal distribution over observed variables fits the data. Often, we're interested in going a step further, and want to approximate the true joint distribution over observed and latent variables, including the true prior and posterior distributions over latent variables. This is known to be generally impossible due to unidentifiability of the model. We address this issue by showing that for a broad family of deep latent-variable models, identification of the true joint distribution over observed and latent variables is actually possible up to very simple transformations, thus achieving a principled and powerful form of disentanglement.

- Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. [[paper]](https://arxiv.org/abs/1906.02530) [[code]](https://github.com/google-research/google-research/tree/master/uq_benchmark_2019)
  - Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, Jasper Snoek. *NeurIPS 2019*
  - Key Word: Uncerntainty Estimation; Distribution Shift.
  - <details><summary>Digest</summary> Quantifying uncertainty is especially critical in real-world settings, which often involve input distributions that are shifted from the training distribution due to a variety of factors including sample bias and non-stationarity. In such settings, well calibrated uncertainty estimates convey information about when a model's output should (or should not) be trusted.

- A Simple Baseline for Bayesian Uncertainty in Deep Learning. [[paper]](https://arxiv.org/abs/1902.02476) [[code]](https://github.com/wjmaddox/swa_gaussian)
  - Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, Andrew Gordon Wilson. *NeurIPS 2019*
  - Key Word: Calibration; Bayesian Deep Learning.
  - <details><summary>Digest</summary> We propose SWA-Gaussian (SWAG), a simple, scalable, and general purpose approach for uncertainty representation and calibration in deep learning. Stochastic Weight Averaging (SWA), which computes the first moment of stochastic gradient descent (SGD) iterates with a modified learning rate schedule, has recently been shown to improve generalization in deep learning.  

- Artificial Intelligence for Social Good. [[paper]](https://arxiv.org/abs/1901.05406)
  - Gregory D. Hager, Ann Drobnis, Fei Fang, Rayid Ghani, Amy Greenwald, Terah Lyons, David C. Parkes, Jason Schultz, Suchi Saria, Stephen F. Smith, Milind Tambe.
  - Key Word: Report; Social Good.
  - <details><summary>Digest</summary> The Computing Community Consortium (CCC), along with the White House Office of Science and Technology Policy (OSTP), and the Association for the Advancement of Artificial Intelligence (AAAI), co-sponsored a public workshop on Artificial Intelligence for Social Good on June 7th, 2016 in Washington, DC. This was one of five workshops that OSTP co-sponsored and held around the country to spur public dialogue on artificial intelligence, machine learning, and to identify challenges and opportunities related to AI. In the AI for Social Good workshop, the successful deployments and the potential use of AI in various topics that are essential for social good were discussed, including but not limited to urban computing, health, environmental sustainability, and public welfare. This report highlights each of these as well as a number of crosscutting issues.

### Others: 2018

- Towards Reverse-Engineering Black-Box Neural Networks. [[paper]](https://arxiv.org/abs/1711.01768) [[code]](https://github.com/coallaoh/WhitenBlackBox)
  - Seong Joon Oh, Max Augustin, Bernt Schiele, Mario Fritz. *ICLR 2018*
  - Key Word: Model Extraction Attacks; Membership inference Attacks.
  - <details><summary>Digest</summary> This work shows that such attributes of neural networks can be exposed from a sequence of queries. This has multiple implications. On the one hand, our work exposes the vulnerability of black-box neural networks to different types of attacks -- we show that the revealed internal information helps generate more effective adversarial examples against the black box model.  

## Related Awesome Lists

### Robustness Lists

- [A Complete List of All (arXiv) Adversarial Example Papers](https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html)

- [Backdoor Learning Resources](https://github.com/THUYimingLi/backdoor-learning-resources) ![ ](https://img.shields.io/github/stars/THUYimingLi/backdoor-learning-resources) ![ ](https://img.shields.io/github/last-commit/THUYimingLi/backdoor-learning-resources)

- [Paper of Robust ML](https://github.com/P2333/Papers-of-Robust-ML) ![ ](https://img.shields.io/github/stars/P2333/Papers-of-Robust-ML) ![ ](https://img.shields.io/github/last-commit/P2333/Papers-of-Robust-ML)

- [The Papers of Adversarial Examples](https://github.com/xiaosen-wang/Adversarial-Examples-Paper) ![ ](https://img.shields.io/github/stars/xiaosen-wang/Adversarial-Examples-Paper) ![ ](https://img.shields.io/github/last-commit/xiaosen-wang/Adversarial-Examples-Paper)

### Privacy Lists

- [Awesome Attacks on Machine Learning Privacy](https://github.com/stratosphereips/awesome-ml-privacy-attacks) ![ ](https://img.shields.io/github/stars/stratosphereips/awesome-ml-privacy-attacks) ![ ](https://img.shields.io/github/last-commit/stratosphereips/awesome-ml-privacy-attacks)

- [Aweosme Privacy](https://github.com/Guyanqi/Awesome-Privacy) ![ ](https://img.shields.io/github/stars/Guyanqi/Awesome-Privacy) ![ ](https://img.shields.io/github/last-commit/Guyanqi/Awesome-Privacy)

- [Privacy-Preserving-Machine-Learning-Resources](https://github.com/Ye-D/PPML-Resource) ![ ](https://img.shields.io/github/stars/Ye-D/PPML-Resource) ![ ](https://img.shields.io/github/last-commit/Ye-D/PPML-Resource)

- [Awesome Privacy Papers for Visual Data](https://github.com/brighter-ai/awesome-privacy-papers) ![ ](https://img.shields.io/github/stars/brighter-ai/awesome-privacy-papers) ![ ](https://img.shields.io/github/last-commit/brighter-ai/awesome-privacy-papers)

### Fairness Lists

- [Awesome Fairness Papers](https://github.com/uclanlp/awesome-fairness-papers) ![ ](https://img.shields.io/github/stars/uclanlp/awesome-fairness-papers) ![ ](https://img.shields.io/github/last-commit/uclanlp/awesome-fairness-papers)

- [Awesome Fairness in AI](https://github.com/datamllab/awesome-fairness-in-ai) ![ ](https://img.shields.io/github/stars/datamllab/awesome-fairness-in-ai) ![ ](https://img.shields.io/github/last-commit/datamllab/awesome-fairness-in-ai)

### Interpretability Lists

- [Awesome Machine Learning Interpretability](https://github.com/jphall663/awesome-machine-learning-interpretability) ![ ](https://img.shields.io/github/stars/jphall663/awesome-machine-learning-interpretability) ![ ](https://img.shields.io/github/last-commit/jphall663/awesome-machine-learning-interpretability)

- [Awesome Interpretable Machine Learning](https://github.com/lopusz/awesome-interpretable-machine-learning) ![ ](https://img.shields.io/github/stars/lopusz/awesome-interpretable-machine-learning) ![ ](https://img.shields.io/github/last-commit/lopusz/awesome-interpretable-machine-learning)

- [Awesome Explainable AI](https://github.com/wangyongjie-ntu/Awesome-explainable-AI) ![ ](https://img.shields.io/github/stars/wangyongjie-ntu/Awesome-explainable-AI) ![ ](https://img.shields.io/github/last-commit/wangyongjie-ntu/Awesome-explainable-AI)

- [Awesome Deep Learning Interpretability](https://github.com/oneTaken/awesome_deep_learning_interpretability) ![ ](https://img.shields.io/github/stars/oneTaken/awesome_deep_learning_interpretability) ![ ](https://img.shields.io/github/last-commit/oneTaken/awesome_deep_learning_interpretability)

### Open-World Lists

- [Awesome Open Set Recognition list](https://github.com/iCGY96/awesome_OpenSetRecognition_list) ![ ](https://img.shields.io/github/stars/iCGY96/awesome_OpenSetRecognition_list) ![ ](https://img.shields.io/github/last-commit/iCGY96/awesome_OpenSetRecognition_list)

- [Awesome Novel Class Discovery](https://github.com/JosephKJ/Awesome-Novel-Class-Discovery) ![ ](https://img.shields.io/github/stars/JosephKJ/Awesome-Novel-Class-Discovery) ![ ](https://img.shields.io/github/last-commit/JosephKJ/Awesome-Novel-Class-Discovery)

- [Awesome Open-World-Learning](https://github.com/zhoudw-zdw/Awesome-open-world-learning) ![ ](https://img.shields.io/github/stars/zhoudw-zdw/Awesome-open-world-learning) ![ ](https://img.shields.io/github/last-commit/zhoudw-zdw/Awesome-open-world-learning)

### Blockchain Lists

- [Blockchain Papers](https://github.com/decrypto-org/blockchain-papers) ![ ](https://img.shields.io/github/stars/decrypto-org/blockchain-papers) ![ ](https://img.shields.io/github/last-commit/decrypto-org/blockchain-papers)

- [Awesome Blockchain AI](https://github.com/steven2358/awesome-blockchain-ai) ![ ](https://img.shields.io/github/stars/steven2358/awesome-blockchain-ai) ![ ](https://img.shields.io/github/last-commit/steven2358/awesome-blockchain-ai)

### Other Lists

- [Awesome Causality Algorithms](https://github.com/rguo12/awesome-causality-algorithms) ![ ](https://img.shields.io/github/stars/rguo12/awesome-causality-algorithms) ![ ](https://img.shields.io/github/last-commit/rguo12/awesome-causality-algorithms)

- [Awesome AI Security](https://github.com/DeepSpaceHarbor/Awesome-AI-Security) ![ ](https://img.shields.io/github/stars/DeepSpaceHarbor/Awesome-AI-Security) ![ ](https://img.shields.io/github/last-commit/DeepSpaceHarbor/Awesome-AI-Security)

- [A curated list of AI Security & Privacy events](https://github.com/ZhengyuZhao/AI-Security-and-Privacy-Events) ![ ](https://img.shields.io/github/stars/ZhengyuZhao/AI-Security-and-Privacy-Events) ![ ](https://img.shields.io/github/last-commit/ZhengyuZhao/AI-Security-and-Privacy-Events)

- [Awesome Deep Phenomena](https://github.com/MinghuiChen43/awesome-deep-phenomena) ![ ](https://img.shields.io/github/stars/MinghuiChen43/awesome-deep-phenomena) ![ ](https://img.shields.io/github/last-commit/MinghuiChen43/awesome-deep-phenomena)

## Toolboxes

### Robustness Toolboxes

- [Cleverhans](https://github.com/cleverhans-lab/cleverhans) ![ ](https://img.shields.io/github/stars/cleverhans-lab/cleverhans)
  - This repository contains the source code for CleverHans, a Python library to benchmark machine learning systems' vulnerability to adversarial examples.

- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) ![ ](https://img.shields.io/github/stars/Trusted-AI/adversarial-robustness-toolbox)
  - Adversarial Robustness Toolbox (ART) is a Python library for Machine Learning Security. ART provides tools that enable developers and researchers to evaluate, defend, certify and verify Machine Learning models and applications against the adversarial threats of Evasion, Poisoning, Extraction, and Inference.

- [Advtorch](https://github.com/BorealisAI/advertorch) ![ ](https://img.shields.io/github/stars/BorealisAI/advertorch)
  - Advtorch is a Python toolbox for adversarial robustness research. The primary functionalities are implemented in PyTorch. Specifically, AdverTorch contains modules for generating adversarial perturbations and defending against adversarial examples, also scripts for adversarial training.
  
- [RobustBench](https://github.com/RobustBench/robustbench) ![ ](https://img.shields.io/github/stars/RobustBench/robustbench)
  - A standardized benchmark for adversarial robustness.

### Privacy Toolboxes

- [Diffprivlib](https://github.com/IBM/differential-privacy-library) ![ ](https://img.shields.io/github/stars/IBM/differential-privacy-library)
  - Diffprivlib is a general-purpose library for experimenting with, investigating and developing applications in, differential privacy.

- [Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter) ![ ](https://img.shields.io/github/stars/privacytrustlab/ml_privacy_meter)
  - Privacy Meter is an open-source library to audit data privacy in statistical and machine learning algorithms.

- [PrivacyRaven](https://github.com/trailofbits/PrivacyRaven) ![ ](https://img.shields.io/github/stars/trailofbits/PrivacyRaven)
  - PrivacyRaven is a privacy testing library for deep learning systems.

### Fairness Toolboxes

- [AI Fairness 360](https://github.com/Trusted-AI/AIF360) ![ ](https://img.shields.io/github/stars/Trusted-AI/AIF360)
  - The AI Fairness 360 toolkit is an extensible open-source library containing techniques developed by the research community to help detect and mitigate bias in machine learning models throughout the AI application lifecycle.

- [Fairlearn](https://github.com/fairlearn/fairlearn) ![ ](https://img.shields.io/github/stars/fairlearn/fairlearn)
  - Fairlearn is a Python package that empowers developers of artificial intelligence (AI) systems to assess their system's fairness and mitigate any observed unfairness issues.

- [Aequitas](https://github.com/dssg/aequitas) ![ ](https://img.shields.io/github/stars/dssg/aequitas)
  - Aequitas is an open-source bias audit toolkit for data scientists, machine learning researchers, and policymakers to audit machine learning models for discrimination and bias, and to make informed and equitable decisions around developing and deploying predictive tools.

### Interpretability Toolboxes

- [Lime](https://github.com/marcotcr/lime) ![ ](https://img.shields.io/github/stars/marcotcr/lime)
  - This project is about explaining what machine learning classifiers (or models) are doing.
  
- [InterpretML](https://github.com/interpretml/interpret/) ![ ](https://img.shields.io/github/stars/interpretml/interpret)
  - InterpretML is an open-source package that incorporates state-of-the-art machine learning interpretability techniques under one roof.

- [Deep Visualization Toolbox](https://github.com/yosinski/deep-visualization-toolbox) ![ ](https://img.shields.io/github/stars/yosinski/deep-visualization-toolbox)
  - This is the code required to run the Deep Visualization Toolbox, as well as to generate the neuron-by-neuron visualizations using regularized optimization.
  
- [Captum](https://github.com/pytorch/captum) ![ ](https://img.shields.io/github/stars/pytorch/captum)
  - Captum is a model interpretability and understanding library for PyTorch.
  
- [Alibi](https://github.com/SeldonIO/alibi) ![ ](https://img.shields.io/github/stars/SeldonIO/alibi)
  - Alibi is an open source Python library aimed at machine learning model inspection and interpretation.
  
- [AI Explainability 360](https://github.com/Trusted-AI/AIX360) ![ ](https://img.shields.io/github/stars/Trusted-AI/AIX360)
  - The AI Explainability 360 toolkit is an open-source library that supports interpretability and explainability of datasets and machine learning models.

### Other Toolboxes

- [Causal Inference 360](https://github.com/IBM/causallib) ![ ](https://img.shields.io/github/stars/IBM/causallib)
  - A Python package for inferring causal effects from observational data.

## Workshops

### Robustness Workshops

- [Workshop on Adversarial Robustness In the Real World (ECCV 2022)](https://eccv22-arow.github.io/)

- [Shift Happens Workshop (ICML 2022)](https://shift-happens-benchmark.github.io/)

- [Principles of Distribution Shift (ICML 2022)](https://sites.google.com/view/icml-2022-pods)

- [New Frontiers in Adversarial Machine Learning (ICML 2022)](https://advml-frontier.github.io/)

- [Workshop on Spurious Correlations, Invariance, and Stability (ICML 2022)](https://sites.google.com/view/scis-workshop/home)

- [Robust and reliable machine learning in the real world (ICLR 2021)](https://sites.google.com/connect.hku.hk/robustml-2021/home)

- [Distribution Shifts Connecting Methods and Applications (NeurIPS 2021)](https://sites.google.com/view/distshift2021)

- [Workshop on Adversarial Robustness In the Real World (ICCV 2021)](https://iccv21-adv-workshop.github.io/)

- [Uncertainty and Robustness in Deep Learning Workshop (ICML 2021)](https://sites.google.com/view/udlworkshop2021/home)

- [Uncertainty and Robustness in Deep Learning Workshop (ICML 2020)](https://sites.google.com/view/udlworkshop2020/home)

### Privacy Workshops

- [Theory and Practice of Differential Privacy (ICML 2022)](https://tpdp.journalprivacyconfidentiality.org/2022/)

### Interpretability Workshops

- [Interpretable Machine Learning in Healthcare (ICML 2022)](https://sites.google.com/view/imlh2022)

### Other Workshops

- [International Workshop on Trustworthy Federated Learning (IJCAI 2022)](https://federated-learning.org/fl-ijcai-2022/)

- [Workshop on AI Safety (IJCAI 2022)](https://www.aisafetyw.org/)

- [1st Workshop on Formal Verification of Machine Learning (ICML 2022)](https://www.ml-verification.com/)

- [Workshop on Distribution-Free Uncertainty Quantification (ICML 2022)](https://sites.google.com/berkeley.edu/dfuq-22/home)

- [First Workshop on Causal Representation Learning (UAI 2022)](https://crl-uai-2022.github.io/)

## Tutorials

### Robustness Tutorials

- [Tutorial on Domain Generalization (IJCAI-ECAI 2022)](https://dgresearch.github.io/)

- [Practical Adversarial Robustness in Deep Learning: Problems and Solutions (CVPR 2021)](https://sites.google.com/view/par-2021)

- [A Tutorial about Adversarial Attacks & Defenses (KDD 2021)](https://sites.google.com/view/kdd21-tutorial-adv-robust/)

- [Adversarial Robustness of Deep Learning Models (ECCV 2020)](https://sites.google.com/umich.edu/eccv-2020-adv-robustness)

- [Adversarial Robustness: Theory and Practice (NeurIPS 2018)](https://nips.cc/Conferences/2018/ScheduleMultitrack?event=10978) [[Note]](https://adversarial-ml-tutorial.org/)

- [Adversarial Machine Learning Tutorial (AAAI 2018)](https://aaai18adversarial.github.io/index.html#)

## Talks

### Robustness Talks

- [Ian Goodfellow: Adversarial Machine Learning (ICLR 2019)](https://www.youtube.com/watch?v=sucqskXRkss)

## Blogs

### Robustness Blogs

- [Pixels still beat text: Attacking the OpenAI CLIP model with text patches and adversarial pixel perturbations](https://stanislavfort.github.io/blog/OpenAI_CLIP_stickers_and_adversarial_examples/)

- [Adversarial examples for the OpenAI CLIP in its zero-shot classification regime and their semantic generalization](https://stanislavfort.github.io/blog/OpenAI_CLIP_adversarial_examples/)

- [A Discussion of Adversarial Examples Are Not Bugs, They Are Features](https://distill.pub/2019/advex-bugs-discussion/)

### Interpretability Blogs

- [Multimodal Neurons in Artificial Neural Networks](https://distill.pub/2021/multimodal-neurons/)

- [Curve Detectors](https://distill.pub/2020/circuits/curve-detectors/)

- [Visualizing the Impact of Feature Attribution Baselines](https://distill.pub/2020/attribution-baselines/)

- [Visualizing Neural Networks with the Grand Tour](https://distill.pub/2020/grand-tour/)

### Other Blogs

- [Cleverhans Blog - Ian Goodfellow, Nicolas Papernot](http://www.cleverhans.io/)

## Other Resources

- [AI Security and Privacy (AISP) Seminar Series](http://scl.sribd.cn/seminar/index.html)

- [ML Safety Newsletter](https://newsletter.mlsafety.org/)

- [Trustworthy ML Initiative](https://www.trustworthyml.org/home)

- [Trustworthy AI Project](https://www.trustworthyaiproject.eu/)

- [ECE1784H: Trustworthy Machine Learning (Course, Fall 2019) - Nicolas Papernot](https://www.papernot.fr/teaching/f19-trustworthy-ml)

- [A School for all Seasons on Trustworthy Machine Learning (Course) - Reza Shokri, Nicolas Papernot](https://trustworthy-machine-learning.github.io/)

- [Trustworthy Machine Learning (Book)](http://www.trustworthymachinelearning.com/)

## Contributing

TBD
