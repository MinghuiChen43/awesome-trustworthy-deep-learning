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

<details>
<summary>Survey: 2022</summary>


- A Comprehensive Survey on Trustworthy Recommender Systems. [[paper]](https://arxiv.org/abs/2209.10117)
  - Wenqi Fan, Xiangyu Zhao, Xiao Chen, Jingran Su, Jingtong Gao, Lin Wang, Qidong Liu, Yiqi Wang, Han Xu, Lei Chen, Qing Li.
  - Key Word: Recommender Systems; Survey.
  - <details><summary>Digest</summary> We provide a comprehensive overview of Trustworthy Recommender systems (TRec) with a specific focus on six of the most important aspects; namely, Safety & Robustness, Nondiscrimination & Fairness, Explainability, Privacy, Environmental Well-being, and Accountability & Auditability. For each aspect, we summarize the recent related technologies and discuss potential research directions to help achieve trustworthy recommender systems in the future.

- Non-Imaging Medical Data Synthesis for Trustworthy AI: A Comprehensive Survey. [[paper]](https://arxiv.org/abs/2209.09239)
  - Xiaodan Xing, Huanjun Wu, Lichao Wang, Iain Stenson, May Yong, Javier Del Ser, Simon Walsh, Guang Yang.
  - Key Word: Non-Imaging Medical Data Generation; Healthcare; Survey.
  - <details><summary>Digest</summary> State-of-the-art data synthesis algorithms, especially deep learning algorithms, focus more on imaging data while neglecting the synthesis of non-imaging healthcare data, including clinical measurements, medical signals and waveforms, and electronic healthcare records (EHRs). Thus, in this paper, we will review the synthesis algorithms, particularly for non-imaging medical data, with the aim of providing trustworthy AI in this domain. This tutorial-styled review paper will provide comprehensive descriptions of non-imaging medical data synthesis on aspects including algorithms, evaluations, limitations and future research directions.

- Trustworthy Reinforcement Learning Against Intrinsic Vulnerabilities: Robustness, Safety, and Generalizability. [[paper]](https://arxiv.org/abs/2209.08025)
  - Mengdi Xu, Zuxin Liu, Peide Huang, Wenhao Ding, Zhepeng Cen, Bo Li, Ding Zhao.
  - Key Word: Reinforcement Learning; Distribution Shifts; Robustness; Safety; Survey.
  - <details><summary>Digest</summary> A trustworthy reinforcement learning algorithm should be competent in solving challenging real-world problems, including {robustly} handling uncertainties, satisfying {safety} constraints to avoid catastrophic failures, and {generalizing} to unseen scenarios during deployments. This study aims to overview these main perspectives of trustworthy reinforcement learning considering its intrinsic vulnerabilities on robustness, safety, and generalizability. 

- Trustworthy Recommender Systems. [[paper]](https://arxiv.org/abs/2208.06265)
  - Shoujin Wang, Xiuzhen Zhang, Yan Wang, Huan Liu, Francesco Ricci.
  - Key Word: Survey; Recommender Systems; Trustworthy Recommendation.
  - <details><summary>Digest</summary> Recent years have witnessed an increasing number of threats to RSs, coming from attacks, system and user generated noise, system bias. As a result, it has become clear that a strict focus on RS accuracy is limited and the research must consider other important factors, e.g., trustworthiness. For end users, a trustworthy RS (TRS) should not only be accurate, but also transparent, unbiased and fair as well as robust to noise or attacks. These observations actually led to a paradigm shift of the research on RSs: from accuracy-oriented RSs to TRSs. However, researchers lack a systematic overview and discussion of the literature in this novel and fast developing field of TRSs. To this end, in this paper, we provide an overview of TRSs, including a discussion of the motivation and basic concepts of TRSs, a presentation of the challenges in building TRSs, and a perspective on the future directions in this area.

- Trustworthy Graph Neural Networks: Aspects, Methods and Trends. [[paper]](https://arxiv.org/abs/2205.07424)
  - He Zhang, Bang Wu, Xingliang Yuan, Shirui Pan, Hanghang Tong, Jian Pei.
  - Key Word: Survey; Graph Neural Networks.
  - <details><summary>Digest</summary> We propose a comprehensive roadmap to build trustworthy GNNs from the view of the various computing technologies involved. In this survey, we introduce basic concepts and comprehensively summarise existing efforts for trustworthy GNNs from six aspects, including robustness, explainability, privacy, fairness, accountability, and environmental well-being. Additionally, we highlight the intricate cross-aspect relations between the above six aspects of trustworthy GNNs. Finally, we present a thorough overview of trending directions for facilitating the research and industrialisation of trustworthy GNNs.

- A Survey on AI Sustainability: Emerging Trends on Learning Algorithms and Research Challenges. [[paper]](https://arxiv.org/abs/2205.03824)
  - Zhenghua Chen, Min Wu, Alvin Chan, Xiaoli Li, Yew-Soon Ong.
  - Key Word: Survey; Sustainability.
  - <details><summary>Digest</summary> The technical trend in realizing the successes has been towards increasing complex and large size AI models so as to solve more complex problems at superior performance and robustness. This rapid progress, however, has taken place at the expense of substantial environmental costs and resources. Besides, debates on the societal impacts of AI, such as fairness, safety and privacy, have continued to grow in intensity. These issues have presented major concerns pertaining to the sustainable development of AI. In this work, we review major trends in machine learning approaches that can address the sustainability problem of AI.

</details>

<!-- ### Survey: 2021 -->

<details>
<summary>Survey: 2021</summary>

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

</details>

<!-- ### Survey: 2020 -->

<details>
<summary>Survey: 2020</summary>

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

</details>

<!-- ### Survey: 2019 -->

<details>
<summary>Survey: 2019</summary>

- The relationship between trust in AI and trustworthy machine learning technologies. [[paper]](https://arxiv.org/abs/1912.00782)
  - Ehsan Toreini, Mhairi Aitken, Kovila Coopamootoo, Karen Elliott, Carlos Gonzalez Zelaya, Aad van Moorsel. *FAT 2020*
  - Key Word: Survey; Social Science.
  - <details><summary>Digest</summary> To build AI-based systems that users and the public can justifiably trust one needs to understand how machine learning technologies impact trust put in these services. To guide technology developments, this paper provides a systematic approach to relate social science concepts of trust with the technologies used in AI-based services and products. We conceive trust as discussed in the ABI (Ability, Benevolence, Integrity) framework and use a recently proposed mapping of ABI on qualities of technologies. We consider four categories of machine learning technologies, namely these for Fairness, Explainability, Auditability and Safety (FEAS) and discuss if and how these possess the required qualities.

</details>

<!-- ### Survey: 2018 -->

<details>
<summary>Survey: 2018</summary>

- A Survey of Safety and Trustworthiness of Deep Neural Networks: Verification, Testing, Adversarial Attack and Defence, and Interpretability. [[paper]](https://arxiv.org/abs/1812.08342)
  - Xiaowei Huang, Daniel Kroening, Wenjie Ruan, James Sharp, Youcheng Sun, Emese Thamo, Min Wu, Xinping Yi. *Computer Science Review*
  - Key Word: Survey.
  - <details><summary>Digest</summary> This survey paper conducts a review of the current research effort into making DNNs safe and trustworthy, by focusing on four aspects: verification, testing, adversarial attack and defence, and interpretability. In total, we survey 202 papers, most of which were published after 2017.

</details>

## Out-of-Distribution Generalization

<details>
<summary>Out-of-Distribution Generalization: 2022</summary>

- Generalizing to unseen domains: a survey on domain generalization. [[paper]](https://jd92.wang/assets/files/DG_survey_TKDE22.pdf)
  - Jindong Wang , Cuiling Lan, Chang Liu, Yidong Ouyang, Tao Qin, Wang Lu, Yiqiang Chen, Wenjun Zeng, and Philip S. Yu. *IEEE TKDE 2022*
  - Key Word: Out-of-distribution generalization; Domain generalization
  - <details><summary>Digest</summary> This paper presents the first review of recent advances in this area. First, we provide a formal definition of domain generalization and discuss several related fields. We then thoroughly review the theories related to domain generalization and carefully analyze the theory behind generalization.

- Stable learning establishes some common ground between causal inference and machine learning. [[paper]](https://www.nature.com/articles/s42256-022-00445-z)
  - Peng Cui, Susan Athey. *Nature Machine Intelligence*
  - Key Word: Stable Learning; Causal Inference.
  - <details><summary>Digest</summary> With the aim of bridging the gap between the tradition of precise modelling in causal inference and black-box approaches from machine learning, stable learning is proposed and developed as a source of common ground. This Perspective clarifies a source of risk for machine learning models and discusses the benefits of bringing causality into learning.

- CrossMatch: Cross-Classifier Consistency Regularization for Open-Set Single Domain Generalization. [[paper]](https://openreview.net/forum?id=48RBsJwGkJf)
  - Ronghang Zhu, Sheng Li. *ICLR 2022*
  - Key Word: Single Domain Generalization, Open-Set Recognition.
  - <details><summary>Digest</summary> We propose a challenging and untouched problem: Open-Set Single Domain Generalization (OS-SDG), where target domains include unseen categories out of source label space. The goal of OS-SDG is to learn a model, with only one source domain, to classify a target sample with correct class if it belongs to source label space, or assign it to unknown classes. We design a CrossMatch approach to improve the performance of SDG methods on identifying unknown classes by leveraging a multi-binary classifier.

- Invariant Causal Representation Learning for Out-of-Distribution Generalization. [[paper]](https://openreview.net/forum?id=-e4EXDWXnSn)
  - Chaochao Lu, Yuhuai Wu, José Miguel Hernández-Lobato, Bernhard Schölkopf. *ICLR 2022*
  - Key Word: Out-of-Distribution Generalization; Invariant Causal Prediction; Causal Representation Learning.
  - <details><summary>Digest</summary> We propose invariant Causal Representation Learning (iCaRL), an approach that enables out-of-distribution (OOD) generalization in the nonlinear setting (i.e., nonlinear representations and nonlinear classifiers). It builds upon a practical and general assumption: the prior over the data representation (i.e., a set of latent variables encoding the data) given the target and the environment belongs to general exponential family distributions, i.e., a more flexible conditionally non-factorized prior that can actually capture complicated dependences between the latent variables.

- Target Conditioned Representation Independence (TCRI); From Domain-Invariant to Domain-General Representations. [[paper]](https://arxiv.org/abs/2212.11342)
  - Olawale Salaudeen, Oluwasanmi Koyejo.
  - Key Word: Domain Generalization; Invaraint Leanring.
  - <details><summary>Digest</summary> We propose a Target Conditioned Representation Independence (TCRI) objective for domain generalization. TCRI addresses the limitations of existing domain generalization methods due to incomplete constraints. Specifically, TCRI implements regularizers motivated by conditional independence constraints that are sufficient to strictly learn complete sets of invariant mechanisms, which we show are necessary and sufficient for domain generalization.

- Recycling diverse models for out-of-distribution generalization. [[paper]](https://arxiv.org/abs/2212.10445)
  - Alexandre Ramé, Kartik Ahuja, Jianyu Zhang, Matthieu Cord, Léon Bottou, David Lopez-Paz.
  - Key Word: Out-of-Distribution Generalization; Fine-Tuning; Weight Averaging.
  - <details><summary>Digest</summary> We propose model recycling, a simple strategy that leverages multiple fine-tunings of the same foundation model on diverse auxiliary tasks, and repurposes them as rich and diverse initializations for the target task. Specifically, model recycling fine-tunes in parallel each specialized model on the target task, and then averages the weights of all target fine-tunings into a final model. 

- Domain Generalization by Learning and Removing Domain-specific Features. [[paper]](https://arxiv.org/abs/2212.07101) [[code]](https://github.com/yulearningg/LRDG)
  - Yu Ding, Lei Wang, Bin Liang, Shuming Liang, Yang Wang, Fang Chen.
  - Key Word: Domain Generalization; Domain-Specific Features.
  - <details><summary>Digest</summary> We propose a novel framework called Learning and Removing Domain-specific features for Generalization (LRDG) that learns a domain-invariant model by tactically removing domain-specific features from the input images. Specifically, we design a classifier to effectively learn the domain-specific features for each source domain, respectively. We then develop an encoder-decoder network to map each input image into a new image space where the learned domain-specific features are removed. 

- Minimax Optimal Estimation of Stability Under Distribution Shift. [[paper]](https://arxiv.org/abs/2212.06338)
  - Hongseok Namkoong, Yuanzhe Ma, Peter W. Glynn.
  - Key Word: Stability; Robustness; Distributional Uncertainty; Model Misspecification.
  - <details><summary>Digest</summary> The performance of decision policies and prediction models often deteriorates when applied to environments different from the ones seen during training. To ensure reliable operation, we propose and analyze the stability of a system under distribution shift, which is defined as the smallest change in the underlying environment that causes the system's performance to deteriorate beyond a permissible threshold. In contrast to standard tail risk measures and distributionally robust losses that require the specification of a plausible magnitude of distribution shift, the stability measure is defined in terms of a more intuitive quantity: the level of acceptable performance degradation.

- Spuriosity Rankings: Sorting Data for Spurious Correlation Robustness. [[paper]](https://arxiv.org/abs/2212.02648)
  - Mazda Moayeri, Wenxiao Wang, Sahil Singla, Soheil Feizi.
  - Key Word: Mitigating Spurious Correlations; Spurious Correlation Benchmarks; Natural Image-based Interpretability.
  - <details><summary>Digest</summary> We present a framework for ranking images within their class based on the strength of spurious cues present. By measuring the gap in accuracy on the highest and lowest ranked images (we call this spurious gap), we assess spurious feature reliance for 89 diverse ImageNet models, finding that even the best models underperform in images with weak spurious presence. 

- Avoiding spurious correlations via logit correction. [[paper]](https://arxiv.org/abs/2212.01433)
  - Sheng Liu, Xu Zhang, Nitesh Sekhar, Yue Wu, Prateek Singhal, Carlos Fernandez-Granda.
  - Key Word: Mitigating Spurious Correlations; Logit Correction.
  - <details><summary>Digest</summary> We explicitly consider a situation where potential spurious correlations are present in the majority of training data. In contrast with existing approaches, which use the ERM model outputs to detect the samples without spurious correlations, and either heuristically upweighting or upsampling those samples; we propose the logit correction (LC) loss, a simple yet effective improvement on the softmax cross-entropy loss, to correct the sample logit. 

- Malign Overfitting: Interpolation Can Provably Preclude Invariance. [[paper]](https://arxiv.org/abs/2211.15724)
  - Yoav Wald, Gal Yona, Uri Shalit, Yair Carmon.
  - Key Word: Invariant Learning; Overfitting; Interpolation.
  - <details><summary>Digest</summary> In this work we provide a theoretical justification for these observations. We prove that -- even in the simplest of settings -- any interpolating learning rule (with arbitrarily small margin) will not satisfy these invariance properties. We then propose and analyze an algorithm that -- in the same setting -- successfully learns a non-interpolating classifier that is provably invariant. We validate our theoretical observations on simulated data and the Waterbirds dataset.

- Chroma-VAE: Mitigating Shortcut Learning with Generative Classifiers. [[paper]](https://arxiv.org/abs/2211.15231)
  - Wanqian Yang, Polina Kirichenko, Micah Goldblum, Andrew Gordon Wilson. *NeurIPS 2022*
  - Key Word: Shortcut Learning; Generative Classifiers; Group Robustness.
  - <details><summary>Digest</summary> We show that generative models alone are not sufficient to prevent shortcut learning, despite an incentive to recover a more comprehensive representation of the data than discriminative approaches. However, we observe that shortcuts are preferentially encoded with minimal information, a fact that generative models can exploit to mitigate shortcut learning. In particular, we propose Chroma-VAE, a two-pronged approach where a VAE classifier is initially trained to isolate the shortcut in a small latent subspace, allowing a secondary classifier to be trained on the complementary, shortcut-free latent subspace. 

- Wild-Time: A Benchmark of in-the-Wild Distribution Shift over Time. [[paper]](https://arxiv.org/abs/2211.14238) [[code]](https://github.com/huaxiuyao/wild-time)
  - Huaxiu Yao, Caroline Choi, Bochuan Cao, Yoonho Lee, Pang Wei Koh, Chelsea Finn. *NeurIPS 2022*
  - Key Word: Temporal Distribution Shift.
  - <details><summary>Digest</summary> By leveraging timestamp metadata, models can potentially learn from trends in past distribution shifts and extrapolate into the future. While recent works have studied distribution shifts, temporal shifts remain underexplored. To address this gap, we curate Wild-Time, a benchmark of 5 datasets that reflect temporal distribution shifts arising in a variety of real-world applications, including patient prognosis and news classification. On these datasets, we systematically benchmark 13 prior approaches, including methods in domain generalization, continual learning, self-supervised learning, and ensemble learning. 

- First Steps Toward Understanding the Extrapolation of Nonlinear Models to Unseen Domains. [[paper]](https://arxiv.org/abs/2211.11719)
  - Kefan Dong, Tengyu Ma.
  - Key Word: Domain Generalization; Domain Generalization; Extrapolation.
  - <details><summary>Digest</summary> This paper makes some initial steps towards analyzing the extrapolation of nonlinear models for structured domain shift. We primarily consider settings where the marginal distribution of each coordinate of the data (or subset of coordinates) do not shift significantly across the training and test distributions, but the joint distribution may have a much bigger shift. 

- Invariant Learning via Diffusion Dreamed Distribution Shifts. [[paper]](https://arxiv.org/abs/2211.10370)
  - Priyatham Kattakinda, Alexander Levine, Soheil Feizi.
  - Key Word: Foreground and Background; Distribution Shifts; Text-to-Image Diffusion Models.
  - <details><summary>Digest</summary> We propose such a dataset called Diffusion Dreamed Distribution Shifts (D3S). D3S consists of synthetic images generated through StableDiffusion using text prompts and image guides obtained by pasting a sample foreground image onto a background template image. Using this scalable approach we generate 120K images of objects from all 1000 ImageNet classes in 10 diverse backgrounds.

- ImageNet-X: Understanding Model Mistakes with Factor of Variation Annotations. [[paper]](https://arxiv.org/abs/2211.01866)
  - Badr Youbi Idrissi, Diane Bouchacourt, Randall Balestriero, Ivan Evtimov, Caner Hazirbas, Nicolas Ballas, Pascal Vincent, Michal Drozdzal, David Lopez-Paz, Mark Ibrahim.
  - Key Word: Out-of-Distribution Generalization; Dataset.
  - <details><summary>Digest</summary> Deep learning vision systems are widely deployed across applications where reliability is critical. However, even today's best models can fail to recognize an object when its pose, lighting, or background varies. While existing benchmarks surface examples challenging for models, they do not explain why such mistakes arise. To address this need, we introduce ImageNet-X, a set of sixteen human annotations of factors such as pose, background, or lighting the entire ImageNet-1k validation set as well as a random subset of 12k training images. 

- Functional Indirection Neural Estimator for Better Out-of-distribution Generalization. [[paper]](https://arxiv.org/abs/2210.12739)
  - Kha Pham, Hung Le, Man Ngo, Truyen Tran. *NeurIPS 2022*
  - Key Word: Out-of-Distribution Generalization; Functional Indirection Neural Estimator.
  - <details><summary>Digest</summary> We hypothesize that OOD generalization may be achieved by performing analogy-making and indirection in the functional space instead of the data space as in current methods. To realize this, we design FINE (Functional Indirection Neural Estimator), a neural framework that learns to compose functions that map data input to output on-the-fly. FINE consists of a backbone network and a trainable semantic memory of basis weight matrices. 

- Just Mix Once: Worst-group Generalization by Group Interpolation. [[paper]](https://arxiv.org/abs/2210.12195)
  - Giorgio Giannone, Serhii Havrylov, Jordan Massiah, Emine Yilmaz, Yunlong Jiao. 
  - Key Word: Out-of-Distribution Generalization; Data Augmentation; Mixup; Group Robustness.
  - <details><summary>Digest</summary> A recent line of work leverages self-supervision and oversampling to improve generalization on minority groups without group annotation. We propose to unify and generalize these approaches using a class-conditional variant of mixup tailored for worst-group generalization. Our approach, Just Mix Once (JM1), interpolates samples during learning, augmenting the training distribution with a continuous mixture of groups. JM1 is domain agnostic and computationally efficient, can be used with any level of group annotation, and performs on par or better than the state-of-the-art on worst-group generalization. 

- On Feature Learning in the Presence of Spurious Correlations. [[paper]](https://arxiv.org/abs/2210.11369) [[code]](https://github.com/izmailovpavel/spurious_feature_learning)
  - Pavel Izmailov, Polina Kirichenko, Nate Gruver, Andrew Gordon Wilson. *NeurIPS 2022*
  - Key Word: Spurious Correlations; Feature Learning.
  - <details><summary>Digest</summary> We evaluate the amount of information about the core (non-spurious) features that can be decoded from the representations learned by standard empirical risk minimization (ERM) and specialized group robustness training. Following recent work on Deep Feature Reweighting (DFR), we evaluate the feature representations by re-training the last layer of the model on a held-out set where the spurious correlation is broken. 

- Freeze then Train: Towards Provable Representation Learning under Spurious Correlations and Feature Noise. [[paper]](https://arxiv.org/abs/2210.11075)
  - Haotian Ye, James Zou, Linjun Zhang.
  - Key Word: Feature Learning; Spurious Correlations.
  - <details><summary>Digest</summary> We find that core features are only learned well when they are less noisy than spurious features, which is not necessarily true in practice. We provide both theories and experiments to support this finding and to illustrate the importance of feature noise. Moreover, we propose an algorithm called Freeze then Train (FTT), that first freezes certain salient features and then trains the rest of the features using ERM.

- ZooD: Exploiting Model Zoo for Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2210.09236)
  - Qishi Dong, Awais Muhammad, Fengwei Zhou, Chuanlong Xie, Tianyang Hu, Yongxin Yang, Sung-Ho Bae, Zhenguo Li. *NeurIPS 2022*
  - Key Word: Out-of-Distribution Generalization; Pre-Training.
  - <details><summary>Digest</summary> We propose ZooD, a paradigm for PTMs ranking and ensemble with feature selection. Our proposed metric ranks PTMs by quantifying inter-class discriminability and inter-domain stability of the features extracted by the PTMs in a leave-one-domain-out cross-validation manner. The top-K ranked models are then aggregated for the target OoD task. To avoid accumulating noise induced by model ensemble, we propose an efficient variational EM algorithm to select informative features. We evaluate our paradigm on a diverse model zoo consisting of 35 models for various OoD tasks.

- Improving Out-of-Distribution Generalization by Adversarial Training with Structured Priors. [[paper]](https://arxiv.org/abs/2210.06807) [[code]](https://github.com/NOVAglow646/NIPS22-MAT-and-LDAT-for-OOD)
  - Qixun Wang, Yifei Wang, Hong Zhu, Yisen Wang. *NeurIPS 2022*
  - Key Word: Out-of-Distribution Generalization; Adversarial Training.
  - <details><summary>Digest</summary> We empirically show that sample-wise AT has limited improvement on OOD performance. Specifically, we find that AT can only maintain performance at smaller scales of perturbation while Universal AT (UAT) is more robust to larger-scale perturbations. This provides us with clues that adversarial perturbations with universal (low dimensional) structures can enhance the robustness against large data distribution shifts that are common in OOD scenarios. Inspired by this, we propose two AT variants with low-rank structures to train OOD-robust models.

- Revisiting adapters with adversarial training. [[paper]](https://arxiv.org/abs/2210.04886)
  - Sylvestre-Alvise Rebuffi, Francesco Croce, Sven Gowal. 
  - Key Word: Adversarial Training; Adapters.
  - <details><summary>Digest</summary> We improve upon the top-1 accuracy of a non-adversarially trained ViT-B16 model by +1.12% on ImageNet (reaching 83.76% top-1 accuracy). Second, and more importantly, we show that training with adapters enables model soups through linear combinations of the clean and adversarial tokens. These model soups, which we call adversarial model soups, allow us to trade-off between clean and robust accuracy without sacrificing efficiency. Finally, we show that we can easily adapt the resulting models in the face of distribution shifts. 

- FLamby: Datasets and Benchmarks for Cross-Silo Federated Learning in Realistic Healthcare Settings. [[paper]](https://arxiv.org/abs/2210.04620)
  - Jean Ogier du Terrail, Samy-Safwan Ayed, Edwige Cyffers, Felix Grimberg, Chaoyang He, Regis Loeb, Paul Mangold, Tanguy Marchand, Othmane Marfoq, Erum Mushtaq, Boris Muzellec, Constantin Philippenko, Santiago Silva, Maria Teleńczuk, Shadi Albarqouni, Salman Avestimehr, Aurélien Bellet, Aymeric Dieuleveut, Martin Jaggi, Sai Praneeth Karimireddy, Marco Lorenzi, Giovanni Neglia, Marc Tommasi, Mathieu Andreux. *NeurIPS 2022*
  - Key Word: Federated Learning; Healthcare; Benchmarks.
  - <details><summary>Digest</summary> We propose a novel cross-silo dataset suite focused on healthcare, FLamby (Federated Learning AMple Benchmark of Your cross-silo strategies), to bridge the gap between theory and practice of cross-silo FL. FLamby encompasses 7 healthcare datasets with natural splits, covering multiple tasks, modalities, and data volumes, each accompanied with baseline training code. 

- Coresets for Wasserstein Distributionally Robust Optimization Problems. [[paper]](https://arxiv.org/abs/2210.04260)
  - Ruomin Huang, Jiawei Huang, Wenjie Liu, Hu Ding. *NeurIPS 2022*
  - Key Word: Coresets; Distributionally Robust Optimization.
  - <details><summary>Digest</summary> We introduce a unified framework to construct the ϵ-coreset for the general WDRO problems. Though it is challenging to obtain a conventional coreset for WDRO due to the uncertainty issue of ambiguous data, we show that we can compute a ''dual coreset'' by using the strong duality property of WDRO.

- Attention Diversification for Domain Generalization. [[paper]](https://arxiv.org/abs/2210.04206) [[code]](https://github.com/hikvision-research/DomainGeneralization)
  - Rang Meng, Xianfeng Li, Weijie Chen, Shicai Yang, Jie Song, Xinchao Wang, Lei Zhang, Mingli Song, Di Xie, Shiliang Pu. *ECCV 2022*
  - Key Word: Domain Generalization; Attention Diversification.
  - <details><summary>Digest</summary> We find the devils lie in the fact that models trained on different domains merely bias to different domain-specific features yet overlook diverse task-related features. Under this guidance, a novel Attention Diversification framework is proposed, in which Intra-Model and Inter-Model Attention Diversification Regularization are collaborated to reassign appropriate attention to diverse task-related features. 

- Tikhonov Regularization is Optimal Transport Robust under Martingale Constraints. [[paper]](https://arxiv.org/abs/2210.01413)
  - Jiajin Li, Sirui Lin, Jose Blanchet, Viet Anh Nguyen. *NeurIPS 2022*
  - Key Word: Optimal Transport; Distributionally Robust Optimization; Tikhonov Regularization.
  - <details><summary>Digest</summary> Distributionally robust optimization has been shown to offer a principled way to regularize learning models. In this paper, we find that Tikhonov regularization is distributionally robust in an optimal transport sense (i.e., if an adversary chooses distributions in a suitable optimal transport neighborhood of the empirical measure), provided that suitable martingale constraints are also imposed. Further, we introduce a relaxation of the martingale constraints which not only provides a unified viewpoint to a class of existing robust methods but also leads to new regularization tools. 

- Nuisances via Negativa: Adjusting for Spurious Correlations via Data Augmentation. [[paper]](https://arxiv.org/abs/2210.01302)
  - Aahlad Puli, Nitish Joshi, He He, Rajesh Ranganath. 
  - Key Word: Data Augmentation; Robustness to Spurious Correlations.
  - <details><summary>Digest</summary> We develop an alternative way to produce robust models by data augmentation. These data augmentations corrupt semantic information to produce models that identify and adjust for where nuisances drive predictions. We study semantic corruptions in powering different robust-modeling methods for multiple out-of distribution (OOD) tasks like classifying waterbirds, natural language inference, and detecting Cardiomegaly in chest X-rays.

- Federated Representation Learning via Maximal Coding Rate Reduction. [[paper]](https://arxiv.org/abs/2210.00299)
  - Juan Cervino, Navid NaderiAlizadeh, Alejandro Ribeiro.
  - Key Word: Personalized Federated Learning; Maximal Coding Rate Reduction.
  - <details><summary>Digest</summary> We propose a federated methodology to learn low-dimensional representations from a dataset that is distributed among several clients. In particular, we move away from the commonly-used cross-entropy loss in federated learning, and seek to learn shared low-dimensional representations of the data in a decentralized manner via the principle of maximal coding rate reduction (MCR2). Our proposed method, which we refer to as FLOW, utilizes MCR2 as the objective of choice, hence resulting in representations that are both between-class discriminative and within-class compressible. 

- MaskTune: Mitigating Spurious Correlations by Forcing to Explore. [[paper]](https://arxiv.org/abs/2210.00055) [[code]](https://github.com/aliasgharkhani/masktune)
  - Saeid Asgari Taghanaki, Aliasghar Khani, Fereshte Khani, Ali Gholami, Linh Tran, Ali Mahdavi-Amiri, Ghassan Hamarneh. *NeurIPS 2022*
  - Key Word: Input Masking; Robustness to Spurious Correlations.
  - <details><summary>Digest</summary> A fundamental challenge of over-parameterized deep learning models is learning meaningful data representations that yield good performance on a downstream task without over-fitting spurious input features. This work proposes MaskTune, a masking strategy that prevents over-reliance on spurious (or a limited number of) features. MaskTune forces the trained model to explore new features during a single epoch finetuning by masking previously discovered features. MaskTune, unlike earlier approaches for mitigating shortcut learning, does not require any supervision, such as annotating spurious features or labels for subgroup samples in a dataset. 

- Domain Generalization -- A Causal Perspective. [[paper]](https://arxiv.org/abs/2209.15177)
  - Paras Sheth, Raha Moraffah, K. Selçuk Candan, Adrienne Raglin, Huan Liu.
  - Key Word: Domain Generalization; Causality; Survey.
  - <details><summary>Digest</summary> We present a comprehensive survey on causal domain generalization models from the aspects of the problem and causal theories. Furthermore, this survey includes in-depth insights into publicly accessible datasets and benchmarks for domain generalization in various domains. Finally, we conclude the survey with insights and discussions on future research directions. Finally, we conclude the survey with insights and discussions on future research directions.

- A Closer Look at Learned Optimization: Stability, Robustness, and Inductive Biases. [[paper]](https://arxiv.org/abs/2209.11208)
  - James Harrison, Luke Metz, Jascha Sohl-Dickstein. *NeurIPS 2022*
  - Key Word: Optimizer; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> We use tools from dynamical systems to investigate the inductive biases and stability properties of optimization algorithms, and apply the resulting insights to designing inductive biases for blackbox optimizers. Our investigation begins with a noisy quadratic model, where we characterize conditions in which optimization is stable, in terms of eigenvalues of the training dynamics. We then introduce simple modifications to a learned optimizer's architecture and meta-training procedure which lead to improved stability, and improve the optimizer's inductive bias.

- UMIX: Improving Importance Weighting for Subpopulation Shift via Uncertainty-Aware Mixup. [[paper]](https://arxiv.org/abs/2209.08928)
  - Zongbo Han, Zhipeng Liang, Fan Yang, Liu Liu, Lanqing Li, Yatao Bian, Peilin Zhao, Bingzhe Wu, Changqing Zhang, Jianhua Yao. *NeurIPS 2022*
  - Key Word: Importance Weighting; Subpopulation Shift; Mixup.
  - <details><summary>Digest</summary> We propose a simple yet practical framework, called uncertainty-aware mixup (Umix), to mitigate the overfitting issue in over-parameterized models by reweighting the "mixed" samples according to the sample uncertainty. The training-trajectories-based uncertainty estimation is equipped in the proposed Umix for each sample to flexibly characterize the subpopulation distribution. We also provide insightful theoretical analysis to verify that Umix achieves better generalization bounds over prior works.

- Importance Tempering: Group Robustness for Overparameterized Models. [[paper]](https://arxiv.org/abs/2209.08745)
  - Yiping Lu, Wenlong Ji, Zachary Izzo, Lexing Ying.
  - Ke Word: Importance Tempering; Label Shift; Neural Collapse; Spurious Correlations.
  - <details><summary>Digest</summary> We propose importance tempering to improve the decision boundary and achieve consistently better results for overparameterized models. Theoretically, we justify that the selection of group temperature can be different under label shift and spurious correlation setting. At the same time, we also prove that properly selected temperatures can extricate the minority collapse for imbalanced classification. 

- On-Device Domain Generalization. [[paper]](https://arxiv.org/abs/2209.07521) [[code]](https://github.com/KaiyangZhou/on-device-dg)
  - Kaiyang Zhou, Yuanhan Zhang, Yuhang Zang, Jingkang Yang, Chen Change Loy, Ziwei Liu.
  - Key Word: Domain Generalization; Knowledge Distillation.
  - <details><summary>Digest</summary> We find that knowledge distillation is a strong candidate for solving the problem: it outperforms state-of-the-art DG methods that were developed using large models with a large margin. Moreover, we observe that the teacher-student performance gap on test data with domain shift is bigger than that on in-distribution data. To improve DG for tiny neural networks without increasing the deployment cost, we propose a simple idea called out-of-distribution knowledge distillation (OKD), which aims to teach the student how the teacher handles (synthetic) out-of-distribution data and is proved to be a promising framework for solving the problem. 

- ID and OOD Performance Are Sometimes Inversely Correlated on Real-world Datasets. [[paper]](https://arxiv.org/abs/2209.00613)
  - Damien Teney, Seong Joon Oh, Ehsan Abbasnejad.
  - Key Word: Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> This short paper shows that inverse correlations between ID and OOD performance do happen in real-world benchmarks. They may have been missed in past studies because of a biased selection of models. We show an example of the pattern on the WILDS-Camelyon17 dataset, using models from multiple training epochs and random seeds. Our observations are particularly striking on models trained with a regularizer that diversifies the solutions to the ERM objective.

- DR-DSGD: A Distributionally Robust Decentralized Learning Algorithm over Graphs. [[paper]](https://arxiv.org/abs/2208.13810)
  - Chaouki Ben Issaid, Anis Elgabli, Mehdi Bennis. *TMLR*
  - Key Word: Robust Federated Learning; Fairness in Federated Learning; Decentralized Learning.
  - <details><summary>Digest</summary> We propose to solve a regularized distributionally robust learning problem in the decentralized setting, taking into account the data distribution shift. By adding a Kullback-Liebler regularization function to the robust min-max optimization problem, the learning problem can be reduced to a modified robust minimization problem and solved efficiently. Leveraging the newly formulated optimization problem, we propose a robust version of Decentralized Stochastic Gradient Descent (DSGD), coined Distributionally Robust Decentralized Stochastic Gradient Descent (DR-DSGD). 

- Shortcut Learning of Large Language Models in Natural Language Understanding: A Survey. [[paper]](https://arxiv.org/abs/2208.11857)
  - Mengnan Du, Fengxiang He, Na Zou, Dacheng Tao, Xia Hu.
  - Key Word: Survey; Shortcut Learning; Out-of-Distribution Generalization; Large Language Models.
  - <details><summary>Digest</summary> Large language models (LLMs) have achieved state-of-the-art performance on a series of natural language understanding tasks. However, these LLMs might rely on dataset bias and artifacts as shortcuts for prediction. This has significantly hurt their Out-of-Distribution (OOD) generalization and adversarial robustness. In this paper, we provide a review of recent developments that address the robustness challenge of LLMs. We first introduce the concepts and robustness challenge of LLMs. We then introduce methods to identify shortcut learning behavior in LLMs, characterize the reasons for shortcut learning, as well as introduce mitigation solutions. Finally, we identify key challenges and introduce the connections of this line of research to other directions.

- A Unified Causal View of Domain Invariant Representation Learning. [[paper]](https://arxiv.org/abs/2208.06987) [[code]](https://github.com/zihao12/causal_da_code)
  - Zihao Wang, Victor Veitch.
  - Key Word: Causality; Data Augmentation; Invariant Learning.
  - <details><summary>Digest</summary> Machine learning methods can be unreliable when deployed in domains that differ from the domains on which they were trained. To address this, we may wish to learn representations of data that are domain-invariant in the sense that we preserve data structure that is stable across domains, but throw out spuriously-varying parts. There are many representation-learning approaches of this type, including methods based on data augmentation, distributional invariances, and risk invariance. Unfortunately, when faced with any particular real-world domain shift, it is unclear which, if any, of these methods might be expected to work. The purpose of this paper is to show how the different methods relate to each other, and clarify the real-world circumstances under which each is expected to succeed. The key tool is a new notion of domain shift relying on the idea that causal relationships are invariant, but non-causal relationships (e.g., due to confounding) may vary.

- Class Is Invariant to Context and Vice Versa: On Learning Invariance for Out-Of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2208.03462) [[code]](https://github.com/simpleshinobu/irmcon)
  - Jiaxin Qi, Kaihua Tang, Qianru Sun, Xian-Sheng Hua, Hanwang Zhang. *ECCV 2022*
  - Key Word: Invarinat Learning; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> We argue that the widely adopted assumption in prior work, the context bias can be directly annotated or estimated from biased class prediction, renders the context incomplete or even incorrect. In contrast, we point out the everoverlooked other side of the above principle: context is also invariant to class, which motivates us to consider the classes (which are already labeled) as the varying environments to resolve context bias (without context labels). We implement this idea by minimizing the contrastive loss of intra-class sample similarity while assuring this similarity to be invariant across all classes.

- Self-Distilled Vision Transformer for Domain Generalization. [[paper]](https://arxiv.org/abs/2207.12392) [[code]](https://github.com/maryam089/SDViT)
  - Maryam Sultana, Muzammal Naseer, Muhammad Haris Khan, Salman Khan, Fahad Shahbaz Khan. *ECCV 2022*
  - Key Word: Domain Generalization; Vision Transformers; Self Distillation.
  - <details><summary>Digest</summary> We attempt to explore ViTs towards addressing the DG problem. Similar to CNNs, ViTs also struggle in out-of-distribution scenarios and the main culprit is overfitting to source domains. Inspired by the modular architecture of ViTs, we propose a simple DG approach for ViTs, coined as self-distillation for ViTs. It reduces the overfitting to source domains by easing the learning of input-output mapping problem through curating non-zero entropy supervisory signals for intermediate transformer blocks.

- Equivariance and Invariance Inductive Bias for Learning from Insufficient Data. [[paper]](https://arxiv.org/abs/2207.12258) [[code]](https://github.com/Wangt-CN/EqInv)
  - Tan Wang, Qianru Sun, Sugiri Pranata, Karlekar Jayashree, Hanwang Zhang. *ECCV 2022*
  - Key Word: Visual Inductive Bias; Data-Efficient Learning; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> First, compared to sufficient data, we show why insufficient data renders the model more easily biased to the limited training environments that are usually different from testing. For example, if all the training swan samples are "white", the model may wrongly use the "white" environment to represent the intrinsic class swan. Then, we justify that equivariance inductive bias can retain the class feature while invariance inductive bias can remove the environmental feature, leaving the class feature that generalizes to any environmental changes in testing. To impose them on learning, for equivariance, we demonstrate that any off-the-shelf contrastive-based self-supervised feature learning method can be deployed; for invariance, we propose a class-wise invariant risk minimization (IRM) that efficiently tackles the challenge of missing environmental annotation in conventional IRM.

- Domain-invariant Feature Exploration for Domain Generalization. [[paper]](https://arxiv.org/abs/2207.12020) [[code]](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG)
  - Wang Lu, Jindong Wang, Haoliang Li, Yiqiang Chen, Xing Xie.
  - Key Word: Domain Generalization; Fourier Features.
  - <details><summary>Digest</summary> We argue that domain-invariant features should be originating from both internal and mutual sides. Internal invariance means that the features can be learned with a single domain and the features capture intrinsic semantics of data, i.e., the property within a domain, which is agnostic to other domains. Mutual invariance means that the features can be learned with multiple domains (cross-domain) and the features contain common information, i.e., the transferable features w.r.t. other domains.

- Discrete Key-Value Bottleneck. [[paper]](https://arxiv.org/abs/2207.11240)
  - Frederik Träuble, Anirudh Goyal, Nasim Rahaman, Michael Mozer, Kenji Kawaguchi, Yoshua Bengio, Bernhard Schölkopf.
  - Key Word: Distribution Shifts; Catastrophic Forgetting; Memory Augmented Models.
  - <details><summary>Digest</summary> In the present work, we propose a model architecture to address this issue, building upon a discrete bottleneck containing pairs of separate and learnable (key, value) codes. In this setup, we follow the encode; process the representation via a discrete bottleneck; and decode paradigm, where the input is fed to the pretrained encoder, the output of the encoder is used to select the nearest keys, and the corresponding values are fed to the decoder to solve the current task. The model can only fetch and re-use a limited number of these (key, value) pairs during inference, enabling localized and context-dependent model updates.

- Domain generalization for activity recognition via adaptive feature fusion. [[paper]](https://arxiv.org/abs/2207.11221)
  - Xin Qin, Jindong Wang, Yiqiang Chen, Wang Lu, and Xinlong Jiang. *ACM TIST 2022*
  - Key Word: Domain generalization; Activity recognition.
  - <details><summary>Digest</summary> We propose Adaptive Feature Fusion for Activity Recognition (AFFAR), a domain generalization approach that learns to fuse the domain-invariant and domain-specific representations to improve the model’s generalization performance. AFFAR takes the best of both worlds where domain-invariant representations enhance the transferability across domains and domain-specific representations leverage the model discrimination power from each domain. 

- UniFed: A Benchmark for Federated Learning Frameworks. [[paper]](https://arxiv.org/abs/2207.10308) [[code]](https://github.com/ai-secure/flbenchmark-toolkit)
  - Xiaoyuan Liu, Tianneng Shi, Chulin Xie, Qinbin Li, Kangping Hu, Haoyu Kim, Xiaojun Xu, Bo Li, Dawn Song.
  - Key Word: Federated Learning; Benchmark; Privacy.
  - <details><summary>Digest</summary> Federated Learning (FL) has become a practical and popular paradigm in machine learning. However, currently, there is no systematic solution that covers diverse use cases. Practitioners often face the challenge of how to select a matching FL framework for their use case. In this work, we present UniFed, the first unified benchmark for standardized evaluation of the existing open-source FL frameworks. With 15 evaluation scenarios, we present both qualitative and quantitative evaluation results of nine existing popular open-sourced FL frameworks, from the perspectives of functionality, usability, and system performance. We also provide suggestions on framework selection based on the benchmark conclusions and point out future improvement directions.

- Grounding Visual Representations with Texts for Domain Generalization. [[paper]](https://arxiv.org/abs/2207.10285) [[code]](https://github.com/mswzeus/gvrt)
  - Seonwoo Min, Nokyung Park, Siwon Kim, Seunghyun Park, Jinkyu Kim. *ECCV 2022*
  - Key Word: Domain Generalization; Visual and Textual Explanations.
  - <details><summary>Digest</summary> We introduce two modules to ground visual representations with texts containing typical reasoning of humans: (1) Visual and Textual Joint Embedder and (2) Textual Explanation Generator. The former learns the image-text joint embedding space where we can ground high-level class-discriminative information into the model. The latter leverages an explainable model and generates explanations justifying the rationale behind its decision. To the best of our knowledge, this is the first work to leverage the vision-and-language cross-modality approach for the domain generalization task.

- Tackling Long-Tailed Category Distribution Under Domain Shifts. [[paper]](https://arxiv.org/abs/2207.10150) [[code]](https://github.com/guxiao0822/lt-ds)
  - Xiao Gu, Yao Guo, Zeju Li, Jianing Qiu, Qi Dou, Yuxuan Liu, Benny Lo, Guang-Zhong Yang. *ECCV 2022*
  - Key Word: Long-Tailed Category Distribution; Domain Generalization; Cross-Modal Learning.
  - <details><summary>Digest</summary> We took a step forward and looked into the problem of long-tailed classification under domain shifts. We designed three novel core functional blocks including Distribution Calibrated Classification Loss, Visual-Semantic Mapping and Semantic-Similarity Guided Augmentation. Furthermore, we adopted a meta-learning framework which integrates these three blocks to improve domain generalization on unseen target domains.

- Overcoming Shortcut Learning in a Target Domain by Generalizing Basic Visual Factors from a Source Domain. [[paper]](https://arxiv.org/abs/2207.10002) [[code]](https://github.com/boschresearch/sourcegen)
  - Piyapat Saranrittichai, Chaithanya Kumar Mummadi, Claudia Blaiotta, Mauricio Munoz, Volker Fischer. *ECCV 2022*
  - Key Word: Compositional Generalization; Domain Generalization; Learning Independent Representations.
  - <details><summary>Digest</summary> Shortcut learning occurs when a deep neural network overly relies on spurious correlations in the training dataset in order to solve downstream tasks. Prior works have shown how this impairs the compositional generalization capability of deep learning models. To address this problem, we propose a novel approach to mitigate shortcut learning in uncontrolled target domains. Our approach extends the training set with an additional dataset (the source domain), which is specifically designed to facilitate learning independent representations of basic visual factors. We benchmark our idea on synthetic target domains where we explicitly control shortcut opportunities as well as real-world target domains.

- Probable Domain Generalization via Quantile Risk Minimization. [[paper]](https://arxiv.org/abs/2207.09944) [[code]](https://github.com/cianeastwood/qrm)
  - Cian Eastwood, Alexander Robey, Shashank Singh, Julius von Kügelgen, Hamed Hassani, George J. Pappas, Bernhard Schölkopf.
  - Key Word: Domain Generalization; Causality; Invariant Learning.
  - <details><summary>Digest</summary> A recent study found that no DG algorithm outperformed empirical risk minimization in terms of average performance. In this work, we argue that DG is neither a worst-case problem nor an average-case problem, but rather a probabilistic one. To this end, we propose a probabilistic framework for DG, which we call Probable Domain Generalization, wherein our key idea is that distribution shifts seen during training should inform us of probable shifts at test time. To realize this, we explicitly relate training and test domains as draws from the same underlying meta-distribution, and propose a new optimization problem -- Quantile Risk Minimization (QRM) -- which requires that predictors generalize with high probability.

- Assaying Out-Of-Distribution Generalization in Transfer Learning. [[paper]](https://arxiv.org/abs/2207.09239)
  - Florian Wenzel, Andrea Dittadi, Peter Vincent Gehler, Carl-Johann Simon-Gabriel, Max Horn, Dominik Zietlow, David Kernert, Chris Russell, Thomas Brox, Bernt Schiele, Bernhard Schölkopf, Francesco Locatello.
  - Key Word: Out-of-Distribution Generalization; Transfer Learning; Calibration; Adversarial Robustness; Corruption Robustness; Invariant Learning.
  - <details><summary>Digest</summary> Since out-of-distribution generalization is a generally ill-posed problem, various proxy targets (e.g., calibration, adversarial robustness, algorithmic corruptions, invariance across shifts) were studied across different research programs resulting in different recommendations. While sharing the same aspirational goal, these approaches have never been tested under the same experimental conditions on real data. In this paper, we take a unified view of previous work, highlighting message discrepancies that we address empirically, and providing recommendations on how to measure the robustness of a model and how to improve it. To this end, we collect 172 publicly available dataset pairs for training and out-of-distribution evaluation of accuracy, calibration error, adversarial attacks, environment invariance, and synthetic corruptions. We fine-tune over 31k networks, from nine different architectures in the many- and few-shot setting.

- On the Strong Correlation Between Model Invariance and Generalization. [[paper]](https://arxiv.org/abs/2207.07065)
  - Weijian Deng, Stephen Gould, Liang Zheng.
  - Key Word: Predicting Generalization Gap; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> First, we introduce effective invariance (EI), a simple and reasonable measure of model invariance which does not rely on image labels. Given predictions on a test image and its transformed version, EI measures how well the predictions agree and with what level of confidence. Second, using invariance scores computed by EI, we perform large-scale quantitative correlation studies between generalization and invariance, focusing on rotation and grayscale transformations. From a model-centric view, we observe generalization and invariance of different models exhibit a strong linear relationship, on both in-distribution and out-of-distribution datasets. From a dataset-centric view, we find a certain model's accuracy and invariance linearly correlated on different test sets.

- Improved OOD Generalization via Conditional Invariant Regularizer. [[paper]](https://arxiv.org/abs/2207.06687)
  - Mingyang Yi, Ruoyu Wang, Jiachen Sun, Zhenguo Li, Zhi-Ming Ma.
  - Key Word: Out-of-Distribution Generalization; Conditional Spurious Variation.
  - <details><summary>Digest</summary> Recently, generalization on out-of-distribution (OOD) data with correlation shift has attracted great attention. The correlation shift is caused by the spurious attributes that correlate to the class label, as the correlation between them may vary in training and test data. For such a problem, we show that given the class label, the conditionally independent models of spurious attributes are OOD generalizable. Based on this, a metric Conditional Spurious Variation (CSV) which controls OOD generalization error, is proposed to measure such conditional independence. To improve the OOD generalization, we regularize the training process with the proposed CSV.

- Models Out of Line: A Fourier Lens on Distribution Shift Robustness. [[paper]](https://arxiv.org/abs/2207.04075)
  - Sara Fridovich-Keil, Brian R. Bartoldson, James Diffenderfer, Bhavya Kailkhura, Peer-Timo Bremer.
  - Key Word: Predicting Out-of-Distribution Generalization; Frequency Analysis.
  - <details><summary>Digest</summary> There still is no clear understanding of the conditions on OOD data and model properties that are required to observe effective robustness. We approach this issue by conducting a comprehensive empirical study of diverse approaches that are known to impact OOD robustness on a broad range of natural and synthetic distribution shifts of CIFAR-10 and ImageNet. In particular, we view the "effective robustness puzzle" through a Fourier lens and ask how spectral properties of both models and OOD data influence the corresponding effective robustness.

- Predicting is not Understanding: Recognizing and Addressing Underspecification in Machine Learning. [[paper]](https://arxiv.org/abs/2207.02598)
  - Damien Teney, Maxime Peyrard, Ehsan Abbasnejad. *ECCV 2022*
  - Key Word: Out-of-Distribution Generalization; Underspecification; Ensembles; Feature Diversity.
  - <details><summary>Digest</summary> We formalize the concept of underspecification and propose a method to identify and partially address it. We train multiple models with an independence constraint that forces them to implement different functions. They discover predictive features that are otherwise ignored by standard empirical risk minimization (ERM), which we then distill into a global model with superior OOD performance. Importantly, we constrain the models to align with the data manifold to ensure that they discover meaningful features. We demonstrate the method on multiple datasets in computer vision (collages, WILDS-Camelyon17, GQA) and discuss general implications of underspecification. Most notably, in-domain performance cannot serve for OOD model selection without additional assumptions.

- Neural Networks and the Chomsky Hierarchy. [[paper]](https://arxiv.org/abs/2207.02098) [[code]](https://github.com/deepmind/neural_networks_chomsky_hierarchy)
  - Grégoire Delétang, Anian Ruoss, Jordi Grau-Moya, Tim Genewein, Li Kevin Wenliang, Elliot Catt, Marcus Hutter, Shane Legg, Pedro A. Ortega.
  - Key Word: Chomsky Hierarchy; Out-of-Distribution Generalization;
  - <details><summary>Digest</summary> Reliable generalization lies at the heart of safe ML and AI. However, understanding when and how neural networks generalize remains one of the most important unsolved problems in the field. In this work, we conduct an extensive empirical study (2200 models, 16 tasks) to investigate whether insights from the theory of computation can predict the limits of neural network generalization in practice. We demonstrate that grouping tasks according to the Chomsky hierarchy allows us to forecast whether certain architectures will be able to generalize to out-of-distribution inputs.

- Multi-modal Robustness Analysis Against Language and Visual Perturbations. [[paper]](https://arxiv.org/abs/2207.02159) [[code]](https://github.com/Maddy12/MultiModalVideoRobustness/tree/master/code)  
  - Madeline C. Schiappa, Yogesh S. Rawat, Shruti Vyas, Vibhav Vineet, Hamid Palangi.
  - Key Word: Corruption Robustness; Multi-modal Robustness; Text-to-Video Retrieval.
  - <details><summary>Digest</summary> Joint visual and language modeling on large-scale datasets has recently shown a good progress in multi-modal tasks when compared to single modal learning. However, robustness of these approaches against real-world perturbations has not been studied. In this work, we perform the first extensive robustness study of such models against various real-world perturbations focusing on video and language. We focus on text-to-video retrieval and propose two large-scale benchmark datasets, MSRVTT-P and YouCook2-P, which utilize 90 different visual and 35 different textual perturbations.

- Predicting Out-of-Domain Generalization with Local Manifold Smoothness. [[paper]](https://arxiv.org/abs/2207.02093)
  - Nathan Ng, Kyunghyun Cho, Neha Hulkund, Marzyeh Ghassemi.
  - Key Word: Measures of Complexity; Predicting Out-of-Distribution Generalization; Measuring Function Smoothness.
  - <details><summary>Digest</summary> Recent work has proposed a variety of complexity measures that directly predict or theoretically bound the generalization capacity of a model. However, these methods rely on a strong set of assumptions that in practice are not always satisfied. Motivated by the limited settings in which existing measures can be applied, we propose a novel complexity measure based on the local manifold smoothness of a classifier. We define local manifold smoothness as a classifier's output sensitivity to perturbations in the manifold neighborhood around a given test point. Intuitively, a classifier that is less sensitive to these perturbations should generalize better.

- Benchmarking the Robustness of Deep Neural Networks to Common Corruptions in Digital Pathology. [[paper]](https://arxiv.org/abs/2206.14973) [[code]](https://github.com/superjamessyx/robustness_benchmark)
  - Yunlong Zhang, Yuxuan Sun, Honglin Li, Sunyi Zheng, Chenglu Zhu, Lin Yang. *MICCAI 2022*
  - Key Word: Corruption Robustness; Digital Pathology.
  - <details><summary>Digest</summary> When designing a diagnostic model for a clinical application, it is crucial to guarantee the robustness of the model with respect to a wide range of image corruptions. Herein, an easy-to-use benchmark is established to evaluate how deep neural networks perform on corrupted pathology images. Specifically, corrupted images are generated by injecting nine types of common corruptions into validation images. Besides, two classification and one ranking metrics are designed to evaluate the prediction and confidence performance under corruption.

- Towards out of distribution generalization for problems in mechanics. [[paper]](https://arxiv.org/abs/2206.14917)
  - Lingxiao Yuan, Harold S. Park, Emma Lejeune.
  - Key Word: Out-of-Distribution Generalization; Invariant Learning.
  - <details><summary>Digest</summary> Out-of-distribution (OOD) generalization assumes that the test data may shift (i.e., violate the i.i.d. assumption). To date, multiple methods have been proposed to improve the OOD generalization of ML methods. However, because of the lack of benchmark datasets for OOD regression problems, the efficiency of these OOD methods on regression problems, which dominate the mechanics field, remains unknown. To address this, we investigate the performance of OOD generalization methods for regression problems in mechanics. Specifically, we identify three OOD problems: covariate shift, mechanism shift, and sampling bias. For each problem, we create two benchmark examples that extend the Mechanical MNIST dataset collection, and we investigate the performance of popular OOD generalization methods on these mechanics-specific regression problems.

- Guillotine Regularization: Improving Deep Networks Generalization by Removing their Head. [[paper]](https://arxiv.org/abs/2206.13378)
  - Florian Bordes, Randall Balestriero, Quentin Garrido, Adrien Bardes, Pascal Vincent.
  - Key Word: Pre-training; Self-Supervion; Fine-tuning; Regularization; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> One unexpected technique that emerged in recent years consists in training a Deep Network (DN) with a Self-Supervised Learning (SSL) method, and using this network on downstream tasks but with its last few layers entirely removed. This usually skimmed-over trick is actually critical for SSL methods to display competitive performances. For example, on ImageNet classification, more than 30 points of percentage can be gained that way. This is a little vexing, as one would hope that the network layer at which invariance is explicitly enforced by the SSL criterion during training (the last layer) should be the one to use for best generalization performance downstream. But it seems not to be, and this study sheds some light on why. This trick, which we name Guillotine Regularization (GR), is in fact a generically applicable form of regularization that has also been used to improve generalization performance in transfer learning scenarios. In this work, through theory and experiments, we formalize GR and identify the underlying reasons behind its success in SSL methods.

- Agreement-on-the-Line: Predicting the Performance of Neural Networks under Distribution Shift. [[paper]](https://arxiv.org/abs/2206.13089)
  - Christina Baek, Yiding Jiang, Aditi Raghunathan, Zico Kolter.
  - Key Word: estimating Generalization Error; Distribution Shift.
  - <details><summary>Digest</summary> Recently, Miller et al. showed that a model's in-distribution (ID) accuracy has a strong linear correlation with its out-of-distribution (OOD) accuracy on several OOD benchmarks -- a phenomenon they dubbed ''accuracy-on-the-line''. While a useful tool for model selection (i.e., the model most likely to perform the best OOD is the one with highest ID accuracy), this fact does not help estimate the actual OOD performance of models without access to a labeled OOD validation set. In this paper, we show a similar but surprising phenomenon also holds for the agreement between pairs of neural network classifiers: whenever accuracy-on-the-line holds, we observe that the OOD agreement between the predictions of any two pairs of neural networks (with potentially different architectures) also observes a strong linear correlation with their ID agreement.

- Memory-Guided Multi-View Multi-Domain Fake News Detection. [[paper]](https://arxiv.org/abs/2206.12808)
  - Yongchun Zhu, Qiang Sheng, Juan Cao, Qiong Nan, Kai Shu, Minghui Wu, Jindong Wang, and Fuzhen Zhuang. *IEEE TKDE 2022*
  - Key Word: Multi-domain learning; out-of-distribution generalization.
  - <details><summary>Digest</summary> We propose a Memory-guided Multi-view Multi-domain Fake News Detection Framework (M3FEND) to address these two challenges. We model news pieces from a multi-view perspective, including semantics, emotion, and style. Specifically, we propose a Domain Memory Bank to enrich domain information which could discover potential domain labels based on seen news pieces and model domain characteristics. Then, with enriched domain information as input, a Domain Adapter could adaptively aggregate discriminative information from multiple views for news in various domains.

- Gated Domain Units for Multi-source Domain Generalization. [[paper]](https://arxiv.org/abs/2206.12444)
  - Simon Föll, Alina Dubatovka, Eugen Ernst, Martin Maritsch, Patrik Okanovic, Gudrun Thäter, Joachim M. Buhmann, Felix Wortmann, Krikamol Muandet.
  - Key Word: Multi-Source Domain Generalization; Invariant Elementary Distributions.
  - <details><summary>Digest</summary> Distribution shift (DS) is a common problem that deteriorates the performance of learning machines. To overcome this problem, we postulate that real-world distributions are composed of elementary distributions that remain invariant across different domains. We call this an invariant elementary distribution (I.E.D.) assumption. This invariance thus enables knowledge transfer to unseen domains. To exploit this assumption in domain generalization (DG), we developed a modular neural network layer that consists of Gated Domain Units (GDUs). Each GDU learns an embedding of an individual elementary domain that allows us to encode the domain similarities during the training. During inference, the GDUs compute similarities between an observation and each of the corresponding elementary distributions which are then used to form a weighted ensemble of learning machines.

- On Certifying and Improving Generalization to Unseen Domains. [[paper]](https://arxiv.org/abs/2206.12364) [[code]](https://github.com/akshaymehra24/CertifiableDG)
  - Akshay Mehra, Bhavya Kailkhura, Pin-Yu Chen, Jihun Hamm.
  - Key Word: Certified Domain Generalization; Distributionally Robust Optimization.
  - <details><summary>Digest</summary> We demonstrate that the accuracy of the models trained with DG methods varies significantly across unseen domains, generated from popular benchmark datasets. This highlights that the performance of DG methods on a few benchmark datasets may not be representative of their performance on unseen domains in the wild. To overcome this roadblock, we propose a universal certification framework based on distributionally robust optimization (DRO) that can efficiently certify the worst-case performance of any DG method. This enables a data-independent evaluation of a DG method complementary to the empirical evaluations on benchmark datasets.

- Out of distribution robustness with pre-trained Bayesian neural networks. [[paper]](https://arxiv.org/abs/2206.12361)
  - Xi Wang, Laurence Aitchison.
  - Key Word: Corruption Robustness; Pre-training; Bayesian Neural Networks.
  - <details><summary>Digest</summary> We develop ShiftMatch, a new training-data-dependent likelihood for out of distribution (OOD) robustness in Bayesian neural networks (BNNs). ShiftMatch is inspired by the training-data-dependent "EmpCov" priors from Izmailov et al. (2021a) and efficiently matches test-time spatial correlations to those at training time. Critically, ShiftMatch is designed to leave neural network training unchanged, allowing it to use publically available samples from pretrained BNNs. Using pre-trained HMC samples, ShiftMatch gives strong performance improvements on CIFAR-10-C, outperforms EmpCov priors, and is perhaps the first Bayesian method capable of convincingly outperforming plain deep ensembles.

- Invariant Causal Mechanisms through Distribution Matching. [[paper]](https://arxiv.org/abs/2206.11646)
  - Mathieu Chevalley, Charlotte Bunne, Andreas Krause, Stefan Bauer.
  - Key Word: Domain Generalization; Causal Inference.
  - <details><summary>Digest</summary> Learning representations that capture the underlying data generating process is a key problem for data efficient and robust use of neural networks. One key property for robustness which the learned representation should capture and which recently received a lot of attention is described by the notion of invariance. In this work we provide a causal perspective and new algorithm for learning invariant representations. Empirically we show that this algorithm works well on a diverse set of tasks and in particular we observe state-of-the-art performance on domain generalization, where we are able to significantly boost the score of existing models.

- On Pre-Training for Federated Learning. [[paper]](https://arxiv.org/abs/2206.11488)
  - Hong-You Chen, Cheng-Hao Tu, Ziwei Li, Han-Wei Shen, Wei-Lun Chao.
  - Key Word: Pre-training; Federated Learning; Training with Sythetic Data.
  - <details><summary>Digest</summary> In most of the literature on federated learning (FL), neural networks are initialized with random weights. In this paper, we present an empirical study on the effect of pre-training on FL. Specifically, we aim to investigate if pre-training can alleviate the drastic accuracy drop when clients' decentralized data are non-IID. We focus on FedAvg, the fundamental and most widely used FL algorithm. We found that pre-training does largely close the gap between FedAvg and centralized learning under non-IID data, but this does not come from alleviating the well-known model drifting problem in FedAvg's local training. Instead, how pre-training helps FedAvg is by making FedAvg's global aggregation more stable. When pre-training using real data is not feasible for FL, we propose a novel approach to pre-train with synthetic data.

- Fighting Fire with Fire: Avoiding DNN Shortcuts through Priming. [[paper]](https://arxiv.org/abs/2206.10816) [[code]](https://github.com/AlvinWen428/fighting-fire-with-fire)
  - Chuan Wen, Jianing Qian, Jierui Lin, Jiaye Teng, Dinesh Jayaraman, Yang Gao. *ICML 2022*
  - Key Word: Shortcut Removal; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> We show empirically that DNNs can be coaxed to avoid poor shortcuts by providing an additional "priming" feature computed from key input features, usually a coarse output estimate. Priming relies on approximate domain knowledge of these task-relevant key input features, which is often easy to obtain in practical settings. For example, one might prioritize recent frames over past frames in a video input for visual imitation learning, or salient foreground over background pixels for image classification.

- Mitigating Data Heterogeneity in Federated Learning with Data Augmentation. [[paper]](https://arxiv.org/abs/2206.09979)
  - Artur Back de Luca, Guojun Zhang, Xi Chen, Yaoliang Yu.
  - Key Word: Federated Learning; Domain Generalization; Data Augmentation.
  - <details><summary>Digest</summary> While many approaches in DG tackle data heterogeneity from the algorithmic perspective, recent evidence suggests that data augmentation can induce equal or greater performance. Motivated by this connection, we present federated versions of popular DG algorithms, and show that by applying appropriate data augmentation, we can mitigate data heterogeneity in the federated setting, and obtain higher accuracy on unseen clients. Equipped with data augmentation, we can achieve state-of-the-art performance using even the most basic Federated Averaging algorithm, with much sparser communication.

- Decoupled Federated Learning for ASR with Non-IID Data. [[paper]](https://arxiv.org/abs/2206.09102)
  - Han Zhu, Jindong Wang , Gaofeng Cheng, Pengyuan Zhang, and Yonghong Yan. *Interspeech 2022*
  - Key Word: Non-IID; Federated learning; Speech recognition.
  - <details><summary>Digest</summary> We tackle the non-IID issue in FL-based ASR with personalized FL, which learns personalized models for each client. Concretely, we propose two types of personalized FL approaches for ASR. Firstly, we adapt the personalization layer based FL for ASR, which keeps some layers locally to learn personalization models. Secondly, to reduce the communication and computation costs, we propose decoupled federated learning (DecoupleFL).

- How robust are pre-trained models to distribution shift? [[paper]](https://arxiv.org/abs/2206.08871)
  - Yuge Shi, Imant Daunhawer, Julia E. Vogt, Philip H.S. Torr, Amartya Sanyal.
  - Key Word: Distribution Shifts; Self-Supervised Pre-Trainig.
  - <details><summary>Digest</summary> The vulnerability of machine learning models to spurious correlations has mostly been discussed in the context of supervised learning (SL). However, there is a lack of insight on how spurious correlations affect the performance of popular self-supervised learning (SSL) and auto-encoder based models (AE). In this work, we shed light on this by evaluating the performance of these models on both real world and synthetic distribution shift datasets. Following observations that the linear head itself can be susceptible to spurious correlations, we develop a novel evaluation scheme with the linear head trained on out-of-distribution (OOD) data, to isolate the performance of the pre-trained models from a potential bias of the linear head used for evaluation.

- Rectify ViT Shortcut Learning by Visual Saliency. [[paper]](https://arxiv.org/abs/2206.08567)
  - Chong Ma, Lin Zhao, Yuzhong Chen, David Weizhong Liu, Xi Jiang, Tuo Zhang, Xintao Hu, Dinggang Shen, Dajiang Zhu, Tianming Liu.
  - Key Word: Shortcut Learning; Vision Transformers; Eye Gaze Heatmap.
  - <details><summary>Digest</summary> We propose a novel and effective saliency-guided vision transformer (SGT) model to rectify shortcut learning in ViT with the absence of eye-gaze data. Specifically, a computational visual saliency model is adopted to predict saliency maps for input image samples. Then, the saliency maps are used to distil the most informative image patches. In the proposed SGT, the self-attention among image patches focus only on the distilled informative ones.

- GOOD: A Graph Out-of-Distribution Benchmark. [[paper]](https://arxiv.org/abs/2206.08452) [[code]](https://github.com/divelab/good)
  - Shurui Gui, Xiner Li, Limei Wang, Shuiwang Ji.
  - Key Word: Graph Neural Networks; Covariate Shifts; Concept Shifts.
  - <details><summary>Digest</summary> Currently, there lacks a systematic benchmark tailored to graph OOD method evaluation. In this work, we aim at developing an OOD benchmark, known as GOOD, for graphs specifically. We explicitly make distinctions between covariate and concept shifts and design data splits that accurately reflect different shifts. We consider both graph and node prediction tasks as there are key differences when designing shifts. Overall, GOOD contains 8 datasets with 14 domain selections. When combined with covariate, concept, and no shifts, we obtain 42 different splits. We provide performance results on 7 commonly used baseline methods with 10 random runs. This results in 294 dataset-model combinations in total.

- Modeling the Data-Generating Process is Necessary for Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2206.07837)
  - Jivat Neet Kaur, Emre Kiciman, Amit Sharma.
  - Key Word: Out-of-Distribution Generalization; Multi-attribute Distribution Shifts; Causal Graph.
  - <details><summary>Digest</summary> Real-world data collected from multiple domains can have multiple, distinct distribution shifts over multiple attributes. However, state-of-the art advances in domain generalization (DG) algorithms focus only on specific shifts over a single attribute. We introduce datasets with multi-attribute distribution shifts and find that existing DG algorithms fail to generalize. To explain this, we use causal graphs to characterize the different types of shifts based on the relationship between spurious attributes and the classification label. Each multi-attribute causal graph entails different constraints over observed variables, and therefore any algorithm based on a single, fixed independence constraint cannot work well across all shifts. We present Causally Adaptive Constraint Minimization (CACM), a new algorithm for identifying the correct independence constraints for regularization.

- What makes domain generalization hard? [[paper]](https://arxiv.org/abs/2206.07802)
  - Spandan Madan, Li You, Mengmi Zhang, Hanspeter Pfister, Gabriel Kreiman.
  - Key Word: Domain Generalization; Scene Context.
  - <details><summary>Digest</summary> While several methodologies have been proposed for the daunting task of domain generalization, understanding what makes this task challenging has received little attention. Here we present SemanticDG (Semantic Domain Generalization): a benchmark with 15 photo-realistic domains with the same geometry, scene layout and camera parameters as the popular 3D ScanNet dataset, but with controlled domain shifts in lighting, materials, and viewpoints. Using this benchmark, we investigate the impact of each of these semantic shifts on generalization independently.

- Pareto Invariant Risk Minimization. [[paper]](https://arxiv.org/abs/2206.07766)
  - Yongqiang Chen, Kaiwen Zhou, Yatao Bian, Binghui Xie, Kaili Ma, Yonggang Zhang, Han Yang, Bo Han, James Cheng.
  - Key Word: Invariant Learning; Multi-Task Learning.
  - <details><summary>Digest</summary> Despite the success of invariant risk minimization (IRM) in tackling the Out-of-Distribution generalization problem, IRM can compromise the optimality when applied in practice. The practical variants of IRM, e.g., IRMv1, have been shown to have significant gaps with IRM and thus could fail to capture the invariance even in simple problems. Moreover, the optimization procedure in IRMv1 involves two intrinsically conflicting objectives, and often requires careful tuning for the objective weights. To remedy the above issues, we reformulate IRM as a multi-objective optimization problem, and propose a new optimization scheme for IRM, called PAreto Invariant Risk Minimization (PAIR).

- Semantic-Discriminative Mixup for Generalizable Sensor-based Cross-domain Activity Recognition. [[paper]](http://arxiv.org/abs/2206.06629)
  - Wang Lu, Jindong Wang, Yiqiang Chen, Sinno Pan, Chunyu Hu, and Xin Qin. *ACM UbiComp 2022*
  - Key Word: Domain generalization; Activity recognition.
  - <details><summary>Digest</summary> We propose a novel method called Semantic-Discriminative Mixup (SDMix) for generalizable cross-domain HAR. Firstly, we introduce semantic-aware Mixup that considers the activity semantic ranges to overcome the semantic inconsistency brought by domain differences. Secondly, we introduce the large margin loss to enhance the discrimination of Mixup to prevent misclassification brought by noisy virtual labels.

- Invariant Structure Learning for Better Generalization and Causal Explainability. [[paper]](https://arxiv.org/abs/2206.06469)
  - Yunhao Ge, Sercan Ö. Arik, Jinsung Yoon, Ao Xu, Laurent Itti, Tomas Pfister.
  - Key Word: Causal Structure Discovery; Explainability; Invariant Learning.
  - <details><summary>Digest</summary> Learning the causal structure behind data is invaluable for improving generalization and obtaining high-quality explanations. We propose a novel framework, Invariant Structure Learning (ISL), that is designed to improve causal structure discovery by utilizing generalization as an indication. ISL splits the data into different environments, and learns a structure that is invariant to the target across different environments by imposing a consistency constraint. An aggregation mechanism then selects the optimal classifier based on a graph structure that reflects the causal mechanisms in the data more accurately compared to the structures learnt from individual environments.

- Causal Balancing for Domain Generalization. [[paper]](https://arxiv.org/abs/2206.05263)
  - Xinyi Wang, Michael Saxon, Jiachen Li, Hongyang Zhang, Kun Zhang, William Yang Wang.
  - Key Word: Invariant Learning; Causal Semantic Generative Model.
  - <details><summary>Digest</summary> While current domain generalization methods usually focus on enforcing certain invariance properties across different domains by new loss function designs, we propose a balanced mini-batch sampling strategy to reduce the domain-specific spurious correlations in the observed training distributions. More specifically, we propose a two-phased method that 1) identifies the source of spurious correlations, and 2) builds balanced mini-batches free from spurious correlations by matching on the identified source.

- GSmooth: Certified Robustness against Semantic Transformations via Generalized Randomized Smoothing. [[paper]](https://arxiv.org/abs/2206.04310)
  - Zhongkai Hao, Chengyang Ying, Yinpeng Dong, Hang Su, Jun Zhu, Jian Song. *ICML 2022*
  - Key Word: Certified Semantic Robustness.
  - <details><summary>Digest</summary> Existing methods are insufficient or unable to provably defend against semantic transformations, especially those without closed-form expressions (such as defocus blur and pixelate), which are more common in practice and often unrestricted. To fill up this gap, we propose generalized randomized smoothing (GSmooth), a unified theoretical framework for certifying robustness against general semantic transformations via a novel dimension augmentation strategy. Under the GSmooth framework, we present a scalable algorithm that uses a surrogate image-to-image network to approximate the complex transformation.

- Sparse Fusion Mixture-of-Experts are Domain Generalizable Learners. [[paper]](https://arxiv.org/abs/2206.04046) [[code]](https://github.com/Luodian/SF-MoE-DG)
  - Bo Li, Jingkang Yang, Jiawei Ren, Yezhen Wang, Ziwei Liu.
  - Key Word: Domain Generalization; Vision Transformer; Sparse Mixture-of-Experts.
  - <details><summary>Digest</summary> We reveal the mixture-of-experts (MoE) model's generalizability on DG by leveraging to distributively handle multiple aspects of the predictive features across domains. To this end, we propose Sparse Fusion Mixture-of-Experts (SF-MoE), which incorporates sparsity and fusion mechanisms into the MoE framework to keep the model both sparse and predictive. SF-MoE has two dedicated modules: 1) sparse block and 2) fusion block, which disentangle and aggregate the diverse learned signals of an object, respectively.

- Toward Certified Robustness Against Real-World Distribution Shifts. [[paper]](https://arxiv.org/abs/2206.03669)
  - Haoze Wu, Teruhiro Tagomori, Alexander Robey, Fengjun Yang, Nikolai Matni, George Pappas, Hamed Hassani, Corina Pasareanu, Clark Barrett.
  - Key Word: Certified Robustness; Distribution Shift.
  - <details><summary>Digest</summary> We propose a general meta-algorithm for handling sigmoid activations which leverages classical notions of counter-example-guided abstraction refinement. The key idea is to "lazily" refine the abstraction of sigmoid functions to exclude spurious counter-examples found in the previous abstraction, thus guaranteeing progress in the verification process while keeping the state-space small.

- Can CNNs Be More Robust Than Transformers? [[paper]](https://arxiv.org/abs/2206.03452) [[code]](https://github.com/UCSC-VLAA/RobustCNN)
  - Zeyu Wang, Yutong Bai, Yuyin Zhou, Cihang Xie.
  - Key Word: Transformers; Distribution Shift.
  - <details><summary>Digest</summary> We question that belief by closely examining the design of Transformers. Our findings lead to three highly effective architecture designs for boosting robustness, yet simple enough to be implemented in several lines of code, namely a) patchifying input images, b) enlarging kernel size, and c) reducing activation layers and normalization layers.

- Distributionally Invariant Learning: Rationalization and Practical Algorithms. [[paper]](https://arxiv.org/abs/2206.02990)
  - Jiashuo Liu, Jiayun Wu, Jie Peng, Zheyan Shen, Bo Li, Peng Cui.
  - Key Word: Invariant Learning.
  - <details><summary>Digest</summary> We come up with the distributional invariance property as a relaxed alternative to the strict invariance, which considers the invariance only among sub-populations down to a prescribed scale and allows a certain degree of variation. We reformulate the invariant learning problem under latent heterogeneity into a relaxed form that pursues the distributional invariance, based on which we propose our novel Distributionally Invariant Learning (DIL) framework as well as two implementations named DIL-MMD and DIL-KL.

- Generalized Federated Learning via Sharpness Aware Minimization. [[paper]](https://arxiv.org/abs/2206.02618)
  - Zhe Qu, Xingyu Li, Rui Duan, Yao Liu, Bo Tang, Zhuo Lu. *ICML 2022*
  - Key Word: Personalized Federated Learning.
  - <details><summary>Digest</summary> We revisit the solutions to the distribution shift problem in FL with a focus on local learning generality. To this end, we propose a general, effective algorithm, FedSAM, based on Sharpness Aware Minimization (SAM) local optimizer, and develop a momentum FL algorithm to bridge local and global models, MoFedSAM. Theoretically, we show the convergence analysis of these two algorithms and demonstrate the generalization bound of FedSAM. Empirically, our proposed algorithms substantially outperform existing FL studies and significantly decrease the learning deviation.

- An Optimal Transport Approach to Personalized Federated Learning. [[paper]](https://arxiv.org/abs/2206.02468) [[code]](https://github.com/farzanfarnia/FedOT)
  - Farzan Farnia, Amirhossein Reisizadeh, Ramtin Pedarsani, Ali Jadbabaie.
  - Key Word: Personalized Federated Learning; Optimal Transport.
  - <details><summary>Digest</summary> We focus on this problem and propose a novel personalized Federated Learning scheme based on Optimal Transport (FedOT) as a learning algorithm that learns the optimal transport maps for transferring data points to a common distribution as well as the prediction model under the applied transport map. To formulate the FedOT problem, we extend the standard optimal transport task between two probability distributions to multi-marginal optimal transport problems with the goal of transporting samples from multiple distributions to a common probability domain. We then leverage the results on multi-marginal optimal transport problems to formulate FedOT as a min-max optimization problem and analyze its generalization and optimization properties.

- AugLoss: A Learning Methodology for Real-World Dataset Corruption. [[paper]](https://arxiv.org/abs/2206.02286)
  - Kyle Otstot, John Kevin Cava, Tyler Sypherd, Lalitha Sankar.
  - Key Word: Corruption Robustness; Data Augmentation.
  - <details><summary>Digest</summary> As a step towards addressing both problems simultaneously, we introduce AugLoss, a simple but effective methodology that achieves robustness against both train-time noisy labeling and test-time feature distribution shifts by unifying data augmentation and robust loss functions. We conduct comprehensive experiments in varied settings of real-world dataset corruption to showcase the gains achieved by AugLoss compared to previous state-of-the-art methods.

- Drawing out of Distribution with Neuro-Symbolic Generative Models. [[paper]](https://arxiv.org/abs/2206.01829)
  - Yichao Liang, Joshua B. Tenenbaum, Tuan Anh Le, N. Siddharth.
  - Key Word: Out-of-Distribution Generalization; Neuro-Symbolic Generative Models.
  - <details><summary>Digest</summary> Learning general-purpose representations from perceptual inputs is a hallmark of human intelligence. For example, people can write out numbers or characters, or even draw doodles, by characterizing these tasks as different instantiations of the same generic underlying process -- compositional arrangements of different forms of pen strokes. Crucially, learning to do one task, say writing, implies reasonable competence at another, say drawing, on account of this shared process. We present Drawing out of Distribution (DooD), a neuro-symbolic generative model of stroke-based drawing that can learn such general-purpose representations. In contrast to prior work, DooD operates directly on images, requires no supervision or expensive test-time inference, and performs unsupervised amortised inference with a symbolic stroke model that better enables both interpretability and generalization.

- On the Generalization of Wasserstein Robust Federated Learning. [[paper]](https://arxiv.org/abs/2206.01432)
  - Tung-Anh Nguyen, Tuan Dung Nguyen, Long Tan Le, Canh T. Dinh, Nguyen H. Tran.
  - Key Word: Wasserstein Distributionally Robust Optimization; Federated Learning.
  - <details><summary>Digest</summary> In federated learning, participating clients typically possess non-i.i.d. data, posing a significant challenge to generalization to unseen distributions. To address this, we propose a Wasserstein distributionally robust optimization scheme called WAFL. Leveraging its duality, we frame WAFL as an empirical surrogate risk minimization problem, and solve it using a local SGD-based algorithm with convergence guarantees. We show that the robustness of WAFL is more general than related approaches, and the generalization bound is robust to all adversarial distributions inside the Wasserstein ball (ambiguity set).

- Federated Learning under Distributed Concept Drift. [[paper]](https://arxiv.org/abs/2206.00799)
  - Ellango Jothimurugesan, Kevin Hsieh, Jianyu Wang, Gauri Joshi, Phillip B. Gibbons.
  - Key Word: Concept Drift; Federated Learning.
  - <details><summary>Digest</summary> Our work is the first to explicitly study data heterogeneity in both dimensions. We first demonstrate that prior solutions to drift adaptation, with their single global model, are ill-suited to staggered drifts, necessitating multi-model solutions. We identify the problem of drift adaptation as a time-varying clustering problem, and we propose two new clustering algorithms for reacting to drifts based on local drift detection and hierarchical clustering.

- Evolving Domain Generalization. [[paper]](https://arxiv.org/abs/2206.00047)
  - Wei Wang, Gezheng Xu, Ruizhi Pu, Jiaqi Li, Fan Zhou, Changjian Shui, Charles Ling, Christian Gagné, Boyu Wang.
  - Key Word: Domain Generalization.
  - <details><summary>Digest</summary> Domain generalization aims to learn a predictive model from multiple different but related source tasks that can generalize well to a target task without the need of accessing any target data. Existing domain generalization methods ignore the relationship between tasks, implicitly assuming that all the tasks are sampled from a stationary environment. Therefore, they can fail when deployed in an evolving environment. To this end, we formulate and study the \emph{evolving domain generalization} (EDG) scenario, which exploits not only the source data but also their evolving pattern to generate a model for the unseen task.

- Evaluating Robustness to Dataset Shift via Parametric Robustness Sets. [[paper]](https://arxiv.org/abs/2205.15947)
  - Nikolaj Thams, Michael Oberst, David Sontag.
  - Key Word: Distributionally Robust Optimization.
  - <details><summary>Digest</summary> We give a method for proactively identifying small, plausible shifts in distribution which lead to large differences in model performance. To ensure that these shifts are plausible, we parameterize them in terms of interpretable changes in causal mechanisms of observed variables. This defines a parametric robustness set of plausible distributions and a corresponding worst-case loss. While the loss under an individual parametric shift can be estimated via reweighting techniques such as importance sampling, the resulting worst-case optimization problem is non-convex, and the estimate may suffer from large variance.

- PAC Generalisation via Invariant Representations. [[paper]](https://arxiv.org/abs/2205.15196)
  - Advait Parulekar, Karthikeyan Shanmugam, Sanjay Shakkottai.
  - Key Word: Invariant Learning; Causal Structure Learning; Domain Adaptation.
  - <details><summary>Digest</summary> We study the following question: If a representation is approximately invariant with respect to a given number of training interventions, will it continue to be approximately invariant on a larger collection of unseen SEMs? This larger collection of SEMs is generated through a parameterized family of interventions. Inspired by PAC learning, we obtain finite-sample out-of-distribution generalization guarantees for approximate invariance that holds probabilistically over a family of linear SEMs without faithfulness assumptions.

- The Missing Invariance Principle Found -- the Reciprocal Twin of Invariant Risk Minimization. [[paper]](https://arxiv.org/abs/2205.14546)
  - Dongsung Huh, Avinash Baidya.
  - Key Word: Invariant Learning.
  - <details><summary>Digest</summary> We identify a fundamental flaw of IRM formulation that causes the failure. We then introduce a complementary notion of invariance, MRI, that is based on conserving the class-conditioned feature expectation across environments, that corrects for the flaw in IRM. Further, we introduce a simplified, practical version of the MRI formulation called as MRI-v1. We note that this constraint is convex which confers it with an advantage over the practical version of IRM, IRM-v1, which imposes non-convex constraints. We prove that in a general linear problem setting, MRI-v1 can guarantee invariant predictors given sufficient environments.

- Exploiting Adapters for Cross-lingual Low-resource Speech Recognition. [[paper]](https://arxiv.org/abs/2105.11905)
  - Wenxin Hou, Han Zhu, Yidong Wang, Jindong Wang, Tao Qin, Renjun Xu, and Takahiro Shinozaki. *TASLP 2022*
  - Key Word: Cross-domain learning; Speech recognition.
  - <details><summary>Digest</summary> We propose a novel algorithms called SimAdapter for explicitly learning knowledge from adapters. Our algorithm leverages adapters which can be easily integrated into the Transformer structure.MetaAdapter leverages meta-learning to transfer the general knowledge from training data to the test language. SimAdapter aims to learn the similarities between the source and target languages during fine-tuning using the adapters.

- FL Games: A federated learning framework for distribution shifts. [[paper]](https://arxiv.org/abs/2205.11101)
  - Sharut Gupta, Kartik Ahuja, Mohammad Havaei, Niladri Chatterjee, Yoshua Bengio.
  - Key Word: Distribution Shifts; Federated Learning.
  - <details><summary>Digest</summary> We argue that in order to generalize better across non-i.i.d. clients, it is imperative to only learn correlations that are stable and invariant across domains. We propose FL Games, a game-theoretic framework for federated learning for learning causal features that are invariant across clients. While training to achieve the Nash equilibrium, the traditional best response strategy suffers from high-frequency oscillations. We demonstrate that FL Games effectively resolves this challenge and exhibits smooth performance curves.

- Federated Learning Aggregation: New Robust Algorithms with Guarantees. [[paper]](https://arxiv.org/abs/2205.10864)
  - Adnan Ben Mansour, Gaia Carenini, Alexandre Duplessis, David Naccache.
  - Key Word: Federated Learning; Model Aggregation.
  - <details><summary>Digest</summary> We carry out a complete general mathematical convergence analysis to evaluate aggregation strategies in a federated learning framework. From this, we derive novel aggregation algorithms which are able to modify their model architecture by differentiating client contributions according to the value of their losses.

- Interpolating Compressed Parameter Subspaces. [[paper]](https://arxiv.org/abs/2205.09891)
  - Siddhartha Datta, Nigel Shadbolt.
  - Key Word: Distribution Shifts; Weight Averaging; Test-time distributions; Task interpolation.
  - <details><summary>Digest</summary> Inspired by recent work on neural subspaces and mode connectivity, we revisit parameter subspace sampling for shifted and/or interpolatable input distributions (instead of a single, unshifted distribution). We enforce a compressed geometric structure upon a set of trained parameters mapped to a set of train-time distributions, denoting the resulting subspaces as Compressed Parameter Subspaces (CPS). We show the success and failure modes of the types of shifted distributions whose optimal parameters reside in the CPS. We find that ensembling point-estimates within a CPS can yield a high average accuracy across a range of test-time distributions, including backdoor, adversarial, permutation, stylization and rotation perturbations. 

- Improving Robustness against Real-World and Worst-Case Distribution Shifts through Decision Region Quantification. [[paper]](https://arxiv.org/abs/2205.09619)
  - Leo Schwinn, Leon Bungert, An Nguyen, René Raab, Falk Pulsmeyer, Doina Precup, Björn Eskofier, Dario Zanca. *ICML 2022*
  - Key Word: Decision Region Quantification; Corruption Robustness; Distribution Shift.
  - <details><summary>Digest</summary> We propose the Decision Region Quantification (DRQ) algorithm to improve the robustness of any differentiable pre-trained model against both real-world and worst-case distribution shifts in the data. DRQ analyzes the robustness of local decision regions in the vicinity of a given data point to make more reliable predictions. We theoretically motivate the DRQ algorithm by showing that it effectively smooths spurious local extrema in the decision surface.

- FedILC: Weighted Geometric Mean and Invariant Gradient Covariance for Federated Learning on Non-IID Data. [[paper]](https://arxiv.org/abs/2205.09305) [[code]](https://github.com/mikemikezhu/FedILC)
  - Mike He Zhu, Léna Néhale Ezzine, Dianbo Liu, Yoshua Bengio.
  - Key Word: Regularization; Federated Learning.
  - <details><summary>Digest</summary> We propose the Federated Invariant Learning Consistency (FedILC) approach, which leverages the gradient covariance and the geometric mean of Hessians to capture both inter-silo and intra-silo consistencies of environments and unravel the domain shift problems in federated networks.

- Causality Inspired Representation Learning for Domain Generalization. [[paper]](https://arxiv.org/abs/2203.14237) [[code]](https://github.com/BIT-DA/CIRL)
  - Fangrui Lv, Jian Liang, Shuang Li, Bin Zang, Chi Harold Liu, Ziteng Wang, Di Liu. *CVPR 2022*
  - Key Word: Domain Generalization; Causality.
  - <details><summary>Digest</summary> We introduce a general structural causal model to formalize the DG problem. Specifically, we assume that each input is constructed from a mix of causal factors (whose relationship with the label is invariant across domains) and non-causal factors (category-independent), and only the former cause the classification judgments. Our goal is to extract the causal factors from inputs and then reconstruct the invariant causal mechanisms.

- Closing the Generalization Gap of Cross-silo Federated Medical Image Segmentation. [[paper]](https://arxiv.org/abs/2203.10144)
  - An Xu, Wenqi Li, Pengfei Guo, Dong Yang, Holger Roth, Ali Hatamizadeh, Can Zhao, Daguang Xu, Heng Huang, Ziyue Xu.
  - Key Word: Personalized Federated Learning; Medical Image Segmentation.
  - <details><summary>Digest</summary> We propose a novel training framework FedSM to avoid the client drift issue and successfully close the generalization gap compared with the centralized training for medical image segmentation tasks for the first time. We also propose a novel personalized FL objective formulation and a new method SoftPull to solve it in our proposed framework FedSM.

- Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. [[paper]](https://arxiv.org/abs/2203.05482) [[code]](https://github.com/mlfoundations/model-soups)
  - Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, Ludwig Schmidt. *ICML 2022*
  - Key Word: Wegiht Averaging; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> We show that averaging the weights of multiple models fine-tuned with different hyperparameter configurations often improves accuracy and robustness. Unlike a conventional ensemble, we may average many models without incurring any additional inference or memory costs -- we call the results "model soups." When fine-tuning large pre-trained models such as CLIP, ALIGN, and a ViT-G pre-trained on JFT, our soup recipe provides significant improvements over the best model in a hyperparameter sweep on ImageNet.

- Continual Feature Selection: Spurious Features in Continual Learning. [[paper]](https://arxiv.org/abs/2203.01012)
  - Timothée Lesort.
  - Key Word: Spurious Correlations; Continual Learning.
  - <details><summary>Digest</summary> This paper studies spurious features' influence on continual learning algorithms. We show that continual learning algorithms solve tasks by selecting features that are not generalizable. Our experiments highlight that continual learning algorithms face two related problems: (1) spurious features and (2) local spurious features. The first one is due to a covariate shift between training and testing data, while the second is due to the limited access to data at each training step. We study (1) through a consistent set of continual learning experiments varying spurious correlation amount and data distribution support. We show that (2) is a major cause of performance decrease in continual learning along with catastrophic forgetting. 

- Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution. [[paper]](https://arxiv.org/abs/2202.10054) [[code]](https://github.com/AnanyaKumar/transfer_learning)
  - Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, Percy Liang.
  - Key Word: Fine-Tuning; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> We show theoretically that this tradeoff between ID and OOD accuracy arises even in a simple setting: fine-tuning overparameterized two-layer linear networks. We prove that the OOD error of fine-tuning is high when we initialize with a fixed or random head -- this is because while fine-tuning learns the head, the lower layers of the neural network change simultaneously and distort the pretrained features. Our analysis suggests that the easy two-step strategy of linear probing then full fine-tuning (LP-FT), sometimes used as a fine-tuning heuristic, combines the benefits of both fine-tuning and linear probing. 

- Uncertainty Modeling for Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2202.03958) [[code]](https://github.com/lixiaotong97/dsu)
  - Xiaotong Li, Yongxing Dai, Yixiao Ge, Jun Liu, Ying Shan, Ling-Yu Duan. *ICLR 2022*
  - Key Word: Out-of-Distribution Generalization; Uncertainty.
  - <details><summary>Digest</summary> We improve the network generalization ability by modeling the uncertainty of domain shifts with synthesized feature statistics during training. Specifically, we hypothesize that the feature statistic, after considering the potential uncertainties, follows a multivariate Gaussian distribution. Hence, each feature statistic is no longer a deterministic value, but a probabilistic point with diverse distribution possibilities. With the uncertain feature statistics, the models can be trained to alleviate the domain perturbations and achieve better robustness against potential domain shifts.

- Benchmarking and Analyzing Point Cloud Classification under Corruptions. [[paper]](https://arxiv.org/abs/2202.03377) [[code]](https://github.com/jiawei-ren/modelnetc)
  - Jiawei Ren, Liang Pan, Ziwei Liu. *ICML 2022*
  - Key Word: Corruption Robustness; Point Cloud Classification; Benchmarks.
  - <details><summary>Digest</summary> 3D perception, especially point cloud classification, has achieved substantial progress. However, in real-world deployment, point cloud corruptions are inevitable due to the scene complexity, sensor inaccuracy, and processing imprecision. In this work, we aim to rigorously benchmark and analyze point cloud classification under corruptions. To conduct a systematic investigation, we first provide a taxonomy of common 3D corruptions and identify the atomic corruptions. Then, we perform a comprehensive evaluation on a wide range of representative point cloud models to understand their robustness and generalizability.

- Handling Distribution Shifts on Graphs: An Invariance Perspective. [[paper]](https://arxiv.org/abs/2202.02466) [[code]](https://github.com/qitianwu/graphood-eerm)
  - Qitian Wu, Hengrui Zhang, Junchi Yan, David Wipf. *ICLR 2022*
  - Key Word: Distribution Shifts; Graph Neural Networks.
  - <details><summary>Digest</summary> We formulate the OOD problem on graphs and develop a new invariant learning approach, Explore-to-Extrapolate Risk Minimization (EERM), that facilitates graph neural networks to leverage invariance principles for prediction. EERM resorts to multiple context explorers (specified as graph structure editers in our case) that are adversarially trained to maximize the variance of risks from multiple virtual environments.

- Certifying Out-of-Domain Generalization for Blackbox Functions. [[paper]](https://arxiv.org/abs/2202.01679)
  - Maurice Weber, Linyi Li, Boxin Wang, Zhikuan Zhao, Bo Li, Ce Zhang.
  - Key Word: Certified Distributional Robustness; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> We focus on the problem of certifying distributional robustness for black box models and bounded losses, without other assumptions. We propose a novel certification framework given bounded distance of mean and variance of two distributions. Our certification technique scales to ImageNet-scale datasets, complex models, and a diverse range of loss functions. We then focus on one specific application enabled by such scalability and flexibility, i.e., certifying out-of-domain generalization for large neural networks and loss functions such as accuracy and AUC.

- Provable Domain Generalization via Invariant-Feature Subspace Recovery. [[paper]](https://arxiv.org/abs/2201.12919) [[code]](https://github.com/haoxiang-wang/isr)  
  - Haoxiang Wang, Haozhe Si, Bo Li, Han Zhao. *ICML 2022*
  - Key Word: Domain Generalization; Invariant Learning.
  - <details><summary>Digest</summary> we propose to achieve domain generalization with Invariant-feature Subspace Recovery (ISR). Our first algorithm, ISR-Mean, can identify the subspace spanned by invariant features from the first-order moments of the class-conditional distributions, and achieve provable domain generalization with ds+1 training environments under the data model of Rosenfeld et al. (2021). Our second algorithm, ISR-Cov, further reduces the required number of training environments to O(1) using the information of second-order moments.

- Certifying Model Accuracy under Distribution Shifts. [[paper]](https://arxiv.org/abs/2201.12440)
  - Aounon Kumar, Alexander Levine, Tom Goldstein, Soheil Feizi.
  - Key Word: Certified Distributional Robustness; Corruption Robustness.
  - <details><summary>Digest</summary> Certified robustness in machine learning has primarily focused on adversarial perturbations of the input with a fixed attack budget for each point in the data distribution. In this work, we present provable robustness guarantees on the accuracy of a model under bounded Wasserstein shifts of the data distribution. We show that a simple procedure that randomizes the input of the model within a transformation space is provably robust to distributional shifts under the transformation. Our framework allows the datum-specific perturbation size to vary across different points in the input distribution and is general enough to include fixed-sized perturbations as well.

</details>

<!-- ### Out-of-Distribution Generalization: 2021 -->

<details>
<summary>Out-of-Distribution Generalization: 2021</summary>

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

- Adarnn: Adaptive learning and forecasting of time series. [[paper]](https://arxiv.org/abs/2108.04443)
  - Yuntao Du, Jindong Wang, Wenjie Feng, Sinno Pan, Tao Qin, Renjun Xu, and Chongjun Wang. *CIKM 2021*
  - Key Word: Out-of-distribution prediction; Time series analysis
  - <details><summary>Digest</summary> This paper proposes Adaptive RNNs (AdaRNN) to tackle the TCS problem by building an adaptive model that generalizes well on the unseen test data. AdaRNN is sequentially composed of two novel algorithms. First, we propose Temporal Distribution Characterization to better characterize the distribution information in the TS. Second, we propose Temporal Distribution Matching to reduce the distribution mismatch in TS to learn the adaptive TS model. AdaRNN is a general framework with flexible distribution distances integrated.

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
  - Kaiyang Zhou, Ziwei Liu, Yu Qiao, Tao Xiang, Chen Change Loy. *TPAMI*
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

</details>

<!-- ### Out-of-Distribution Generalization: 2020 -->

<details>
<summary>Out-of-Distribution Generalization: 2020</summary>

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

- Learning causal semantic representation for out-of-distribution prediction. [[paper]](https://arxiv.org/abs/2011.01681)
  - Chang Liu, Xinwei Sun, Jindong Wang , Haoyue Tang, Tao Li, Tao Qin, Wei Chen, and Tie-Yan Liu. *NeurIPS 2021*
  - Key Word: Out-of-distribution generalization; Causality
  - <details><summary>Digest</summary> We propose a Causal Semantic Generative model (CSG) based on a causal reasoning so that the two factors are modeled separately, and develop methods for OOD prediction from a single training domain, which is common and challenging. The methods are based on the causal invariance principle, with a novel design in variational Bayes for both efficient learning and easy prediction.

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

</details>

<!-- ### Out-of-Distribution Generalization: 2019 -->

<details>
<summary>Out-of-Distribution Generalization: 2019</summary>

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

</details>

<!-- ### Out-of-Distribution Generalization: 2018 -->

<details>
<summary>Out-of-Distribution Generalization: 2018</summary>

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

</details>

## Evasion Attacks and Defenses

<details>
<summary>Evasion Attacks and Defenses: 2022</summary>

- ReMoS: Reducing Defect Inheritance in Transfer Learning via Relevant Model Slicing. [[paper]](https://jd92.wang/assets/files/icse22-remos.pdf)
  - Ziqi Zhang, Yuanchun Li, Jindong Wang , Bingyan Liu, Ding Li, Xiangqun Chen, Yao Guo, and Yunxin Liu. *ICSE 2022*
  - Key Word: Attack defect inheritance; Transfer learning
  - <details><summary>Digest</summary> We propose ReMoS, a relevant model slicing technique to reduce defect inheritance during transfer learning while retaining useful knowledge from the teacher model. Specifically, ReMoS computes a model slice (a subset of model weights) that is relevant to the student task based on the neuron coverage information obtained by profiling the teacher neurons.

- Implicit Bias of Adversarial Training for Deep Neural Networks. [[paper]](https://openreview.net/forum?id=l8It-0lE5e7)
  - Bochen Lv, Zhanxing Zhu. *ICLR 2022*
  - Key Word: Adversarial Training.
  - <details><summary>Digest</summary> We provide theoretical understandings of the implicit bias imposed by adversarial training for homogeneous deep neural networks without any explicit regularization. In particular, for deep linear networks adversarially trained by gradient descent on a linearly separable dataset, we prove that the direction of the product of weight matrices converges to the direction of the max-margin solution of the original dataset.

- Reducing Excessive Margin to Achieve a Better Accuracy vs. Robustness Trade-off. [[paper]](https://openreview.net/forum?id=Azh9QBQ4tR7) [[code]](https://github.com/imrahulr/hat)
  - Rahul Rade, Seyed-Mohsen Moosavi-Dezfooli. *ICLR 2022*
  - Key Word: Adversarial Training.
  - <details><summary>Digest</summary> We closely examine the changes induced in the decision boundary of a deep network during adversarial training. We find that adversarial training leads to unwarranted increase in the margin along certain adversarial directions, thereby hurting accuracy. Motivated by this observation, we present a novel algorithm, called Helper-based Adversarial Training (HAT), to reduce this effect by incorporating additional wrongly labelled examples during training.

- Invariance-Aware Randomized Smoothing Certificates. [[paper]](https://arxiv.org/abs/2211.14207)
  - Jan Schuchardt, Stephan Günnemann. *NeurIPS 2022*
  - Key Word: Randomized Smoothing; Group Invariance.
  - <details><summary>Digest</summary> Building models that comply with the invariances inherent to different domains, such as invariance under translation or rotation, is a key aspect of applying machine learning to real world problems like molecular property prediction, medical imaging, protein folding or LiDAR classification. For the first time, we study how the invariances of a model can be leveraged to provably guarantee the robustness of its predictions. We propose a gray-box approach, enhancing the powerful black-box randomized smoothing technique with white-box knowledge about invariances.

- Benchmarking Adversarially Robust Quantum Machine Learning at Scale. [[paper]](https://arxiv.org/abs/2211.12681)
  - Maxwell T. West, Sarah M. Erfani, Christopher Leckie, Martin Sevior, Lloyd C.L. Hollenberg, Muhammad Usman.
  - Key Word: Adversarial Robustness; Quantum Machine Learning.
  - <details><summary>Digest</summary> We benchmark the robustness of quantum ML networks, such as quantum variational classifiers (QVC), at scale by performing rigorous training for both simple and complex image datasets and through a variety of high-end adversarial attacks. Our results show that QVCs offer a notably enhanced robustness against classical adversarial attacks by learning features which are not detected by the classical neural networks, indicating a possible quantum advantage for ML tasks. 

- Improving Robust Generalization by Direct PAC-Bayesian Bound Minimization. [[paper]](https://arxiv.org/abs/2211.12624)
  - Zifan Wang, Nan Ding, Tomer Levinboim, Xi Chen, Radu Soricut.
  - Key Word: Adversarial Robustness; PAC-Bayes.
  - <details><summary>Digest</summary> We consider a different form of the robust PAC-Bayesian bound and directly minimize it with respect to the model posterior. The derivation of the optimal solution connects PAC-Bayesian learning to the geometry of the robust loss surface through a Trace of Hessian (TrH) regularizer that measures the surface flatness. In practice, we restrict the TrH regularizer to the top layer only, which results in an analytical solution to the bound whose computational cost does not depend on the network depth. 

- Fairness Increases Adversarial Vulnerability. [[paper]](https://arxiv.org/abs/2211.11835)
  - Cuong Tran, Keyu Zhu, Ferdinando Fioretto, Pascal Van Henternyck.
  - Key Word: Adversarial Robustness; Fairness.
  - <details><summary>Digest</summary> This paper shows the existence of a dichotomy between fairness and robustness, and analyzes when achieving fairness decreases the model robustness to adversarial samples. The reported analysis sheds light on the factors causing such contrasting behavior, suggesting that distance to the decision boundary across groups as a key explainer for this behavior.

- Towards Robust Dataset Learning. [[paper]](https://arxiv.org/abs/2211.10752)
  - Yihan Wu, Xinda Li, Florian Kerschbaum, Heng Huang, Hongyang Zhang.
  - Key Word: Adversarial Robustness; Dataset Distillation.
  - <details><summary>Digest</summary> We study the problem of learning a robust dataset such that any classifier naturally trained on the dataset is adversarially robust. Such a dataset benefits the downstream tasks as natural training is much faster than adversarial training, and demonstrates that the desired property of robustness is transferable between models and data. We propose a principled, tri-level optimization to formulate the robust dataset learning problem. We show that, under an abstraction model that characterizes robust vs. non-robust features, the proposed method provably learns a robust dataset. 

- Diagnostics for Deep Neural Networks with Automated Copy/Paste Attacks. [[paper]](https://arxiv.org/abs/2211.10024) [[code]](https://github.com/thestephencasper/snafue)
  - Stephen Casper, Kaivalya Hariharan, Dylan Hadfield-Menell.
  - Key Word: Adversarial Patch.
  - <details><summary>Digest</summary> First, we introduce a method termed Search for Natural Adversarial Features Using Embeddings (SNAFUE) which offers a fully-automated method for finding "copy/paste" attacks in which one natural image can be pasted into another in order to induce an unrelated misclassification. Second, we use this to red team an ImageNet classifier and identify hundreds of easily-describable sets of vulnerabilities. Third, we compare this approach with other interpretability tools by attempting to rediscover trojans. 

- Towards Good Practices in Evaluating Transfer Adversarial Attacks. [[paper]](https://arxiv.org/abs/2211.09565) [[code]](https://github.com/zhengyuzhao/transferattackeval)
  - Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes.
  - Key Word: Transfer Adversarial Attacks; Benchmarks.
  - <details><summary>Digest</summary> We design good practices to address these limitations. We first introduce a new attack categorization, which enables our systematic analyses of similar attacks in each specific category. Our analyses lead to new findings that complement or even challenge existing knowledge. Furthermore, we comprehensively evaluate 23 representative attacks against 9 defenses on ImageNet. We pay particular attention to stealthiness, by adopting diverse imperceptibility metrics and looking into new, finer-grained characteristics.

- Improved techniques for deterministic l2 robustness. [[paper]](https://arxiv.org/abs/2211.08453) [[code]](https://github.com/singlasahil14/improved_l2_robustness)
  - Sahil Singla, Soheil Feizi. *NeurIPS 2022*
  - Key Word: Provable Adversarial Defenses; Provably Lipschitz Neural Networks.
  - <details><summary>Digest</summary> 1-Lipschitz CNNs are usually designed by enforcing each layer to have an orthogonal Jacobian matrix (for all inputs) to prevent the gradients from vanishing during backpropagation. However, their performance often significantly lags behind that of heuristic methods to enforce Lipschitz constraints where the resulting CNN is not provably 1-Lipschitz. In this work, we reduce this gap by introducing (a) a procedure to certify robustness of 1-Lipschitz CNNs by replacing the last linear layer with a 1-hidden layer MLP that significantly improves their performance for both standard and provably robust accuracy, (b) a method to significantly reduce the training time per epoch for Skew Orthogonal Convolution (SOC) layers (>30\% reduction for deeper networks) and (c) a class of pooling layers using the mathematical property that the l2 distance of an input to a manifold is 1-Lipschitz. 

- Are AlphaZero-like Agents Robust to Adversarial Perturbations? [[paper]](https://arxiv.org/abs/2211.03769) [[code]](https://github.com/lan-lc/adversarial_example_of_Go)
  - Li-Cheng Lan, Huan Zhang, Ti-Rong Wu, Meng-Yu Tsai, I-Chen Wu, Cho-Jui Hsieh. *NeurIPS 2022*
  - Key Word: Adversarial Robustness; AlphaZero; Reinforcement Learning.
  - <details><summary>Digest</summary> We first extend the concept of adversarial examples to the game of Go: we generate perturbed states that are ''semantically'' equivalent to the original state by adding meaningless moves to the game, and an adversarial state is a perturbed state leading to an undoubtedly inferior action that is obvious even for Go beginners. However, searching the adversarial state is challenging due to the large, discrete, and non-differentiable search space. To tackle this challenge, we develop the first adversarial attack on Go AIs that can efficiently search for adversarial states by strategically reducing the search space.

- Physically Adversarial Attacks and Defenses in Computer Vision: A Survey. [[paper]](https://arxiv.org/abs/2211.01671)
  - Xingxing Wei, Bangzheng Pu, Jiefan Lu, Baoyuan Wu.
  - Key Word: Physical Adversarial Attacks; Survey.
  - <details><summary>Digest</summary> We summarize a survey versus the current physically adversarial attacks and physically adversarial defenses in computer vision. To establish a taxonomy, we organize the current physical attacks from attack tasks, attack forms, and attack methods, respectively. Thus, readers can have a systematic knowledge about this topic from different aspects. For the physical defenses, we establish the taxonomy from pre-processing, in-processing, and post-processing for the DNN models to achieve a full coverage of the adversarial defenses. 

- DensePure: Understanding Diffusion Models towards Adversarial Robustness. [[paper]](https://arxiv.org/abs/2211.00322)
  - Chaowei Xiao, Zhongzhu Chen, Kun Jin, Jiongxiao Wang, Weili Nie, Mingyan Liu, Anima Anandkumar, Bo Li, Dawn Song.
  - Key Word: Diffusion Models; Adversarial Robustness.
  - <details><summary>Digest</summary> Diffusion models have been recently employed to improve certified robustness through the process of denoising. However, the theoretical understanding of why diffusion models are able to improve the certified robustness is still lacking, preventing from further improvement. In this study, we close this gap by analyzing the fundamental properties of diffusion models and establishing the conditions under which they can enhance certified robustness. This deeper understanding allows us to propose a new method DensePure, designed to improve the certified robustness of a pretrained model (i.e. classifier).

- Adversarial Training with Complementary Labels: On the Benefit of Gradually Informative Attacks. [[paper]](https://arxiv.org/abs/2211.00269)
  - Jianan Zhou, Jianing Zhu, Jingfeng Zhang, Tongliang Liu, Gang Niu, Bo Han, Masashi Sugiyama. *NeurIPS 2022*
  - Key Word: Complementary Learning; Adversarial Training.
  - <details><summary>Digest</summary> To push AT towards more practical scenarios, we explore a brand new yet challenging setting, i.e., AT with complementary labels (CLs), which specify a class that a data sample does not belong to. However, the direct combination of AT with existing methods for CLs results in consistent failure, but not on a simple baseline of two-stage training. In this paper, we further explore the phenomenon and identify the underlying challenges of AT with CLs as intractable adversarial optimization and low-quality adversarial examples.

- FI-ODE: Certified and Robust Forward Invariance in Neural ODEs. [[paper]](https://arxiv.org/abs/2210.16940) [[code]](https://github.com/yjhuangcd/FI-ODE)
  - Yujia Huang, Ivan Dario Jimenez Rodriguez, Huan Zhang, Yuanyuan Shi, Yisong Yue.
  - Key Word: Certified Adversarial Robustness; Neural Ordinary Differential Equations.
  - <details><summary>Digest</summary> We study how to certifiably enforce forward invariance properties in neural ODEs. Forward invariance implies that the hidden states of the ODE will stay in a ``good'' region, and a robust version would hold even under adversarial perturbations to the input. Such properties can be used to certify desirable behaviors such as adversarial robustness (the hidden states stay in the region that generates accurate classification even under input perturbations) and safety in continuous control (the system never leaves some safe set). 

- Universal Adversarial Directions. [[paper]](https://arxiv.org/abs/2210.15997)
  - Ching Lam Choi, Farzan Farnia.
  - Key Word: Universal Adversarial Perturbations.
  - <details><summary>Digest</summary> We study the transferability of UAPs by analyzing equilibrium in the universal adversarial example game between the classifier and UAP adversary players. We show that under mild assumptions the universal adversarial example game lacks a pure Nash equilibrium, indicating UAPs' suboptimal transferability across DNN classifiers. To address this issue, we propose Universal Adversarial Directions (UADs) which only fix a universal direction for adversarial perturbations and allow the perturbations' magnitude to be chosen freely across samples. 

- Efficient and Effective Augmentation Strategy for Adversarial Training. [[paper]](https://arxiv.org/abs/2210.15318) [[code]](https://github.com/val-iisc/DAJAT)
  - Sravanti Addepalli, Samyak Jain, R.Venkatesh Babu. *NeurIPS 2022*
  - Key Word: Adversarial Training; Data Augmentation.
  - <details><summary>Digest</summary> We first explain this contrasting behavior by viewing augmentation during training as a problem of domain generalization, and further propose Diverse Augmentation-based Joint Adversarial Training (DAJAT) to use data augmentations effectively in adversarial training. We aim to handle the conflicting goals of enhancing the diversity of the training dataset and training with data that is close to the test distribution by using a combination of simple and complex augmentations with separate batch normalization layers during training. 

- Accelerating Certified Robustness Training via Knowledge Transfer. [[paper]](https://arxiv.org/abs/2210.14283)
  - Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati. *NeurIPS 2022*
  - Key Word: Certified Adversarial Robustness; Transfer Learning.
  - <details><summary>Digest</summary> We propose Certified Robustness Transfer (CRT), a general-purpose framework for reducing the computational overhead of any certifiably robust training method through knowledge transfer. Given a robust teacher, our framework uses a novel training loss to transfer the teacher's robustness to the student. We provide theoretical and empirical validation of CRT. 

- Hindering Adversarial Attacks with Implicit Neural Representations. [[paper]](https://arxiv.org/abs/2210.13982)
  - Andrei A. Rusu, Dan A. Calian, Sven Gowal, Raia Hadsell.
  - Key Word: Adversarial Attacks; Implicit Neural Representations.
  - <details><summary>Digest</summary> We introduce the Lossy Implicit Network Activation Coding (LINAC) defence, an input transformation which successfully hinders several common adversarial attacks on CIFAR-10 classifiers for perturbations up to ϵ=8/255 in L∞ norm and ϵ=0.5 in L2 norm. Implicit neural representations are used to approximately encode pixel colour intensities in 2D images such that classifiers trained on transformed data appear to have robustness to small perturbations without adversarial training or large drops in performance.

- Chaos Theory and Adversarial Robustness. [[paper]](https://arxiv.org/abs/2210.13235)
  - Jonathan S. Kent. 
  - Key Word: Adversarial Robustness; Chaos Theory.
  - <details><summary>Digest</summary> Neural Networks, being susceptible to adversarial attacks, should face a strict level of scrutiny before being deployed in critical or adversarial applications. This paper uses ideas from Chaos Theory to explain, analyze, and quantify the degree to which Neural Networks are susceptible to or robust against adversarial attacks. Our results show that susceptibility to attack grows significantly with the depth of the model, which has significant safety implications for the design of Neural Networks for production environments. We also demonstrate how to quickly and easily approximate the certified robustness radii for extremely large models, which until now has been computationally infeasible to calculate directly, as well as show a clear relationship between our new susceptibility metric and post-attack accuracy.

- Nash Equilibria and Pitfalls of Adversarial Training in Adversarial Robustness Games. [[paper]](https://arxiv.org/abs/2210.12606)
  - Maria-Florina Balcan, Rattana Pukdee, Pradeep Ravikumar, Hongyang Zhang. 
  - Key Word: Adversarial Training; Game Theory.
  - <details><summary>Digest</summary> Adversarial training is a standard technique for training adversarially robust models. In this paper, we study adversarial training as an alternating best-response strategy in a 2-player zero-sum game. We prove that even in a simple scenario of a linear classifier and a statistical model that abstracts robust vs. non-robust features, the alternating best response strategy of such game may not converge. On the other hand, a unique pure Nash equilibrium of the game exists and is provably robust. We support our theoretical results with experiments, showing the non-convergence of adversarial training and the robustness of Nash equilibrium.

- Evolution of Neural Tangent Kernels under Benign and Adversarial Training. [[paper]](https://arxiv.org/abs/2210.12030) [[code]](https://github.com/yolky/adversarial_ntk_evolution)
  - Noel Loo, Ramin Hasani, Alexander Amini, Daniela Rus. *NeurIPS 2022*
  - Key Word: Adversarial Training; Neural Tangent Kernel.
  - <details><summary>Digest</summary> We perform an empirical study of the evolution of the empirical NTK under standard and adversarial training, aiming to disambiguate the effect of adversarial training on kernel learning and lazy training. We find under adversarial training, the empirical NTK rapidly converges to a different kernel (and feature map) than standard training. 

- LOT: Layer-wise Orthogonal Training on Improving l2 Certified Robustness. [[paper]](https://arxiv.org/abs/2210.11620) [[code]](https://github.com/AI-secure/Layerwise-Orthogonal-Training)
  - Xiaojun Xu, Linyi Li, Bo Li. *NeurIPS 2022*
  - Key Word: Certified Adversarial Robustness.
  - <details><summary>Digest</summary> We propose a layer-wise orthogonal training method (LOT) to effectively train 1-Lipschitz convolution layers via parametrizing an orthogonal matrix with an unconstrained matrix. We then efficiently compute the inverse square root of a convolution kernel by transforming the input domain to the Fourier frequency domain. On the other hand, as existing works show that semi-supervised training helps improve empirical robustness, we aim to bridge the gap and prove that semi-supervised learning also improves the certified robustness of Lipschitz-bounded models. 

- On the Adversarial Robustness of Mixture of Experts. [[paper]](https://arxiv.org/abs/2210.10253)
  - Joan Puigcerver, Rodolphe Jenatton, Carlos Riquelme, Pranjal Awasthi, Srinadh Bhojanapalli.
  - Key Word: Adversarial Robustness; Mixture of Experts.
  - <details><summary>Digest</summary> Recently, Bubeck and Sellke proved a lower bound on the Lipschitz constant of functions that fit the training data in terms of their number of parameters. This raises an interesting open question, do -- and can -- functions with more parameters, but not necessarily more computational cost, have better robustness? We study this question for sparse Mixture of Expert models (MoEs), that make it possible to scale up the model size for a roughly constant computational cost. We theoretically show that under certain conditions on the routing and the structure of the data, MoEs can have significantly smaller Lipschitz constants than their dense counterparts. 

- Scaling Adversarial Training to Large Perturbation Bounds. [[paper]](https://arxiv.org/abs/2210.09852) [[code]](https://github.com/val-iisc/OAAT)
  - Sravanti Addepalli, Samyak Jain, Gaurang Sriramanan, R.Venkatesh Babu. *ECCV 2022*
  - Key Word: Adversarial Training.
  - <details><summary>Digest</summary> We aim to achieve adversarial robustness within larger bounds, against perturbations that may be perceptible, but do not change human (or Oracle) prediction. The presence of images that flip Oracle predictions and those that do not makes this a challenging setting for adversarial robustness. We discuss the ideal goals of an adversarial defense algorithm beyond perceptual limits, and further highlight the shortcomings of naively extending existing training algorithms to higher perturbation bounds. In order to overcome these shortcomings, we propose a novel defense, Oracle-Aligned Adversarial Training (OA-AT), to align the predictions of the network with that of an Oracle during adversarial training. 

- Adv-Attribute: Inconspicuous and Transferable Adversarial Attack on Face Recognition. [[paper]](https://arxiv.org/abs/2210.06871)
  - Shuai Jia, Bangjie Yin, Taiping Yao, Shouhong Ding, Chunhua Shen, Xiaokang Yang, Chao Ma. *NeurIPS 2022*
  - Key Word: Adversarial Transferability; Face Recognition.
  - <details><summary>Digest</summary> In this work, instead of performing perturbations on the low-level pixels, we propose to generate attacks through perturbing on the high-level semantics to improve attack transferability. Specifically, a unified flexible framework, Adversarial Attributes (Adv-Attribute), is designed to generate inconspicuous and transferable attacks on face recognition, which crafts the adversarial noise and adds it into different attributes based on the guidance of the difference in face recognition features from the target. 

- Double Bubble, Toil and Trouble: Enhancing Certified Robustness through Transitivity. [[paper]](https://arxiv.org/abs/2210.06077) [[code]](https://github.com/andrew-cullen/doublebubble)
  - Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani, Benjamin I.P. Rubinstein. *NeurIPS 2022*
  - Key Word: Certified Adversarial Robustness; Randomised Smoothing.
  - <details><summary>Digest</summary> We demonstrate how today's "optimal" certificates can be improved by exploiting both the transitivity of certifications, and the geometry of the input space, giving rise to what we term Geometrically-Informed Certified Robustness. By considering the smallest distance to points on the boundary of a set of certifications this approach improves certifications for more than 80% of Tiny-Imagenet instances, yielding an on average 5% increase in the associated certification. 

- Boosting the Transferability of Adversarial Attacks with Reverse Adversarial Perturbation. [[paper]](https://arxiv.org/abs/2210.05968) [[code]](https://github.com/SCLBD/Transfer_attack_RAP)
  - Zeyu Qin, Yanbo Fan, Yi Liu, Li Shen, Yong Zhang, Jue Wang, Baoyuan Wu. *NeurIPS 2022*
  - Key Word: Adversarial Transferability.
  - <details><summary>Digest</summary> We propose a novel attack method, dubbed reverse adversarial perturbation (RAP). Specifically, instead of minimizing the loss of a single adversarial point, we advocate seeking adversarial example located at a region with unified low loss value, by injecting the worst-case perturbation (the reverse adversarial perturbation) for each step of the optimization procedure. The adversarial attack with RAP is formulated as a min-max bi-level optimization problem. 

- Robust Models are less Over-Confident. [[paper]](https://arxiv.org/abs/2210.05938) [[code]](https://github.com/GeJulia/robustness_confidences_evaluation)
  - Julia Grabinski, Paul Gavrikov, Janis Keuper, Margret Keuper. *NeurIPS 2022*
  - Key Word: Adversarial Robustness; Confidence Calibration.
  - <details><summary>Digest</summary> We empirically analyze a variety of adversarially trained models that achieve high robust accuracies when facing state-of-the-art attacks and we show that AT has an interesting side-effect: it leads to models that are significantly less overconfident with their decisions, even on clean data than non-robust models. Further, our analysis of robust models shows that not only AT but also the model's building blocks (like activation functions and pooling) have a strong influence on the models' prediction confidences.

- What Can the Neural Tangent Kernel Tell Us About Adversarial Robustness? [[paper]](https://arxiv.org/abs/2210.05577)
  - Nikolaos Tsilivis, Julia Kempe. *NeurIPS 2022*
  - Key Word: Adversarial Robustness; Neural Tangent Kernel.
  - <details><summary>Digest</summary> We show how NTKs allow to generate adversarial examples in a ``training-free'' fashion, and demonstrate that they transfer to fool their finite-width neural net counterparts in the ''lazy'' regime. We leverage this connection to provide an alternative view on robust and non-robust features, which have been suggested to underlie the adversarial brittleness of neural nets. Specifically, we define and study features induced by the eigendecomposition of the kernel to better understand the role of robust and non-robust features, the reliance on both for standard classification and the robustness-accuracy trade-off. 

- ViewFool: Evaluating the Robustness of Visual Recognition to Adversarial Viewpoints. [[paper]](https://arxiv.org/abs/2210.03895)
  - Yinpeng Dong, Shouwei Ruan, Hang Su, Caixin Kang, Xingxing Wei, Jun Zhu. *NeurIPS 2022*
  - Key Word: Adversarial Robustness; Robustness to 3D Variations; Novel View Synthesis; Neural Rendering.
  - <details><summary>Digest</summary> We propose a novel method called ViewFool to find adversarial viewpoints that mislead visual recognition models. By encoding real-world objects as neural radiance fields (NeRF), ViewFool characterizes a distribution of diverse adversarial viewpoints under an entropic regularizer, which helps to handle the fluctuations of the real camera pose and mitigate the reality gap between the real objects and their neural representations. 

- Towards Out-of-Distribution Adversarial Robustness. [[paper]](https://arxiv.org/abs/2210.03150)
  - Adam Ibrahim, Charles Guille-Escuret, Ioannis Mitliagkas, Irina Rish, David Krueger, Pouya Bashivan.
  - Key Word: Adversarial Robustness; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> Adversarial robustness continues to be a major challenge for deep learning. A core issue is that robustness to one type of attack often fails to transfer to other attacks. While prior work establishes a theoretical trade-off in robustness against different Lp norms, we show that there is potential for improvement against many commonly used attacks by adopting a domain generalisation approach. Concretely, we treat each type of attack as a domain, and apply the Risk Extrapolation method (REx), which promotes similar levels of robustness against all training attacks. 

- Pre-trained Adversarial Perturbations. [[paper]](https://arxiv.org/abs/2210.03372)
  - Yuanhao Ban, Yinpeng Dong. *NeurIPS 2022*
  - Key Word: Adversarial Robustness; Self-Supervision.
  - <details><summary>Digest</summary> We delve into the robustness of pre-trained models by introducing Pre-trained Adversarial Perturbations (PAPs), which are universal perturbations crafted for the pre-trained models to maintain the effectiveness when attacking fine-tuned ones without any knowledge of the downstream tasks. To this end, we propose a Low-Level Layer Lifting Attack (L4A) method to generate effective PAPs by lifting the neuron activations of low-level layers of the pre-trained models. 

- Practical Adversarial Attacks on Spatiotemporal Traffic Forecasting Models. [[paper]](https://arxiv.org/abs/2210.02447) [[code]](https://github.com/kdd-hkust/Adv-ST)
  - Fan Liu, Hao Liu, Wenzhao Jiang. *NeurIPS 2022*
  - Key Word: Spatiotemporal Traffic Forecasting; Adversarial Attack.
  - <details><summary>Digest</summary> We investigate the vulnerability of spatiotemporal traffic forecasting models and propose a practical adversarial spatiotemporal attack framework. Specifically, instead of simultaneously attacking all geo-distributed data sources, an iterative gradient-guided node saliency method is proposed to identify the time-dependent set of victim nodes. Furthermore, we devise a spatiotemporal gradient descent based scheme to generate real-valued adversarial traffic states under a perturbation constraint. Meanwhile, we theoretically demonstrate the worst performance bound of adversarial traffic forecasting attacks.

- On the Robustness of Deep Clustering Models: Adversarial Attacks and Defenses. [[paper]](https://arxiv.org/abs/2210.01940)
  - Anshuman Chhabra, Ashwin Sekhari, Prasant Mohapatra. *NeurIPS 2022*
  - Key Word: Deep Clustering; Adversarial Robustness.
  - <details><summary>Digest</summary> While traditional clustering approaches have been analyzed from a robustness perspective, no prior work has investigated adversarial attacks and robustness for deep clustering models in a principled manner. To bridge this gap, we propose a blackbox attack using Generative Adversarial Networks (GANs) where the adversary does not know which deep clustering model is being used, but can query it for outputs. We analyze our attack against multiple state-of-the-art deep clustering models and real-world datasets, and find that it is highly successful. We then employ some natural unsupervised defense approaches, but find that these are unable to mitigate our attack.

- Natural Color Fool: Towards Boosting Black-box Unrestricted Attacks. [[paper]](https://arxiv.org/abs/2210.02041) [[code]](https://github.com/ylhz/Natural-Color-Fool)
  - Shengming Yuan, Qilong Zhang, Lianli Gao, Yaya Cheng, Jingkuan Song. *NeurIPS 2022*
  - Key Word: Unrestricted Color Attacks.
  - <details><summary>Digest</summary> Unrestricted color attacks, which manipulate semantically meaningful color of an image, have shown their stealthiness and success in fooling both human eyes and deep neural networks. However, current works usually sacrifice the flexibility of the uncontrolled setting to ensure the naturalness of adversarial examples. As a result, the black-box attack performance of these methods is limited. To boost transferability of adversarial examples without damaging image quality, we propose a novel Natural Color Fool (NCF) which is guided by realistic color distributions sampled from a publicly available dataset and optimized by our neighborhood search and initialization reset.

- Rethinking Lipschitz Neural Networks for Certified L-infinity Robustness. [[paper]](https://arxiv.org/abs/2210.01787)
  - Bohang Zhang, Du Jiang, Di He, Liwei Wang. *NeurIPS 2022*
  - Key Word: Certified L-infinity Robustness; Randomized Smoothing.
  - <details><summary>Digest</summary> We derive two fundamental impossibility results that hold for any standard Lipschitz network: one for robust classification on finite datasets, and the other for Lipschitz function approximation. These results identify that networks built upon norm-bounded affine layers and Lipschitz activations intrinsically lose expressive power even in the two-dimensional case, and shed light on how recently proposed Lipschitz networks (e.g., GroupSort and ℓ∞-distance nets) bypass these impossibilities by leveraging order statistic functions.

- Perceptual Attacks of No-Reference Image Quality Models with Human-in-the-Loop. [[paper]](https://arxiv.org/abs/2210.00933)
  - Weixia Zhang, Dingquan Li, Xiongkuo Min, Guangtao Zhai, Guodong Guo, Xiaokang Yang, Kede Ma. *NeurIPS 2022*
  - Key Word: Adversarial Robustness; No-reference Image Quality Assessment.
  - <details><summary>Digest</summary> No-reference image quality assessment (NR-IQA) aims to quantify how humans perceive visual distortions of digital images without access to their undistorted references. NR-IQA models are extensively studied in computational vision, and are widely used for performance evaluation and perceptual optimization of man-made vision systems. Here we make one of the first attempts to examine the perceptual robustness of NR-IQA models. Under a Lagrangian formulation, we identify insightful connections of the proposed perceptual attack to previous beautiful ideas in computer vision and machine learning. 

- Understanding Adversarial Robustness Against On-manifold Adversarial Examples. [[paper]](https://arxiv.org/abs/2210.00430)
  - Jiancong Xiao, Liusha Yang, Yanbo Fan, Jue Wang, Zhi-Quan Luo.
  - Key Word: Adversarial Robustness; On-Manifold Adversarial Examples.
  - <details><summary>Digest</summary> We revisit the off-manifold assumption and want to study a question: at what level is the poor performance of neural networks against adversarial attacks due to on-manifold adversarial examples? Since the true data manifold is unknown in practice, we consider two approximated on-manifold adversarial examples on both real and synthesis datasets. On real datasets, we show that on-manifold adversarial examples have greater attack rates than off-manifold adversarial examples on both standard-trained and adversarially-trained models. On synthetic datasets, theoretically, We prove that on-manifold adversarial examples are powerful, yet adversarial training focuses on off-manifold directions and ignores the on-manifold adversarial examples. 

- Adaptive Weight Decay: On The Fly Weight Decay Tuning for Improving Robustness. [[paper]](https://arxiv.org/abs/2210.00094)
  - Amin Ghiasi, Ali Shafahi, Reza Ardekani.
  - Key Word: Weight Decay; Adversarial Robustness; Learning with Label Noise.
  - <details><summary>Digest</summary> We introduce adaptive weight decay, which automatically tunes the hyper-parameter for weight decay during each training iteration. For classification problems, we propose changing the value of the weight decay hyper-parameter on the fly based on the strength of updates from the classification loss (i.e., gradient of cross-entropy), and the regularization loss (i.e., ℓ2-norm of the weights). We show that this simple modification can result in large improvements in adversarial robustness -- an area which suffers from robust overfitting -- without requiring extra data. 

- Your Out-of-Distribution Detection Method is Not Robust! [[paper]](https://arxiv.org/abs/2209.15246) [[code]](https://github.com/rohban-lab/atd)
  - Mohammad Azizmalayeri, Arshia Soltani Moakhar, Arman Zarei, Reihaneh Zohrabi, Mohammad Taghi Manzuri, Mohammad Hossein Rohban. *NeurIPS 2022*
  - Key Word: Out-of-Distribution Detection; Adversarial Robustness.
  - <details><summary>Digest</summary> We re-examine these defenses against an end-to-end PGD attack on in/out data with larger perturbation sizes, e.g. up to commonly used ϵ=8/255 for the CIFAR-10 dataset. Surprisingly, almost all of these defenses perform worse than a random detection under the adversarial setting.

- A Survey on Physical Adversarial Attack in Computer Vision. [[paper]](https://arxiv.org/abs/2209.14262)
  - Donghua Wang, Wen Yao, Tingsong Jiang, Guijiang Tang, Xiaoqian Chen.
  - Key Word: Physical Adversarial Attacks; Survey.
  - <details><summary>Digest</summary> We review the development of physical adversarial attacks in DNN-based computer vision tasks, including image recognition tasks, object detection tasks, and semantic segmentation. For the sake of completeness of the algorithm evolution, we will briefly introduce the works that do not involve the physical adversarial attack. 

- GAMA: Generative Adversarial Multi-Object Scene Attacks. [[paper]](https://arxiv.org/abs/2209.09502)
  - Abhishek Aich, Calvin Khang-Ta, Akash Gupta, Chengyu Song, Srikanth V. Krishnamurthy, M. Salman Asif, Amit K. Roy-Chowdhury.
  - Key Word: Multi-object Scene based Generative Adversarial Attack; Multi-Modal Machine Learning.
  - <details><summary>Digest</summary> This paper presents the first approach of using generative models for adversarial attacks on multi-object scenes. In order to represent the relationships between different objects in the input scene, we leverage upon the open-sourced pre-trained vision-language model CLIP (Contrastive Language-Image Pre-training), with the motivation to exploit the encoded semantics in the language space along with the visual space. We call this attack approach Generative Adversarial Multi-object scene Attacks (GAMA). GAMA demonstrates the utility of the CLIP model as an attacker's tool to train formidable perturbation generators for multi-object scenes. 

- Part-Based Models Improve Adversarial Robustness. [[paper]](https://arxiv.org/abs/2209.09117) [[code]](https://github.com/chawins/adv-part-model)
  - Chawin Sitawarin, Kornrapat Pongmala, Yizheng Chen, Nicholas Carlini, David Wagner.
  - Key Word: Adversarial Robustness; Part-Based Models.
  - <details><summary>Digest</summary> We believe that the richer form of annotation helps guide neural networks to learn more robust features without requiring more samples or larger models. Our model combines a part segmentation model with a tiny classifier and is trained end-to-end to simultaneously segment objects into parts and then classify the segmented object.

- AdvDO: Realistic Adversarial Attacks for Trajectory Prediction. [[paper]](https://arxiv.org/abs/2209.08744)
  - Yulong Cao, Chaowei Xiao, Anima Anandkumar, Danfei Xu, Marco Pavone.
  - Key Word: Adversarial Attacks; Trajectory Prediciton.
  - <details><summary>Digest</summary> Trajectory prediction is essential for autonomous vehicles (AVs) to plan correct and safe driving behaviors. While many prior works aim to achieve higher prediction accuracy, few study the adversarial robustness of their methods. To bridge this gap, we propose to study the adversarial robustness of data-driven trajectory prediction systems. We devise an optimization-based adversarial attack framework that leverages a carefully-designed differentiable dynamic model to generate realistic adversarial trajectories. 

- Enhance the Visual Representation via Discrete Adversarial Training. [[paper]](https://arxiv.org/abs/2209.07735) [[code]](https://github.com/alibaba/easyrobust)
  - Xiaofeng Mao, Yuefeng Chen, Ranjie Duan, Yao Zhu, Gege Qi, Shaokai Ye, Xiaodan Li, Rong Zhang, Hui Xue. *NeurIPS 2022*
  - Key Word: Adversarial Training; Discrete Visual Representation Learning.
  - <details><summary>Digest</summary> We propose Discrete Adversarial Training (DAT). DAT leverages VQGAN to reform the image data to discrete text-like inputs, i.e. visual words. Then it minimizes the maximal risk on such discrete images with symbolic adversarial perturbations. We further give an explanation from the perspective of distribution to demonstrate the effectiveness of DAT.

- Explicit Tradeoffs between Adversarial and Natural Distributional Robustness. [[paper]](https://arxiv.org/abs/2209.07592)
  - Mazda Moayeri, Kiarash Banihashem, Soheil Feizi. *NeurIPS 2022*
  - Key Word: Natural Distributional Robustness; Adversarial Robustness; Spurious Correlations.
  - <details><summary>Digest</summary> We bridge this gap and show that in fact, explicit tradeoffs exist between adversarial and natural distributional robustness. We first consider a simple linear regression setting on Gaussian data with disjoint sets of core and spurious features. In this setting, through theoretical and empirical analysis, we show that (i) adversarial training with ℓ1 and ℓ2 norms increases the model reliance on spurious features; (ii) For ℓ∞ adversarial training, spurious reliance only occurs when the scale of the spurious features is larger than that of the core features; (iii) adversarial training can have an unintended consequence in reducing distributional robustness, specifically when spurious correlations are changed in the new test domain. 

- A Light Recipe to Train Robust Vision Transformers. [[paper]](https://arxiv.org/abs/2209.07399) [[code]](https://github.com/dedeswim/vits-robustness-torch)
  - Edoardo Debenedetti, Vikash Sehwag, Prateek Mittal. 
  - Key Word: Adversarially Robust Vision Transformers.
  - <details><summary>Digest</summary> We ask whether Vision Transformers (ViTs) can serve as an underlying architecture for improving the adversarial robustness of machine learning models against evasion attacks. While earlier works have focused on improving Convolutional Neural Networks, we show that also ViTs are highly suitable for adversarial training to achieve competitive performance. We achieve this objective using a custom adversarial training recipe, discovered using rigorous ablation studies on a subset of the ImageNet dataset. The canonical training recipe for ViTs recommends strong data augmentation, in part to compensate for the lack of vision inductive bias of attention modules, when compared to convolutions. 

- Adversarially Robust Learning: A Generic Minimax Optimal Learner and Characterization. [[paper]](https://arxiv.org/abs/2209.07369)
  - Omar Montasser, Steve Hanneke, Nathan Srebro. *NeurIPS 2022*
  - Key Word: Adversarial Robustness.
  - <details><summary>Digest</summary> We present a minimax optimal learner for the problem of learning predictors robust to adversarial examples at test-time. Interestingly, we find that this requires new algorithmic ideas and approaches to adversarially robust learning. In particular, we show, in a strong negative sense, the suboptimality of the robust learner proposed by Montasser, Hanneke, and Srebro (2019) and a broader family of learners we identify as local learners. Our results are enabled by adopting a global perspective, specifically, through a key technical contribution: the global one-inclusion graph, which may be of independent interest, that generalizes the classical one-inclusion graph due to Haussler, Littlestone, and Warmuth (1994). 

- On the interplay of adversarial robustness and architecture components: patches, convolution and attention. [[paper]](https://arxiv.org/abs/2209.06953)
  - Francesco Croce, Matthias Hein.
  - Key Word: Adversarial Robustness.
  - <details><summary>Digest</summary> In recent years novel architecture components for image classification have been developed, starting with attention and patches used in transformers. While prior works have analyzed the influence of some aspects of architecture components on the robustness to adversarial attacks, in particular for vision transformers, the understanding of the main factors is still limited. We compare several (non)-robust classifiers with different architectures and study their properties, including the effect of adversarial training on the interpretability of the learnt features and robustness to unseen threat models. 

- Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation. [[paper]](https://arxiv.org/abs/2209.05980)
  - Maksym Yatsura, Kaspar Sakmann, N. Grace Hua, Matthias Hein, Jan Hendrik Metze.
  - Key Word: Certified Recovery; Certified Detection; Image Reconstruction.
  - <details><summary>Digest</summary> Adversarial patch attacks are an emerging security threat for real world deep learning applications. We present Demasked Smoothing, the first approach (up to our knowledge) to certify the robustness of semantic segmentation models against this threat model. Previous work on certifiably defending against patch attacks has mostly focused on image classification task and often required changes in the model architecture and additional training which is undesirable and computationally expensive. In Demasked Smoothing, any segmentation model can be applied without particular training, fine-tuning, or restriction of the architecture. 

- Adversarial Coreset Selection for Efficient Robust Training. [[paper]](https://arxiv.org/abs/2209.05785)
  - Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie. *ECCV 2022*
  - Key Word: Adversarial Training; Coreset Selection.
  - <details><summary>Digest</summary> By leveraging the theory of coreset selection, we show how selecting a small subset of training data provides a principled approach to reducing the time complexity of robust training. To this end, we first provide convergence guarantees for adversarial coreset selection. In particular, we show that the convergence bound is directly related to how well our coresets can approximate the gradient computed over the entire training data. Motivated by our theoretical analysis, we propose using this gradient approximation error as our adversarial coreset selection objective to reduce the training set size effectively.

- The Space of Adversarial Strategies. [[paper]](https://arxiv.org/abs/2209.04521)
  - Ryan Sheatsley, Blaine Hoak, Eric Pauley, Patrick McDaniel. *USENIX Security 2022*
  - Key Word: Adversarial Attacks; Benchmark.
  - <details><summary>Digest</summary> We propose a systematic approach to characterize worst-case (i.e., optimal) adversaries. We first introduce an extensible decomposition of attacks in adversarial machine learning by atomizing attack components into surfaces and travelers. With our decomposition, we enumerate over components to create 576 attacks (568 of which were previously unexplored). Next, we propose the Pareto Ensemble Attack (PEA): a theoretical attack that upper-bounds attack performance. 

- Securing the Spike: On the Transferabilty and Security of Spiking Neural Networks to Adversarial Examples. [[paper]](https://arxiv.org/abs/2209.03358)
  - Key Word: Spiking Neural Networks; Transferable Adversarail Examples.
  - <details><summary>Digest</summary> We advance the field of adversarial machine learning through experimentation and analyses of three important SNN security attributes. First, we show that successful white-box adversarial attacks on SNNs are highly dependent on the underlying surrogate gradient technique. Second, we analyze the transferability of adversarial examples generated by SNNs and other state-of-the-art architectures like Vision Transformers and Big Transfer CNNs. We demonstrate that SNNs are not often deceived by adversarial examples generated by Vision Transformers and certain types of CNNs. 

- Bag of Tricks for FGSM Adversarial Training. [[paper]](https://arxiv.org/abs/2209.02684) [[code]](https://github.com/UCSC-VLAA/Bag-of-Tricks-for-FGSM-AT)
  - Zichao Li, Li Liu, Zeyu Wang, Yuyin Zhou, Cihang Xie.
  - Key Word: Fast Adversarial Training.
  - <details><summary>Digest</summary> Existing methods use gradient regularizers or random initialization tricks to attenuate this issue, whereas they either take high computational cost or lead to lower robust accuracy. In this work, we provide the first study, which thoroughly examines a collection of tricks from three perspectives: Data Initialization, Network Structure, and Optimization, to overcome the catastrophic overfitting in FGSM-AT. Surprisingly, we find that simple tricks, i.e., a) masking partial pixels (even without randomness), b) setting a large convolution stride and smooth activation functions, or c) regularizing the weights of the first convolutional layer, can effectively tackle the overfitting issue.

- A Novel Plug-and-Play Approach for Adversarially Robust Generalization. [[paper]](https://arxiv.org/abs/2208.09449)
  - Deepak Maurya, Adarsh Barik, Jean Honorio.
  - Key Word: Adversarial Robustness.
  - <details><summary>Digest</summary> Our main focus is to provide a plug-and-play solution that can be incorporated in the existing machine learning algorithms with minimal changes. To that end, we derive the closed-form ready-to-use solution for several widely used loss functions with a variety of norm constraints on adversarial perturbation. Finally, we validate our approach by showing significant performance improvement on real-world datasets for supervised problems such as regression and classification, as well as for unsupervised problems such as matrix completion and learning graphical models, with very little computational overhead.

- Adversarial Attacks on Image Generation With Made-Up Words. [[paper]](https://arxiv.org/abs/2208.04135)
  - Raphaël Millière.
  - Key Word: Adversarial Attacks; Text-Guided Image Generation; Prompting.
  - <details><summary>Digest</summary> Text-guided image generation models can be prompted to generate images using nonce words adversarially designed to robustly evoke specific visual concepts. Two approaches for such generation are introduced: macaronic prompting, which involves designing cryptic hybrid words by concatenating subword units from different languages; and evocative prompting, which involves designing nonce words whose broad morphological features are similar enough to that of existing words to trigger robust visual associations. The two methods can also be combined to generate images associated with more specific visual concepts. The implications of these techniques for the circumvention of existing approaches to content moderation, and particularly the generation of offensive or harmful images, are discussed.

- Federated Adversarial Learning: A Framework with Convergence Analysis. [[paper]](https://arxiv.org/abs/2208.03635)
  - Xiaoxiao Li, Zhao Song, Jiaming Yang.
  - Key Word: Federated Learning; Adversarial Robustness; Convergence via Over-parameterization.
  - <details><summary>Digest</summary> We formulate a general form of federated adversarial learning (FAL) that is adapted from adversarial learning in the centralized setting. On the client side of FL training, FAL has an inner loop to generate adversarial samples for adversarial training and an outer loop to update local model parameters. On the server side, FAL aggregates local model updates and broadcast the aggregated model. We design a global robust training loss and formulate FAL training as a min-max optimization problem. Unlike the convergence analysis in classical centralized training that relies on the gradient direction, it is significantly harder to analyze the convergence in FAL for three reasons: 1) the complexity of min-max optimization, 2) model not updating in the gradient direction due to the multi-local updates on the client-side before aggregation and 3) inter-client heterogeneity. We address these challenges by using appropriate gradient approximation and coupling techniques and present the convergence analysis in the over-parameterized regime.

- Understanding Adversarial Robustness of Vision Transformers via Cauchy Problem. [[paper]](https://arxiv.org/abs/2208.00906) [[code]](https://github.com/trustai/ode4robustvit)
  - Zheng Wang, Wenjie Ruan. *ECML-PKDD 2022*
  - Key Word: Vision Transformers; Cauchy Problem; Adversarial Robustness.
  - <details><summary>Digest</summary> We aim to introduce a principled and unified theoretical framework to investigate such an argument on ViT's robustness. We first theoretically prove that, unlike Transformers in Natural Language Processing, ViTs are Lipschitz continuous. Then we theoretically analyze the adversarial robustness of ViTs from the perspective of the Cauchy Problem, via which we can quantify how the robustness propagates through layers.

- Is current research on adversarial robustness addressing the right problem? [[paper]](https://arxiv.org/abs/2208.00539)
  - Ali Borji.
  - Key Word: Adversarial Robustness; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> Short answer: Yes, Long answer: No! Indeed, research on adversarial robustness has led to invaluable insights helping us understand and explore different aspects of the problem. Many attacks and defenses have been proposed over the last couple of years. The problem, however, remains largely unsolved and poorly understood. Here, I argue that the current formulation of the problem serves short term goals, and needs to be revised for us to achieve bigger gains. Specifically, the bound on perturbation has created a somewhat contrived setting and needs to be relaxed. This has misled us to focus on model classes that are not expressive enough to begin with. Instead, inspired by human vision and the fact that we rely more on robust features such as shape, vertices, and foreground objects than non-robust features such as texture, efforts should be steered towards looking for significantly different classes of models.

- LGV: Boosting Adversarial Example Transferability from Large Geometric Vicinity. [[paper]](https://arxiv.org/abs/2207.13129) [[code]](https://github.com/framartin/lgv-geometric-transferability)
  - Martin Gubri, Maxime Cordy, Mike Papadakis, Yves Le Traon, Koushik Sen. *ECCV 2022*
  - Key Word: Adversarial Transferability.
  - <details><summary>Digest</summary> We propose transferability from Large Geometric Vicinity (LGV), a new technique to increase the transferability of black-box adversarial attacks. LGV starts from a pretrained surrogate model and collects multiple weight sets from a few additional training epochs with a constant and high learning rate. LGV exploits two geometric properties that we relate to transferability.

- Improving Adversarial Robustness via Mutual Information Estimation. [[paper]](https://arxiv.org/abs/2207.12203) [[code]](https://github.com/dwDavidxd/MIAT)
  - Dawei Zhou, Nannan Wang, Xinbo Gao, Bo Han, Xiaoyu Wang, Yibing Zhan, Tongliang Liu. *ICML 2022*
  - Key Word: Mutual information; Adversarial Robustness.
  - <details><summary>Digest</summary> We investigate the dependence between outputs of the target model and input adversarial samples from the perspective of information theory, and propose an adversarial defense method. Specifically, we first measure the dependence by estimating the mutual information (MI) between outputs and the natural patterns of inputs (called natural MI) and MI between outputs and the adversarial patterns of inputs (called adversarial MI), respectively.

- Can we achieve robustness from data alone? [[paper]](https://arxiv.org/abs/2207.11727)
  - Nikolaos Tsilivis, Jingtong Su, Julia Kempe.
  - Key Word: Dataset Distillation; Distributionally Robust Optimization; Adversarial Augmentation; Adversarial Robustness.
  - <details><summary>Digest</summary> We devise a meta-learning method for robust classification, that optimizes the dataset prior to its deployment in a principled way, and aims to effectively remove the non-robust parts of the data. We cast our optimization method as a multi-step PGD procedure on kernel regression, with a class of kernels that describe infinitely wide neural nets (Neural Tangent Kernels - NTKs).

- Proving Common Mechanisms Shared by Twelve Methods of Boosting Adversarial Transferability. [[paper]](https://arxiv.org/abs/2207.11694)
  - Quanshi Zhang, Xin Wang, Jie Ren, Xu Cheng, Shuyun Lin, Yisen Wang, Xiangming Zhu.
  - Key Word: Adversarial Transferability; Interaction.
  - <details><summary>Digest</summary> This paper summarizes the common mechanism shared by twelve previous transferability-boosting methods in a unified view, i.e., these methods all reduce game-theoretic interactions between regional adversarial perturbations. To this end, we focus on the attacking utility of all interactions between regional adversarial perturbations, and we first discover and prove the negative correlation between the adversarial transferability and the attacking utility of interactions.

- Calibrated ensembles can mitigate accuracy tradeoffs under distribution shift. [[paper]](https://arxiv.org/abs/2207.08977)
  - Ananya Kumar, Tengyu Ma, Percy Liang, Aditi Raghunathan. *UAI 2022*
  - Key Word: Calibration; Distribution Shift.
  - <details><summary>Digest</summary> We find that ID-calibrated ensembles -- where we simply ensemble the standard and robust models after calibrating on only ID data -- outperforms prior state-of-the-art (based on self-training) on both ID and OOD accuracy. On eleven natural distribution shift datasets, ID-calibrated ensembles obtain the best of both worlds: strong ID accuracy and OOD accuracy. We analyze this method in stylized settings, and identify two important conditions for ensembles to perform well both ID and OOD: (1) we need to calibrate the standard and robust models (on ID data, because OOD data is unavailable), (2) OOD has no anticorrelated spurious features.

- Prior-Guided Adversarial Initialization for Fast Adversarial Training. [[paper]](https://arxiv.org/abs/2207.08859) [[code]](https://github.com/jiaxiaojunQAQ/FGSM-PGI)
  - Xiaojun Jia, Yong Zhang, Xingxing Wei, Baoyuan Wu, Ke Ma, Jue Wang, Xiaochun Cao. *ECCV 2022*
  - Key Word: Fast Adversarial Training; Regularization.
  - <details><summary>Digest</summary> We explore the difference between the training processes of SAT and FAT and observe that the attack success rate of adversarial examples (AEs) of FAT gets worse gradually in the late training stage, resulting in overfitting. The AEs are generated by the fast gradient sign method (FGSM) with a zero or random initialization. Based on the observation, we propose a prior-guided FGSM initialization method to avoid overfitting after investigating several initialization strategies, improving the quality of the AEs during the whole training process.

- Watermark Vaccine: Adversarial Attacks to Prevent Watermark Removal. [[paper]](https://arxiv.org/abs/2207.08178) [[code]](https://github.com/thinwayliu/Watermark-Vaccine)
  - Xinwei Liu, Jian Liu, Yang Bai, Jindong Gu, Tao Chen, Xiaojun Jia, Xiaochun Cao. *ECCV 2022*
  - Key Word: Adversarial Attacks; Visible Watermark Removal; Watermark Protection.
  - <details><summary>Digest</summary> As a common security tool, visible watermarking has been widely applied to protect copyrights of digital images. However, recent works have shown that visible watermarks can be removed by DNNs without damaging their host images. Such watermark-removal techniques pose a great threat to the ownership of images. Inspired by the vulnerability of DNNs on adversarial perturbations, we propose a novel defence mechanism by adversarial machine learning for good. From the perspective of the adversary, blind watermark-removal networks can be posed as our target models; then we actually optimize an imperceptible adversarial perturbation on the host images to proactively attack against watermark-removal networks, dubbed Watermark Vaccine.

- Adversarially-Aware Robust Object Detector. [[paper]](https://arxiv.org/abs/2207.06202) [[code]](https://github.com/7eu7d7/RobustDet)
  - Ziyi Dong, Pengxu Wei, Liang Lin. *ECCV 2022*
  - Key Word: Adversarial Robustness; Object Detection.
  - <details><summary>Digest</summary> We empirically explore the model training for adversarial robustness in object detection, which greatly attributes to the conflict between learning clean images and adversarial images. To mitigate this issue, we propose a Robust Detector (RobustDet) based on adversarially-aware convolution to disentangle gradients for model learning on clean and adversarial images. RobustDet also employs the Adversarial Image Discriminator (AID) and Consistent Features with Reconstruction (CFR) to ensure a reliable robustness.

- Frequency Domain Model Augmentation for Adversarial Attack. [[paper]](https://arxiv.org/abs/2207.05382) [[code]](https://github.com/yuyang-long/ssa)
  - Yuyang Long, Qilong Zhang, Boheng Zeng, Lianli Gao, Xianglong Liu, Jian Zhang, Jingkuan Song. *ECCV 2022*
  - Key Word: Frequency; Adversarial Attacks.
  - <details><summary>Digest</summary> For black-box attacks, the gap between the substitute model and the victim model is usually large, which manifests as a weak attack performance. Motivated by the observation that the transferability of adversarial examples can be improved by attacking diverse models simultaneously, model augmentation methods which simulate different models by using transformed images are proposed. However, existing transformations for spatial domain do not translate to significantly diverse augmented models. To tackle this issue, we propose a novel spectrum simulation attack to craft more transferable adversarial examples against both normally trained and defense models.

- Not all broken defenses are equal: The dead angles of adversarial accuracy. [[paper]](https://arxiv.org/abs/2207.04129)
  - Raphael Olivier, Bhiksha Raj.
  - Key Word: Adversarial Defenses.
  - <details><summary>Digest</summary> Many defenses, when evaluated against a strong attack, do not provide accuracy improvements while still contributing partially to adversarial robustness. Popular certification methods suffer from the same issue, as they provide a lower bound to accuracy. To capture finer robustness properties we propose a new metric for L2 robustness, adversarial angular sparsity, which partially answers the question "how many adversarial examples are there around an input". We demonstrate its usefulness by evaluating both "strong" and "weak" defenses. We show that some state-of-the-art defenses, delivering very similar accuracy, can have very different sparsity on the inputs that they are not robust on. We also show that some weak defenses actually decrease robustness, while others strengthen it in a measure that accuracy cannot capture.

- Demystifying the Adversarial Robustness of Random Transformation Defenses. [[paper]](https://arxiv.org/abs/2207.03574) [[code]](https://github.com/wagner-group/demystify-random-transform)
  - Chawin Sitawarin, Zachary Golan-Strieb, David Wagner. *ICML 2022*
  - Key Word: Adversarial Defenses; Random Transformation.
  - <details><summary>Digest</summary> Defenses using random transformations (RT) have shown impressive results, particularly BaRT (Raff et al., 2019) on ImageNet. However, this type of defense has not been rigorously evaluated, leaving its robustness properties poorly understood. Their stochastic properties make evaluation more challenging and render many proposed attacks on deterministic models inapplicable. First, we show that the BPDA attack (Athalye et al., 2018a) used in BaRT's evaluation is ineffective and likely overestimates its robustness. We then attempt to construct the strongest possible RT defense through the informed selection of transformations and Bayesian optimization for tuning their parameters. Furthermore, we create the strongest possible attack to evaluate our RT defense.

- Removing Batch Normalization Boosts Adversarial Training. [[paper]](https://arxiv.org/abs/2207.01156) [[code]](https://github.com/amazon-research/normalizer-free-robust-training)
  - Key Word: Batch Normalization; Adversarial Training.
  - <details><summary>Digest</summary> Our normalizer-free robust training (NoFrost) method extends recent advances in normalizer-free networks to AT for its unexplored advantage on handling the mixture distribution challenge. We show that NoFrost achieves adversarial robustness with only a minor sacrifice on clean sample accuracy. On ImageNet with ResNet50, NoFrost achieves 74.06% clean accuracy, which drops merely 2.00% from standard training. In contrast, BN-based AT obtains 59.28% clean accuracy, suffering a significant 16.78% drop from standard training.

- Efficient Adversarial Training With Data Pruning. [[paper]](https://arxiv.org/abs/2207.00694)
  - Maximilian Kaufmann, Yiren Zhao, Ilia Shumailov, Robert Mullins, Nicolas Papernot.
  - Key Word: Adversarial Training; Data Pruning.
  - <details><summary>Digest</summary> We demonstrate data pruning-a method for increasing adversarial training efficiency through data sub-sampling.We empirically show that data pruning leads to improvements in convergence and reliability of adversarial training, albeit with different levels of utility degradation. For example, we observe that using random sub-sampling of CIFAR10 to drop 40% of data, we lose 8% adversarial accuracy against the strongest attackers, while by using only 20% of data we lose 14% adversarial accuracy and reduce runtime by a factor of 3. Interestingly, we discover that in some settings data pruning brings benefits from both worlds-it both improves adversarial accuracy and training time.

- Adversarial Robustness is at Odds with Lazy Training. [[paper]](https://arxiv.org/abs/2207.00411)
  - Yunjuan Wang, Enayat Ullah, Poorya Mianjy, Raman Arora.
  - Key Word: Adversarial Robustness; Lazy Training.
  - <details><summary>Digest</summary> Recent works show that random neural networks are vulnerable against adversarial attacks [Daniely and Schacham, 2020] and that such attacks can be easily found using a single step of gradient descent [Bubeck et al., 2021]. In this work, we take it one step further and show that a single gradient step can find adversarial examples for networks trained in the so-called lazy regime. This regime is interesting because even though the neural network weights remain close to the initialization, there exist networks with small generalization error, which can be found efficiently using first-order methods. Our work challenges the model of the lazy regime, the dominant regime in which neural networks are provably efficiently learnable. We show that the networks trained in this regime, even though they enjoy good theoretical computational guarantees, remain vulnerable to adversarial examples.

- Increasing Confidence in Adversarial Robustness Evaluations. [[paper]](https://arxiv.org/abs/2206.13991)  
  - Roland S. Zimmermann, Wieland Brendel, Florian Tramer, Nicholas Carlini.
  - Key Word: Adversarial Robustness.
  - <details><summary>Digest</summary> Hundreds of defenses have been proposed to make deep neural networks robust against minimal (adversarial) input perturbations. However, only a handful of these defenses held up their claims because correctly evaluating robustness is extremely challenging: Weak attacks often fail to find adversarial examples even if they unknowingly exist, thereby making a vulnerable network look robust. In this paper, we propose a test to identify weak attacks, and thus weak defense evaluations. Our test slightly modifies a neural network to guarantee the existence of an adversarial example for every sample. Consequentially, any correct attack must succeed in breaking this modified network.

- Defending Multimodal Fusion Models against Single-Source Adversaries. [[paper]](https://arxiv.org/abs/2206.12714)
  - Karren Yang, Wan-Yi Lin, Manash Barman, Filipe Condessa, Zico Kolter. *CVPR 2021*
  - Key Word: Adversarial Robustness; Multimodal Fusion Models.
  - <details><summary>Digest</summary> We investigate the robustness of multimodal neural networks against worst-case (i.e., adversarial) perturbations on a single modality. We first show that standard multimodal fusion models are vulnerable to single-source adversaries: an attack on any single modality can overcome the correct information from multiple unperturbed modalities and cause the model to fail. This surprising vulnerability holds across diverse multimodal tasks and necessitates a solution. Motivated by this finding, we propose an adversarially robust fusion strategy that trains the model to compare information coming from all the input sources, detect inconsistencies in the perturbed modality compared to the other modalities, and only allow information from the unperturbed modalities to pass through.

- Adversarial Robustness of Deep Neural Networks: A Survey from a Formal Verification Perspective. [[paper]](https://arxiv.org/abs/2206.12227)
  - Mark Huasong Meng, Guangdong Bai, Sin Gee Teo, Zhe Hou, Yan Xiao, Yun Lin, Jin Song Dong.
  - Key Word: Adversarial Robustness; Survey.
  - <details><summary>Digest</summary> We survey existing literature in adversarial robustness verification for neural networks and collect 39 diversified research works across machine learning, security, and software engineering domains. We systematically analyze their approaches, including how robustness is formulated, what verification techniques are used, and the strengths and limitations of each technique. We provide a taxonomy from a formal verification perspective for a comprehensive understanding of this topic. We classify the existing techniques based on property specification, problem reduction, and reasoning strategies.

- Measuring Representational Robustness of Neural Networks Through Shared Invariances. [[paper]](https://arxiv.org/abs/2206.11939) [[code]](https://github.com/nvedant07/stir)
  - Vedant Nanda, Till Speicher, Camila Kolling, John P. Dickerson, Krishna P. Gummadi, Adrian Weller. *ICML 2022*
  - Key Word: Representational Similarity; Adversarial Robustness.
  - <details><summary>Digest</summary> A major challenge in studying robustness in deep learning is defining the set of ``meaningless'' perturbations to which a given Neural Network (NN) should be invariant. Most work on robustness implicitly uses a human as the reference model to define such perturbations. Our work offers a new view on robustness by using another reference NN to define the set of perturbations a given NN should be invariant to, thus generalizing the reliance on a reference ``human NN'' to any NN. This makes measuring robustness equivalent to measuring the extent to which two NNs share invariances, for which we propose a measure called STIR. STIR re-purposes existing representation similarity measures to make them suitable for measuring shared invariances.

- Adversarially trained neural representations may already be as robust as corresponding biological neural representations. [[paper]](https://arxiv.org/abs/2206.11228)
  - Chong Guo, Michael J. Lee, Guillaume Leclerc, Joel Dapello, Yug Rao, Aleksander Madry, James J. DiCarlo.
  - Key Word: Adversarial Robustness; Biological Neural Representation.
  - <details><summary>Digest</summary> We develop a method for performing adversarial visual attacks directly on primate brain activity. We then leverage this method to demonstrate that the above-mentioned belief might not be well founded. Specifically, we report that the biological neurons that make up visual systems of primates exhibit susceptibility to adversarial perturbations that is comparable in magnitude to existing (robustly trained) artificial neural networks.

- Guided Diffusion Model for Adversarial Purification from Random Noise. [[paper]](https://arxiv.org/abs/2206.10875)
  - Quanlin Wu, Hang Ye, Yuntian Gu.
  - Key Word: Adversarial Purification; Diffusion Model.
  - <details><summary>Digest</summary> In this paper, we propose a novel guided diffusion purification approach to provide a strong defense against adversarial attacks. Our model achieves 89.62% robust accuracy under PGD-L_inf attack (eps = 8/255) on the CIFAR-10 dataset. We first explore the essential correlations between unguided diffusion models and randomized smoothing, enabling us to apply the models to certified robustness. The empirical results show that our models outperform randomized smoothing by 5% when the certified L2 radius r is larger than 0.5.

- Robust Universal Adversarial Perturbations. [[paper]](https://arxiv.org/abs/2206.10858)
  - Changming Xu, Gagandeep Singh.
  - Key Word: Transferable Adversarial Example; Universal Adversarial Perturbations.
  - <details><summary>Digest</summary> We introduce a new concept and formulation of robust universal adversarial perturbations. Based on our formulation, we build a novel, iterative algorithm that leverages probabilistic robustness bounds for generating UAPs robust against transformations generated by composing arbitrary sub-differentiable transformation functions.

- (Certified!!) Adversarial Robustness for Free! [[paper]](https://arxiv.org/abs/2206.10550)
  - Nicholas Carlini, Florian Tramer, Krishnamurthy (Dj)Dvijotham, J. Zico Kolter.
  - Key Word: Certified Adversarial Robustness; Randomized Smoothing; Diffusion Models.
  - <details><summary>Digest</summary> In this paper we show how to achieve state-of-the-art certified adversarial robustness to 2-norm bounded perturbations by relying exclusively on off-the-shelf pretrained models. To do so, we instantiate the denoised smoothing approach of Salman et al. by combining a pretrained denoising diffusion probabilistic model and a standard high-accuracy classifier. This allows us to certify 71% accuracy on ImageNet under adversarial perturbations constrained to be within a 2-norm of 0.5, an improvement of 14 percentage points over the prior certified SoTA using any approach, or an improvement of 30 percentage points over denoised smoothing. We obtain these results using only pretrained diffusion models and image classifiers, without requiring any fine tuning or retraining of model parameters.

- Understanding Robust Learning through the Lens of Representation Similarities. [[paper]](https://arxiv.org/abs/2206.09868) [[code]](https://github.com/inspire-group/robust_representation_similarity)
  - Christian Cianfarani, Arjun Nitin Bhagoji, Vikash Sehwag, Ben Zhao, Prateek Mittal.
  - Key Word: Adversarial Robustness; Representation Similarity.
  - <details><summary>Digest</summary> We aim to understand how the properties of representations learned by robust training differ from those obtained from standard, non-robust training. This is critical to diagnosing numerous salient pitfalls in robust networks, such as, degradation of performance on benign inputs, poor generalization of robustness, and increase in over-fitting. We utilize a powerful set of tools known as representation similarity metrics, across three vision datasets, to obtain layer-wise comparisons between robust and non-robust DNNs with different architectures, training procedures and adversarial constraints.

- Diversified Adversarial Attacks based on Conjugate Gradient Method. [[paper]](https://arxiv.org/abs/2206.09628) [[code]](https://github.com/yamamura-k/ACG)
  - Keiichiro Yamamura, Haruki Sato, Nariaki Tateiwa, Nozomi Hata, Toru Mitsutake, Issa Oe, Hiroki Ishikura, Katsuki Fujisawa. *ICML 2022*
  - Key Word: Adversarial Attacks.
  - <details><summary>Digest</summary> Although existing methods based on the steepest descent have achieved high attack success rates, ill-conditioned problems occasionally reduce their performance. To address this limitation, we utilize the conjugate gradient (CG) method, which is effective for this type of problem, and propose a novel attack algorithm inspired by the CG method, named the Auto Conjugate Gradient (ACG) attack. The results of large-scale evaluation experiments conducted on the latest robust models show that, for most models, ACG was able to find more adversarial examples with fewer iterations than the existing SOTA algorithm Auto-PGD (APGD).

- On the Role of Generalization in Transferability of Adversarial Examples. [[paper]](https://arxiv.org/abs/2206.09238)
  - Yilin Wang, Farzan Farnia.
  - Key Word: Transferable Adversarial Example.
  - <details><summary>Digest</summary> We aim to demonstrate the role of the generalization properties of the substitute classifier used for generating adversarial examples in the transferability of the attack scheme to unobserved NN classifiers. To do this, we apply the max-min adversarial example game framework and show the importance of the generalization properties of the substitute NN in the success of the black-box attack scheme in application to different NN classifiers. We prove theoretical generalization bounds on the difference between the attack transferability rates on training and test samples.

- Understanding Robust Overfitting of Adversarial Training and Beyond. [[paper]](https://arxiv.org/abs/2206.08675) [[code]](https://github.com/chaojianyu/understanding-robust-overfitting)
  - Chaojian Yu, Bo Han, Li Shen, Jun Yu, Chen Gong, Mingming Gong, Tongliang Liu. *ICML 2022*
  - Key Word: Adversarial Training; Robust Overfitting.
  - <details><summary>Digest</summary> Robust overfitting widely exists in adversarial training of deep networks. The exact underlying reasons for this are still not completely understood. Here, we explore the causes of robust overfitting by comparing the data distribution of non-overfit (weak adversary) and overfitted (strong adversary) adversarial training, and observe that the distribution of the adversarial data generated by weak adversary mainly contain small-loss data. However, the adversarial data generated by strong adversary is more diversely distributed on the large-loss data and the small-loss data. Given these observations, we further designed data ablation adversarial training and identify that some small-loss data which are not worthy of the adversary strength cause robust overfitting in the strong adversary mode. To relieve this issue, we propose minimum loss constrained adversarial training (MLCAT): in a minibatch, we learn large-loss data as usual, and adopt additional measures to increase the loss of the small-loss data.

- Query-Efficient and Scalable Black-Box Adversarial Attacks on Discrete Sequential Data via Bayesian Optimization. [[paper]](https://arxiv.org/abs/2206.08575)
  - Deokjae Lee, Seungyong Moon, Junhyeok Lee, Hyun Oh Song. *ICML 2022*
  - Key Word: Black-Box Adversarial Attacks.
  - <details><summary>Digest</summary> Existing black-box attacks, mostly based on greedy algorithms, find adversarial examples using pre-computed key positions to perturb, which severely limits the search space and might result in suboptimal solutions. To this end, we propose a query-efficient black-box attack using Bayesian optimization, which dynamically computes important positions using an automatic relevance determination (ARD) categorical kernel. We introduce block decomposition and history subsampling techniques to improve the scalability of Bayesian optimization when an input sequence becomes long.

- Adversarial Patch Attacks and Defences in Vision-Based Tasks: A Survey. [[paper]](https://arxiv.org/abs/2206.08304)
  - Abhijith Sharma, Yijun Bian, Phil Munz, Apurva Narayan.
  - Key Word: Adversarial Patach Attacks and Defenses; Survey.
  - <details><summary>Digest</summary> Adversarial attacks in deep learning models, especially for safety-critical systems, are gaining more and more attention in recent years, due to the lack of trust in the security and robustness of AI models. Yet the more primitive adversarial attacks might be physically infeasible or require some resources that are hard to access like the training data, which motivated the emergence of patch attacks. In this survey, we provide a comprehensive overview to cover existing techniques of adversarial patch attacks, aiming to help interested researchers quickly catch up with the progress in this field. We also discuss existing techniques for developing detection and defences against adversarial patches, aiming to help the community better understand this field and its applications in the real world.

- Catastrophic overfitting is a bug but also a feature. [[paper]](https://arxiv.org/abs/2206.08242) [[code]](https://github.com/gortizji/co_features)
  - Guillermo Ortiz-Jiménez, Pau de Jorge, Amartya Sanyal, Adel Bibi, Puneet K. Dokania, Pascal Frossard, Gregory Rogéz, Philip H.S. Torr.
  - Key Word: Adversarial Robustness; Robust Overfitting.
  - <details><summary>Digest</summary> We find that the interplay between the structure of the data and the dynamics of AT plays a fundamental role in CO. Specifically, through active interventions on typical datasets of natural images, we establish a causal link between the structure of the data and the onset of CO in single-step AT methods. This new perspective provides important insights into the mechanisms that lead to CO and paves the way towards a better understanding of the general dynamics of robust model construction.

- Linearity Grafting: Relaxed Neuron Pruning Helps Certifiable Robustness. [[paper]](https://arxiv.org/abs/2206.07839) [[code]](https://github.com/VITA-Group/Linearity-Grafting)
  - Tianlong Chen, Huan Zhang, Zhenyu Zhang, Shiyu Chang, Sijia Liu, Pin-Yu Chen, Zhangyang Wang. *ICML 2022*
  - Key Word: Certified Adversarial Robustness; Pruning.
  - <details><summary>Digest</summary> Certifiable robustness is a highly desirable property for adopting deep neural networks (DNNs) in safety-critical scenarios, but often demands tedious computations to establish. The main hurdle lies in the massive amount of non-linearity in large DNNs. To trade off the DNN expressiveness (which calls for more non-linearity) and robustness certification scalability (which prefers more linearity), we propose a novel solution to strategically manipulate neurons, by "grafting" appropriate levels of linearity. The core of our proposal is to first linearize insignificant ReLU neurons, to eliminate the non-linear components that are both redundant for DNN performance and harmful to its certification. We then optimize the associated slopes and intercepts of the replaced linear activations for restoring model performance while maintaining certifiability. Hence, typical neuron pruning could be viewed as a special case of grafting a linear function of the fixed zero slopes and intercept, that might overly restrict the network flexibility and sacrifice its performance.

- Adversarial Vulnerability of Randomized Ensembles. [[paper]](https://arxiv.org/abs/2206.06737) [[code]](https://github.com/hsndbk4/arc)
  - Hassan Dbouk, Naresh R. Shanbhag. *ICML 2022*
  - Key Word: Adaptive Adversarial Attacks; Ensemble Adversarial Training; Randomized Smoothing.
  - <details><summary>Digest</summary> Recently, works on randomized ensembles have empirically demonstrated significant improvements in adversarial robustness over standard adversarially trained (AT) models with minimal computational overhead, making them a promising solution for safety-critical resource-constrained applications. However, this impressive performance raises the question: Are these robustness gains provided by randomized ensembles real? In this work we address this question both theoretically and empirically. We first establish theoretically that commonly employed robustness evaluation methods such as adaptive PGD provide a false sense of security in this setting.

- Meet You Halfway: Explaining Deep Learning Mysteries. [[paper]](https://arxiv.org/abs/2206.04463)
  - Oriel BenShmuel.
  - Key Word: Adversarial Robustness.
  - <details><summary>Digest</summary> We introduce a new conceptual framework attached with a formal description that aims to shed light on the network's behavior and interpret the behind-the-scenes of the learning process. Our framework provides an explanation for inherent questions concerning deep learning. Particularly, we clarify: (1) Why do neural networks acquire generalization abilities? (2) Why do adversarial examples transfer between different models?. We provide a comprehensive set of experiments that support this new framework, as well as its underlying theory.

- Early Transferability of Adversarial Examples in Deep Neural Networks. [[paper]](https://arxiv.org/abs/2206.04472)
  - Oriel BenShmuel.
  - Key Word: Adversarial Transferability.
  - <details><summary>Digest</summary> This paper will describe and analyze a new phenomenon that was not known before, which we call "Early Transferability". Its essence is that the adversarial perturbations transfer among different networks even at extremely early stages in their training. In fact, one can initialize two networks with two different independent choices of random weights and measure the angle between their adversarial perturbations after each step of the training. What we discovered was that these two adversarial directions started to align with each other already after the first few training steps (which typically use only a small fraction of the available training data), even though the accuracy of the two networks hadn't started to improve from their initial bad values due to the early stage of the training.

- Gradient Obfuscation Gives a False Sense of Security in Federated Learning. [[paper]](https://arxiv.org/abs/2206.04055)
  - Kai Yue, Richeng Jin, Chau-Wai Wong, Dror Baron, Huaiyu Dai.
  - Key Word: Federated Learning; Adversarial Robustness; Privacy.
  - <details><summary>Digest</summary> We present a new data reconstruction attack framework targeting the image classification task in federated learning. We show that commonly adopted gradient postprocessing procedures, such as gradient quantization, gradient sparsification, and gradient perturbation, may give a false sense of security in federated learning. Contrary to prior studies, we argue that privacy enhancement should not be treated as a byproduct of gradient compression.

- Building Robust Ensembles via Margin Boosting. [[paper]](https://arxiv.org/abs/2206.03362) [[code]](https://github.com/zdhNarsil/margin-boosting)
  - Dinghuai Zhang, Hongyang Zhang, Aaron Courville, Yoshua Bengio, Pradeep Ravikumar, Arun Sai Suggala. *ICML 2022*
  - Key Word: Adversarial Robustness; Boosting.
  - <details><summary>Digest</summary> In the context of adversarial robustness, a single model does not usually have enough power to defend against all possible adversarial attacks, and as a result, has sub-optimal robustness. Consequently, an emerging line of work has focused on learning an ensemble of neural networks to defend against adversarial attacks. In this work, we take a principled approach towards building robust ensembles. We view this problem from the perspective of margin-boosting and develop an algorithm for learning an ensemble with maximum margin.

- Adversarial Unlearning: Reducing Confidence Along Adversarial Directions. [[paper]](https://arxiv.org/abs/2206.01367) [[code]](https://github.com/ars22/RCAD-regularizer)
  - Amrith Setlur, Benjamin Eysenbach, Virginia Smith, Sergey Levine. **
  - Key Word: Adversarial Training; Entropy Maximization.
  - <details><summary>Digest</summary> We propose a complementary regularization strategy that reduces confidence on self-generated examples. The method, which we call RCAD (Reducing Confidence along Adversarial Directions), aims to reduce confidence on out-of-distribution examples lying along directions adversarially chosen to increase training loss. In contrast to adversarial training, RCAD does not try to robustify the model to output the original label, but rather regularizes it to have reduced confidence on points generated using much larger perturbations than in conventional adversarial training.

- An Analytic Framework for Robust Training of Artificial Neural Networks. [[paper]](https://arxiv.org/abs/2205.13502)
  - Ramin Barati, Reza Safabakhsh, Mohammad Rahmati.
  - Key Word: Adversarial Robustness; Geometric and Analytic Modeling.
  - <details><summary>Digest</summary> Many studies investigate the phenomenon by proposing a simplified model of how adversarial examples occur and validate it by predicting some aspect of the phenomenon. While these studies cover many different characteristics of the adversarial examples, they have not reached a holistic approach to the geometric and analytic modeling of the phenomenon. This paper propose a formal framework to study the phenomenon in learning theory and make use of complex analysis and holomorphicity to offer a robust learning rule for artificial neural networks. 

- Diffusion Models for Adversarial Purification. [[paper]](https://arxiv.org/abs/2205.07460) [[code]](https://diffpure.github.io/)
  - Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, Anima Anandkumar. *ICML 2022*
  - Key Word: Adversarial Purification; Diffusion Models.
  - <details><summary>Digest</summary> We propose DiffPure that uses diffusion models for adversarial purification: Given an adversarial example, we first diffuse it with a small amount of noise following a forward diffusion process, and then recover the clean image through a reverse generative process. To evaluate our method against strong adaptive attacks in an efficient and scalable way, we propose to use the adjoint method to compute full gradients of the reverse generative process.

- Self-Ensemble Adversarial Training for Improved Robustness. [[paper]](https://arxiv.org/abs/2203.09678) [[code]](https://github.com/whj363636/self-ensemble-adversarial-training)
  - Hongjun Wang, Yisen Wang. *ICLR 2022*
  - Key Word: Adversarial Robustness.
  - <details><summary>Digest</summary> We are dedicated to the weight states of models through the training process and devise a simple but powerful Self-Ensemble Adversarial Training (SEAT) method for yielding a robust classifier by averaging weights of history models. This considerably improves the robustness of the target model against several well known adversarial attacks, even merely utilizing the naive cross-entropy loss to supervise.

- A Unified Wasserstein Distributional Robustness Framework for Adversarial Training. [[paper]](https://arxiv.org/abs/2202.13437) [[code]](https://github.com/tuananhbui89/unified-distributional-robustness)
  - Tuan Anh Bui, Trung Le, Quan Tran, He Zhao, Dinh Phung. *ICLR 2022*
  - Key Word: Adversarial Robustness; Distribution Shift.
  - <details><summary>Digest</summary> This paper presents a unified framework that connects Wasserstein distributional robustness with current state-of-the-art AT methods. We introduce a new Wasserstein cost function and a new series of risk functions, with which we show that standard AT methods are special cases of their counterparts in our framework.

- Make Some Noise: Reliable and Efficient Single-Step Adversarial Training. [[paper]](https://arxiv.org/abs/2202.01181) [[code]](https://github.com/pdejorge/n-fgsm)
  - Pau de Jorge, Adel Bibi, Riccardo Volpi, Amartya Sanyal, Philip H. S. Torr, Grégory Rogez, Puneet K. Dokania.
  - Key Word: Adversarial Training; Robust Overfitting.
  - <details><summary>Digest</summary> We methodically revisit the role of noise and clipping in single-step adversarial training. Contrary to previous intuitions, we find that using a stronger noise around the clean sample combined with not clipping is highly effective in avoiding CO for large perturbation radii. Based on these observations, we then propose Noise-FGSM (N-FGSM) that, while providing the benefits of single-step adversarial training, does not suffer from CO. Empirical analyses on a large suite of experiments show that N-FGSM is able to match or surpass the performance of previous single-step methods while achieving a 3× speed-up.

</details>

<!-- ### Evasion Attacks and Defenses: 2021 -->

<details>
<summary>Evasion Attacks and Defenses: 2021</summary>

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

</details>

<!-- ### Evasion Attacks and Defenses: 2020 -->

<details>
<summary>Evasion Attacks and Defenses: 2020</summary>

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

</details>

<!-- ### Evasion Attacks and Defenses: 2019 -->

<details>
<summary>Evasion Attacks and Defenses: 2019</summary>

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

</details>

<!-- ### Evasion Attacks and Defenses: 2018 -->

<details>
<summary>Evasion Attacks and Defenses: 2018</summary>

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

</details>

<!-- ### Evasion Attacks and Defenses: 2017 -->

<details>
<summary>Evasion Attacks and Defenses: 2017</summary>


- Certifying Some Distributional Robustness with Principled Adversarial Training. [[paper]](https://arxiv.org/abs/1710.10571)
  - Aman Sinha, Hongseok Namkoong, Riccardo Volpi, John Duchi. *ICLR 2018*
  - Key Word: Certificated Adversarial Robustness.
  - <details><summary>Digest</summary> By considering a Lagrangian penalty formulation of perturbing the underlying data distribution in a Wasserstein ball, we provide a training procedure that augments model parameter updates with worst-case perturbations of training data. For smooth losses, our procedure provably achieves moderate levels of robustness with little computational or statistical cost relative to empirical risk minimization.  

- One pixel attack for fooling deep neural networks. [[paper]](https://arxiv.org/abs/1710.08864)
  - Jiawei Su, Danilo Vasconcellos Vargas, Sakurai Kouichi. *IEEE Transactions on Evolutionary Computation*
  - Key Word: Adversarial Attacks.
  - <details><summary>Digest</summary> In this paper, we analyze an attack in an extremely limited scenario where only one pixel can be modified. For that we propose a novel method for generating one-pixel adversarial perturbations based on differential evolution (DE). It requires less adversarial information (a black-box attack) and can fool more types of networks due to the inherent features of DE.  

</details>

<!-- ### Evasion Attacks and Defenses: 2016 -->

<details>
<summary>Evasion Attacks and Defenses: 2016</summary>

- Practical Black-Box Attacks against Machine Learning. [[paper]](https://arxiv.org/abs/1602.02697)
  - Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z. Berkay Celik, Ananthram Swami. *AsiaCCS 2017*
  - Key Word: Adversarial Attacks.
  - <details><summary>Digest</summary> We introduce the first practical demonstration of an attacker controlling a remotely hosted DNN with no such knowledge. Our attack evades a category of defenses, which we call gradient masking, previously proposed to increase resilience to adversarial examples.  

</details>

## Poisoning Attacks and Defenses

<details>
<summary>Poisoning Attacks and Defenses: 2022</summary>

- Hidden Poison: Machine Unlearning Enables Camouflaged Poisoning Attacks. [[paper]](https://arxiv.org/abs/2212.10717) [[code]](https://github.com/jimmy-di/camouflage-poisoning)
  - Jimmy Z. Di, Jack Douglas, Jayadev Acharya, Gautam Kamath, Ayush Sekhari.
  - Key Word: Poisoning Attacks; Machine Unlearning.
  - <details><summary>Digest</summary> We introduce camouflaged data poisoning attacks, a new attack vector that arises in the context of machine unlearning and other settings when model retraining may be induced. An adversary first adds a few carefully crafted points to the training dataset such that the impact on the model's predictions is minimal. The adversary subsequently triggers a request to remove a subset of the introduced points at which point the attack is unleashed and the model's predictions are negatively affected.

- Fine-Tuning Is All You Need to Mitigate Backdoor Attacks. [[paper]](https://arxiv.org/abs/2212.09067)
  - Zeyang Sha, Xinlei He, Pascal Berrang, Mathias Humbert, Yang Zhang.
  - Key Word: Backdoor Defenses; Fine-Tuning.
  - <details><summary>Digest</summary> We show that fine-tuning, one of the most common and easy-to-adopt machine learning training operations, can effectively remove backdoors from machine learning models while maintaining high model utility. Extensive experiments over three machine learning paradigms show that fine-tuning and our newly proposed super-fine-tuning achieve strong defense performance. 

- FLIP: A Provable Defense Framework for Backdoor Mitigation in Federated Learning. [[paper]](https://arxiv.org/abs/2210.12873)
  - Kaiyuan Zhang, Guanhong Tao, Qiuling Xu, Siyuan Cheng, Shengwei An, Yingqi Liu, Shiwei Feng, Guangyu Shen, Pin-Yu Chen, Shiqing Ma, Xiangyu Zhang.
  - Key Word: Backdoor Defenses; Federated Learning.
  - <details><summary>Digest</summary> We theoretically analyze the connection among cross-entropy loss, attack success rate, and clean accuracy in this setting. Moreover, we propose a trigger reverse engineering based defense and show that our method can achieve robustness improvement with guarantee (i.e., reducing the attack success rate) without affecting benign accuracy. 

- FedRecover: Recovering from Poisoning Attacks in Federated Learning using Historical Information. [[paper]](https://arxiv.org/abs/2210.10936)
  - Xiaoyu Cao, Jinyuan Jia, Zaixi Zhang, Neil Zhenqiang Gong. *S&P 2023*
  - Key Word: Federated Learning; Poisoning Defenses.
  - <details><summary>Digest</summary> We propose FedRecover, which can recover an accurate global model from poisoning attacks with small cost for the clients. Our key idea is that the server estimates the clients' model updates instead of asking the clients to compute and communicate them during the recovery process. In particular, the server stores the global models and clients' model updates in each round, when training the poisoned global model. During the recovery process, the server estimates a client's model update in each round using its stored historical information. 

- Not All Poisons are Created Equal: Robust Training against Data Poisoning. [[paper]](https://arxiv.org/abs/2210.09671) [[code]](https://github.com/YuYang0901/EPIC)
  - Yu Yang, Tian Yu Liu, Baharan Mirzasoleiman. *ICML 2022*
  - Key Word: Poisoning Defenses.
  - <details><summary>Digest</summary> We propose an efficient defense mechanism that significantly reduces the success rate of various data poisoning attacks, and provides theoretical guarantees for the performance of the model. Targeted attacks work by adding bounded perturbations to a randomly selected subset of training data to match the targets' gradient or representation. We show that: (i) under bounded perturbations, only a number of poisons can be optimized to have a gradient that is close enough to that of the target and make the attack successful; (ii) such effective poisons move away from their original class and get isolated in the gradient space; (iii) dropping examples in low-density gradient regions during training can successfully eliminate the effective poisons, and guarantees similar training dynamics to that of training on full data. 

- Towards Fair Classification against Poisoning Attacks. [[paper]](https://arxiv.org/abs/2210.09503)
  - Han Xu, Xiaorui Liu, Yuxuan Wan, Jiliang Tang. 
  - Key Word: Poisoning Attacks; Fairness.
  - <details><summary>Digest</summary> We study the poisoning scenario where the attacker can insert a small fraction of samples into training data, with arbitrary sensitive attributes as well as other predictive features. We demonstrate that the fairly trained classifiers can be greatly vulnerable to such poisoning attacks, with much worse accuracy & fairness trade-off, even when we apply some of the most effective defenses (originally proposed to defend traditional classification tasks).

- Thinking Two Moves Ahead: Anticipating Other Users Improves Backdoor Attacks in Federated Learning. [[paper]](https://arxiv.org/abs/2210.09305) [[code]](https://github.com/YuxinWenRick/thinking-two-moves-ahead)
  - Yuxin Wen, Jonas Geiping, Liam Fowl, Hossein Souri, Rama Chellappa, Micah Goldblum, Tom Goldstein.
  - Key Word: Backdoor Attacks; Federated Learning.
  - <details><summary>Digest</summary> Existing attacks do not account for future behaviors of other users, and thus require many sequential updates and their effects are quickly erased. We propose an attack that anticipates and accounts for the entire federated learning pipeline, including behaviors of other clients, and ensures that backdoors are effective quickly and persist even after multiple rounds of community updates. 

- Marksman Backdoor: Backdoor Attacks with Arbitrary Target Class. [[paper]](https://arxiv.org/abs/2210.09194)
  - Khoa D. Doan, Yingjie Lao, Ping Li. *NeurIPS 2022*
  - Key Word: Backdoor Attacks.
  - <details><summary>Digest</summary> This paper exploits a novel backdoor attack with a much more powerful payload, denoted as Marksman, where the adversary can arbitrarily choose which target class the model will misclassify given any input during inference. To achieve this goal, we propose to represent the trigger function as a class-conditional generative model and to inject the backdoor in a constrained optimization framework, where the trigger function learns to generate an optimal trigger pattern to attack any target class at will while simultaneously embedding this generative backdoor into the trained model. 

- Trap and Replace: Defending Backdoor Attacks by Trapping Them into an Easy-to-Replace Subnetwork. [[paper]](https://arxiv.org/abs/2210.06428)
  - Haotao Wang, Junyuan Hong, Aston Zhang, Jiayu Zhou, Zhangyang Wang. *NeurIPS 2022*
  - Key Word: Backdoor Defenses.
  - <details><summary>Digest</summary> We propose a brand-new backdoor defense strategy, which makes it much easier to remove the harmful influence of backdoor samples from the model. Our defense strategy, Trap and Replace, consists of two stages. In the first stage, we bait and trap the backdoors in a small and easy-to-replace subnetwork.  In the second stage, we replace the poisoned light-weighted classification head with an untainted one, by re-training it from scratch only on a small holdout dataset with clean samples, while fixing the stem network. 

- Few-shot Backdoor Attacks via Neural Tangent Kernels. [[paper]](https://arxiv.org/abs/2210.05929)
  - Jonathan Hayase, Sewoong Oh.
  - Key Word: Backdoor Attacks; Neural Tangent Kernel.
  - <details><summary>Digest</summary> In a backdoor attack, an attacker injects corrupted examples into the training set. The goal of the attacker is to cause the final trained model to predict the attacker's desired target label when a predefined trigger is added to test inputs. Central to these attacks is the trade-off between the success rate of the attack and the number of corrupted training examples injected. We pose this attack as a novel bilevel optimization problem: construct strong poison examples that maximize the attack success rate of the trained model. We use neural tangent kernels to approximate the training dynamics of the model being attacked and automatically learn strong poison examples. 

- Backdoor Attacks in the Supply Chain of Masked Image Modeling. [[paper]](https://arxiv.org/abs/2210.01632)
  - Xinyue Shen, Xinlei He, Zheng Li, Yun Shen, Michael Backes, Yang Zhang.
  - Key Word: Backdoor Attacks; Masked Image Modeling.
  - <details><summary>Digest</summary> We are the first to systematically threat modeling on SSL in every phase of the model supply chain, i.e., pre-training, release, and downstream phases. Our evaluation shows that models built with MIM are vulnerable to existing backdoor attacks in release and downstream phases and are compromised by our proposed method in pre-training phase. For instance, on CIFAR10, the attack success rate can reach 99.62%, 96.48%, and 98.89% in the downstream phase, release phase, and pre-training phase, respectively. 

- FLCert: Provably Secure Federated Learning against Poisoning Attacks. [[paper]](https://arxiv.org/abs/2210.00584)
  - Xiaoyu Cao, Zaixi Zhang, Jinyuan Jia, Neil Zhenqiang Gong. *TIFS*
  - Key Word: Poisoning Defenses; Federated Learning.
  - <details><summary>Digest</summary> We aim to bridge the gap by proposing FLCert, an ensemble federated learning framework, that is provably secure against poisoning attacks with a bounded number of malicious clients. Our key idea is to divide the clients into groups, learn a global model for each group of clients using any existing federated learning method, and take a majority vote among the global models to classify a test input. 

- Data Poisoning Attacks Against Multimodal Encoders. [[paper]](https://arxiv.org/abs/2209.15266)
  - Ziqing Yang, Xinlei He, Zheng Li, Michael Backes, Mathias Humbert, Pascal Berrang, Yang Zhang.
  - Key Word: Data Poisoning; Multimodal Learning.
  - <details><summary>Digest</summary> We instead focus on answering two questions: (1) Is the linguistic modality also vulnerable to poisoning attacks? and (2) Which modality is most vulnerable? To answer the two questions, we conduct three types of poisoning attacks against CLIP, the most representative multimodal contrastive learning framework.

- Augmentation Backdoors. [[paper]](https://arxiv.org/abs/2209.15139) [[code]](https://github.com/slkdfjslkjfd/augmentation_backdoors)
  - Joseph Rance, Yiren Zhao, Ilia Shumailov, Robert Mullins.
  - Key Word: Data Augmentation; Backdoor Attacks.
  - <details><summary>Digest</summary> We present three backdoor attacks that can be covertly inserted into data augmentation. Our attacks each insert a backdoor using a different type of computer vision augmentation transform, covering simple image transforms, GAN-based augmentation, and composition-based augmentation. By inserting the backdoor using these augmentation transforms, we make our backdoors difficult to detect, while still supporting arbitrary backdoor functionality. 

- RIBAC: Towards Robust and Imperceptible Backdoor Attack against Compact DNN. [[paper]](https://arxiv.org/abs/2208.10608) [[code]](https://github.com/huyvnphan/eccv2022-ribac)
  - Huy Phan, Cong Shi, Yi Xie, Tianfang Zhang, Zhuohang Li, Tianming Zhao, Jian Liu, Yan Wang, Yingying Chen, Bo Yuan. *ECCV 2022*
  - Key Word: Backdoor Attack; Model Compression.
  - <details><summary>Digest</summary> We propose to study and develop Robust and Imperceptible Backdoor Attack against Compact DNN models (RIBAC). By performing systematic analysis and exploration on the important design knobs, we propose a framework that can learn the proper trigger patterns, model parameters and pruning masks in an efficient way. Thereby achieving high trigger stealthiness, high attack success rate and high model efficiency simultaneously.

- Lethal Dose Conjecture on Data Poisoning. [[paper]](https://arxiv.org/abs/2208.03309)
  - Wenxiao Wang, Alexander Levine, Soheil Feizi.
  - Key Word: Data poisoning; Deep Partition Aggregation; Finite Aggregation.
  - <details><summary>Digest</summary> Deep Partition Aggregation (DPA) and its extension, Finite Aggregation (FA) are recent approaches for provable defenses against data poisoning, where they predict through the majority vote of many base models trained from different subsets of training set using a given learner. The conjecture implies that both DPA and FA are (asymptotically) optimal -- if we have the most data-efficient learner, they can turn it into one of the most robust defenses against data poisoning. This outlines a practical approach to developing stronger defenses against poisoning via finding data-efficient learners.

- Data-free Backdoor Removal based on Channel Lipschitzness. [[paper]](https://arxiv.org/abs/2208.03111) [[code]](https://github.com/rkteddy/channel-Lipschitzness-based-pruning)
  - Runkai Zheng, Rongjun Tang, Jianze Li, Li Liu. *ECCV 2022*
  - Key Word: Backdoor Defense; Lipschitz Constant; Model pruning.
  - <details><summary>Digest</summary> We introduce a novel concept called Channel Lipschitz Constant (CLC), which is defined as the Lipschitz constant of the mapping from the input images to the output of each channel. Then we provide empirical evidences to show the strong correlation between an Upper bound of the CLC (UCLC) and the trigger-activated change on the channel activation. Since UCLC can be directly calculated from the weight matrices, we can detect the potential backdoor channels in a data-free manner, and do simple pruning on the infected DNN to repair the model.

- Just Rotate it: Deploying Backdoor Attacks via Rotation Transformation. [[paper]](https://arxiv.org/abs/2207.10825)
  - Tong Wu, Tianhao Wang, Vikash Sehwag, Saeed Mahloujifar, Prateek Mittal.
  - Key Word: Backdoor Attacks; Object Detection.
  - <details><summary>Digest</summary> Our method constructs the poisoned dataset by rotating a limited amount of objects and labeling them incorrectly; once trained with it, the victim's model will make undesirable predictions during run-time inference. It exhibits a significantly high attack success rate while maintaining clean performance through comprehensive empirical studies on image classification and object detection tasks.

- Suppressing Poisoning Attacks on Federated Learning for Medical Imaging. [[paper]](https://arxiv.org/abs/2207.10804) [[code]](https://github.com/Naiftt/SPAFD)
  - Naif Alkhunaizi, Dmitry Kamzolov, Martin Takáč, Karthik Nandakumar.
  - Key Word: Poisoning Attacks; Federated Learning; Medical Imaging; Healthcare.
  - <details><summary>Digest</summary> We propose a robust aggregation rule called Distance-based Outlier Suppression (DOS) that is resilient to byzantine failures. The proposed method computes the distance between local parameter updates of different clients and obtains an outlier score for each client using Copula-based Outlier Detection (COPOD). The resulting outlier scores are converted into normalized weights using a softmax function, and a weighted average of the local parameters is used for updating the global model.

- When does Bias Transfer in Transfer Learning? [[paper]](https://arxiv.org/abs/2207.02842) [[code]](https://github.com/MadryLab/bias-transfer)
  - Hadi Salman, Saachi Jain, Andrew Ilyas, Logan Engstrom, Eric Wong, Aleksander Madry.
  - Key Word: Backdoor Attacks; Bias Transfer.
  - <details><summary>Digest</summary> Using transfer learning to adapt a pre-trained "source model" to a downstream "target task" can dramatically increase performance with seemingly no downside. In this work, we demonstrate that there can exist a downside after all: bias transfer, or the tendency for biases of the source model to persist even after adapting the model to the target class. Through a combination of synthetic and natural experiments, we show that bias transfer both (a) arises in realistic settings (such as when pre-training on ImageNet or other standard datasets) and (b) can occur even when the target dataset is explicitly de-biased.

- Backdoor Attack is A Devil in Federated GAN-based Medical Image Synthesis. [[paper]](https://arxiv.org/abs/2207.00762)
  - Ruinan Jin, Xiaoxiao Li.
  - Key Word: Backdoor Attacks; Federated Learning; Generative Adversarial Nets; Medical Image; Healthcare.
  - <details><summary>Digest</summary> We propose a way of attacking federated GAN (FedGAN) by treating the discriminator with a commonly used data poisoning strategy in backdoor attack classification models. We demonstrate that adding a small trigger with size less than 0.5 percent of the original image size can corrupt the FL-GAN model. Based on the proposed attack, we provide two effective defense strategies: global malicious detection and local training regularization.

- BackdoorBench: A Comprehensive Benchmark of Backdoor Learning. [[paper]](https://arxiv.org/abs/2206.12654) [[code]](https://github.com/sclbd/backdoorbench)
  - Baoyuan Wu, Hongrui Chen, Mingda Zhang, Zihao Zhu, Shaokui Wei, Danni Yuan, Chao Shen, Hongyuan Zha.
  - Key Word: Backdoor Learning; Benchmark.
  - <details><summary>Digest</summary> We find that the evaluations of new methods are often unthorough to verify their claims and real performance, mainly due to the rapid development, diverse settings, as well as the difficulties of implementation and reproducibility. Without thorough evaluations and comparisons, it is difficult to track the current progress and design the future development roadmap of the literature. To alleviate this dilemma, we build a comprehensive benchmark of backdoor learning, called BackdoorBench. It consists of an extensible modular based codebase (currently including implementations of 8 state-of-the-art (SOTA) attack and 9 SOTA defense algorithms), as well as a standardized protocol of a complete backdoor learning. We also provide comprehensive evaluations of every pair of 8 attacks against 9 defenses, with 5 poisoning ratios, based on 5 models and 4 datasets, thus 8,000 pairs of evaluations in total.

- zPROBE: Zero Peek Robustness Checks for Federated Learning. [[paper]](https://arxiv.org/abs/2206.12100)
  - Zahra Ghodsi, Mojan Javaheripi, Nojan Sheybani, Xinqiao Zhang, Ke Huang, Farinaz Koushanfar.
  - Key Word: Byzantine Attacks; Federated Learning; Zero-Knowledge Proof.
  - <details><summary>Digest</summary> We establish the first private robustness check that uses high break point rank-based statistics on aggregated model updates. By exploiting randomized clustering, we significantly improve the scalability of our defense without compromising privacy. We leverage the derived statistical bounds in zero-knowledge proofs to detect and remove malicious updates without revealing the private user updates. Our novel framework, zPROBE, enables Byzantine resilient and secure federated learning.

- Natural Backdoor Datasets. [[paper]](https://arxiv.org/abs/2206.10673) [[code]](https://github.com/uchicago-sandlab/naturalbackdoors)
  - Emily Wenger, Roma Bhattacharjee, Arjun Nitin Bhagoji, Josephine Passananti, Emilio Andere, Haitao Zheng, Ben Y. Zhao.
  - Key Word: Natural Backdoor Attacks.
  - <details><summary>Digest</summary> Extensive literature on backdoor poison attacks has studied attacks and defenses for backdoors using "digital trigger patterns." In contrast, "physical backdoors" use physical objects as triggers, have only recently been identified, and are qualitatively different enough to resist all defenses targeting digital trigger backdoors. Research on physical backdoors is limited by access to large datasets containing real images of physical objects co-located with targets of classification. Building these datasets is time- and labor-intensive. This works seeks to address the challenge of accessibility for research on physical backdoor attacks. We hypothesize that there may be naturally occurring physically co-located objects already present in popular datasets such as ImageNet. Once identified, a careful relabeling of these data can transform them into training samples for physical backdoor attacks. We propose a method to scalably identify these subsets of potential triggers in existing datasets, along with the specific classes they can poison.

- Neurotoxin: Durable Backdoors in Federated Learning. [[paper]](https://arxiv.org/abs/2206.10341)
  - Zhengming Zhang, Ashwinee Panda, Linyue Song, Yaoqing Yang, Michael W. Mahoney, Joseph E. Gonzalez, Kannan Ramchandran, Prateek Mittal. *ICML 2022*
  - Key Word: Backdoor Attacks; Federated Learning.
  - <details><summary>Digest</summary> Prior work has shown that backdoors can be inserted into FL models, but these backdoors are often not durable, i.e., they do not remain in the model after the attacker stops uploading poisoned updates. Thus, since training typically continues progressively in production FL systems, an inserted backdoor may not survive until deployment. Here, we propose Neurotoxin, a simple one-line modification to existing backdoor attacks that acts by attacking parameters that are changed less in magnitude during training.

- Backdoor Attacks on Vision Transformers. [[paper]](https://arxiv.org/abs/2206.08477)  [[code]](https://github.com/ucdvision/backdoor_transformer)
  - Akshayvarun Subramanya, Aniruddha Saha, Soroush Abbasi Koohpayegani, Ajinkya Tejankar, Hamed Pirsiavash.
  - Key Word: Backdoor Attacks; Vision Transformers.
  - <details><summary>Digest</summary> We are the first to show that ViTs are vulnerable to backdoor attacks. We also find an intriguing difference between ViTs and CNNs - interpretation algorithms effectively highlight the trigger on test images for ViTs but not for CNNs. Based on this observation, we propose a test-time image blocking defense for ViTs which reduces the attack success rate by a large margin.

- Autoregressive Perturbations for Data Poisoning. [[paper]](https://arxiv.org/abs/2206.03693) [[code]](https://github.com/psandovalsegura/autoregressive-poisoning)
  - Pedro Sandoval-Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein, David W. Jacobs.
  - Key Word: Poisoning Attacks.
  - <details><summary>Digest</summary> We introduce autoregressive (AR) poisoning, a method that can generate poisoned data without access to the broader dataset. The proposed AR perturbations are generic, can be applied across different datasets, and can poison different architectures. Compared to existing unlearnable methods, our AR poisons are more resistant against common defenses such as adversarial training and strong data augmentations. Our analysis further provides insight into what makes an effective data poison.

- Backdoor Defense via Decoupling the Training Process. [[paper]](https://arxiv.org/abs/2202.03423) [[code]](https://github.com/sclbd/dbd)
  - Kunzhe Huang, Yiming Li, Baoyuan Wu, Zhan Qin, Kui Ren. *ICLR 2022*
  - Key Word: Backdoor Defenses.
  - <details><summary>Digest</summary>  We reveal that poisoned samples tend to cluster together in the feature space of the attacked DNN model, which is mostly due to the end-to-end supervised training paradigm. Inspired by this observation, we propose a novel backdoor defense via decoupling the original end-to-end training process into three stages. Specifically, we first learn the backbone of a DNN model via self-supervised learning based on training samples without their labels.

- Post-Training Detection of Backdoor Attacks for Two-Class and Multi-Attack Scenarios. [[paper]](https://arxiv.org/abs/2201.08474) [[code]](https://github.com/zhenxianglance/2classbadetection)
  - Zhen Xiang, David J. Miller, George Kesidis. *ICLR 2022*
  - Key Word: Backdoor Detection; Adversarial Training.
  - <details><summary>Digest</summary> We propose a detection framework based on BP reverse-engineering and a novel expected transferability (ET) statistic. We show that our ET statistic is effective using the same detection threshold, irrespective of the classification domain, the attack configuration, and the BP reverse-engineering algorithm that is used.

</details>

<!-- ### Poisoning Attacks and Defenses: 2021 -->

<details>
<summary>Poisoning Attacks and Defenses: 2021</summary>

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

</details>

<!-- ### Poisoning Attacks and Defenses: 2020 -->

<details>
<summary>Poisoning Attacks and Defenses: 2020</summary>

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

</details>

<!-- ### Poisoning Attacks and Defenses: 2019 -->

<details>
<summary>Poisoning Attacks and Defenses: 2019</summary>

- Universal Litmus Patterns: Revealing Backdoor Attacks in CNNs. [[paper]](https://arxiv.org/abs/1906.10842) [[code]](https://github.com/UMBCvision/Universal-Litmus-Patterns)
  - Soheil Kolouri, Aniruddha Saha, Hamed Pirsiavash, Heiko Hoffmann. *CVPR 2020*
  - Key Word: Backdoor Detection.
  - <details><summary>Digest</summary> We introduce a benchmark technique for detecting backdoor attacks (aka Trojan attacks) on deep convolutional neural networks (CNNs). We introduce the concept of Universal Litmus Patterns (ULPs), which enable one to reveal backdoor attacks by feeding these universal patterns to the network and analyzing the output (i.e., classifying the network as `clean' or `corrupted'). This detection is fast because it requires only a few forward passes through a CNN.

</details>

<!-- ### Poisoning Attacks and Defenses: 2017

- BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain. [[paper]](https://arxiv.org/abs/1708.06733)
  - Tianyu Gu, Brendan Dolan-Gavitt, Siddharth Garg.
  - Key Word: Backdoor Attacks; Neuron Activation.
  - <details><summary>Digest</summary>  In this paper we show that outsourced training introduces new security risks: an adversary can create a maliciously trained network (a backdoored neural network, or a BadNet that has state-of-the-art performance on the user's training and validation samples, but behaves badly on specific attacker-chosen inputs.   -->

## Privacy

<details>
<summary>Privacy: 2022</summary>

- SoK: Let The Privacy Games Begin! A Unified Treatment of Data Inference Privacy in Machine Learning. [[paper]](https://arxiv.org/abs/2212.10986)
  - Ahmed Salem, Giovanni Cherubin, David Evans, Boris Köpf, Andrew Paverd, Anshuman Suri, Shruti Tople, Santiago Zanella-Béguelin.
  - Key Word: Survey; Privacy Attacks; Membership Inference Attacks; Attribute Inference Attacks; Property Inference Attacks; Data Reconstruction.
  - <details><summary>Digest</summary> Deploying machine learning models in production may allow adversaries to infer sensitive information about training data. There is a vast literature analyzing different types of inference risks, ranging from membership inference to reconstruction attacks. Inspired by the success of games (i.e., probabilistic experiments) to study security properties in cryptography, some authors describe privacy inference risks in machine learning using a similar game-based style. However, adversary capabilities and goals are often stated in subtly different ways from one presentation to the other, which makes it hard to relate and compose results. In this paper, we present a game-based framework to systematize the body of knowledge on privacy inference risks in machine learning.

- Leveraging Unlabeled Data to Track Memorization. [[paper]](https://arxiv.org/abs/2212.04461)
  - Mahsa Forouzesh, Hanie Sedghi, Patrick Thiran.
  - Key Word: Memorization; Noisy Label.
  - <details><summary>Digest</summary> Deep neural networks may easily memorize noisy labels present in real-world data, which degrades their ability to generalize. It is therefore important to track and evaluate the robustness of models against noisy label memorization. We propose a metric, called susceptibility, to gauge such memorization for neural networks. Susceptibility is simple and easy to compute during training. 

- Vicious Classifiers: Data Reconstruction Attack at Inference Time. [[paper]](https://arxiv.org/abs/2212.04223)
  - Mohammad Malekzadeh, Deniz Gunduz.
  - Key Word: Data Privacy; Data Reconstruction Attack.
  - <details><summary>Digest</summary> We show the possibility of jointly training a target model (to be run at users' side) and an attack model for data reconstruction (to be secretly used at server's side). We introduce the ''reconstruction risk'': a new measure for assessing the quality of reconstructed data that better captures the privacy risk of such attacks. Experimental results on 6 benchmark datasets show that for low-complexity data types, or for tasks with larger number of classes, a user's personal data can be approximately reconstructed from the outputs of a single target inference task. 

- Reconstructing Training Data from Model Gradient, Provably. [[paper]](https://arxiv.org/abs/2212.03714)
  - Zihan Wang, Jason Lee, Qi Lei.
  - Key Word: Federated Learning; Gradient Inversion; Tensor Decomposition.
  - <details><summary>Digest</summary> We present a surprising result: even without training or memorizing the data, we can fully reconstruct the training samples from a single gradient query at a randomly chosen parameter value. We prove the identifiability of the training data under mild conditions: with shallow or deep neural networks and a wide range of activation functions. 

- Leveraging Algorithmic Fairness to Mitigate Blackbox Attribute Inference Attacks. [[paper]](https://arxiv.org/abs/2211.10209)
  - Jan Aalmoes, Vasisht Duddu, Antoine Boutet.
  - Key Word: Fairness; Attribute Inference Attacks.
  - <details><summary>Digest</summary> We propose an practical and effective attribute inference attack that accounts for this imbalance using an adaptive threshold over the attack model's predictions. We exhaustively evaluate our proposed attack on multiple datasets and show that the adaptive threshold over the model's predictions drastically improves the attack accuracy over prior work. 

- Provable Membership Inference Privacy. [[paper]](https://arxiv.org/abs/2211.06582)
  - Zachary Izzo, Jinsung Yoon, Sercan O. Arik, James Zou.
  - Key Word: Membership Inference Attacks; Differential Privacy.
  - <details><summary>Digest</summary> We propose a novel privacy notion, membership inference privacy (MIP), to address these challenges. We give a precise characterization of the relationship between MIP and DP, and show that MIP can be achieved using less amount of randomness compared to the amount required for guaranteeing DP, leading to a smaller drop in utility. 

- MetaFed: Federated Learning among Federations with Cyclic Knowledge Distillation for Personalized Healthcare. [[paper]](https://arxiv.org/pdf/2206.08516.pdf)
  - Yiqiang Chen, Wang Lu, Xin Qin, Jindong Wang, and Xing Xie. *IJCAI'FL 2022*
  - Key Word: Federated learning; Personalized FL
  - <details><summary>Digest</summary> MetaFed is to facilitate trustworthy FL between different federations. MetaFed obtains a personalized model for each federation without a central server via the proposed Cyclic Knowledge Distillation. Specifically, MetaFed treats each federation as a meta distribution and aggregates knowledge of each federation in a cyclic manner. 

- Privacy Induces Robustness: Information-Computation Gaps and Sparse Mean Estimation. [[paper]](https://arxiv.org/abs/2211.00724)
  - Kristian Georgiev, Samuel B. Hopkins.
  - Key Word: Differential Privacy; Automatic Robustness.
  - <details><summary>Digest</summary> We establish a simple connection between robust and differentially-private algorithms: private mechanisms which perform well with very high probability are automatically robust in the sense that they retain accuracy even if a constant fraction of the samples they receive are adversarially corrupted. Since optimal mechanisms typically achieve these high success probabilities, our results imply that optimal private mechanisms for many basic statistics problems are robust.

- Machine Unlearning of Federated Clusters. [[paper]](https://arxiv.org/abs/2210.16424)
  - Chao Pan, Jin Sima, Saurav Prakash, Vishal Rana, Olgica Milenkovic.
  - Key Word: Machine Unlearning; Federated Learning.
  - <details><summary>Digest</summary> This work proposes the first known unlearning mechanism for federated clustering with privacy criteria that support simple, provable, and efficient data removal at the client and server level. The gist of our approach is to combine special initialization procedures with quantization methods that allow for secure aggregation of estimated local cluster counts at the server unit. As part of our platform, we introduce secure compressed multiset aggregation (SCMA), which is of independent interest for secure sparse model aggregation. 

- Analyzing Privacy Leakage in Machine Learning via Multiple Hypothesis Testing: A Lesson From Fano. [[paper]](https://arxiv.org/abs/2210.13662)
  - Chuan Guo, Alexandre Sablayrolles, Maziar Sanjabi.
  - Key Word: Differential Privacy; Reconstruction Attacks; Fano’s Inequality.
  - <details><summary>Digest</summary> We study data reconstruction attacks for discrete data and analyze it under the framework of multiple hypothesis testing. We utilize different variants of the celebrated Fano's inequality to derive upper bounds on the inferential power of a data reconstruction adversary when the model is trained differentially privately. Importantly, we show that if the underlying private data takes values from a set of size M, then the target privacy parameter ϵ can be O(logM) before the adversary gains significant inferential power.

- On the Robustness of Dataset Inference. [[paper]](https://arxiv.org/abs/2210.13631)
  - Sebastian Szyller, Rui Zhang, Jian Liu, N. Asokan.
  - Key Word: Dataset Inference; Data Ownership Verification.
  - <details><summary>Digest</summary> A fingerprinting technique introduced at ICLR '21, Dataset Inference (DI), has been shown to offer better robustness and efficiency than prior methods. The authors of DI provided a correctness proof for linear (suspect) models. However, in the same setting, we prove that DI suffers from high false positives (FPs) -- it can incorrectly identify an independent model trained with non-overlapping data from the same distribution as stolen. We further prove that DI also triggers FPs in realistic, non-linear suspect models. 

- A General Framework for Auditing Differentially Private Machine Learning. [[paper]](https://arxiv.org/abs/2210.08643)
  - Fred Lu, Joseph Munoz, Maya Fuchs, Tyler LeBlond, Elliott Zaresky-Williams, Edward Raff, Francis Ferraro, Brian Testa. *NeurIPS 2022*
  - Key Word: Auditing Differential Privacy.
  - <details><summary>Digest</summary> We present a framework to statistically audit the privacy guarantee conferred by a differentially private machine learner in practice. While previous works have taken steps toward evaluating privacy loss through poisoning attacks or membership inference, they have been tailored to specific models or have demonstrated low statistical power. Our work develops a general methodology to empirically evaluate the privacy of differentially private machine learning implementations, combining improved privacy search and verification methods with a toolkit of influence-based poisoning attacks. 

- Differentially Private Deep Learning with ModelMix. [[paper]](https://arxiv.org/abs/2210.03843)
  - Hanshen Xiao, Jun Wan, Srinivas Devadas. 
  - Key Word: Differential Privacy; Clipped Stochastic Gradient Descent.
  - <details><summary>Digest</summary> We provide rigorous analyses for both the utility guarantees and privacy amplification of ModelMix. In particular, we present a formal study on the effect of gradient clipping in DP-SGD, which provides theoretical instruction on how hyper-parameters should be selected. We also introduce a refined gradient clipping method, which can further sharpen the privacy loss in private learning when combined with ModelMix.

- Fine-Tuning with Differential Privacy Necessitates an Additional Hyperparameter Search. [[paper]](https://arxiv.org/abs/2210.02156)
  - Yannis Cattan, Christopher A. Choquette-Choo, Nicolas Papernot, Abhradeep Thakurta.
  - Key Word: Fine-Tuning; Differential Privacy.
  - <details><summary>Digest</summary> In this work, we identify an oversight of existing approaches for differentially private fine tuning. They do not tailor the fine-tuning approach to the specifics of learning with privacy. Our main result is to show how carefully selecting the layers being fine-tuned in the pretrained neural network allows us to establish new state-of-the-art tradeoffs between privacy and accuracy. 

- Data Leakage in Tabular Federated Learning. [[paper]](https://arxiv.org/abs/2210.01785)
  - Mark Vero, Mislav Balunović, Dimitar I. Dimitrov, Martin Vechev.
  - Key Word: Federated Learning; Data Leakage Attacks.
  - <details><summary>Digest</summary> Compared to the NLP and image domains, reconstruction of tabular data poses several unique challenges: (i) categorical features introduce a significantly more difficult mixed discrete-continuous optimization problem, (ii) the mix of categorical and continuous features causes high variance in the final reconstructions, and (iii) structured data makes it difficult for the adversary to judge reconstruction quality. In this work, we tackle these challenges and propose the first comprehensive reconstruction attack on tabular data, called TabLeak. TabLeak is based on three key ingredients: (i) a softmax structural prior, implicitly converting the mixed discrete-continuous optimization problem into an easier fully continuous one, (ii) a way to reduce the variance of our reconstructions through a pooled ensembling scheme exploiting the structure of tabular data, and (iii) an entropy measure which can successfully assess reconstruction quality. 

- Certified Data Removal in Sum-Product Networks. [[paper]](https://arxiv.org/abs/2210.01451)
  - Alexander Becker, Thomas Liebig. *ICKG 2022*
  - Key Word: Machine Unlearning; Sum-Product Networks.
  - <details><summary>Digest</summary> Data protection regulations like the GDPR or the California Consumer Privacy Act give users more control over the data that is collected about them. Deleting the collected data is often insufficient to guarantee data privacy since it is often used to train machine learning models, which can expose information about the training data. Thus, a guarantee that a trained model does not expose information about its training data is additionally needed. In this paper, we present UnlearnSPN -- an algorithm that removes the influence of single data points from a trained sum-product network and thereby allows fulfilling data privacy requirements on demand.

- Membership Inference Attacks Against Text-to-image Generation Models. [[paper]](https://arxiv.org/abs/2210.00968)
  - Yixin Wu, Ning Yu, Zheng Li, Michael Backes, Yang Zhang.
  - Key Word: Membership Inference Attacks; Text-to-Image Generation.
  - <details><summary>Digest</summary> We perform the first privacy analysis of text-to-image generation models through the lens of membership inference. Specifically, we propose three key intuitions about membership information and design four attack methodologies accordingly. We conduct comprehensive evaluations on two mainstream text-to-image generation models including sequence-to-sequence modeling and diffusion-based modeling. 

- UnGANable: Defending Against GAN-based Face Manipulation. [[paper]](https://arxiv.org/abs/2210.00957)
  - Zheng Li, Ning Yu, Ahmed Salem, Michael Backes, Mario Fritz, Yang Zhang. *USENIX Security 2023*
  - Key Word: GAN Inversion Attacks; Latent Code Manipulation.
  - <details><summary>Digest</summary> We propose the first defense system, namely UnGANable, against GAN-inversion-based face manipulation. In specific, UnGANable focuses on defending GAN inversion, an essential step for face manipulation. Its core technique is to search for alternative images (called cloaked images) around the original images (called target images) in image space. 

- Untargeted Backdoor Watermark: Towards Harmless and Stealthy Dataset Copyright Protection. [[paper]](https://arxiv.org/abs/2210.00875) [[code]](https://github.com/thuyimingli/untargeted_backdoor_watermark)
  - Yiming Li, Yang Bai, Yong Jiang, Yong Yang, Shu-Tao Xia, Bo Li. *NeurIPS 2022*
  - Key Word: Data Watermarking; Defenses against Data Leakage.
  - <details><summary>Digest</summary> We explore the untargeted backdoor watermarking scheme, where the abnormal model behaviors are not deterministic. Specifically, we introduce two dispersibilities and prove their correlation, based on which we design the untargeted backdoor watermark under both poisoned-label and clean-label settings. We also discuss how to use the proposed untargeted backdoor watermark for dataset ownership verification. 

- No Free Lunch in "Privacy for Free: How does Dataset Condensation Help Privacy". [[paper]](https://arxiv.org/abs/2209.14987)
  - Nicholas Carlini, Vitaly Feldman, Milad Nasr.
  - Key Word: Differential Privacy; Dataset Condensation.
  - <details><summary>Digest</summary> New methods designed to preserve data privacy require careful scrutiny. Failure to preserve privacy is hard to detect, and yet can lead to catastrophic results when a system implementing a ``privacy-preserving'' method is attacked. A recent work selected for an Outstanding Paper Award at ICML 2022 (Dong et al., 2022) claims that dataset condensation (DC) significantly improves data privacy when training machine learning models. This claim is supported by theoretical analysis of a specific dataset condensation technique and an empirical evaluation of resistance to some existing membership inference attacks.

- Algorithms that Approximate Data Removal: New Results and Limitations. [[paper]](https://arxiv.org/abs/2209.12269)
  - Vinith M. Suriyakumar, Ashia C. Wilson. *NeurIPS 2022*
  - Key Word: Data Removal; Machine Unlearning.
  - <details><summary>Digest</summary> We study the problem of deleting user data from machine learning models trained using empirical risk minimization. Our focus is on learning algorithms which return the empirical risk minimizer and approximate unlearning algorithms that comply with deletion requests that come streaming minibatches. Leveraging the infintesimal jacknife, we develop an online unlearning algorithm that is both computationally and memory efficient. Unlike prior memory efficient unlearning algorithms, we target models that minimize objectives with non-smooth regularizers, such as the commonly used ℓ1, elastic net, or nuclear norm penalties.

- Deep Learning-based Anonymization of Chest Radiographs: A Utility-preserving Measure for Patient Privacy. [[paper]](https://arxiv.org/abs/2209.11531)
  - Kai Packhäuser, Sebastian Gündel, Florian Thamm, Felix Denzinger, Andreas Maier.
  - Key Word: Image Anonymization; Patient Privacy; Data Utility; Chest Radiographs.
  - <details><summary>Digest</summary> We propose the first deep learning-based approach to targetedly anonymize chest radiographs while maintaining data utility for diagnostic and machine learning purposes. Our model architecture is a composition of three independent neural networks that, when collectively used, allow for learning a deformation field that is able to impede patient re-identification. The individual influence of each component is investigated with an ablation study. Quantitative results on the ChestX-ray14 dataset show a reduction of patient re-identification from 81.8% to 58.6% in the area under the receiver operating characteristic curve (AUC) with little impact on the abnormality classification performance. 

- In Differential Privacy, There is Truth: On Vote Leakage in Ensemble Private Learning. [[paper]](https://arxiv.org/abs/2209.10732)
  - Jiaqi Wang, Roei Schuster, Ilia Shumailov, David Lie, Nicolas Papernot. *NeurIPS 2022*
  - Key Word: Differential Privacy.
  - <details><summary>Digest</summary> We observe that this use of noise, which makes PATE predictions stochastic, enables new forms of leakage of sensitive information. For a given input, our adversary exploits this stochasticity to extract high-fidelity histograms of the votes submitted by the underlying teachers. From these histograms, the adversary can learn sensitive attributes of the input such as race, gender, or age. 

- Dataset Inference for Self-Supervised Models. [[paper]](https://arxiv.org/abs/2209.09024)
  - Adam Dziedzic, Haonan Duan, Muhammad Ahmad Kaleem, Nikita Dhawan, Jonas Guan, Yannis Cattan, Franziska Boenisch, Nicolas Papernot. *NeurIPS 2022*
  - Key Word: Dataset Inferece; Model Stealing Attacks.
  - <details><summary>Digest</summary> We introduce a new dataset inference defense, which uses the private training set of the victim encoder model to attribute its ownership in the event of stealing. The intuition is that the log-likelihood of an encoder's output representations is higher on the victim's training data than on test data if it is stolen from the victim, but not if it is independently trained. We compute this log-likelihood using density estimation models. As part of our evaluation, we also propose measuring the fidelity of stolen encoders and quantifying the effectiveness of the theft detection without involving downstream tasks; instead, we leverage mutual information and distance measurements. 

- Membership Inference Attacks and Generalization: A Causal Perspective. [[paper]](https://arxiv.org/abs/2209.08615)
  - Teodora Baluta, Shiqi Shen, S. Hitarth, Shruti Tople, Prateek Saxena.
  - Key Word: Membership Inference Attacks; Causal Reasoning.
  - <details><summary>Digest</summary> We propose the first approach to explain MI attacks and their connection to generalization based on principled causal reasoning. We offer causal graphs that quantitatively explain the observed MI attack performance achieved for 6 attack variants. We refute several prior non-quantitative hypotheses that over-simplify or over-estimate the influence of underlying causes, thereby failing to capture the complex interplay between several factors. 

- Renyi Differential Privacy of Propose-Test-Release and Applications to Private and Robust Machine Learning. [[paper]](https://arxiv.org/abs/2209.07716)
  - Jiachen T. Wang, Saeed Mahloujifar, Shouda Wang, Ruoxi Jia, Prateek Mittal. *NeurIPS 2022*
  - Key Word: Renyi Differential Privacy; Propose-Test-Release.
  - <details><summary>Digest</summary> Propose-Test-Release (PTR) is a differential privacy framework that works with local sensitivity of functions, instead of their global sensitivity. This framework is typically used for releasing robust statistics such as median or trimmed mean in a differentially private manner. While PTR is a common framework introduced over a decade ago, using it in applications such as robust SGD where we need many adaptive robust queries is challenging. This is mainly due to the lack of Renyi Differential Privacy (RDP) analysis, an essential ingredient underlying the moments accountant approach for differentially private deep learning. In this work, we generalize the standard PTR and derive the first RDP bound for it when the target function has bounded global sensitivity.

- CLIPping Privacy: Identity Inference Attacks on Multi-Modal Machine Learning Models. [[paper]](https://arxiv.org/abs/2209.07341)
  - Dominik Hintersdorf, Lukas Struppek, Kristian Kersting.
  - Key Word: Multi-Modal Machine Learning; Identity Inference Attacks.
  - <details><summary>Digest</summary> Image-text models like CLIP have not yet been looked at in the context of privacy attacks. While membership inference attacks aim to tell whether a specific data point was used for training, we introduce a new type of privacy attack, named identity inference attack (IDIA), designed for multi-modal image-text models like CLIP. Using IDIAs, an attacker can reveal whether a particular person, was part of the training data by querying the model in a black-box fashion with different images of the same person. 

- M^4I: Multi-modal Models Membership Inference. [[paper]](https://arxiv.org/abs/2209.06997) [[code]](https://github.com/MultimodalMI/Multimodal-membership-inference)
  - Pingyi Hu, Zihan Wang, Ruoxi Sun, Hu Wang, Minhui Xue. *NeurIPS 2022*
  - Key Word: Multi-modal Machine Learning; Membership inference.
  - <details><summary>Digest</summary> we propose Multi-modal Models Membership Inference (M^4I) with two attack methods to infer the membership status, named metric-based (MB) M^4I and feature-based (FB) M^4I, respectively. More specifically, MB M^4I adopts similarity metrics while attacking to infer target data membership. FB M^4I uses a pre-trained shadow multi-modal feature extractor to achieve the purpose of data inference attack by comparing the similarities from extracted input and output features. 

- Black-box Ownership Verification for Dataset Protection via Backdoor Watermarking. [[paper]](https://arxiv.org/abs/2209.06015) [[code]](https://github.com/THUYimingLi/DVBW)
  - Yiming Li, Mingyan Zhu, Xue Yang, Yong Jiang, Shu-Tao Xia.
  - Key Word: Black-box Ownership Verification; Backdoor Watermarking.
  - <details><summary>Digest</summary> We propose to embed external patterns via backdoor watermarking for the ownership verification to protect them. Our method contains two main parts, including dataset watermarking and dataset verification. Specifically, we exploit poison-only backdoor attacks (e.g., BadNets) for dataset watermarking and design a hypothesis-test-guided method for dataset verification.  

- A Framework for Evaluating Privacy-Utility Trade-off in Vertical Federated Learning. [[paper]](https://arxiv.org/abs/2209.03885)
  - Yan Kang, Jiahuan Luo, Yuanqin He, Xiaojin Zhang, Lixin Fan, Qiang Yang.
  - Key Word: Vertical Federated Learning; Privacy Attacks.
  - <details><summary>Digest</summary> We propose an evaluation framework that formulates the privacy-utility evaluation problem. We then use this framework as a guide to comprehensively evaluate a broad range of protection mechanisms against most of the state-of-the-art privacy attacks for three widely-deployed VFL algorithms. These evaluations may help FL practitioners select appropriate protection mechanisms given specific requirements. 

- A Survey of Machine Unlearning. [[paper]](https://arxiv.org/abs/2209.02299) [[code]](https://github.com/tamlhp/awesome-machine-unlearning)
  - Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen.
  - Key Word: Machine Unlearning; Survey.
  - <details><summary>Digest</summary> In this survey paper, we seek to provide a thorough investigation of machine unlearning in its definitions, scenarios, mechanisms, and applications. Specifically, as a categorical collection of state-of-the-art research, we hope to provide a broad reference for those seeking a primer on machine unlearning and its various formulations, design requirements, removal requests, algorithms, and uses in a variety of ML applications. Furthermore, we hope to outline key findings and trends in the paradigm as well as highlight new areas of research that have yet to see the application of machine unlearning, but could nonetheless benefit immensely. 

- Are Attribute Inference Attacks Just Imputation? [[paper]](https://arxiv.org/abs/2209.01292) [[code]](https://github.com/bargavj/EvaluatingDPML)
  - Bargav Jayaraman, David Evans. *CCS 2022*
  - Kew Word: Attribute Inference Attacks; Data Imputation.
  - <details><summary>Digest</summary> Our main conclusions are: (1) previous attribute inference methods do not reveal more about the training data from the model than can be inferred by an adversary without access to the trained model, but with the same knowledge of the underlying distribution as needed to train the attribute inference attack; (2) black-box attribute inference attacks rarely learn anything that cannot be learned without the model; but (3) white-box attacks, which we introduce and evaluate in the paper, can reliably identify some records with the sensitive value attribute that would not be predicted without having access to the model. 

- On the Privacy Risks of Cell-Based NAS Architectures. [[paper]](https://arxiv.org/abs/2209.01688) [[code]](https://github.com/miraclehh/nas_privacy)
  - Hai Huang, Zhikun Zhang, Yun Shen, Michael Backes, Qi Li, Yang Zhang. *CCS 2022*
  - Key Word: Cell-based Neural Architecture Search; Membership Inference Attack.
  - <details><summary>Digest</summary> We fill this gap and systematically measure the privacy risks of NAS architectures. Leveraging the insights from our measurement study, we further explore the cell patterns of cell-based NAS architectures and evaluate how the cell patterns affect the privacy risks of NAS-searched architectures. Through extensive experiments, we shed light on how to design robust NAS architectures against privacy attacks, and also offer a general methodology to understand the hidden correlation between the NAS-searched architectures and other privacy risks.

- Group Property Inference Attacks Against Graph Neural Networks. [[paper]](https://arxiv.org/abs/2209.01100)
  - Xiuling Wang, Wendy Hui Wang. *CCS 2022*
  - Key Word: Property inference attack; Graph neural networks.
  - <details><summary>Digest</summary> We perform the first systematic study of group property inference attacks (GPIA) against GNNs. First, we consider a taxonomy of threat models under both black-box and white-box settings with various types of adversary knowledge, and design six different attacks for these settings. We evaluate the effectiveness of these attacks through extensive experiments on three representative GNN models and three real-world graphs. Our results demonstrate the effectiveness of these attacks whose accuracy outperforms the baseline approaches. 

- An Introduction to Machine Unlearning. [[paper]](https://arxiv.org/abs/2209.00939)
  - Salvatore Mercuri, Raad Khraishi, Ramin Okhrati, Devesh Batra, Conor Hamill, Taha Ghasempour, Andrew Nowlan.
  - Key Word: Machine Unlearning; Exact Unlearning; Approximate Unlearning; Data Removal; Data Privacy.
  - <details><summary>Digest</summary> Removing the influence of a specified subset of training data from a machine learning model may be required to address issues such as privacy, fairness, and data quality. Retraining the model from scratch on the remaining data after removal of the subset is an effective but often infeasible option, due to its computational expense. The past few years have therefore seen several novel approaches towards efficient removal, forming the field of "machine unlearning", however, many aspects of the literature published thus far are disparate and lack consensus. In this paper, we summarise and compare seven state-of-the-art machine unlearning algorithms, consolidate definitions of core concepts used in the field, reconcile different approaches for evaluating algorithms, and discuss issues related to applying machine unlearning in practice.

- Trading Off Privacy, Utility and Efficiency in Federated Learning. [[paper]](https://arxiv.org/abs/2209.00230)
  - Xiaojin Zhang, Yan Kang, Kai Chen, Lixin Fan, Qiang Yang. 
  - Key Word: Federated Learning; Privacy.
  - <details><summary>Digest</summary> We propose a unified federated learning framework that reconciles horizontal and vertical federated learning. Based on this framework, we formulate and quantify the trade-offs between privacy leakage, utility loss, and efficiency reduction, which leads us to the No-Free-Lunch (NFL) theorem for the federated learning system. 

- Membership Inference Attacks by Exploiting Loss Trajectory. [[paper]](https://arxiv.org/abs/2208.14933)
  - Yiyong Liu, Zhengyu Zhao, Michael Backes, Yang Zhang. *CCS 2022*
  - Key Word: Membership Inference Attacks.
  - <details><summary>Digest</summary> Existing attack methods have commonly exploited the output information (mostly, losses) solely from the given target model. As a result, in practical scenarios where both the member and non-member samples yield similarly small losses, these methods are naturally unable to differentiate between them. To address this limitation, in this paper, we propose a new attack method, called \system, which can exploit the membership information from the whole training process of the target model for improving the attack performance. 

- SNAP: Efficient Extraction of Private Properties with Poisoning. [[paper]](https://arxiv.org/abs/2208.12348)
  - Harsh Chaudhari, John Abascal, Alina Oprea, Matthew Jagielski, Florian Tramèr, Jonathan Ullman.
  - Key Word: Property Inference Attacks; Poisoning Attacks.
  - <details><summary>Digest</summary> We consider the setting of property inference attacks in which the attacker can poison a subset of the training dataset and query the trained target model. Motivated by our theoretical analysis of model confidences under poisoning, we design an efficient property inference attack, SNAP, which obtains higher attack success and requires lower amounts of poisoning than the state-of-the-art poisoning-based property inference attack by Mahloujifar et al.

- Auditing Membership Leakages of Multi-Exit Networks. [[paper]](https://arxiv.org/abs/2208.11180)
  - Zheng Li, Yiyong Liu, Xinlei He, Ning Yu, Michael Backes, Yang Zhang. *CCS 2022*
  - Key Word: Membership Inference Attacks; Multi-Existing.
  - <details><summary>Digest</summary> In this paper, we perform the first privacy analysis of multi-exit networks through the lens of membership leakages. In particular, we first leverage the existing attack methodologies to quantify the multi-exit networks' vulnerability to membership leakages. Our experimental results show that multi-exit networks are less vulnerable to membership leakages and the exit (number and depth) attached to the backbone model is highly correlated with the attack performance.

- On the Design of Privacy-Aware Cameras: a Study on Deep Neural Networks. [[paper]](https://arxiv.org/abs/2208.11372) [[code]](https://github.com/upciti/privacy-by-design-semseg)
  - Marcela Carvalho, Oussama Ennaffi, Sylvain Chateau, Samy Ait Bachir.
  - Key Word:  Smart City; Privacy-Aware Camera.
  - <details><summary>Digest</summary> In this paper, the effect of camera distortions is studied using Deep Learning techniques commonly used to extract sensitive data. To do so, we simulate out-of-focus images corresponding to a realistic conventional camera with fixed focal length, aperture, and focus, as well as grayscale images coming from a monochrome camera. We then prove, through an experimental study, that we can build a privacy-aware camera that cannot extract personal information such as license plate numbers.  

- Joint Privacy Enhancement and Quantization in Federated Learning. [[paper]](https://arxiv.org/abs/2208.10888) [[code]](https://github.com/langnatalie/jopeq)
  - Natalie Lang, Elad Sofer, Tomer Shaked, Nir Shlezinger.
  - Key Word: Federated Learning; Quantization; Privacy.
  - <details><summary>Digest</summary> We propose a method coined joint privacy enhancement and quantization (JoPEQ), which jointly implements lossy compression and privacy enhancement in FL settings. In particular, JoPEQ utilizes vector quantization based on random lattice, a universal compression technique whose byproduct distortion is statistically equivalent to additive noise. This distortion is leveraged to enhance privacy by augmenting the model updates with dedicated multivariate privacy preserving noise.

- Split-U-Net: Preventing Data Leakage in Split Learning for Collaborative Multi-Modal Brain Tumor Segmentation. [[paper]](https://arxiv.org/abs/2208.10553)
  - Holger R. Roth, Ali Hatamizadeh, Ziyue Xu, Can Zhao, Wenqi Li, Andriy Myronenko, Daguang Xu.
  - Key Word: Split Learning; Vertical Federated Learning; Multi-Modal Brain Tumor Segmentation; Data Inversion.
  - <details><summary>Digest</summary> We propose "Split-U-Net" and successfully apply SL for collaborative biomedical image segmentation. Nonetheless, SL requires the exchanging of intermediate activation maps and gradients to allow training models across different feature spaces, which might leak data and raise privacy concerns. Therefore, we also quantify the amount of data leakage in common SL scenarios for biomedical image segmentation and provide ways to counteract such leakage by applying appropriate defense strategies.

- Membership-Doctor: Comprehensive Assessment of Membership Inference Against Machine Learning Models. [[paper]](https://arxiv.org/abs/2208.10445)
  - Xinlei He, Zheng Li, Weilin Xu, Cory Cornelius, Yang Zhang.
  - Key Word: Membership Inference Attacks and Defenses; Benchmark.
  - <details><summary>Digest</summary> We fill this gap by presenting a large-scale measurement of different membership inference attacks and defenses. We systematize membership inference through the study of nine attacks and six defenses and measure the performance of different attacks and defenses in the holistic evaluation. We then quantify the impact of the threat model on the results of these attacks. We find that some assumptions of the threat model, such as same-architecture and same-distribution between shadow and target models, are unnecessary. We are also the first to execute attacks on the real-world data collected from the Internet, instead of laboratory datasets.

- SoK: Machine Learning with Confidential Computing. [[paper]](https://arxiv.org/abs/2208.10134)
  - Fan Mo, Zahra Tarkhani, Hamed Haddadi.
  - Key Word: Survey; Confidential Computing; Trusted Execution Environment; Intergrity.
  - <details><summary>Digest</summary> We systematize the findings on confidential computing-assisted ML security and privacy techniques for providing i) confidentiality guarantees and ii) integrity assurances. We further identify key challenges and provide dedicated analyses of the limitations in existing Trusted Execution Environment (TEE) systems for ML use cases. We discuss prospective works, including grounded privacy definitions, partitioned ML executions, dedicated TEE designs for ML, TEE-aware ML, and ML full pipeline guarantee. These potential solutions can help achieve a much strong TEE-enabled ML for privacy guarantees without introducing computation and system costs.

- Inferring Sensitive Attributes from Model Explanations. [[paper]](https://arxiv.org/abs/2208.09967) [[code]](https://github.com/vasishtduddu/attinfexplanations)
  - Vasisht Duddu, Antoine Boutet. *CIKM 2022*
  - Key Word: We focus on the specific privacy risk of attribute inference attack wherein an adversary infers sensitive attributes of an input (e.g., race and sex) given its model explanations. We design the first attribute inference attack against model explanations in two threat models where model builder either (a) includes the sensitive attributes in training data and input or (b) censors the sensitive attributes by not including them in the training data and input.

- On the Privacy Effect of Data Enhancement via the Lens of Memorization. [[paper]](https://arxiv.org/abs/2208.08270)
  - Xiao Li, Qiongxiu Li, Zhanhao Hu, Xiaolin Hu.
  - Key Word: Membership Inference Attacks; Data Augmentation; Adversarial Training.
  - <details><summary>Digest</summary> We propose to investigate privacy from a new perspective called memorization. Through the lens of memorization, we find that previously deployed MIAs produce misleading results as they are less likely to identify samples with higher privacy risks as members compared to samples with low privacy risks. To solve this problem, we deploy a recent attack that can capture the memorization degrees of individual samples for evaluation. Through extensive experiments, we unveil non-trivial findings about the connections between three important properties of machine learning models, including privacy, generalization gap, and adversarial robustness.

- Private Domain Adaptation from a Public Source. [[paper]](https://arxiv.org/abs/2208.06135)
  - Raef Bassily, Mehryar Mohri, Ananda Theertha Suresh.
  - Key Word: Domain Adaptation; Differential Privacy; Frank-Wolfe Algorithm; Mirror Descent Algorithm.
  - <details><summary>Digest</summary> In regression problems with no privacy constraints on the source or target data, a discrepancy minimization algorithm based on several theoretical guarantees was shown to outperform a number of other adaptation algorithm baselines. Building on that approach, we design differentially private discrepancy-based algorithms for adaptation from a source domain with public labeled data to a target domain with unlabeled private data. The design and analysis of our private algorithms critically hinge upon several key properties we prove for a smooth approximation of the weighted discrepancy, such as its smoothness with respect to the ℓ1-norm and the sensitivity of its gradient.

- Dropout is NOT All You Need to Prevent Gradient Leakage. [[paper]](https://arxiv.org/abs/2208.06163)
  - Daniel Scheliga, Patrick Mäder, Marco Seeland.
  - Key Word: Dropout; Gradient Inversion Attacks.
  - <details><summary>Digest</summary> Recent observations suggest that dropout could mitigate gradient leakage and improve model utility if added to neural networks. Unfortunately, this phenomenon has not been systematically researched yet. In this work, we thoroughly analyze the effect of dropout on iterative gradient inversion attacks. We find that state of the art attacks are not able to reconstruct the client data due to the stochasticity induced by dropout during model training. Nonetheless, we argue that dropout does not offer reliable protection if the dropout induced stochasticity is adequately modeled during attack optimization. Consequently, we propose a novel Dropout Inversion Attack (DIA) that jointly optimizes for client data and dropout masks to approximate the stochastic client model.

- On the Fundamental Limits of Formally (Dis)Proving Robustness in Proof-of-Learning. [[paper]](https://arxiv.org/abs/2208.03567)
  - Congyu Fang, Hengrui Jia, Anvith Thudi, Mohammad Yaghini, Christopher A. Choquette-Choo, Natalie Dullerud, Varun Chandrasekaran, Nicolas Papernot.
  - Key Word: Proof-of-Learning; Adversarial Examples.
  - <details><summary>Digest</summary> Proof-of-learning (PoL) proposes a model owner use machine learning training checkpoints to establish a proof of having expended the necessary compute for training. The authors of PoL forego cryptographic approaches and trade rigorous security guarantees for scalability to deep learning by being applicable to stochastic gradient descent and adaptive variants. This lack of formal analysis leaves the possibility that an attacker may be able to spoof a proof for a model they did not train. We contribute a formal analysis of why the PoL protocol cannot be formally (dis)proven to be robust against spoofing adversaries. To do so, we disentangle the two roles of proof verification in PoL: (a) efficiently determining if a proof is a valid gradient descent trajectory, and (b) establishing precedence by making it more expensive to craft a proof after training completes (i.e., spoofing).

- Semi-Leak: Membership Inference Attacks Against Semi-supervised Learning. [[paper]](https://arxiv.org/abs/2207.12535) [[code]](https://github.com/xinleihe/Semi-Leak)
  - Xinlei He, Hongbin Liu, Neil Zhenqiang Gong, Yang Zhang. *ECCV 2022*
  - Key Word: Membership Inference Attacks; Semi-Supervised Learning.
  - <details><summary>Digest</summary> We take a different angle by studying the training data privacy of SSL. Specifically, we propose the first data augmentation-based membership inference attacks against ML models trained by SSL. Given a data sample and the black-box access to a model, the goal of membership inference attack is to determine whether the data sample belongs to the training dataset of the model. Our evaluation shows that the proposed attack can consistently outperform existing membership inference attacks and achieves the best performance against the model trained by SSL.

- Learnable Privacy-Preserving Anonymization for Pedestrian Images. [[paper]](https://arxiv.org/abs/2207.11677) [[code]](https://github.com/whuzjw/privacy-reid)
  - Junwu Zhang, Mang Ye, Yao Yang. *MM 2022*
  - Key Word: Privacy Protection; Person Re-Identification.
  - <details><summary>Digest</summary> This paper studies a novel privacy-preserving anonymization problem for pedestrian images, which preserves personal identity information (PII) for authorized models and prevents PII from being recognized by third parties. Conventional anonymization methods unavoidably cause semantic information loss, leading to limited data utility. Besides, existing learned anonymization techniques, while retaining various identity-irrelevant utilities, will change the pedestrian identity, and thus are unsuitable for training robust re-identification models. To explore the privacy-utility trade-off for pedestrian images, we propose a joint learning reversible anonymization framework, which can reversibly generate full-body anonymous images with little performance drop on person re-identification tasks.

- Certified Neural Network Watermarks with Randomized Smoothing. [[paper]](https://arxiv.org/abs/2207.07972) [[code]](https://github.com/arpitbansal297/Certified_Watermarks)
  - Arpit Bansal, Ping-yeh Chiang, Michael Curry, Rajiv Jain, Curtis Wigington, Varun Manjunatha, John P Dickerson, Tom Goldstein. *ICML 2022*
  - Key Word: Watermarking Neural Networks; Certified Defenses; Randomized Smoothing.
  - <details><summary>Digest</summary> Watermarking is a commonly used strategy to protect creators' rights to digital images, videos and audio. Recently, watermarking methods have been extended to deep learning models -- in principle, the watermark should be preserved when an adversary tries to copy the model. However, in practice, watermarks can often be removed by an intelligent adversary. Several papers have proposed watermarking methods that claim to be empirically resistant to different types of removal attacks, but these new techniques often fail in the face of new or better-tuned adversaries. In this paper, we propose a certifiable watermarking method. Using the randomized smoothing technique proposed in Chiang et al., we show that our watermark is guaranteed to be unremovable unless the model parameters are changed by more than a certain l2 threshold.

- RelaxLoss: Defending Membership Inference Attacks without Losing Utility. [[paper]](https://arxiv.org/abs/2207.05801) [[code]](https://github.com/DingfanChen/RelaxLoss)
  - Dingfan Chen, Ning Yu, Mario Fritz. *ICLR 2022*
  - Key Word: Membership Inference Attacks and Defenses.
  - <details><summary>Digest</summary> We propose a novel training framework based on a relaxed loss with a more achievable learning target, which leads to narrowed generalization gap and reduced privacy leakage. RelaxLoss is applicable to any classification model with added benefits of easy implementation and negligible overhead.

- High-Dimensional Private Empirical Risk Minimization by Greedy Coordinate Descent. [[paper]](https://arxiv.org/abs/2207.01560)  
  - Paul Mangold, Aurélien Bellet, Joseph Salmon, Marc Tommasi.
  - Key Word: Differentially Private Empirical Risk Minimization.
  - <details><summary>Digest</summary> In this paper, we study differentially private empirical risk minimization (DP-ERM). It has been shown that the (worst-case) utility of DP-ERM reduces as the dimension increases. This is a major obstacle to privately learning large machine learning models. In high dimension, it is common for some model's parameters to carry more information than others. To exploit this, we propose a differentially private greedy coordinate descent (DP-GCD) algorithm. At each iteration, DP-GCD privately performs a coordinate-wise gradient step along the gradients' (approximately) greatest entry.

- When Does Differentially Private Learning Not Suffer in High Dimensions? [[paper]](https://arxiv.org/abs/2207.00160)
  - Xuechen Li, Daogao Liu, Tatsunori Hashimoto, Huseyin A. Inan, Janardhan Kulkarni, Yin Tat Lee, Abhradeep Guha Thakurta.
  - Key Word: Differentially Private Learning; Large Language Models.
  - <details><summary>Digest</summary> Large pretrained models can be privately fine-tuned to achieve performance approaching that of non-private models. A common theme in these results is the surprising observation that high-dimensional models can achieve favorable privacy-utility trade-offs. This seemingly contradicts known results on the model-size dependence of differentially private convex learning and raises the following research question: When does the performance of differentially private learning not degrade with increasing model size? We identify that the magnitudes of gradients projected onto subspaces is a key factor that determines performance. To precisely characterize this for private convex learning, we introduce a condition on the objective that we term restricted Lipschitz continuity and derive improved bounds for the excess empirical and population risks that are dimension-independent under additional conditions.

- Measuring Forgetting of Memorized Training Examples. [[paper]](https://arxiv.org/abs/2207.00099)
  - Matthew Jagielski, Om Thakkar, Florian Tramèr, Daphne Ippolito, Katherine Lee, Nicholas Carlini, Eric Wallace, Shuang Song, Abhradeep Thakurta, Nicolas Papernot, Chiyuan Zhang.
  - Key Word: Membership Inference Attacks; Reconstruction Attacks.
  - <details><summary>Digest</summary> We connect these phenomena. We propose a technique to measure to what extent models ``forget'' the specifics of training examples, becoming less susceptible to privacy attacks on examples they have not seen recently. We show that, while non-convexity can prevent forgetting from happening in the worst-case, standard image and speech models empirically do forget examples over time. We identify nondeterminism as a potential explanation, showing that deterministically trained models do not forget. Our results suggest that examples seen early when training with extremely large datasets -- for instance those examples used to pre-train a model -- may observe privacy benefits at the expense of examples seen later.

- Why patient data cannot be easily forgotten? [[paper]](https://arxiv.org/abs/2206.14541)
  - Ruolin Su, Xiao Liu, Sotirios A. Tsaftaris. *MICCAI 2022*
  - Key Word: Privacy; Patient-wise Forgetting; Scrubbing.
  - <details><summary>Digest</summary> We study the influence of patient data on model performance and formulate two hypotheses for a patient's data: either they are common and similar to other patients or form edge cases, i.e. unique and rare cases. We show that it is not possible to easily forget patient data. We propose a targeted forgetting approach to perform patient-wise forgetting. Extensive experiments on the benchmark Automated Cardiac Diagnosis Challenge dataset showcase the improved performance of the proposed targeted forgetting approach as opposed to a state-of-the-art method.

- Approximate Data Deletion in Generative Models. [[paper]](https://arxiv.org/abs/2206.14439)
  - Zhifeng Kong, Scott Alfeld.
  - Key Word: Density Ratio Based Framework; Machine Unlearning; Generative Model.
  - <details><summary>Digest</summary> Many approximate data deletion methods have been developed for supervised learning. Unsupervised learning, in contrast, remains largely an open problem when it comes to (approximate or exact) efficient data deletion. In this paper, we propose a density-ratio-based framework for generative models. Using this framework, we introduce a fast method for approximate data deletion and a statistical test for estimating whether or not training points have been deleted. 

- Data Leakage in Federated Averaging. [[paper]](https://arxiv.org/abs/2206.12395)
  - Dimitar I. Dimitrov, Mislav Balunović, Nikola Konstantinov, Martin Vechev.
  - Key Word: Federated Learning; Gradient Inversion Attacks and Defenses.
  - <details><summary>Digest</summary> Recent attacks have shown that user data can be reconstructed from FedSGD updates, thus breaking privacy. However, these attacks are of limited practical relevance as federated learning typically uses the FedAvg algorithm. It is generally accepted that reconstructing data from FedAvg updates is much harder than FedSGD as: (i) there are unobserved intermediate weight updates, (ii) the order of inputs matters, and (iii) the order of labels changes every epoch. In this work, we propose a new optimization-based attack which successfully attacks FedAvg by addressing the above challenges. First, we solve the optimization problem using automatic differentiation that forces a simulation of the client's update for the reconstructed labels and inputs so as to match the received client update. Second, we address the unknown input order by treating images at different epochs as independent during optimization, while relating them with a permutation invariant prior. Third, we reconstruct the labels by estimating the parameters of existing FedSGD attacks at every FedAvg step.

- A Framework for Understanding Model Extraction Attack and Defense. [[paper]](https://arxiv.org/abs/2206.11480)
  - Xun Xian, Mingyi Hong, Jie Ding.
  - Key Word: Model Extraction Attack and Defense.
  - <details><summary>Digest</summary> To study the fundamental tradeoffs between model utility from a benign user's view and privacy from an adversary's view, we develop new metrics to quantify such tradeoffs, analyze their theoretical properties, and develop an optimization problem to understand the optimal adversarial attack and defense strategies. The developed concepts and theory match the empirical findings on the `equilibrium' between privacy and utility. In terms of optimization, the key ingredient that enables our results is a unified representation of the attack-defense problem as a min-max bi-level problem.

- The Privacy Onion Effect: Memorization is Relative. [[paper]](https://arxiv.org/abs/2206.10469)
  - Nicholas Carlini, Matthew Jagielski, Nicolas Papernot, Andreas Terzis, Florian Tramer, Chiyuan Zhang.
  - Key Word: Memorization; Differential Privacy; Membership Inference Attacks and Defenses; Machine Unlearning.
  - <details><summary>Digest</summary> Machine learning models trained on private datasets have been shown to leak their private data. While recent work has found that the average data point is rarely leaked, the outlier samples are frequently subject to memorization and, consequently, privacy leakage. We demonstrate and analyse an Onion Effect of memorization: removing the "layer" of outlier points that are most vulnerable to a privacy attack exposes a new layer of previously-safe points to the same attack. We perform several experiments to study this effect, and understand why it occurs. The existence of this effect has various consequences. For example, it suggests that proposals to defend against memorization without training with rigorous privacy guarantees are unlikely to be effective. Further, it suggests that privacy-enhancing technologies such as machine unlearning could actually harm the privacy of other users.

- Certified Graph Unlearning. [[paper]](https://arxiv.org/abs/2206.09140)
  - Eli Chien, Chao Pan, Olgica Milenkovic.
  - Key Word: Machine Unlearning; Certified Data Removal; Graph Neural Networks.
  - <details><summary>Digest</summary> Graph-structured data is ubiquitous in practice and often processed using graph neural networks (GNNs). With the adoption of recent laws ensuring the ``right to be forgotten'', the problem of graph data removal has become of significant importance. To address the problem, we introduce the first known framework for \emph{certified graph unlearning} of GNNs. In contrast to standard machine unlearning, new analytical and heuristic unlearning challenges arise when dealing with complex graph data. First, three different types of unlearning requests need to be considered, including node feature, edge and node unlearning. Second, to establish provable performance guarantees, one needs to address challenges associated with feature mixing during propagation. The underlying analysis is illustrated on the example of simple graph convolutions (SGC) and their generalized PageRank (GPR) extensions, thereby laying the theoretical foundation for certified unlearning of GNNs.

- Fully Privacy-Preserving Federated Representation Learning via Secure Embedding Aggregation. [[paper]](https://arxiv.org/abs/2206.09097)
  - Jiaxiang Tang, Jinbao Zhu, Songze Li, Kai Zhang, Lichao Sun.
  - Key Word: Federated Learning; Privacy.
  - <details><summary>Digest</summary> We consider a federated representation learning framework, where with the assistance of a central server, a group of N distributed clients train collaboratively over their private data, for the representations (or embeddings) of a set of entities (e.g., users in a social network). Under this framework, for the key step of aggregating local embeddings trained at the clients in a private manner, we develop a secure embedding aggregation protocol named SecEA, which provides information-theoretical privacy guarantees for the set of entities and the corresponding embeddings at each client simultaneously, against a curious server and up to T < N/2 colluding clients.

- I Know What You Trained Last Summer: A Survey on Stealing Machine Learning Models and Defences. [[paper]](https://arxiv.org/abs/2206.08451)
  - Daryna Oliynyk, Rudolf Mayer, Andreas Rauber.
  - Key Word: Model Extraction Attacks; Survey.
  - <details><summary>Digest</summary> Adversaries can create a copy of the model with (almost) identical behavior using the the prediction labels only. While many variants of this attack have been described, only scattered defence strategies have been proposed, addressing isolated threats. This raises the necessity for a thorough systematisation of the field of model stealing, to arrive at a comprehensive understanding why these attacks are successful, and how they could be holistically defended against. We address this by categorising and comparing model stealing attacks, assessing their performance, and exploring corresponding defence techniques in different settings.

- Reconstructing Training Data from Trained Neural Networks. [[paper]](https://arxiv.org/abs/2206.07758) [[code]](https://giladude1.github.io/reconstruction/)
  - Niv Haim, Gal Vardi, Gilad Yehudai, Ohad Shamir, Michal Irani.
  - Key Word: Reconstruction Attacks.
  - <details><summary>Digest</summary> We propose a novel reconstruction scheme that stems from recent theoretical results about the implicit bias in training neural networks with gradient-based methods. To the best of our knowledge, our results are the first to show that reconstructing a large portion of the actual training samples from a trained neural network classifier is generally possible. This has negative implications on privacy, as it can be used as an attack for revealing sensitive training data. We demonstrate our method for binary MLP classifiers on a few standard computer vision datasets.

- A Survey on Gradient Inversion: Attacks, Defenses and Future Directions. [[paper]](https://arxiv.org/abs/2206.07284)
  - Rui Zhang, Song Guo, Junxiao Wang, Xin Xie, Dacheng Tao. *IJCAI 2022*
  - Key Word: Gradient Inversion Attacks and Defenses; Survey.
  - <details><summary>Digest</summary> Recent studies have shown that the training samples can be recovered from gradients, which are called Gradient Inversion (GradInv) attacks. However, there remains a lack of extensive surveys covering recent advances and thorough analysis of this issue. In this paper, we present a comprehensive survey on GradInv, aiming to summarize the cutting-edge research and broaden the horizons for different domains.

- Self-Supervised Pretraining for Differentially Private Learning. [[paper]](https://arxiv.org/abs/2206.07125)
  - Arash Asadian, Evan Weidner, Lei Jiang.
  - Key Word: Self-Supervised Pretraining; Differential Privacy.
  - <details><summary>Digest</summary> We demonstrate self-supervised pretraining (SSP) is a scalable solution to deep learning with differential privacy (DP) regardless of the size of available public datasets in image classification. When facing the lack of public datasets, we show the features generated by SSP on only one single image enable a private classifier to obtain much better utility than the non-learned handcrafted features under the same privacy budget. When a moderate or large size public dataset is available, the features produced by SSP greatly outperform the features trained with labels on various complex private datasets under the same private budget.

- PrivHAR: Recognizing Human Actions From Privacy-preserving Lens. [[paper]](https://arxiv.org/abs/2206.03891)
  - Carlos Hinojosa, Miguel Marquez, Henry Arguello, Ehsan Adeli, Li Fei-Fei, Juan Carlos Niebles.
  - Key Word: Privacy-Preserving Lens Design; Human Action Recognition; Adversarial Training; Deep Optics.
  - <details><summary>Digest</summary> The accelerated use of digital cameras prompts an increasing concern about privacy and security, particularly in applications such as action recognition. In this paper, we propose an optimizing framework to provide robust visual privacy protection along the human action recognition pipeline. Our framework parameterizes the camera lens to successfully degrade the quality of the videos to inhibit privacy attributes and protect against adversarial attacks while maintaining relevant features for activity recognition. We validate our approach with extensive simulations and hardware experiments.

- Data Stealing Attack on Medical Images: Is it Safe to Export Networks from Data Lakes? [[paper]](https://arxiv.org/abs/2206.03391)
  - Huiyu Li, Nicholas Ayache, Hervé Delingette.
  - Key Word: Data Stealing Attacks; Medical Imaging; Heathcare.
  - <details><summary>Digest</summary> We introduce the concept of data stealing attack during the export of neural networks. It consists in hiding some information in the exported network that allows the reconstruction outside the data lake of images initially stored in that data lake. More precisely, we show that it is possible to train a network that can perform lossy image compression and at the same time solve some utility tasks such as image segmentation.

- On the Privacy Properties of GAN-generated Samples. [[paper]](https://arxiv.org/abs/2206.01349)
  - Zinan Lin, Vyas Sekar, Giulia Fanti. *AISTATS 2021*
  - Key Word: Generative Adversarial Nets; Differential Privacy; Membership Inference Attacks.
  - <details><summary>Digest</summary> The privacy implications of generative adversarial networks (GANs) are a topic of great interest, leading to several recent algorithms for training GANs with privacy guarantees. By drawing connections to the generalization properties of GANs, we prove that under some assumptions, GAN-generated samples inherently satisfy some (weak) privacy guarantees. First, we show that if a GAN is trained on m samples and used to generate n samples, the generated samples are (epsilon, delta)-differentially-private for (epsilon, delta) pairs where delta scales as O(n/m). We show that under some special conditions, this upper bound is tight. Next, we study the robustness of GAN-generated samples to membership inference attacks. We model membership inference as a hypothesis test in which the adversary must determine whether a given sample was drawn from the training dataset or from the underlying data distribution.

- Defense Against Gradient Leakage Attacks via Learning to Obscure Data. [[paper]](https://arxiv.org/abs/2206.00769)
  - Yuxuan Wan, Han Xu, Xiaorui Liu, Jie Ren, Wenqi Fan, Jiliang Tang.
  - Key Word: Gradient Leakage Defenses.
  - <details><summary>Digest</summary> We propose a new defense method to protect the privacy of clients' data by learning to obscure data. Our defense method can generate synthetic samples that are totally distinct from the original samples, but they can also maximally preserve their predictive features and guarantee the model performance. Furthermore, our defense strategy makes the gradient leakage attack and its variants extremely difficult to reconstruct the client data.

- Dataset Distillation using Neural Feature Regression. [[paper]](https://arxiv.org/abs/2206.00719)
  - Yongchao Zhou, Ehsan Nezhadarya, Jimmy Ba.
  - Key Word: Dataset Condensation; Continual Learning; Membership Inference Defenses.
  - <details><summary>Digest</summary> we address these challenges using neural Feature Regression with Pooling (FRePo), achieving the state-of-the-art performance with an order of magnitude less memory requirement and two orders of magnitude faster training than previous methods. The proposed algorithm is analogous to truncated backpropagation through time with a pool of models to alleviate various types of overfitting in dataset distillation. FRePo significantly outperforms the previous methods on CIFAR100, Tiny ImageNet, and ImageNet-1K. Furthermore, we show that high-quality distilled data can greatly improve various downstream applications, such as continual learning and membership inference defense.

- Federated Learning in Non-IID Settings Aided by Differentially Private Synthetic Data. [[paper]](https://arxiv.org/abs/2206.00686)
  - Huancheng Chen, Haris Vikalo.
  - Key Word: Federated Learning; Differential Privacy.
  - <details><summary>Digest</summary> We propose FedDPMS (Federated Differentially Private Means Sharing), an FL algorithm in which clients deploy variational auto-encoders to augment local datasets with data synthesized using differentially private means of latent data representations communicated by a trusted server. Such augmentation ameliorates effects of data heterogeneity across the clients without compromising privacy.

- FETA: Fairness Enforced Verifying, Training, and Predicting Algorithms for Neural Networks. [[paper]](https://arxiv.org/abs/2206.00553)
  - Kiarash Mohammadi, Aishwarya Sivaraman, Golnoosh Farnadi.
  - Key Word: Fairness; Verification.
  - <details><summary>Digest</summary> We study the problem of verifying, training, and guaranteeing individual fairness of neural network models. A popular approach for enforcing fairness is to translate a fairness notion into constraints over the parameters of the model. However, such a translation does not always guarantee fair predictions of the trained neural network model. To address this challenge, we develop a counterexample-guided post-processing technique to provably enforce fairness constraints at prediction time.

- Privacy for Free: How does Dataset Condensation Help Privacy? [[paper]](https://arxiv.org/abs/2206.00240)
  - Tian Dong, Bo Zhao, Lingjuan Lyu. *ICML 2022*
  - Key Word: Privacy; Dataset Condensation.
  - <details><summary>Digest</summary> We for the first time identify that dataset condensation (DC) which is originally designed for improving training efficiency is also a better solution to replace the traditional data generators for private data generation, thus providing privacy for free. To demonstrate the privacy benefit of DC, we build a connection between DC and differential privacy, and theoretically prove on linear feature extractors (and then extended to non-linear feature extractors) that the existence of one sample has limited impact (O(m/n)) on the parameter distribution of networks trained on m samples synthesized from n(n≫m) raw samples by DC.

- Benign Overparameterization in Membership Inference with Early Stopping. [[paper]](https://arxiv.org/abs/2205.14055)
  - Jasper Tan, Daniel LeJeune, Blake Mason, Hamid Javadi, Richard G. Baraniuk.
  - Key Word: Benign Overparameterization; Membership Inference Attacks; Early Stopping.
  - <details><summary>Digest</summary> Does a neural network's privacy have to be at odds with its accuracy? In this work, we study the effects the number of training epochs and parameters have on a neural network's vulnerability to membership inference (MI) attacks, which aim to extract potentially private information about the training data. We first demonstrate how the number of training epochs and parameters individually induce a privacy-utility trade-off: more of either improves generalization performance at the expense of lower privacy. However, remarkably, we also show that jointly tuning both can eliminate this privacy-utility trade-off.

- FaceMAE: Privacy-Preserving Face Recognition via Masked Autoencoders. [[paper]](https://arxiv.org/abs/2205.11090) [[code]](https://github.com/kaiwang960112/FaceMAE)
  - Kai Wang, Bo Zhao, Xiangyu Peng, Zheng Zhu, Jiankang Deng, Xinchao Wang, Hakan Bilen, Yang You.
  - Key Word: Privacy; Face Recognition.
  - <details><summary>Digest</summary> We propose a novel framework FaceMAE, where the face privacy and recognition performance are considered simultaneously. Firstly, randomly masked face images are used to train the reconstruction module in FaceMAE. We tailor the instance relation matching (IRM) module to minimize the distribution gap between real faces and FaceMAE reconstructed ones. During the deployment phase, we use trained FaceMAE to reconstruct images from masked faces of unseen identities without extra training.

- Unlocking High-Accuracy Differentially Private Image Classification through Scale. [[paper]](https://arxiv.org/abs/2204.13650) [[code]](https://github.com/deepmind/jax_privacy)
  - Soham De, Leonard Berrada, Jamie Hayes, Samuel L. Smith, Borja Balle.
  - Key Word: Differential Privacy; Image Classication.
  - <details><summary>Digest</summary> Differential Privacy (DP) provides a formal privacy guarantee preventing adversaries with access to a machine learning model from extracting information about individual training points. Differentially Private Stochastic Gradient Descent (DP-SGD), the most popular DP training method for deep learning, realizes this protection by injecting noise during training. However previous works have found that DP-SGD often leads to a significant degradation in performance on standard image classification benchmarks. Furthermore, some authors have postulated that DP-SGD inherently performs poorly on large models, since the norm of the noise required to preserve privacy is proportional to the model dimension. In contrast, we demonstrate that DP-SGD on over-parameterized models can perform significantly better than previously thought. Combining careful hyper-parameter tuning with simple techniques to ensure signal propagation and improve the convergence rate, we obtain a new SOTA without extra data on CIFAR-10 of 81.4% under (8, 10^{-5})-DP using a 40-layer Wide-ResNet, improving over the previous SOTA of 71.7%.

- SPAct: Self-supervised Privacy Preservation for Action Recognition. [[paper]](https://arxiv.org/abs/2203.15205) [[code]](https://github.com/DAVEISHAN/SPAct)
  - Ishan Rajendrakumar Dave, Chen Chen, Mubarak Shah. *CVPR 2022*
  - Key Word: Self-Supervion; Privacy; Action Recognition.
  - <details><summary>Digest</summary> Recent developments of self-supervised learning (SSL) have unleashed the untapped potential of the unlabeled data. For the first time, we present a novel training framework which removes privacy information from input video in a self-supervised manner without requiring privacy labels. Our training framework consists of three main components: anonymization function, self-supervised privacy removal branch, and action recognition branch. We train our framework using a minimax optimization strategy to minimize the action recognition cost function and maximize the privacy cost function through a contrastive self-supervised loss.

- Robust Unlearnable Examples: Protecting Data Against Adversarial Learning. [[paper]](https://arxiv.org/abs/2203.14533) [[code]](https://github.com/fshp971/robust-unlearnable-examples)
  - Shaopeng Fu, Fengxiang He, Yang Liu, Li Shen, Dacheng Tao. *ICLR 2022*
  - Key Word: Privacy; Adversarial Training.
  - <details><summary>Digest</summary> We first find that the vanilla error-minimizing noise, which suppresses the informative knowledge of data via minimizing the corresponding training loss, could not effectively minimize the adversarial training loss. This explains the vulnerability of error-minimizing noise in adversarial training. Based on the observation, robust error-minimizing noise is then introduced to reduce the adversarial training loss.

- Quantifying Memorization Across Neural Language Models. [[paper]](https://arxiv.org/abs/2202.07646)
  - Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, Chiyuan Zhang.
  - Key Word: Reconstruction Attacks; Membership Inference Attacks; Language Models.
  - <details><summary>Digest</summary> We describe three log-linear relationships that quantify the degree to which LMs emit memorized training data. Memorization significantly grows as we increase (1) the capacity of a model, (2) the number of times an example has been duplicated, and (3) the number of tokens of context used to prompt the model. Surprisingly, we find the situation becomes complicated when generalizing these results across model families. On the whole, we find that memorization in LMs is more prevalent than previously believed and will likely get worse as models continues to scale, at least without active mitigations.

- What Does it Mean for a Language Model to Preserve Privacy? [[paper]](https://arxiv.org/abs/2202.05520)
  - Hannah Brown, Katherine Lee, Fatemehsadat Mireshghallah, Reza Shokri, Florian Tramèr. *FAccT 2022*
  - Key Word: Natural Language Processing; Differential Privacy; Data Sanitization.
  - <details><summary>Digest</summary> We discuss the mismatch between the narrow assumptions made by popular data protection techniques (data sanitization and differential privacy), and the broadness of natural language and of privacy as a social norm. We argue that existing protection methods cannot guarantee a generic and meaningful notion of privacy for language models. We conclude that language models should be trained on text data which was explicitly produced for public use.

- Bounding Training Data Reconstruction in Private (Deep) Learning. [[paper]](https://arxiv.org/abs/2201.12383) [[code]](https://github.com/facebookresearch/bounding_data_reconstruction)
  - Chuan Guo, Brian Karrer, Kamalika Chaudhuri, Laurens van der Maaten. *ICML 2022*
  - Key Word: Reconstruction Attacks; Differential Privacy.
  - <details><summary>Digest</summary> Existing semantic guarantees for DP focus on membership inference, which may overestimate the adversary's capabilities and is not applicable when membership status itself is non-sensitive. In this paper, we derive the first semantic guarantees for DP mechanisms against training data reconstruction attacks under a formal threat model. We show that two distinct privacy accounting methods -- Renyi differential privacy and Fisher information leakage -- both offer strong semantic protection against data reconstruction attacks.

- Variational Model Inversion Attacks. [[paper]](https://arxiv.org/abs/2201.10787) [[code]](https://github.com/wangkua1/vmi)
  - Kuan-Chieh Wang, Yan Fu, Ke Li, Ashish Khisti, Richard Zemel, Alireza Makhzani. *NeurIPS 2021*
  - Key Word: Model Inversion Attacks.
  - <details><summary>Digest</summary> We provide a probabilistic interpretation of model inversion attacks, and formulate a variational objective that accounts for both diversity and accuracy. In order to optimize this variational objective, we choose a variational family defined in the code space of a deep generative model, trained on a public auxiliary dataset that shares some structural similarity with the target dataset.  

- Plug & Play Attacks: Towards Robust and Flexible Model Inversion Attacks. [[paper]](https://arxiv.org/abs/2201.12179) [[code]](https://github.com/LukasStruppek/Plug-and-Play-Attacks)
  - Lukas Struppek, Dominik Hintersdorf, Antonio De Almeida Correia, Antonia Adler, Kristian Kersting. *ICML 2022*
  - Key Word: Model Inversion Attacks.
  - <details><summary>Digest</summary> Previous research has developed generative MIAs that use generative adversarial networks (GANs) as image priors tailored to a specific target model. This makes the attacks time- and resource-consuming, inflexible, and susceptible to distributional shifts between datasets. To overcome these drawbacks, we present Plug & Play Attacks, which relax the dependency between the target model and image prior, and enable the use of a single GAN to attack a wide range of targets, requiring only minor adjustments to the attack.

- Reconstructing Training Data with Informed Adversaries. [[paper]](https://arxiv.org/abs/2201.04845) [[code]](https://github.com/deepmind/informed_adversary_mnist_reconstruction)
  - Borja Balle, Giovanni Cherubin, Jamie Hayes. *S&P 2022*
  - Key Word: Reconstruction Attacks; Differential Privacy.
  - <details><summary>Digest</summary> This work studies this question from the lens of a powerful informed adversary who knows all the training data points except one. By instantiating concrete attacks, we show it is feasible to reconstruct the remaining data point in this stringent threat model. For convex models (e.g. logistic regression), reconstruction attacks are simple and can be derived in closed-form. For more general models (e.g. neural networks), we propose an attack strategy based on training a reconstructor network that receives as input the weights of the model under attack and produces as output the target data point. 

</details>

<!-- ### Privacy: 2021 -->

<details>
<summary>Privacy: 2021</summary>

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

</details>

<!-- ### Privacy: 2020 -->

<details>
<summary>Privacy: 2020</summary>

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

</details>

<!-- ### Privacy: 2019 -->

<details>
<summary>Privacy: 2019</summary>

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

</details>

## Fairness

<details>
<summary>Fairness: 2022</summary>

- Fairify: Fairness Verification of Neural Networks. [[paper]](https://arxiv.org/abs/2212.06140) [[code]](https://github.com/sumonbis/farify)
  - Sumon Biswas, Hridesh Rajan. *ICSE 2023*
  - Key Word: Fairness Verification.
  - <details><summary>Digest</summary> We proposed Fairify, the first SMT-based approach to verify individual fairness property in neural network (NN) models. Individual fairness ensures that any two similar individuals get similar treatment irrespective of their protected attributes e.g., race, sex, age. Verifying this fairness property is hard because of its global nature and the presence of non-linear computation nodes in NN. We proposed sound approach to make individual fairness verification tractable for the developers. The key idea is that many neurons in the NN always remain inactive when a smaller part of the input domain is considered. 

- BiasBed -- Rigorous Texture Bias Evaluation. [[paper]](https://arxiv.org/abs/2211.13190) [[code]](https://github.com/D1noFuzi/BiasBed)
  - Nikolai Kalischek, Rodrigo C. Daudt, Torben Peters, Jan D. Wegner, Konrad Schindler.
  - Key Word: Texture Bias.
  - <details><summary>Digest</summary> We investigate difficulties and limitations when training networks with reduced texture bias. In particular, we also show that proper evaluation and meaningful comparisons between methods are not trivial. We introduce BiasBed, a testbed for texture- and style-biased training, including multiple datasets and a range of existing algorithms. It comes with an extensive evaluation protocol that includes rigorous hypothesis testing to gauge the significance of the results, despite the considerable training instability of some style bias methods.

- Easily Accessible Text-to-Image Generation Amplifies Demographic Stereotypes at Large Scale. [[paper]](https://arxiv.org/abs/2211.03759)
  - Federico Bianchi, Pratyusha Kalluri, Esin Durmus, Faisal Ladhak, Myra Cheng, Debora Nozza, Tatsunori Hashimoto, Dan Jurafsky, James Zou, Aylin Caliskan.
  - Key Word: Stable Diffusion; Stereotype Amplification.
  - <details><summary>Digest</summary> Machine learning models are now able to convert user-written text descriptions into naturalistic images. These models are available to anyone online and are being used to generate millions of images a day. We investigate these models and find that they amplify dangerous and complex stereotypes. Moreover, we find that the amplified stereotypes are difficult to predict and not easily mitigated by users or model owners. The extent to which these image-generation models perpetuate and amplify stereotypes and their mass deployment is cause for serious concern.

- Fairness in Federated Learning via Core-Stability. [[paper]](https://arxiv.org/abs/2211.02091)
  - Bhaskar Ray Chaudhury, Linyi Li, Mintong Kang, Bo Li, Ruta Mehta. *NeurIPS 2022*
  - Key Word: Federated Learning; Fairness.
  - <details><summary>Digest</summary> Federated learning provides an effective paradigm to jointly optimize a model benefited from rich distributed data while protecting data privacy. Nonetheless, the heterogeneity nature of distributed data makes it challenging to define and ensure fairness among local agents. Currently popular egalitarian and weighted equity-based fairness measures suffer from the aforementioned pitfall. In this work, we aim to formally represent this problem and address these fairness issues using concepts from co-operative game theory and social choice theory. 

- FARE: Provably Fair Representation Learning. [[paper]](https://arxiv.org/abs/2210.07213)
  - Nikola Jovanović, Mislav Balunović, Dimitar I. Dimitrov, Martin Vechev.
  - Key Word: Provable Fairness; Group Fairness.
  - <details><summary>Digest</summary> Recent work has shown that prior methods achieve worse accuracy-fairness tradeoffs than originally suggested by their results. This dictates the need for FRL methods that provide provable upper bounds on unfairness of any downstream classifier, a challenge yet unsolved. In this work we address this challenge and propose Fairness with Restricted Encoders (FARE), the first FRL method with provable fairness guarantees. Our key insight is that restricting the representation space of the encoder enables us to derive suitable fairness guarantees, while allowing empirical accuracy-fairness tradeoffs comparable to prior work.

- Fairness via Adversarial Attribute Neighbourhood Robust Learning. [[paper]](https://arxiv.org/abs/2210.06630)
  - Qi Qi, Shervin Ardeshir, Yi Xu, Tianbao Yang.
  - Key Word: Fairness; Robust Loss; Invariant Learning.
  - <details><summary>Digest</summary> To enhance the model performs uniformly well in different sensitive attributes, we propose a principled Robust Adversarial Attribute Neighbourhood (RAAN) loss to debias the classification head and promote a fairer representation distribution across different sensitive attribute groups. The key idea of RAAN is to mitigate the differences of biased representations between different sensitive attribute groups by assigning each sample an adversarial robust weight, which is defined on the representations of adversarial attribute neighbors, i.e, the samples from different protected groups. 

- MEDFAIR: Benchmarking Fairness for Medical Imaging. [[paper]](https://arxiv.org/abs/2210.01725) [[code]](https://github.com/ys-zong/medfair)
  - Yongshuo Zong, Yongxin Yang, Timothy Hospedales.
  - Key Word: Fairness; Medical Imaging.
  - <details><summary>Digest</summary> We introduce MEDFAIR, a framework to benchmark the fairness of machine learning models for medical imaging. MEDFAIR covers eleven algorithms from various categories, nine datasets from different imaging modalities, and three model selection criteria. Through extensive experiments, we find that the under-studied issue of model selection criterion can have a significant impact on fairness outcomes; while in contrast, state-of-the-art bias mitigation algorithms do not significantly improve fairness outcomes over empirical risk minimization (ERM) in both in-distribution and out-of-distribution settings. 

- A Survey of Fairness in Medical Image Analysis: Concepts, Algorithms, Evaluations, and Challenges. [[paper]](https://arxiv.org/abs/2209.13177)
  - Zikang Xu, Jun Li, Qingsong Yao, Han Li, Xin Shi, S. Kevin Zhou.
  - Key Word: Survey; Fairness; Medical Image Analysis; Healthcare.
  - <details><summary>Digest</summary> We first give a comprehensive and precise definition of fairness, following by introducing currently used techniques in fairness issues in MedIA. After that, we list public medical image datasets that contain demographic attributes for facilitating the fairness research and summarize current algorithms concerning fairness in MedIA. 

- Fairness Reprogramming. [[paper]](https://arxiv.org/abs/2209.10222)
  - Guanhua Zhang, Yihua Zhang, Yang Zhang, Wenqi Fan, Qing Li, Sijia Liu, Shiyu Chang.
  - Key Word: Model reprogramming; Fairness.
  - <details><summary>Digest</summary> We propose a new generic fairness learning paradigm, called FairReprogram, which incorporates the model reprogramming technique. Specifically, FairReprogram considers the neural model fixed, and instead appends to the input a set of perturbations, called the fairness trigger, which is tuned towards the fairness criteria under a min-max formulation. 

- Fairness and robustness in anti-causal prediction. [[paper]](https://arxiv.org/abs/2209.09423)
  - Maggie Makar, Alexander D'Amour.
  - Key Word: Distribution Shifts; Causal Structure.
  - <details><summary>Digest</summary> Robustness to distribution shift and fairness have independently emerged as two important desiderata required of modern machine learning models. While these two desiderata seem related, the connection between them is often unclear in practice. Here, we discuss these connections through a causal lens, focusing on anti-causal prediction tasks, where the input to a classifier (e.g., an image) is assumed to be generated as a function of the target label and the protected attribute. By taking this perspective, we draw explicit connections between a common fairness criterion - separation - and a common notion of robustness - risk invariance. 

- It's Not Fairness, and It's Not Fair: The Failure of Distributional Equality and the Promise of Relational Equality in Complete-Information Hiring Games. [[paper]](https://arxiv.org/abs/2209.05602)
  - Benjamin Fish, Luke Stark.
  - Key Word: Group Fairness; Individual Fairness; Causal Fairness.
  - <details><summary>Digest</summary> Existing efforts to formulate computational definitions of fairness have largely focused on distributional notions of equality, where equality is defined by the resources or decisions given to individuals in the system. Yet existing discrimination and injustice is often the result of unequal social relations, rather than an unequal distribution of resources. Here, we show how optimizing for existing computational and economic definitions of fairness and equality fail to prevent unequal social relations. To do this, we provide an example of a self-confirming equilibrium in a simple hiring market that is relationally unequal but satisfies existing distributional notions of fairness.

- Sustaining Fairness via Incremental Learning. [[paper]](https://arxiv.org/abs/2208.12212)
  - Somnath Basu Roy Chowdhury, Snigdha Chaturvedi.
  - Key Word: Fairness; Incremental Learning.
  - <details><summary>Digest</summary> We propose to address this issue by introducing the problem of learning fair representations in an incremental learning setting. To this end, we present Fairness-aware Incremental Representation Learning (FaIRL), a representation learning system that can sustain fairness while incrementally learning new tasks. FaIRL is able to achieve fairness and learn new tasks by controlling the rate-distortion function of the learned representations.

- Bugs in the Data: How ImageNet Misrepresents Biodiversity. [[paper]](https://arxiv.org/abs/2208.11695) [[code]](https://github.com/rolnicklab/imagenetbiodiversity)
  - Alexandra Sasha Luccioni, David Rolnick.
  - Key Word: ImageNet Bias.
  - <details><summary>Digest</summary> We analyze the 13,450 images from 269 classes that represent wild animals in the ImageNet-1k validation set, with the participation of expert ecologists. We find that many of the classes are ill-defined or overlapping, and that 12% of the images are incorrectly labeled, with some classes having >90% of images incorrect. We also find that both the wildlife-related labels and images included in ImageNet-1k present significant geographical and cultural biases, as well as ambiguities such as artificial animals, multiple species in the same image, or the presence of humans.

- Discover and Mitigate Unknown Biases with Debiasing Alternate Networks. [[paper]](https://arxiv.org/abs/2207.10077) [[code]](https://github.com/zhihengli-UR/DebiAN)
  - Zhiheng Li, Anthony Hoogs, Chenliang Xu. *ECCV 2022*
  - Key Word: Bias Identification; Bias Mitigation; Fairness; Unsupervised Debiasing.
  - <details><summary>Digest</summary> Deep image classifiers have been found to learn biases from datasets. To mitigate the biases, most previous methods require labels of protected attributes (e.g., age, skin tone) as full-supervision, which has two limitations: 1) it is infeasible when the labels are unavailable; 2) they are incapable of mitigating unknown biases -- biases that humans do not preconceive. To resolve those problems, we propose Debiasing Alternate Networks (DebiAN), which comprises two networks -- a Discoverer and a Classifier. By training in an alternate manner, the discoverer tries to find multiple unknown biases of the classifier without any annotations of biases, and the classifier aims at unlearning the biases identified by the discoverer.

- Mitigating Algorithmic Bias with Limited Annotations. [[paper]](https://arxiv.org/abs/2207.10018) [[code]](https://github.com/guanchuwang/apod-fairness)
  - Guanchu Wang, Mengnan Du, Ninghao Liu, Na Zou, Xia Hu.
  - Key Word: Fairness; Active Bias Mitigation; Limited Annotations.
  - <details><summary>Digest</summary> When sensitive attributes are not disclosed or available, it is needed to manually annotate a small part of the training data to mitigate bias. However, the skewed distribution across different sensitive groups preserves the skewness of the original dataset in the annotated subset, which leads to non-optimal bias mitigation. To tackle this challenge, we propose Active Penalization Of Discrimination (APOD), an interactive framework to guide the limited annotations towards maximally eliminating the effect of algorithmic bias. The proposed APOD integrates discrimination penalization with active instance selection to efficiently utilize the limited annotation budget, and it is theoretically proved to be capable of bounding the algorithmic bias.

- Bias Mitigation for Machine Learning Classifiers: A Comprehensive Survey. [[paper]](https://arxiv.org/abs/2207.07068)
  - Max Hort, Zhenpeng Chen, Jie M. Zhang, Federica Sarro, Mark Harman.
  - Key Word: Bias Mitigation; Fairness; Survey.
  - <details><summary>Digest</summary> This paper provides a comprehensive survey of bias mitigation methods for achieving fairness in Machine Learning (ML) models. We collect a total of 234 publications concerning bias mitigation for ML classifiers. These methods can be distinguished based on their intervention procedure (i.e., pre-processing, in-processing, post-processing) and the technology they apply. We investigate how existing bias mitigation methods are evaluated in the literature. In particular, we consider datasets, metrics and benchmarking. Based on the gathered insights (e.g., what is the most popular fairness metric? How many datasets are used for evaluating bias mitigation methods?).

- Fair Machine Learning in Healthcare: A Review. [[paper]](https://arxiv.org/abs/2206.14397)
  - Qizhang Feng, Mengnan Du, Na Zou, Xia Hu.
  - Key Word: Fairness; Healthcare; Survey.
  - <details><summary>Digest</summary> Benefiting from the digitization of healthcare data and the development of computing power, machine learning methods are increasingly used in the healthcare domain. Fairness problems have been identified in machine learning for healthcare, resulting in an unfair allocation of limited healthcare resources or excessive health risks for certain groups. Therefore, addressing the fairness problems has recently attracted increasing attention from the healthcare community. However, the intersection of machine learning for healthcare and fairness in machine learning remains understudied. In this review, we build the bridge by exposing fairness problems, summarizing possible biases, sorting out mitigation methods and pointing out challenges along with opportunities for the future.

- Transferring Fairness under Distribution Shifts via Fair Consistency Regularization. [[paper]](https://arxiv.org/abs/2206.12796)
  - Bang An, Zora Che, Mucong Ding, Furong Huang.
  - Key Word: Fairness; Distribution Shifts; Regularization.
  - <details><summary>Digest</summary> We study how to transfer model fairness under distribution shifts, a widespread issue in practice. We conduct a fine-grained analysis of how the fair model is affected under different types of distribution shifts and find that domain shifts are more challenging than subpopulation shifts. Inspired by the success of self-training in transferring accuracy under domain shifts, we derive a sufficient condition for transferring group fairness. Guided by it, we propose a practical algorithm with a fair consistency regularization as the key component.

- Input-agnostic Certified Group Fairness via Gaussian Parameter Smoothing. [[paper]](https://arxiv.org/abs/2206.11423)
  - Jiayin Jin, Zeru Zhang, Yang Zhou, Lingfei Wu.
  - Key Word: Fair Classification; Group Fairness.
  - <details><summary>Digest</summary> Only recently, researchers attempt to provide classification algorithms with provable group fairness guarantees. Most of these algorithms suffer from harassment caused by the requirement that the training and deployment data follow the same distribution. This paper proposes an input-agnostic certified group fairness algorithm, FairSmooth, for improving the fairness of classification models while maintaining the remarkable prediction accuracy. A Gaussian parameter smoothing method is developed to transform base classifiers into their smooth versions. An optimal individual smooth classifier is learnt for each group with only the data regarding the group and an overall smooth classifier for all groups is generated by averaging the parameters of all the individual smooth ones.

- FairGrad: Fairness Aware Gradient Descent. [[paper]](https://arxiv.org/abs/2206.10923)
  - Gaurav Maheshwari, Michaël Perrot.
  - Key Word: Group Fairness.
  - <details><summary>Digest</summary> We propose FairGrad, a method to enforce fairness based on a reweighting scheme that iteratively learns group specific weights based on whether they are advantaged or not. FairGrad is easy to implement and can accommodate various standard fairness definitions. Furthermore, we show that it is comparable to standard baselines over various datasets including ones used in natural language processing and computer vision.

- Active Fairness Auditing. [[paper]](https://arxiv.org/abs/2206.08450)
  - Tom Yan, Chicheng Zhang. *ICML 2022*
  - Key Word: Active Learning.
  - <details><summary>Digest</summary> The fast spreading adoption of machine learning (ML) by companies across industries poses significant regulatory challenges. One such challenge is scalability: how can regulatory bodies efficiently audit these ML models, ensuring that they are fair? In this paper, we initiate the study of query-based auditing algorithms that can estimate the demographic parity of ML models in a query-efficient manner. We propose an optimal deterministic algorithm, as well as a practical randomized, oracle-efficient algorithm with comparable guarantees.

- What-Is and How-To for Fairness in Machine Learning: A Survey, Reflection, and Perspective. [[paper]](https://arxiv.org/abs/2206.04101)
  - Zeyu Tang, Jiji Zhang, Kun Zhang.
  - Key Word: Fairness; Causality; Bias Mitigation; Survey.
  - <details><summary>Digest</summary> Algorithmic fairness has attracted increasing attention in the machine learning community. Various definitions are proposed in the literature, but the differences and connections among them are not clearly addressed. In this paper, we review and reflect on various fairness notions previously proposed in machine learning literature, and make an attempt to draw connections to arguments in moral and political philosophy, especially theories of justice. We also consider fairness inquiries from a dynamic perspective, and further consider the long-term impact that is induced by current prediction and decision.

- How unfair is private learning? [[paper]](https://arxiv.org/abs/2206.03985)
  - Amartya Sanyal, Yaxi Hu, Fanny Yang. *UAI 2022*
  - Key Word: Fairness; Privacy.
  - <details><summary>Digest</summary> As machine learning algorithms are deployed on sensitive data in critical decision making processes, it is becoming increasingly important that they are also private and fair. In this paper, we show that, when the data has a long-tailed structure, it is not possible to build accurate learning algorithms that are both private and results in higher accuracy on minority subpopulations. We further show that relaxing overall accuracy can lead to good fairness even with strict privacy requirements.

- DebiasBench: Benchmark for Fair Comparison of Debiasing in Image Classification. [[paper]](https://arxiv.org/abs/2206.03680)
  - Jungsoo Lee, Juyoung Lee, Sanghun Jung, Jaegul Choo.
  - Key Word: Debiasing; Image Classification; Benchmark.
  - <details><summary>Digest</summary> The goal of this paper is to standardize the inconsistent experimental settings and propose a consistent model parameter selection criterion for debiasing. Based on such unified experimental settings and model parameter selection criterion, we build a benchmark named DebiasBench which includes five datasets and seven debiasing methods. We carefully conduct extensive experiments in various aspects and show that different state-of-the-art methods work best in different datasets, respectively. Even, the vanilla method, the method with no debiasing module, also shows competitive results in datasets with low bias severity.

- How Biased is Your Feature?: Computing Fairness Influence Functions with Global Sensitivity Analysis. [[paper]](https://arxiv.org/abs/2206.00667)
  - Bishwamittra Ghosh, Debabrota Basu, Kuldeep S. Meel.
  - Key Word: Fairness; Influence Function.
  - <details><summary>Digest</summary> We aim to quantify the influence of different features on the bias of a classifier. To this end, we propose a framework of Fairness Influence Function (FIF), and compute it as a scaled difference of conditional variances in the prediction of the classifier. We also instantiate an algorithm, FairXplainer, that uses variance decomposition among the subset of features and a local regressor to compute FIFs accurately, while also capturing the intersectional effects of the features.

- Fairness Transferability Subject to Bounded Distribution Shift. [[paper]](https://arxiv.org/abs/2206.00129)
  - Yatong Chen, Reilly Raab, Jialu Wang, Yang Liu.
  - Key Word: Fairness; Distribution Shift.
  - <details><summary>Digest</summary> We study the transferability of statistical group fairness for machine learning predictors (i.e., classifiers or regressors) subject to bounded distribution shift, a phenomenon frequently caused by user adaptation to a deployed model or a dynamic environment. Herein, we develop a bound characterizing such transferability, flagging potentially inappropriate deployments of machine learning for socially consequential tasks.

- Inducing bias is simpler than you think. [[paper]](https://arxiv.org/abs/2205.15935)
  - Stefano Sarao Mannelli, Federica Gerace, Negar Rostamzadeh, Luca Saglietti.
  - Key Word: Fairness.
  - <details><summary>Digest</summary> We introduce a solvable high-dimensional model of data imbalance, where parametric control over the many bias-inducing factors allows for an extensive exploration of the bias inheritance mechanism. Through the tools of statistical physics, we analytically characterise the typical behaviour of learning models trained in our synthetic framework and find similar unfairness behaviours as those observed on more realistic data. However, we also identify a positive transfer effect between the different subpopulations within the data. This suggests that mixing data with different statistical properties could be helpful, provided the learning model is made aware of this structure.

- Mitigating Dataset Bias by Using Per-sample Gradient. [[paper]](https://arxiv.org/abs/2205.15704)
  - Sumyeong Ahn, Seongyoon Kim, Se-young Yun.
  - Key Word: Debiasing; Benchmark; Invariant Learning.
  - <details><summary>Digest</summary> We propose a debiasing algorithm, called PGD (Per-sample Gradient-based Debiasing), that comprises three steps: (1) training a model on uniform batch sampling, (2) setting the importance of each sample in proportion to the norm of the sample gradient, and (3) training the model using importance-batch sampling, whose probability is obtained in step (2). Compared with existing baselines for various synthetic and real-world datasets, the proposed method showed state-of-the-art accuracy for a the classification task. Furthermore, we describe theoretical understandings about how PGD can mitigate dataset bias.

- Certifying Some Distributional Fairness with Subpopulation Decomposition. [[paper]](https://arxiv.org/abs/2205.15494)
  - Mintong Kang, Linyi Li, Maurice Weber, Yang Liu, Ce Zhang, Bo Li.
  - Key Word: Certified Fairness.
  - <details><summary>Digest</summary> We first formulate the certified fairness of an ML model trained on a given data distribution as an optimization problem based on the model performance loss bound on a fairness constrained distribution, which is within bounded distributional distance with the training distribution. We then propose a general fairness certification framework and instantiate it for both sensitive shifting and general shifting scenarios. In particular, we propose to solve the optimization problem by decomposing the original data distribution into analytical subpopulations and proving the convexity of the subproblems to solve them. We evaluate our certified fairness on six real-world datasets and show that our certification is tight in the sensitive shifting scenario and provides non-trivial certification under general shifting.

- Fairness via Explanation Quality: Evaluating Disparities in the Quality of Post hoc Explanations. [[paper]](https://arxiv.org/abs/2205.07277)
  - Jessica Dai, Sohini Upadhyay, Ulrich Aivodji, Stephen H. Bach, Himabindu Lakkaraju. *AIES 2022*
  - Key Word: Fairness; Interpretability.
  - <details><summary>Digest</summary> We first outline the key properties which constitute explanation quality and where disparities can be particularly problematic. We then leverage these properties to propose a novel evaluation framework which can quantitatively measure disparities in the quality of explanations output by state-of-the-art methods. Using this framework, we carry out a rigorous empirical analysis to understand if and when group-based disparities in explanation quality arise. Our results indicate that such disparities are more likely to occur when the models being explained are complex and highly non-linear. In addition, we also observe that certain post hoc explanation methods (e.g., Integrated Gradients, SHAP) are more likely to exhibit the aforementioned disparities.

- Long-Tailed Recognition via Weight Balancing. [[paper]](https://arxiv.org/abs/2203.14197) [[code]](https://github.com/shadealsha/ltr-weight-balancing)
  - Shaden Alshammari, Yu-Xiong Wang, Deva Ramanan, Shu Kong. *CVPR 2022*
  - Key Word: Long-tailed Recognition; Weight Balancing.
  - <details><summary>Digest</summary> The key to addressing LTR is to balance various aspects including data distribution, training losses, and gradients in learning. We explore an orthogonal direction, weight balancing, motivated by the empirical observation that the naively trained classifier has "artificially" larger weights in norm for common classes (because there exists abundant data to train them, unlike the rare classes). We investigate three techniques to balance weights, L2-normalization, weight decay, and MaxNorm. We first point out that L2-normalization "perfectly" balances per-class weights to be unit norm, but such a hard constraint might prevent classes from learning better classifiers. In contrast, weight decay penalizes larger weights more heavily and so learns small balanced weights; the MaxNorm constraint encourages growing small weights within a norm ball but caps all the weights by the radius. Our extensive study shows that both help learn balanced weights and greatly improve the LTR accuracy. Surprisingly, weight decay, although underexplored in LTR, significantly improves over prior work. Therefore, we adopt a two-stage training paradigm and propose a simple approach to LTR: (1) learning features using the cross-entropy loss by tuning weight decay, and (2) learning classifiers using class-balanced loss by tuning weight decay and MaxNorm.

- Is Fairness Only Metric Deep? Evaluating and Addressing Subgroup Gaps in Deep Metric Learning. [[paper]](https://arxiv.org/abs/2203.12748) [[code]](https://github.com/ndullerud/dml-fairness)
  - Natalie Dullerud, Karsten Roth, Kimia Hamidieh, Nicolas Papernot, Marzyeh Ghassemi. *ICLR 2022*
  - Key Word: Metric Learning; Fairness.
  - <details><summary>Digest</summary> We are the first to evaluate state-of-the-art DML methods trained on imbalanced data, and to show the negative impact these representations have on minority subgroup performance when used for downstream tasks. In this work, we first define fairness in DML through an analysis of three properties of the representation space -- inter-class alignment, intra-class alignment, and uniformity -- and propose finDML, the fairness in non-balanced DML benchmark to characterize representation fairness.

- Linear Adversarial Concept Erasure. [[paper]](https://arxiv.org/abs/2201.12091) [[code]](https://github.com/shauli-ravfogel/rlace-icml)
  - Shauli Ravfogel, Michael Twiton, Yoav Goldberg, Ryan Cotterell. *ICML 2022*
  - Key Word: Fairness; Concept Removal; Bias Mitigation; Interpretability.
  - <details><summary>Digest</summary> We formulate the problem of identifying and erasing a linear subspace that corresponds to a given concept, in order to prevent linear predictors from recovering the concept. We model this problem as a constrained, linear minimax game, and show that existing solutions are generally not optimal for this task. We derive a closed-form solution for certain objectives, and propose a convex relaxation, R-LACE, that works well for others. When evaluated in the context of binary gender removal, the method recovers a low-dimensional subspace whose removal mitigates bias by intrinsic and extrinsic evaluation. We show that the method -- despite being linear -- is highly expressive, effectively mitigating bias in deep nonlinear classifiers while maintaining tractability and interpretability.

</details>

<!-- ### Fairness: 2021 -->

<details>
<summary>Fairness: 2021</summary>

- Fair Normalizing Flows. [[paper]](https://arxiv.org/abs/2106.05937) [[code]](https://github.com/eth-sri/fnf)
  - Mislav Balunović, Anian Ruoss, Martin Vechev. *ICLR 2022*
  - Key Word: Fairness; Normalizing Flows.
  - <details><summary>Digest</summary> We present Fair Normalizing Flows (FNF), a new approach offering more rigorous fairness guarantees for learned representations. Specifically, we consider a practical setting where we can estimate the probability density for sensitive groups. The key idea is to model the encoder as a normalizing flow trained to minimize the statistical distance between the latent representations of different groups.

</details>

<!-- ### Fairness: 2020 -->

<details>
<summary>Fairness: 2020</summary>

- Fairness in the Eyes of the Data: Certifying Machine-Learning Models. [[paper]](https://arxiv.org/abs/2009.01534)
  - Shahar Segal, Yossi Adi, Benny Pinkas, Carsten Baum, Chaya Ganesh, Joseph Keshet.
  - Key Word: Fairness; Privacy.
  - <details><summary>Digest</summary> We present a framework that allows to certify the fairness degree of a model based on an interactive and privacy-preserving test. The framework verifies any trained model, regardless of its training process and architecture. Thus, it allows us to evaluate any deep learning model on multiple fairness definitions empirically.  

</details>

<!-- ### Fairness: 2019 -->

<details>
<summary>Fairness: 2019</summary>

- Training individually fair ML models with Sensitive Subspace Robustness. [[paper]](https://arxiv.org/abs/1907.00020) [[code]](https://github.com/IBM/sensitive-subspace-robustness)
  - Mikhail Yurochkin, Amanda Bower, Yuekai Sun. *ICLR 2020*
  - Key Word: Distributionally Robust Optimization.
  - <details><summary>Digest</summary> We consider training machine learning models that are fair in the sense that their performance is invariant under certain sensitive perturbations to the inputs. For example, the performance of a resume screening system should be invariant under changes to the gender and/or ethnicity of the applicant. We formalize this notion of algorithmic fairness as a variant of individual fairness and develop a distributionally robust optimization approach to enforce it during training.  

</details>

## Interpretability

<details>
<summary>Interpretability: 2022</summary>

- Measuring the Interpretability of Unsupervised Representations via Quantized Reversed Probing. [[paper]](https://openreview.net/forum?id=HFPTzdwN39)
  - Iro Laina, Yuki M Asano, Andrea Vedaldi. *ICLR 2022*
  - Key Word: Interpretability; Unsupervision.
  - <details><summary>Digest</summary> Self-supervised visual representation learning has recently attracted significant research interest. While a common way to evaluate self-supervised representations is through transfer to various downstream tasks, we instead investigate the problem of measuring their interpretability, i.e. understanding the semantics encoded in raw representations. We formulate the latter as estimating the mutual information between the representation and a space of manually labelled concepts.

- Attention-based Interpretability with Concept Transformers. [[paper]](https://openreview.net/forum?id=kAa9eDS0RdO)
  - Mattia Rigotti, Christoph Miksovic, Ioana Giurgiu, Thomas Gschwind, Paolo Scotton. *ICLR 2022*
  - Key Word: Transformers; Interpretability.
  - <details><summary>Digest</summary> We propose the generalization of attention from low-level input features to high-level concepts as a mechanism to ensure the interpretability of attention scores within a given application domain. In particular, we design the ConceptTransformer, a deep learning module that exposes explanations of the output of a model in which it is embedded in terms of attention over user-defined high-level concepts.

- Fooling Explanations in Text Classifiers. [[paper]](https://openreview.net/forum?id=j3krplz_4w6)
  - Adam Ivankay, Ivan Girardi, Chiara Marchiori, Pascal Frossard. *ICLR 2022*
  - Key Word: Attribution Robustness; Natural Language Processing.
  - <details><summary>Digest</summary> It has been shown that explanation methods in vision applications are susceptible to local, imperceptible perturbations that can significantly alter the explanations without changing the predicted classes. We show here that the existence of such perturbations extends to text classifiers as well. Specifically, we introduce TextExplanationFooler (TEF), a novel explanation attack algorithm that alters text input samples imperceptibly so that the outcome of widely-used explanation methods changes considerably while leaving classifier predictions unchanged.

- Explanations of Black-Box Models based on Directional Feature Interactions. [[paper]](https://openreview.net/forum?id=45Mr7LeKR9)
  - Aria Masoomi, Davin Hill, Zhonghui Xu, Craig P Hersh, Edwin K. Silverman, Peter J. Castaldi, Stratis Ioannidis, Jennifer Dy. *ICLR 2022*
  - Key Word: Explainability; Shapley Values; Feature Interactions.
  - <details><summary>Digest</summary> Several recent works explain black-box models by capturing the most influential features for prediction per instance; such explanation methods are univariate, as they characterize importance per feature.  We extend univariate explanation to a higher-order; this enhances explainability, as bivariate methods can capture feature interactions in black-box models, represented as a directed graph.  

- On the Relationship Between Explanation and Prediction: A Causal View. [[paper]](https://arxiv.org/abs/2212.06925)
  - Amir-Hossein Karimi, Krikamol Muandet, Simon Kornblith, Bernhard Schölkopf, Been Kim.
  - Key Word: Quantify Explanation; Causal Inference.
  - <details><summary>Digest</summary> While previous work empirically hinted that explanations (E) may have little relationship with the prediction (Y), there is a lack of conclusive study to quantify this relationship. Our work borrows tools from causal inference to systematically assay this relationship. More specifically, we measure the relationship between E and Y by measuring the treatment effect when intervening on their causal ancestors (hyperparameters) (inputs to generate saliency-based Es or Ys). 

- Spurious Features Everywhere -- Large-Scale Detection of Harmful Spurious Features in ImageNet. [[paper]](https://arxiv.org/abs/2212.04871)
  - Yannic Neuhaus, Maximilian Augustin, Valentyn Boreiko, Matthias Hein.
  - Key Word: Spurious Features; Interpretability.
  - <details><summary>Digest</summary> We develop a framework that allows us to systematically identify spurious features in large datasets like ImageNet. It is based on our neural PCA components and their visualization. Previous work on spurious features of image classifiers often operates in toy settings or requires costly pixel-wise annotations. In contrast, we validate our results by checking that presence of the harmful spurious feature of a class is sufficient to trigger the prediction of that class. 

- Post hoc Explanations may be Ineffective for Detecting Unknown Spurious Correlation. [[paper]](https://arxiv.org/abs/2212.04629)
  - Julius Adebayo, Michael Muelly, Hal Abelson, Been Kim.
  - Key Word: Feature Attributions; Concept Activation; Training Point Ranking.
  - <details><summary>Digest</summary> We investigate whether three types of post hoc model explanations--feature attribution, concept activation, and training point ranking--are effective for detecting a model's reliance on spurious signals in the training data. Specifically, we consider the scenario where the spurious signal to be detected is unknown, at test-time, to the user of the explanation method. We design an empirical methodology that uses semi-synthetic datasets along with pre-specified spurious artifacts to obtain models that verifiably rely on these spurious training signals. 

- Interpreting Neural Networks through the Polytope Lens. [[paper]](https://arxiv.org/abs/2211.12312)
  - Sid Black, Lee Sharkey, Leo Grinsztajn, Eric Winsor, Dan Braun, Jacob Merizian, Kip Parker, Carlos Ramón Guevara, Beren Millidge, Gabriel Alfour, Connor Leahy.
  - Key Word: Interpreting Polytopes; Single Neurons; Polysemanticity and Superposition.
  - <details><summary>Digest</summary> Polysemanticity makes interpreting the network in terms of neurons or directions challenging since we can no longer assign a specific feature to a neural unit. In order to find a basic unit of description that does not suffer from these problems, we zoom in beyond just directions to study the way that piecewise linear activation functions (such as ReLU) partition the activation space into numerous discrete polytopes. We call this perspective the polytope lens. The polytope lens makes concrete predictions about the behavior of neural networks, which we evaluate through experiments on both convolutional image classifiers and language models.

- Data-Centric Debugging: mitigating model failures via targeted data collection. [[paper]](https://arxiv.org/abs/2211.09859)
  - Sahil Singla, Atoosa Malemir Chegini, Mazda Moayeri, Soheil Feiz.
  - Key Word: Debugging and Explainability; Visual Similarity Metrics.
  - <details><summary>Digest</summary> We propose a general methodology for model debugging that can systemically improve model performance on E while maintaining its performance on the original test set. Our key assumption is that we have access to a large pool of weakly (noisily) labeled data F. However, naively adding F to the training would hurt model performance due to the large extent of label noise. Our Data-Centric Debugging (DCD) framework carefully creates a debug-train set by selecting images from F that are perceptually similar to the images in Esample.

- Verifying And Interpreting Neural Networks using Finite Automata. [[paper]](https://arxiv.org/abs/2211.01022) [[code]](https://github.com/marcosaelzer/nn2nfa)
  - Marco Sälzer, Eric Alsmann, Florian Bruse, Martin Lange.
  - Key Word: Finite State Automata; Verification; Interpretation.
  - <details><summary>Digest</summary> Verifying properties and interpreting the behaviour of deep neural networks (DNN) is an important task given their ubiquitous use in applications, including safety-critical ones, and their blackbox nature. We propose an automata-theoric approach to tackling problems arising in DNN analysis. We show that the input-output behaviour of a DNN can be captured precisely by a (special) weak Büchi automaton of exponential size. We show how these can be used to address common verification and interpretation tasks like adversarial robustness, minimum sufficient reasons etc. We report on a proof-of-concept implementation translating DNN to automata on finite words for better efficiency at the cost of losing precision in analysis.

- Efficient Dataset Distillation Using Random Feature Approximation. [[paper]](https://arxiv.org/abs/2210.12067) [[code]](https://github.com/yolky/RFAD)
  - Noel Loo, Ramin Hasani, Alexander Amini, Daniela Rus. *NeurIPS 2022*
  - Key Word: Dataset Distillation; Interpretability; Privacy; Neural Network Gaussian Process; Neural Tangent Kernel; Random Feature.
  - <details><summary>Digest</summary> We propose a novel algorithm that uses a random feature approximation (RFA) of the Neural Network Gaussian Process (NNGP) kernel, which reduces the kernel matrix computation to O(|S|). Our algorithm provides at least a 100-fold speedup over KIP and can run on a single GPU. Our new method, termed an RFA Distillation (RFAD), performs competitively with KIP and other dataset condensation algorithms in accuracy over a range of large-scale datasets, both in kernel regression and finite-width network training. We demonstrate the effectiveness of our approach on tasks involving model interpretability and privacy preservation.

- Diffusion Visual Counterfactual Explanations. [[paper]](https://arxiv.org/abs/2210.11841) [[code]](https://github.com/valentyn1boreiko/DVCEs)
  - Maximilian Augustin, Valentyn Boreiko, Francesco Croce, Matthias Hein. *NeurIPS 2022*
  - Key Word: Counterfactual Explainability; Diffusion Models.
  - <details><summary>Digest</summary> Current approaches for the generation of VCEs are restricted to adversarially robust models and often contain non-realistic artefacts, or are limited to image classification problems with few classes. In this paper, we overcome this by generating Diffusion Visual Counterfactual Explanations (DVCEs) for arbitrary ImageNet classifiers via a diffusion process.

- Self-explaining deep models with logic rule reasoning. [[paper]](https://arxiv.org/abs/2210.07024)  [[code]](https://github.com/archon159/SELOR)
  - Seungeon Lee, Xiting Wang, Sungwon Han, Xiaoyuan Yi, Xing Xie, Meeyoung Cha. *NeurIPS 2022*
  - Key Word: Self-explanation; Logic rule; Interpretability.
  - <details><summary>Digest</summary> We present SELOR, a framework for integrating self-explaining capabilities into a given deep model to achieve both high prediction performance and human precision. By “human precision”, we refer to the degree to which humans agree with the reasons models provide for their predictions. Human precision affects user trust and allows users to collaborate closely with the model. We demonstrate that logic rule explanations naturally satisfy human precision with the expressive power required for good predictive performance. We then illustrate how to enable a deep model to predict and explain with logic rules. Our method does not require predefined logic rule sets or human annotations and can be learned efficiently and easily with widely-used deep learning modules in a differentiable way. Extensive experiments show that our method gives explanations closer to human decision logic than other methods while maintaining the performance of deep learning models. 

- "Help Me Help the AI": Understanding How Explainability Can Support Human-AI Interaction. [[paper]](https://arxiv.org/abs/2210.03735)
  - Sunnie S. Y. Kim, Elizabeth Anne Watkins, Olga Russakovsky, Ruth Fong, Andrés Monroy-Hernández.
  - Key Word: Explainable AI; Human-AI Interaction.
  - <details><summary>Digest</summary> Despite the proliferation of explainable AI (XAI) methods, little is understood about end-users' explainability needs. This gap is critical, because end-users may have needs that XAI methods should but don't yet support. To address this gap and contribute to understanding how explainability can support human-AI interaction, we conducted a study of a real-world AI application via interviews with 20 end-users of Merlin, a bird-identification app.

- WeightedSHAP: analyzing and improving Shapley based feature attributions. [[paper]](https://arxiv.org/abs/2209.13429)
  - Yongchan Kwon, James Zou.
  - Key Word: Model Interpretation; Shapley Value.
  - <details><summary>Digest</summary> Our work performs a rigorous analysis of the potential limitations of Shapley feature attribution. We identify simple settings where the Shapley value is mathematically suboptimal by assigning larger attributions for less influential features. Motivated by this observation, we propose WeightedSHAP, which generalizes the Shapley value and learns which marginal contributions to focus directly from data. 

- Concept Activation Regions: A Generalized Framework For Concept-Based Explanations. [[paper]](https://arxiv.org/abs/2209.11222) [[code]](https://github.com/jonathancrabbe/cars)
  - Jonathan Crabbé, Mihaela van der Schaar. *NeurIPS 2022*
  - Key Word: Concept Activation Vector.
  - <details><summary>Digest</summary> We propose to relax this assumption by allowing concept examples to be scattered across different clusters in the DNN's latent space. Each concept is then represented by a region of the DNN's latent space that includes these clusters and that we call concept activation region (CAR). To formalize this idea, we introduce an extension of the CAV formalism that is based on the kernel trick and support vector classifiers. This CAR formalism yields global concept-based explanations and local concept-based feature importance.

- EMaP: Explainable AI with Manifold-based Perturbations. [[paper]](https://arxiv.org/abs/2209.08453)
  - Minh N. Vu, Huy Q. Mai, My T. Thai.
  - Key Word: Black-box Explanations; Topological Data Analysis; Robust Machine Learning.
  - <details><summary>Digest</summary>  The goal of this work is to introduce a novel perturbation scheme so that more faithful and robust explanations can be obtained. Our study focuses on the impact of perturbing directions on the data topology. We show that perturbing along the orthogonal directions of the input manifold better preserves the data topology, both in the worst-case analysis of the discrete Gromov-Hausdorff distance and in the average-case analysis via persistent homology. 

- Explainable AI for clinical and remote health applications: a survey on tabular and time series data. [[paper]](https://arxiv.org/abs/2209.06528)
  - Flavio Di Martino, Franca Delmastro.
  - Key Word: Explainable AI; Tabular Data; Time-series Data; Healthcare.
  - <details><summary>Digest</summary> To provide an overview of XAI methods that are most suitable for tabular and time series data in the healthcare domain, this paper provides a review of the literature in the last 5 years, illustrating the type of generated explanations and the efforts provided to evaluate their relevance and quality. Specifically, we identify clinical validation, consistency assessment, objective and standardised quality evaluation, and human-centered quality assessment as key features to ensure effective explanations for the end users. 

- Concept-Based Explanations for Tabular Data. [[paper]](https://arxiv.org/abs/2209.05690)
  - Varsha Pendyala, Jihye Choi.
  - Key Word: Concept Activation Vector; Conceptual Sensitivity.
  - <details><summary>Digest</summary> We extend TCAV, the concept attribution approach, to tabular learning, by providing an idea on how to define concepts over tabular data. On a synthetic dataset with ground-truth concept explanations and a real-world dataset, we show the validity of our method in generating interpretability results that match the human-level intuitions. 

- From Shapley Values to Generalized Additive Models and back. [[paper]](https://arxiv.org/abs/2209.04012) [[code]](https://github.com/tml-tuebingen/nshap)
  - Sebastian Bordt, Ulrike von Luxburg.
  - Key Word: Shapley Values; Generalized Additive Models.
  - <details><summary>Digest</summary> We introduce n-Shapley Values, a natural extension of Shapley Values that explain individual predictions with interaction terms up to order n. As n increases, the n-Shapley Values converge towards the Shapley-GAM, a uniquely determined decomposition of the original function. From the Shapley-GAM, we can compute Shapley Values of arbitrary order, which gives precise insights into the limitations of these explanations. We then show that Shapley Values recover generalized additive models of order n, assuming that we allow for interaction terms up to order n in the explanations. 

- The Utility of Explainable AI in Ad Hoc Human-Machine Teaming. [[paper]](https://arxiv.org/abs/2209.03943)
  - Rohan Paleja, Muyleng Ghuy, Nadun Ranawaka Arachchige, Reed Jensen, Matthew Gombolay. *NeurIPS 2021*
  - Key Word: Human-Machine Teaming; Explainable AI.
  - <details><summary>Digest</summary> Recent advances in machine learning have led to growing interest in Explainable AI (xAI) to enable humans to gain insight into the decision-making of machine learning models. Despite this recent interest, the utility of xAI techniques has not yet been characterized in human-machine teaming. Importantly, xAI offers the promise of enhancing team situational awareness (SA) and shared mental model development, which are the key characteristics of effective human-machine teams. Rapidly developing such mental models is especially critical in ad hoc human-machine teaming, where agents do not have a priori knowledge of others' decision-making strategies. In this paper, we present two novel human-subject experiments quantifying the benefits of deploying xAI techniques within a human-machine teaming scenario. 

- Responsibility: An Example-based Explainable AI approach via Training Process Inspection. [[paper]](https://arxiv.org/abs/2209.03433)
  - Faraz Khadivpour, Arghasree Banerjee, Matthew Guzdial.
  - Key Word: Influential Training Samples.
  - <details><summary>Digest</summary> We present a novel XAI approach we call Responsibility that identifies the most responsible training example for a particular decision. This example can then be shown as an explanation: "this is what I (the AI) learned that led me to do that". We present experimental results across a number of domains along with the results of an Amazon Mechanical Turk user study, comparing responsibility and existing XAI methods on an image classification task. Our results demonstrate that responsibility can help improve accuracy for both human end users and secondary ML models.

- Measuring the Interpretability of Unsupervised Representations via Quantized Reverse Probing. [[paper]](https://arxiv.org/abs/2209.03268)
  - Iro Laina, Yuki M. Asano, Andrea Vedaldi. *ICLR 2022*
  - Key Word: Interpretability; Self-Supervised Learning.
  - <details><summary>Digest</summary> While a common way to evaluate self-supervised representations is through transfer to various downstream tasks, we instead investigate the problem of measuring their interpretability, i.e. understanding the semantics encoded in raw representations. We formulate the latter as estimating the mutual information between the representation and a space of manually labelled concepts. To quantify this we introduce a decoding bottleneck: information must be captured by simple predictors, mapping concepts to clusters in representation space. This approach, which we call reverse linear probing, provides a single number sensitive to the semanticity of the representation. 

- "Is your explanation stable?": A Robustness Evaluation Framework for Feature Attribution. [[paper]](https://arxiv.org/abs/2209.01782)
  - Yuyou Gan, Yuhao Mao, Xuhong Zhang, Shouling Ji, Yuwen Pu, Meng Han, Jianwei Yin, Ting Wang. *CCS 2022*
  - Key Word: Attributional Robustness.
  - <details><summary>Digest</summary> We propose a model-agnostic method \emph{Median Test for Feature Attribution} (MeTFA) to quantify the uncertainty and increase the stability of explanation algorithms with theoretical guarantees. MeTFA has the following two functions: (1) examine whether one feature is significantly important or unimportant and generate a MeTFA-significant map to visualize the results; (2) compute the confidence interval of a feature attribution score and generate a MeTFA-smoothed map to increase the stability of the explanation.

- Interpreting Black-box Machine Learning Models for High Dimensional Datasets. [[paper]](https://arxiv.org/abs/2208.13405)
  - Md. Rezaul Karim, Md. Shajalal, Alex Graß, Till Döhmen, Sisay Adugna Chala, Christian Beecks, Stefan Decker.
  - Key Word: Curse of Dimensionality; Interpretability; Attention mechanism; Model Surrogation.
  - <details><summary>Digest</summary> We first train a black-box model on a high-dimensional dataset to learn the embeddings on which the classification is performed. To decompose the inner working principles of the black-box model and to identify top-k important features, we employ different probing and perturbing techniques. We then approximate the behavior of the black-box model by means of an interpretable surrogate model on the top-k feature space. Finally, we derive decision rules and local explanations from the surrogate model to explain individual decisions. 

- SoK: Explainable Machine Learning for Computer Security Applications. [[paper]](https://arxiv.org/abs/2208.10605)
  - Azqa Nadeem, Daniël Vos, Clinton Cao, Luca Pajola, Simon Dieck, Robert Baumgartner, Sicco Verwer.
  - Key Word: Explainable Machine Learning; Cybersecurity; Survey.
  - <details><summary>Digest</summary> Explainable Artificial Intelligence (XAI) is a promising solution to improve the transparency of machine learning (ML) pipelines. We systematize the increasingly growing (but fragmented) microcosm of studies that develop and utilize XAI methods for defensive and offensive cybersecurity tasks. We identify 3 cybersecurity stakeholders, i.e., model users, designers, and adversaries, that utilize XAI for 5 different objectives within an ML pipeline, namely 1) XAI-enabled decision support, 2) applied XAI for security tasks, 3) model verification via XAI, 4) explanation verification & robustness, and 5) offensive use of explanations.

- Auditing Visualizations: Transparency Methods Struggle to Detect Anomalous Behavior. [[paper]](https://arxiv.org/abs/2206.13498)
  - Jean-Stanislas Denain, Jacob Steinhardt.
  - Key Word: Anomalous Models; Feature Attributions;
  - <details><summary>Digest</summary> Transparency methods such as model visualizations provide information that outputs alone might miss, since they describe the internals of neural networks. But can we trust that model explanations reflect model behavior? For instance, can they diagnose abnormal behavior such as backdoors or shape bias? To evaluate model explanations, we define a model as anomalous if it differs from a reference set of normal models, and we test whether transparency methods assign different explanations to anomalous and normal models. We find that while existing methods can detect stark anomalies such as shape bias or adversarial training, they struggle to identify more subtle anomalies such as models trained on incomplete data.

- When adversarial attacks become interpretable counterfactual explanations. [[paper]](https://arxiv.org/abs/2206.06854)
  - Mathieu Serrurier, Franck Mamalet, Thomas Fel, Louis Béthune, Thibaut Boissin.
  - Key Word: Explainability; Interpretability; Saliency Maps.
  - <details><summary>Digest</summary> We argue that, when learning a 1-Lipschitz neural network with the dual loss of an optimal transportation problem, the gradient of the model is both the direction of the transportation plan and the direction to the closest adversarial attack. Traveling along the gradient to the decision boundary is no more an adversarial attack but becomes a counterfactual explanation, explicitly transporting from one class to the other. Through extensive experiments on XAI metrics, we find that the simple saliency map method, applied on such networks, becomes a reliable explanation, and outperforms the state-of-the-art explanation approaches on unconstrained models. The proposed networks were already known to be certifiably robust, and we prove that they are also explainable with a fast and simple method.

- Which Explanation Should I Choose? A Function Approximation Perspective to Characterizing Post hoc Explanations. [[paper]](https://arxiv.org/abs/2206.01254)
  - Tessa Han, Suraj Srinivas, Himabindu Lakkaraju. *NeurIPS 2022*
  - Key Word: Analyzing Post-hoc Explanations.
  - <details><summary>Digest</summary> We adopt a function approximation perspective and formalize the local function approximation (LFA) framework. We show that popular explanation methods are instances of this framework, performing function approximations of the underlying model in different neighborhoods using different loss functions. We introduce a no free lunch theorem for explanation methods which demonstrates that no single method can perform optimally across all neighbourhoods and calls for choosing among methods.

- Concept-level Debugging of Part-Prototype Networks. [[paper]](https://arxiv.org/abs/2205.15769) [[code]](https://github.com/abonte/protopdebug)
  - Andrea Bontempelli, Stefano Teso, Fausto Giunchiglia, Andrea Passerini.
  - Key Word: Part-Prototype Networks; Concept-level Debugging.
  - <details><summary>Digest</summary> We propose ProtoPDebug, an effective concept-level debugger for ProtoPNets in which a human supervisor, guided by the model's explanations, supplies feedback in the form of what part-prototypes must be forgotten or kept, and the model is fine-tuned to align with this supervision. An extensive empirical evaluation on synthetic and real-world data shows that ProtoPDebug outperforms state-of-the-art debuggers for a fraction of the annotation cost.

- Post-hoc Concept Bottleneck Models. [[paper]](https://arxiv.org/abs/2205.15480) [[code]](https://github.com/mertyg/post-hoc-cbm)
  - Mert Yuksekgonul, Maggie Wang, James Zou.
  - Key Word: Concept Bottleneck Models; Model Editing.
  - <details><summary>Digest</summary> We address the limitations of CBMs by introducing Post-hoc Concept Bottleneck models (PCBMs). We show that we can turn any neural network into a PCBM without sacrificing model performance while still retaining interpretability benefits. When concept annotation is not available on the training data, we show that PCBM can transfer concepts from other datasets or from natural language descriptions of concepts. PCBM also enables users to quickly debug and update the model to reduce spurious correlations and improve generalization to new (potentially different) data.

- Towards Better Understanding Attribution Methods. [[paper]](https://arxiv.org/abs/2205.10435)
  - Sukrut Rao, Moritz Böhle, Bernt Schiele. *CVPR 2022*
  - Key Word: Post-hoc Attribution.
  - <details><summary>Digest</summary> Deep neural networks are very successful on many vision tasks, but hard to interpret due to their black box nature. To overcome this, various post-hoc attribution methods have been proposed to identify image regions most influential to the models' decisions. Evaluating such methods is challenging since no ground truth attributions exist. We thus propose three novel evaluation schemes to more reliably measure the faithfulness of those methods, to make comparisons between them more fair, and to make visual inspection more systematic.

- B-cos Networks: Alignment is All We Need for Interpretability. [[paper]](https://arxiv.org/abs/2205.10268) [[code]](https://github.com/moboehle/b-cos)
  - Moritz Böhle, Mario Fritz, Bernt Schiele.
  - Key Word: Weight-Input Alignment.
  - <details><summary>Digest</summary> We present a new direction for increasing the interpretability of deep neural networks (DNNs) by promoting weight-input alignment during training. For this, we propose to replace the linear transforms in DNNs by our B-cos transform. As we show, a sequence (network) of such transforms induces a single linear transform that faithfully summarises the full model computations.

- Discovering Latent Concepts Learned in BERT. [[paper]](https://arxiv.org/abs/2205.07237)
  - Fahim Dalvi, Abdul Rafae Khan, Firoj Alam, Nadir Durrani, Jia Xu, Hassan Sajjad. *ICLR 2022*
  - Key Word: Interpretability; Natural Language Processing.
  - <details><summary>Digest</summary> We study: i) what latent concepts exist in the pre-trained BERT model, ii) how the discovered latent concepts align or diverge from classical linguistic hierarchy and iii) how the latent concepts evolve across layers. Our findings show: i) a model learns novel concepts (e.g. animal categories and demographic groups), which do not strictly adhere to any pre-defined categorization (e.g. POS, semantic tags), ii) several latent concepts are based on multiple properties which may include semantics, syntax, and morphology, iii) the lower layers in the model dominate in learning shallow lexical concepts while the higher layers learn semantic relations and iv) the discovered latent concepts highlight potential biases learned in the model.

- Do Users Benefit From Interpretable Vision? A User Study, Baseline, And Dataset. [[paper]](https://arxiv.org/abs/2204.11642) [[code]](https://github.com/berleon/do_users_benefit_from_interpretable_vision)
  - Leon Sixt, Martin Schuessler, Oana-Iuliana Popescu, Philipp Weiß, Tim Landgraf. *ICLR 2022*
  - Key Word: Interpretability; Human Subject Evaluation.
  - <details><summary>Digest</summary> We assess if participants can identify the relevant set of attributes compared to the ground-truth. Our results show that the baseline outperformed concept-based explanations. Counterfactual explanations from an invertible neural network performed similarly as the baseline.

- Model Agnostic Interpretability for Multiple Instance Learning. [[paper]](https://arxiv.org/abs/2201.11701) [[code]](https://github.com/jaearly/milli)
  - Joseph Early, Christine Evers, Sarvapali Ramchurn. *ICLR 2022*
  - Key Word: Multiple Instance Learning, Interpretability.
  - <details><summary>Digest</summary> In Multiple Instance Learning (MIL), models are trained using bags of instances, where only a single label is provided for each bag. A bag label is often only determined by a handful of key instances within a bag, making it difficult to interpret what information a classifier is using to make decisions. In this work, we establish the key requirements for interpreting MIL models. We then go on to develop several model-agnostic approaches that meet these requirements.

</details>

<!-- ### Interpretability: 2021 -->

<details>
<summary>Interpretability: 2021</summary>

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

</details>

<!-- ### Interpretability: 2020 -->

<details>
<summary>Interpretability: 2020</summary>

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

</details>

<!-- ### Interpretability: 2019 -->

<details>
<summary>Interpretability: 2019</summary>

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

</details>

<!-- ### Interpretability: 2018 -->

<details>
<summary>Interpretability: 2018</summary>

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

</details>


<!-- ### Interpretability: 2017 -->

<details>
<summary>Interpretability: 2017</summary>

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

</details>

<!-- ### Interpretability: 2016 -->

<details>
<summary>Interpretability: 2016</summary>

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

</details>

<!-- ### Interpretability: 2015 -->

<details>
<summary>Interpretability: 2015</summary>

- Explaining NonLinear Classification Decisions with Deep Taylor Decomposition. [[paper]](https://arxiv.org/abs/1512.02479) [[code]](https://github.com/myc159/Deep-Taylor-Decomposition)
  - Grégoire Montavon, Sebastian Bach, Alexander Binder, Wojciech Samek, Klaus-Robert Müller.
  - Key Word: Saliency Maps; Deep Taylor Decomposition.
  - <details><summary>Digest</summary> We introduce a novel methodology for interpreting generic multilayer neural networks by decomposing the network classification decision into contributions of its input elements. Although our focus is on image classification, the method is applicable to a broad set of input data, learning tasks and network architectures.

- Learning Deep Features for Discriminative Localization. [[paper]](https://arxiv.org/abs/1512.04150) [[code]](https://github.com/frgfm/torch-cam)
  - Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, Antonio Torralba. *CVPR 2016*
  - Key Word: Saliency Maps.
  - <details><summary>Digest</summary> We revisit the global average pooling layer proposed, and shed light on how it explicitly enables the convolutional neural network to have remarkable localization ability despite being trained on image-level labels. While this technique was previously proposed as a means for regularizing training, we find that it actually builds a generic localizable deep representation that can be applied to a variety of tasks.

</details>

## Open-World Learning

<details>
<summary>Open-World Learning: 2022</summary>

- Understanding Open-Set Recognition by Jacobian Norm of Representation. [[paper]](https://arxiv.org/abs/2209.11436)
  - Jaewoo Park, Hojin Park, Eunju Jeong, Andrew Beng Jin Teoh.
  - Key Word: Open-Set Recognition; Out-of-Distribution Detection.
  - <details><summary>Digest</summary> This paper analyzes this emergent phenomenon by observing the Jacobian norm of representation. We theoretically show that minimizing the intra-class distances within the known set reduces the Jacobian norm of known class representations while maximizing the inter-class distances within the known set increases the Jacobian norm of the unknown class. The closed-set metric learning thus separates the unknown from the known by forcing their Jacobian norm values to differ. We empirically validate our theoretical framework with ample pieces of evidence using standard OSR datasets. 

- Holistic Segmentation. [[paper]](https://arxiv.org/abs/2209.05407)
  - Stefano Gasperini, Frithjof Winkelmann, Alvaro Marcos-Ramiro, Micheal Schmidt, Nassir Navab, Benjamin Busam, Federico Tombari.
  - Key Word: Open-set Panoptic Segmentation; Open-set Perception; Uncertainty Estimation.
  - <details><summary>Digest</summary> We broaden the scope proposing holistic segmentation: a task to identify and separate unseen unknown objects into instances, without learning from unknowns, while performing panoptic segmentation of known classes. We tackle this new problem with U3HS, which first finds unknowns as highly uncertain regions, then clusters the corresponding instance-aware embeddings into individual objects.

- Fine-grain Inference on Out-of-Distribution Data with Hierarchical Classification. [[paper]](https://arxiv.org/abs/2209.04493)
  - Randolph Linderman, Jingyang Zhang, Nathan Inkawhich, Hai Li, Yiran Chen.
  - Key Word: Fine-grain Out-of-Distribution Detection.
  - <details><summary>Digest</summary> When the OOD sample significantly overlaps with the training data, a binary anomaly detection is not interpretable or explainable, and provides little information to the user. We propose a new model for OOD detection that makes predictions at varying levels of granularity as the inputs become more ambiguous, the model predictions become coarser and more conservative. Consider an animal classifier that encounters an unknown bird species and a car. Both cases are OOD, but the user gains more information if the classifier recognizes that its uncertainty over the particular species is too large and predicts bird instead of detecting it as OOD.

- Measuring Human Perception to Improve Open Set Recognition. [[paper]](https://arxiv.org/abs/2209.03519)
  - Jin Huang, Student Member, Derek Prijatelj, Justin Dulay, Walter Scheirer. 
  - Key Word: Open Set Recognition; Novelty Detection; Visual Psychophysics.
  - <details><summary>Digest</summary> We designed and performed a large-scale behavioral experiment that collected over 200,000 human reaction time measurements associated with object recognition. The data collected indicated reaction time varies meaningfully across objects at the sample level. We therefore designed a new psychophysical loss function that enforces consistency with human behavior in deep networks which exhibit variable reaction time for different images. 

- Open-Set Semi-Supervised Object Detection. [[paper]](https://arxiv.org/abs/2208.13722) [[code]](https://ycliu93.github.io/projects/ossod.html)
  - Yen-Cheng Liu, Chih-Yao Ma, Xiaoliang Dai, Junjiao Tian, Peter Vajda, Zijian He, Zsolt Kira.
  - Key Word: Open-Set Semi-Supervised Learning; Object Detection.
  - <details><summary>Digest</summary> We first find the existing SSOD method obtains a lower performance gain in open-set conditions, and this is caused by the semantic expansion, where the distracting OOD objects are mispredicted as in-distribution pseudo-labels for the semi-supervised training. To address this problem, we consider online and offline OOD detection modules, which are integrated with SSOD methods. 

- Detecting the unknown in Object Detection. [[paper]](https://arxiv.org/abs/2208.11641)
  - Dario Fontanel, Matteo Tarantino, Fabio Cermelli, Barbara Caputo.
  - Key Word: Open-Set Detection; Discovery of Unknown Class.
  - <details><summary>Digest</summary> In this work, we address the problem of detecting unknown objects, known as open-set object detection. We propose a novel training strategy, called UNKAD, able to predict unknown objects without requiring any annotation of them, exploiting non annotated objects that are already present in the background of training images. In particular, exploiting the four-steps training strategy of Faster R-CNN, UNKAD first identifies and pseudo-labels unknown objects and then uses the pseudo-annotations to train an additional unknown class. While UNKAD can directly detect unknown objects, we further combine it with previous unknown detection techniques, showing that it improves their performance at no costs.

- Towards Open Set Video Anomaly Detection. [[paper]](https://arxiv.org/abs/2208.11113)
  - Yuansheng Zhu, Wentao Bao, Qi Yu. *ECCV 2022*
  - Key Word: Open-Set Recognition; Video Anomaly Detection.
  - <details><summary>Digest</summary> Open Set Video Anomaly Detection (OpenVAD) aims to identify abnormal events from video data where both known anomalies and novel ones exist in testing. Unsupervised models learned solely from normal videos are applicable to any testing anomalies but suffer from a high false positive rate. In contrast, weakly supervised methods are effective in detecting known anomalies but could fail in an open world. We develop a novel weakly supervised method for the OpenVAD problem by integrating evidential deep learning (EDL) and normalizing flows (NFs) into a multiple instance learning (MIL) framework.

- Self-Trained Proposal Networks for the Open World. [[paper]](https://arxiv.org/abs/2208.11050)
  - Matthew Inkawhich, Nathan Inkawhich, Hai Li, Yiran Chen.
  - Key Word: Self-Training; Open-Set Detection; Class-Agnostic Object Proposal.
  - <details><summary>Digest</summary> We propose a classification-free Self-Trained Proposal Network (STPN) that leverages a novel self-training optimization strategy combined with dynamically weighted loss functions that account for challenges such as class imbalance and pseudo-label uncertainty. Not only is our model designed to excel in existing optimistic open-world benchmarks, but also in challenging operating environments where there is significant label bias. To showcase this, we devise two challenges to test the generalization of proposal models when the training data contains (1) less diversity within the labeled classes, and (2) fewer labeled instances.

- Single-Stage Open-world Instance Segmentation with Cross-task Consistency Regularization. [[paper]](https://arxiv.org/abs/2208.09023)
  - Xizhe Xue, Dongdong Yu, Lingqiao Liu, Yu Liu, Ying Li, Zehuan Yuan, Ping Song, Mike Zheng Shou.
  - Key Word: Class-agnostic; Open-world Instance Segmentation; Cross-task Consistency Loss.
  - <details><summary>Digest</summary> Open-world instance segmentation (OWIS) aims to segment class-agnostic instances from images, which has a wide range of real-world applications such as autonomous driving. Most existing approaches follow a two-stage pipeline: performing class-agnostic detection first and then class-specific mask segmentation. In contrast, this paper proposes a single-stage framework to produce a mask for each instance directly. Also, instance mask annotations could be noisy in the existing datasets; to overcome this issue, we introduce a new regularization loss. Specifically, we first train an extra branch to perform an auxiliary task of predicting foreground regions, and then encourage the prediction from the auxiliary branch to be consistent with the predictions of the instance masks. The key insight is that such a cross-task consistency loss could act as an error-correcting mechanism to combat the errors in annotations.

- Open Long-Tailed Recognition in a Dynamic World. [[paper]](https://arxiv.org/abs/2208.08349)
  - Ziwei Liu, Zhongqi Miao, Xiaohang Zhan, Jiayun Wang, Boqing Gong, Stella X. Yu. *TPAMI*
  - Key Word: Long-Tailed Recognition; Few-shot Learning; Open-Set Recognition; Active Learning.
  - <details><summary>Digest</summary> Real world data often exhibits a long-tailed and open-ended (with unseen classes) distribution. A practical recognition system must balance between majority (head) and minority (tail) classes, generalize across the distribution, and acknowledge novelty upon the instances of unseen classes (open classes). We define Open Long-Tailed Recognition++ (OLTR++) as learning from such naturally distributed data and optimizing for the classification accuracy over a balanced test set which includes both known and open classes. OLTR++ handles imbalanced classification, few-shot learning, open-set recognition, and active learning in one integrated algorithm, whereas existing classification approaches often focus only on one or two aspects and deliver poorly over the entire spectrum.

- From Known to Unknown: Quality-aware Self-improving Graph Neural Network for Open Set Social Event Detection. [[paper]](https://arxiv.org/abs/2208.06973) [[code]](https://github.com/RingBDStack/open-set-social-event-detection)
  - Jiaqian Ren, Lei Jiang, Hao Peng, Yuwei Cao, Jia Wu, Philip S. Yu, Lifang He.
  - Key Word: Open-set Social Event Detection; Graph Neural Network; Classification.
  - <details><summary>Digest</summary> To address this problem, we design a Quality-aware Self-improving Graph Neural Network (QSGNN) which extends the knowledge from known to unknown by leveraging the best of known samples and reliable knowledge transfer. Specifically, to fully exploit the labeled data, we propose a novel supervised pairwise loss with an additional orthogonal inter-class relation constraint to train the backbone GNN encoder. The learnt, already-known events further serve as strong reference bases for the unknown ones, which greatly prompts knowledge acquisition and transfer. When the model is generalized to unknown data, to ensure the effectiveness and reliability, we further leverage the reference similarity distribution vectors for pseudo pairwise label generation, selection and quality assessment. Besides, we propose a novel quality-guided optimization in which the contributions of pseudo labels are weighted based on consistency.

- Open-world Contrastive Learning. [[paper]](https://arxiv.org/abs/2208.02764)
  - Yiyou Sun, Yixuan Li.
  - Key Word: Contrastive learning; Open-world; Classification.
  - <details><summary>Digest</summary> In this paper, we enrich the landscape of representation learning by tapping into an open-world setting, where unlabeled samples from novel classes can naturally emerge in the wild. To bridge the gap, we introduce a new learning framework, open-world contrastive learning (OpenCon). OpenCon tackles the challenges of learning compact representations for both known and novel classes, and facilitates novelty discovery along the way. We demonstrate the effectiveness of OpenCon on challenging benchmark datasets and establish competitive performance.

- Few-Shot Class-Incremental Learning from an Open-Set Perspective. [[paper]](https://arxiv.org/abs/2208.00147) [[code]](https://github.com/canpeng123/fscil_alice)
  - Can Peng, Kun Zhao, Tianren Wang, Meng Li, Brian C. Lovell. *ECCV 2022*
  - Key Word: Few-shot Class-Incremental Learning; Open-set; One-shot; Classification.
  - <details><summary>Digest</summary> Here we explore the important task of Few-Shot Class-Incremental Learning (FSCIL) and its extreme data scarcity condition of one-shot. An ideal FSCIL model needs to perform well on all classes, regardless of their presentation order or paucity of data. It also needs to be robust to open-set real-world conditions and be easily adapted to the new tasks that always arise in the field. In this paper, we first reevaluate the current task setting and propose a more comprehensive and practical setting for the FSCIL task. Then, inspired by the similarity of the goals for FSCIL and modern face recognition systems, we propose our method -- Augmented Angular Loss Incremental Classification or ALICE. In ALICE, instead of the commonly used cross-entropy loss, we propose to use the angular penalty loss to obtain well-clustered features. As the obtained features not only need to be compactly clustered but also diverse enough to maintain generalization for future incremental classes, we further discuss how class augmentation, data augmentation, and data balancing affect classification performance.

- Open World Learning Graph Convolution for Latency Estimation in Routing Networks. [[paper]](https://arxiv.org/abs/2207.14643)
  - Yifei Jin, Marios Daoutis, Sarunas Girdzijauskas, Aristides Gionis. *IJCNN 2022*
  - Key Word: Open-world Learning; Modeling Network Routing; Software Defined Networking.
  - <details><summary>Digest</summary> Accurate routing network status estimation is a key component in Software Defined Networking. We propose a novel approach for modeling network routing, using Graph Neural Networks. Our method can also be used for network-latency estimation. Supported by a domain-knowledge-assisted graph formulation, our model shares a stable performance across different network sizes and configurations of routing networks, while at the same time being able to extrapolate towards unseen sizes, configurations, and user behavior. We show that our model outperforms most conventional deep-learning-based models, in terms of prediction accuracy, computational resources, inference speed, as well as ability to generalize towards open-world input.

- Visual Recognition by Request. [[paper]](https://arxiv.org/abs/2207.14227) [[code]](https://github.com/chufengt/Visual-Recognition-by-Request)
  - Chufeng Tang, Lingxi Xie, Xiaopeng Zhang, Xiaolin Hu, Qi Tian.
  - Key Word: Visual Recognition by Request; Open-domain; Knowledge Base.
  - <details><summary>Digest</summary> In this paper, we present a novel protocol of annotation and evaluation for visual recognition. Different from traditional settings, the protocol does not require the labeler/algorithm to annotate/recognize all targets (objects, parts, etc.) at once, but instead raises a number of recognition instructions and the algorithm recognizes targets by request. This mechanism brings two beneficial properties to reduce the burden of annotation, namely, (i) variable granularity: different scenarios can have different levels of annotation, in particular, object parts can be labeled only in large and clear instances, (ii) being open-domain: new concepts can be added to the database in minimal costs. To deal with the proposed setting, we maintain a knowledge base and design a query-based visual recognition framework that constructs queries on-the-fly based on the requests. We evaluate the recognition system on two mixed-annotated datasets, CPP and ADE20K, and demonstrate its promising ability of learning from partially labeled data as well as adapting to new concepts with only text labels.

- Towards Open Set 3D Learning: A Benchmark on Object Point Clouds. [[paper]](https://arxiv.org/abs/2207.11554) [[code]](https://github.com/antoalli/3d_os)
  - Antonio Alliegro, Francesco Cappio Borlino, Tatiana Tommasi.
  - Key Word: Open-set 3D Learning; In-domain and Cross-domain; Out-of-distribution.
  - <details><summary>Digest</summary> In this context exploiting 3D data can be a valuable asset since it conveys rich information about the geometry of sensed objects and scenes. This paper provides the first broad study on Open Set 3D learning. We introduce a novel testbed with settings of increasing difficulty in terms of category semantic shift and cover both in-domain (synthetic-to-synthetic) and cross-domain (synthetic-to-real) scenarios. Moreover, we investigate the related out-of-distribution and Open Set 2D literature to understand if and how their most recent approaches are effective on 3D data. Our extensive benchmark positions several algorithms in the same coherent picture, revealing their strengths and limitations.

- Semantic Abstraction: Open-World 3D Scene Understanding from 2D Vision-Language Models. [[paper]](https://arxiv.org/abs/2207.11514) [[code]](https://semantic-abstraction.cs.columbia.edu/)
  - Huy Ha, Shuran Song.
  - Key Word: Open-set Vocabulary; 3D scene understanding; Zero-shot.
  - <details><summary>Digest</summary> We study open-world 3D scene understanding, a family of tasks that require agents to reason about their 3D environment with an open-set vocabulary and out-of-domain visual inputs - a critical skill for robots to operate in the unstructured 3D world. Towards this end, we propose Semantic Abstraction (SemAbs), a framework that equips 2D Vision-Language Models (VLMs) with new 3D spatial capabilities, while maintaining their zero-shot robustness. We achieve this abstraction using relevancy maps extracted from CLIP, and learn 3D spatial and geometric reasoning skills on top of those abstractions in a semantic-agnostic manner. We demonstrate the usefulness of SemAbs on two open-world 3D scene understanding tasks: 1) completing partially observed objects and 2) localizing hidden objects from language descriptions.

- UC-OWOD: Unknown-Classified Open World Object Detection. [[paper]](https://arxiv.org/abs/2207.11455) [[code]](https://github.com/JohnWuzh/UC-OWOD)
  - Quanshi Zhang, Xin Wang, Jie Ren, Xu Cheng, Shuyun Lin, Yisen Wang, Xiangming Zhu. *ECCV 2022*
  - Key Word: Open World Object Detection.
  - <details><summary>Digest</summary> Open World Object Detection (OWOD) is a challenging computer vision problem that requires detecting unknown objects and gradually learning the identified unknown classes. However, it cannot distinguish unknown instances as multiple unknown classes. In this work, we propose a novel OWOD problem called Unknown-Classified Open World Object Detection (UC-OWOD). UC-OWOD aims to detect unknown instances and classify them into different unknown classes. Besides, we formulate the problem and devise a two-stage object detector to solve UC-OWOD.

- Difficulty-Aware Simulator for Open Set Recognition. [[paper]](https://arxiv.org/abs/2207.10024) [[code]](https://github.com/wjun0830/difficulty-aware-simulator)
  - WonJun Moon, Junho Park, Hyun Seok Seong, Cheol-Ho Cho, Jae-Pil Heo. *ECCV 2022*
  - Key Word: Open-set Recognition; Generative Adversarial Network.
  - <details><summary>Digest</summary> We present a novel framework, DIfficulty-Aware Simulator (DIAS), that generates fakes with diverse difficulty levels to simulate the real world. We first investigate fakes from generative adversarial network (GAN) in the classifier's viewpoint and observe that these are not severely challenging. This leads us to define the criteria for difficulty by regarding samples generated with GANs having moderate-difficulty. To produce hard-difficulty examples, we introduce Copycat, imitating the behavior of the classifier. Furthermore, moderate- and easy-difficulty samples are also yielded by our modified GAN and Copycat, respectively.

- More Practical Scenario of Open-set Object Detection: Open at Category Level and Closed at Super-category Level. [[paper]](https://arxiv.org/abs/2207.09775)
  - Yusuke Hosoya, Masanori Suganuma, Takayuki Okatani.
  - Key Word: Open-set Object Detection; Super-category.
  - <details><summary>Digest</summary> We first point out that the scenario of OSOD considered in recent studies, which considers an unlimited variety of unknown objects similar to open-set recognition (OSR), has a fundamental issue. That is, we cannot determine what to detect and what not for such unlimited unknown objects, which is necessary for detection tasks. This issue leads to difficulty with the evaluation of methods' performance on unknown object detection. We then introduce a novel scenario of OSOD, which deals with only unknown objects that share the super-category with known objects. It has many real-world applications, e.g., detecting an increasing number of fine-grained objects. This new setting is free from the above issue and evaluation difficulty. Moreover, it makes detecting unknown objects more realistic owing to the visual similarity between known and unknown objects.

- DenseHybrid: Hybrid Anomaly Detection for Dense Open-set Recognition. [[paper]](https://arxiv.org/abs/2207.02606) [[code]](https://github.com/matejgrcic/DenseHybrid)
  - Matej Grcić, Petra Bevandić, Siniša Šegvić. *ECCV 2022*
  - Key Word: Anomaly detection; Dense anomaly detection; Open-set Recognition.
  - <details><summary>Digest</summary> We design a novel hybrid algorithm based on reinterpreting discriminative logits as a logarithm of the unnormalized joint distribution p̂ (x,y). Our model builds on a shared convolutional representation from which we recover three dense predictions: i) the closed-set class posterior P(y|x), ii) the dataset posterior P(din|x), iii) unnormalized data likelihood p̂ (x). The latter two predictions are trained both on the standard training data and on a generic negative dataset. We blend these two predictions into a hybrid anomaly score which allows dense open-set recognition on large natural images. We carefully design a custom loss for the data likelihood in order to avoid backpropagation through the untractable normalizing constant Z(θ). Experiments evaluate our contributions on standard dense anomaly detection benchmarks as well as in terms of open-mIoU - a novel metric for dense open-set performance.

- Towards Realistic Semi-Supervised Learning. [[paper]](https://arxiv.org/abs/2207.02269) [[code]](https://github.com/nayeemrizve/trssl)
  - Mamshad Nayeem Rizve, Navid Kardan, Mubarak Shah. *ECCV 2022 oral*
  - Key Word: Semi-supervised Learning; Open-world SSL; Discovery of Unknown class.
  - <details><summary>Digest</summary> The standard SSL approach assumes unlabeled data are from the same distribution as annotated data. Recently, a more realistic SSL problem, called open-world SSL, is introduced, where the unannotated data might contain samples from unknown classes. In this paper, we propose a novel pseudo-label based approach to tackle SSL in open-world setting. At the core of our method, we utilize sample uncertainty and incorporate prior knowledge about class distribution to generate reliable class-distribution-aware pseudo-labels for unlabeled data belonging to both known and unknown classes. We also highlight the flexibility of our approach in solving novel class discovery task, demonstrate its stability in dealing with imbalanced data, and complement our approach with a technique to estimate the number of novel classes.

- Open-world Semantic Segmentation for LIDAR Point Clouds. [[paper]](https://arxiv.org/abs/2207.01452) [[code]](https://github.com/jun-cen/open_world_3d_semantic_segmentation)
  - Jun Cen, Peng Yun, Shiwei Zhang, Junhao Cai, Di Luan, Michael Yu Wang, Ming Liu, Mingqian Tang. *ECCV 2022*
  - Key Word: Open-world Semantic Segmentation; LIDAR Point Clouds; Incremental Learning.
  - <details><summary>Digest</summary> In this work, we propose the open-world semantic segmentation task for LIDAR point clouds, which aims to 1) identify both old and novel classes using open-set semantic segmentation, and 2) gradually incorporate novel objects into the existing knowledge base using incremental learning without forgetting old classes. For this purpose, we propose a REdundAncy cLassifier (REAL) framework to provide a general architecture for both the open-set semantic segmentation and incremental learning problems.

- RegMixup: Mixup as a Regularizer Can Surprisingly Improve Accuracy and Out Distribution Robustness. [[paper]](https://arxiv.org/abs/2206.14502) [[code]](https://github.com/FrancescoPinto/RegMixup)
  - Francesco Pinto, Harry Yang, Ser-Nam Lim, Philip H.S. Torr, Puneet K. Dokania. *NeurIPS 2022*
  - Key Word: Mixup; Out-of-Distribution Detection; Out-of-Distribution Robustness.
  - <details><summary>Digest</summary> We observe that Mixup yields much degraded performance on detecting out-of-distribution samples possibly, as we show empirically, because of its tendency to learn models that exhibit high-entropy throughout; making it difficult to differentiate in-distribution samples from out-distribution ones. To show the efficacy of our approach (RegMixup), we provide thorough analyses and experiments on vision datasets (ImageNet & CIFAR-10/100) and compare it with a suite of recent approaches for reliable uncertainty estimation.

- Open Vocabulary Object Detection with Proposal Mining and Prediction Equalization. [[paper]](https://arxiv.org/abs/2206.11134) [[code]](https://github.com/pealing/medet)
  - Peixian Chen, Kekai Sheng, Mengdan Zhang, Yunhang Shen, Ke Li, Chunhua Shen.
  - Key Word: Open-vocabulary Object Detection; Backdoor Adjustment.
  - <details><summary>Digest</summary> Open-vocabulary object detection (OVD) aims to scale up vocabulary size to detect objects of novel categories beyond the training vocabulary. We present MEDet, a novel and effective OVD framework with proposal mining and prediction equalization. First, we design an online proposal mining to refine the inherited vision-semantic knowledge from coarse to fine, allowing for proposal-level detection-oriented feature alignment. Second, based on causal inference theory, we introduce a class-wise backdoor adjustment to reinforce the predictions on novel categories to improve the overall OVD performance.

- Rethinking the Openness of CLIP. [[paper]](https://arxiv.org/abs/2206.01986)
  - Shuhuai Ren, Lei Li, Xuancheng Ren, Guangxiang Zhao, Xu Sun.
  - Key Word: Open-vocabulary; CLIP; Rethinking; In-depth Analysis.
  - <details><summary>Digest</summary> Contrastive Language-Image Pre-training (CLIP) has demonstrated great potential in realizing open-vocabulary image classification in a matching style, because of its holistic use of natural language supervision that covers unconstrained real-world visual concepts. However, it is, in turn, also difficult to evaluate and analyze the openness of CLIP-like models, since they are in theory open to any vocabulary but the actual accuracy varies. To address the insufficiency of conventional studies on openness, we resort to an incremental view and define the extensibility, which essentially approximates the model's ability to deal with new visual concepts, by evaluating openness through vocabulary expansions. Our evaluation based on extensibility shows that CLIP-like models are hardly truly open and their performances degrade as the vocabulary expands to different degrees. Further analysis reveals that the over-estimation of openness is not because CLIP-like models fail to capture the general similarity of image and text features of novel visual concepts, but because of the confusion among competing text features, that is, they are not stable with respect to the vocabulary. In light of this, we propose to improve the openness of CLIP from the perspective of feature space by enforcing the distinguishability of text features. Our method retrieves relevant texts from the pre-training corpus to enhance prompts for inference, which boosts the extensibility and stability of CLIP even without fine-tuning.

- Simple Open-Vocabulary Object Detection with Vision Transformers. [[paper]](https://arxiv.org/abs/2205.06230) [[code]](https://github.com/google-research/scenic)
  - Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, Neil Houlsby. *ECCV 2022*
  - Key Word: Open-vocabulary; Long-tail; Object detection; Vision Transformer.
  - <details><summary>Digest</summary> For object detection, pre-training and scaling approaches are less well established, especially in the long-tailed and open-vocabulary setting, where training data is relatively scarce. In this paper, we propose a strong recipe for transferring image-text models to open-vocabulary object detection. We use a standard Vision Transformer architecture with minimal modifications, contrastive image-text pre-training, and end-to-end detection fine-tuning. Our analysis of the scaling properties of this setup shows that increasing image-level pre-training and model size yield consistent improvements on the downstream detection task. We provide the adaptation strategies and regularizations needed to attain very strong performance on zero-shot text-conditioned and one-shot image-conditioned object detection.

- OSSGAN: Open-Set Semi-Supervised Image Generation. [[paper]](https://arxiv.org/abs/2204.14249) [[code]](https://github.com/raven38/ossgan)
  - Kai Katsumata, Duc Minh Vo, Hideki Nakayama. *CVPR 2022*
  - Key Word: Open-set Semi-supervised Image Generation; Conditional GAN.
  - <details><summary>Digest</summary> We introduce a challenging training scheme of conditional GANs, called open-set semi-supervised image generation, where the training dataset consists of two parts: (i) labeled data and (ii) unlabeled data with samples belonging to one of the labeled data classes, namely, a closed-set, and samples not belonging to any of the labeled data classes, namely, an open-set. Unlike the existing semi-supervised image generation task, where unlabeled data only contain closed-set samples, our task is more general and lowers the data collection cost in practice by allowing open-set samples to appear. Thanks to entropy regularization, the classifier that is trained on labeled data is able to quantify sample-wise importance to the training of cGAN as confidence, allowing us to use all samples in unlabeled data.

- Open-World Instance Segmentation: Exploiting Pseudo Ground Truth From Learned Pairwise Affinity. [[paper]](https://arxiv.org/abs/2204.06107) [[code]](https://sites.google.com/view/generic-grouping/)
  - Weiyao Wang, Matt Feiszli, Heng Wang, Jitendra Malik, Du Tran. *CVPR 2022*
  - Key Word: Open-world Instance Segmentation; Generic Grouping Networks; Pairwise Affinities.
  - <details><summary>Digest</summary> Open-world instance segmentation is the task of grouping pixels into object instances without any pre-determined taxonomy. This is challenging, as state-of-the-art methods rely on explicit class semantics obtained from large labeled datasets, and out-of-domain evaluation performance drops significantly. Here we propose a novel approach for mask proposals, Generic Grouping Networks (GGNs), constructed without semantic supervision. Our approach combines a local measure of pixel affinity with instance-level mask supervision, producing a training regimen designed to make the model as generic as the data diversity allows. We introduce a method for predicting Pairwise Affinities (PA), a learned local relationship between pairs of pixels. PA generalizes very well to unseen categories. From PA we construct a large set of pseudo-ground-truth instance masks; combined with human-annotated instance masks we train GGNs and significantly outperform the SOTA on open-world instance segmentation on various benchmarks including COCO, LVIS, ADE20K, and UVO.

- Full-Spectrum Out-of-Distribution Detection. [[paper]](https://arxiv.org/abs/2204.05306) [[code]](https://github.com/jingkang50/openood)
  - Jingkang Yang, Kaiyang Zhou, Ziwei Liu.
  - Key Word: Benchmark; Anomaly Detection; Open-set Recognition; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> We take into account both shift types and introduce full-spectrum OOD (FS-OOD) detection, a more realistic problem setting that considers both detecting semantic shift and being tolerant to covariate shift; and designs three benchmarks. These new benchmarks have a more fine-grained categorization of distributions (i.e., training ID, covariate-shifted ID, near-OOD, and far-OOD) for the purpose of more comprehensively evaluating the pros and cons of algorithms.

- FS6D: Few-Shot 6D Pose Estimation of Novel Objects. [[paper]](https://arxiv.org/abs/2203.14628) [[code]](https://github.com/ethnhe/FS6D-PyTorch)
  - Yisheng He, Yao Wang, Haoqiang Fan, Jian Sun, Qifeng Chen. *CVPR 2022*
  - Key Word: Open-World 6D Pose Estimation; Few-shot learning.
  - <details><summary>Digest</summary> In this work, we study a new open set problem; the few-shot 6D object poses estimation: estimating the 6D pose of an unknown object by a few support views without extra training. We point out the importance of fully exploring the appearance and geometric relationship between the given support views and query scene patches and propose a dense prototypes matching framework by extracting and matching dense RGBD prototypes with transformers. Moreover, we show that the priors from diverse appearances and shapes are crucial to the generalization capability and thus propose a large-scale RGBD photorealistic dataset (ShapeNet6D) for network pre-training. A simple and effective online texture blending approach is also introduced to eliminate the domain gap from the synthesis dataset, which enriches appearance diversity at a low cost.

- PMAL: Open Set Recognition via Robust Prototype Mining. [[paper]](https://arxiv.org/abs/2203.08569) [[code]](https://github.com/Cogito2012/OpenTAL)
  - Jing Lu, Yunxu Xu, Hao Li, Zhanzhan Cheng, Yi Niu. *AAAI 2022*
  - Key Word: Open-set Recognition; Prototype Learning.
  - <details><summary>Digest</summary> In this work, we propose a novel Prototype Mining And Learning (PMAL) framework. It has a prototype mining mechanism before the phase of optimizing embedding space, explicitly considering two crucial properties, namely high-quality and diversity of the prototype set. Concretely, a set of high-quality candidates are firstly extracted from training samples based on data uncertainty learning, avoiding the interference from unexpected noise. Considering the multifarious appearance of objects even in a single category, a diversity-based strategy for prototype set filtering is proposed. Accordingly, the embedding space can be better optimized to discriminate therein the predefined classes and between known and unknowns.

- OpenTAL: Towards Open Set Temporal Action Localization. [[paper]](https://arxiv.org/abs/2203.05114) [[code]](https://github.com/Cogito2012/OpenTAL)
  - Wentao Bao, Qi Yu, Yu Kong. *CVPR 2022*
  - Key Word: Open-set Temporal Action Localization; Temporal Action Localization; Evidential Deep Learning.
  - <details><summary>Digest</summary> In this paper, we, for the first time, step toward the Open Set TAL (OSTAL) problem and propose a general framework OpenTAL based on Evidential Deep Learning (EDL). Specifically, the OpenTAL consists of uncertainty-aware action classification, actionness prediction, and temporal location regression. With the proposed importance-balanced EDL method, classification uncertainty is learned by collecting categorical evidence majorly from important samples. To distinguish the unknown actions from background video frames, the actionness is learned by the positive-unlabeled learning. The classification uncertainty is further calibrated by leveraging the guidance from the temporal localization quality. The OpenTAL is general to enable existing TAL models for open set scenarios.

</details>

<!-- ### Open-World Learning: 2021 -->

<details>
<summary>Open-World Learning: 2021</summary>

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

</details>

## Environmental Well-being

<details>
<summary>Environmental Well-being: 2022</summary>

- Green Learning: Introduction, Examples and Outlook. [[paper]](https://arxiv.org/abs/2210.00965)
  - C.-C. Jay Kuo, Azad M. Madni.
  - Key Word: Green Learning, Trust Learning.
  - <details><summary>Digest</summary> Rapid advances in artificial intelligence (AI) in the last decade have largely been built upon the wide applications of deep learning (DL). However, the high carbon footprint yielded by larger and larger DL networks becomes a concern for sustainability. Furthermore, DL decision mechanism is somewhat obsecure and can only be verified by test data. Green learning (GL) has been proposed as an alternative paradigm to address these concerns. GL is characterized by low carbon footprints, small model sizes, low computational complexity, and logical transparency. It offers energy-effective solutions in cloud centers as well as mobile/edge devices. GL also provides a clear and logical decision-making process to gain people's trust. 

- Measuring the Carbon Intensity of AI in Cloud Instances. [[paper]](https://arxiv.org/abs/2206.05229)
  - Jesse Dodge, Taylor Prewitt, Remi Tachet Des Combes, Erika Odmark, Roy Schwartz, Emma Strubell, Alexandra Sasha Luccioni, Noah A. Smith, Nicole DeCario, Will Buchanan. *FAccT 2022*
  - Key Word: Carbon Emissions; Cloud.
  - <details><summary>Digest</summary> We provide a framework for measuring software carbon intensity, and propose to measure operational carbon emissions by using location-based and time-specific marginal emissions data per energy unit. We provide measurements of operational software carbon intensity for a set of modern models for natural language processing and computer vision, and a wide range of model sizes, including pretraining of a 6.1 billion parameter language model.

- The Carbon Footprint of Machine Learning Training Will Plateau, Then Shrink. [[paper]](https://arxiv.org/abs/2204.05149)
  - David Patterson, Joseph Gonzalez, Urs Hölzle, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, Jeff Dean.
  - Key Word: Carbon Footprint.
  - <details><summary>Digest</summary> We recommend that ML papers include emissions explicitly to foster competition on more than just model quality. Estimates of emissions in papers that omitted them have been off 100x-100,000x, so publishing emissions has the added benefit of ensuring accurate accounting. Given the importance of climate change, we must get the numbers right to make certain that we work on its biggest challenges.

</details>

<!-- ### Environmental Well-being: 2021 -->

<details>
<summary>Environmental Well-being: 2021</summary>

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

</details>

<!-- ### Environmental Well-being: 2020 -->

<details>
<summary>Environmental Well-being: 2020</summary>

- Can Federated Learning Save The Planet? [[paper]](https://arxiv.org/abs/2010.06537)
  - Xinchi Qiu, Titouan Parcollet, Daniel J. Beutel, Taner Topal, Akhil Mathur, Nicholas D. Lane.
  - Key Word: Federated Learning; Carbon Footprint.
  - <details><summary>Digest</summary> This paper offers the first-ever systematic study of the carbon footprint of FL. First, we propose a rigorous model to quantify the carbon footprint, hence facilitating the investigation of the relationship between FL design and carbon emissions. Then, we compare the carbon footprint of FL to traditional centralized learning. Our findings show FL, despite being slower to converge, can be a greener technology than data center GPUs. Finally, we highlight and connect the reported results to the future challenges and trends in FL to reduce its environmental impact, including algorithms efficiency, hardware capabilities, and stronger industry transparency.

- Carbontracker: Tracking and Predicting the Carbon Footprint of Training Deep Learning Models. [[paper]](https://arxiv.org/abs/2007.03051) [[code]](https://github.com/lfwa/carbontracker/)
  - Lasse F. Wolff Anthony, Benjamin Kanding, Raghavendra Selvan.
  - Key Word: Carbon Footprint.
  - <details><summary>Digest</summary> We present Carbontracker, a tool for tracking and predicting the energy and carbon footprint of training DL models. We propose that energy and carbon footprint of model development and training is reported alongside performance metrics using tools like Carbontracker. We hope this will promote responsible computing in ML and encourage research into energy-efficient deep neural networks.

</details>

<!-- ### Environmental Well-being: 2019 -->

<details>
<summary>Environmental Well-being: 2019</summary>

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

</details>

## Interactions with Blockchain

<details>
<summary>Interactions with Blockchain: 2022</summary>

- Trustworthy Federated Learning via Blockchain. [[paper]](https://arxiv.org/abs/2209.04418)
  - Zhanpeng Yang, Yuanming Shi, Yong Zhou, Zixin Wang, Kai Yang.
  - Key Word: Federated Learning; Blockchain.
  - <details><summary>Digest</summary> We shall propose a decentralized blockchain based FL (B-FL) architecture by using a secure global aggregation algorithm to resist malicious devices, and deploying practical Byzantine fault tolerance consensus protocol with high effectiveness and low energy consumption among multiple edge servers to prevent model tampering from the malicious server. However, to implement B-FL system at the network edge, multiple rounds of cross-validation in blockchain consensus protocol will induce long training latency. We thus formulate a network optimization problem that jointly considers bandwidth and power allocation for the minimization of long-term average training latency consisting of progressive learning rounds.

- A Fast Blockchain-based Federated Learning Framework with Compressed Communications. [[paper]](https://arxiv.org/abs/2208.06095)
  - Laizhong Cui, Xiaoxin Su, Yipeng Zhou. *JSAC*
  - Key Word: Blockchain-based Federated Learning.
  - <details><summary>Digest</summary> To improve the practicality of BFL, we are among the first to propose a fast blockchain-based communication-efficient federated learning framework by compressing communications in BFL, called BCFL. Meanwhile, we derive the convergence rate of BCFL with non-convex loss. To maximize the final model accuracy, we further formulate the problem to minimize the training loss of the convergence rate subject to a limited training time with respect to the compression rate and the block generation rate, which is a bi-convex optimization problem and can be efficiently solved.

- BPFISH: Blockchain and Privacy-preserving FL Inspired Smart Healthcare. [[paper]](https://arxiv.org/abs/2207.11654)
  - Moirangthem Biken Singh, Ajay Pratap.
  - Key Word: Blockchain; Federated Learning; Stable Matching; Differential Privacy; Smart Healthcare.
  - <details><summary>Digest</summary> This paper proposes Federated Learning (FL) based smar t healthcare system where Medical Centers (MCs) train the local model using the data collected from patients and send the model weights to the miners in a blockchain-based robust framework without sharing raw data, keeping privacy preservation into deliberation. We formulate an optimization problem by maximizing the utility and minimizing the loss function considering energy consumption and FL process delay of MCs for learning effective models on distributed healthcare data underlying a blockchain-based framework.

- BEAS: Blockchain Enabled Asynchronous & Secure Federated Machine Learning. [[paper]](https://arxiv.org/abs/2202.02817) [[code]](https://github.com/harpreetvirkk/BEAS)
  - Arup Mondal, Harpreet Virk, Debayan Gupta.
  - Key Word: Arup Mondal, Harpreet Virk, Debayan Gupta.
  - <details><summary>Digest</summary> Federated Learning (FL) enables multiple parties to distributively train a ML model without revealing their private datasets. However, it assumes trust in the centralized aggregator which stores and aggregates model updates. This makes it prone to gradient tampering and privacy leakage by a malicious aggregator. Malicious parties can also introduce backdoors into the joint model by poisoning the training data or model gradients. To address these issues, we present BEAS, the first blockchain-based framework for N-party FL that provides strict privacy guarantees of training data using gradient pruning (showing improved differential privacy compared to existing noise and clipping based techniques).

</details>

<!-- ### Interactions with Blockchain: 2021 -->

<details>
<summary>Interactions with Blockchain: 2021</summary>


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

</details>

<!-- ### Interactions with Blockchain: 2020 -->

<details>
<summary>Interactions with Blockchain: 2020</summary>

- When Federated Learning Meets Blockchain: A New Distributed Learning Paradigm. [[paper]](https://arxiv.org/abs/2009.09338)
  - Chuan Ma, Jun Li, Ming Ding, Long Shi, Taotao Wang, Zhu Han, H. Vincent Poor.
  - Key Word: Federated Leraning; Blockchain.
  - <details><summary>Digest</summary> This work investigates a blockchain assisted decentralized FL (BLADE-FL) framework, which can well prevent the malicious clients from poisoning the learning process, and further provides a self-motivated and reliable learning environment for clients. In detail, the model aggregation process is fully decentralized and the tasks of training for FL and mining for blockchain are integrated into each participant.

- A Blockchain-based Decentralized Federated Learning Framework with Committee Consensus. [[paper]](https://arxiv.org/abs/2004.00773)
  - Yuzheng Li, Chuan Chen, Nan Liu, Huawei Huang, Zibin Zheng, Qiang Yan.
  - Key Word: Blockchain; Smart Contracts; Federated Learning.
  - <details><summary>Digest</summary> Federated learning has been widely studied and applied to various scenarios. In mobile computing scenarios, federated learning protects users from exposing their private data, while cooperatively training the global model for a variety of real-world applications. However, the security of federated learning is increasingly being questioned, due to the malicious clients or central servers' constant attack to the global model or user privacy data. To address these security issues, we proposed a decentralized federated learning framework based on blockchain, i.e., a Blockchain-based Federated Learning framework with Committee consensus (BFLC). The framework uses blockchain for the global model storage and the local model update exchange.

</details>

<!-- ### Interactions with Blockchain: 2019 -->

<details>
<summary>Interactions with Blockchain: 2019</summary>

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

</details>

## Others

<details>
<summary>Others: 2022</summary>

- Advances, challenges and opportunities in creating data for trustworthy AI. [[paper]](https://www.nature.com/articles/s42256-022-00516-1)
  - Weixin Liang, Girmaw Abebe Tadesse, Daniel Ho, Fei-Fei Li, Matei Zaharia, Ce Zhang, James Zou. *Nature Machine Intelligence*
  - Key Word: Trustworthy AI; Data Design; Data Sculpting; Data Strategies; Data Policy.
  - <details><summary>Digest</summary> Automated AI model builders that are publicly available can now achieve top performance in many applications. In contrast, the design and sculpting of the data used to develop AI often rely on bespoke manual work, and they critically affect the trustworthiness of the model. This Perspective discusses key considerations for each stage of the data-for-AI pipeline—starting from data design to data sculpting (for example, cleaning, valuation and annotation) and data evaluation—to make AI more reliable.

- Efficient Conditionally Invariant Representation Learning. [[paper]](https://arxiv.org/abs/2212.08645) [[code]](https://github.com/namratadeka/circe)
  - Roman Pogodin, Namrata Deka, Yazhe Li, Danica J. Sutherland, Victor Veitch, Arthur Gretton.
  - Key Word: Fairness; Domain Invariant Learning; Causal Representation Learning.
  - <details><summary>Digest</summary> We introduce the Conditional Independence Regression CovariancE (CIRCE), a measure of conditional independence for multivariate continuous-valued variables. CIRCE applies as a regularizer in settings where we wish to learn neural features φ(X) of data X to estimate a target Y, while being conditionally independent of a distractor Z given Y. Both Z and Y are assumed to be continuous-valued but relatively low dimensional, whereas X and its features may be complex and high dimensional.

- Navigating causal deep learning. [[paper]](https://arxiv.org/abs/2212.00911)
  - Jeroen Berrevoets, Krzysztof Kacprzyk, Zhaozhi Qian, Mihaela van der Schaar.
  - Key Word: Causality.
  - <details><summary>Digest</summary> This is a living paper that categorises methods in causal deep learning beyond Pearl's ladder of causation. We refine the rungs in Pearl's ladder, while also adding a separate dimension that categorises the parametric assumptions of both input and representation, arriving at the map of causal deep learning. Our map covers machine learning disciplines such as supervised learning, reinforcement learning, generative modelling and beyond. 

- Editing Models with Task Arithmetic. [[paper]](https://arxiv.org/abs/2212.04089) [[code]](https://github.com/mlfoundations/task_vectors)
  - Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi.
  - Key Word: Model Editting; Forgetting; Domain Adaptation; Fine-Tuning; Weight Averaging; Loss Landsacpe.
  - <details><summary>Digest</summary> We propose a new paradigm for steering the behavior of neural networks, centered around task vectors. A task vector specifies a direction in the weight space of a pre-trained model, such that movement in that direction improves performance on the task. We build task vectors by subtracting the weights of a pre-trained model from the weights of the same model after fine-tuning on a task. We show that these task vectors can be modified and combined together through arithmetic operations such as negation and addition, and the behavior of the resulting model is steered accordingly. 

- PAC Verification of Statistical Algorithms. [[paper]](https://arxiv.org/abs/2211.17096)
  - Saachi Mutreja, Jonathan Shafer.  
  - Key Word: PAC Verification.
  - <details><summary>Digest</summary> We develop this notion further in a number of ways. First, we prove a lower bound for PAC verification of Ω(d−−√) i.i.d. samples for hypothesis classes of VC dimension d. Second, we present a protocol for PAC verification of unions of intervals over R that improves upon their proposed protocol for that task, and matches our lower bound. Third, we introduce a natural generalization of their definition to verification of general statistical algorithms, which is applicable to a wider variety of practical algorithms beyond agnostic PAC learning. 

- Holistic Evaluation of Language Models. [[paper]](https://arxiv.org/abs/2211.09110) [[code]](https://github.com/stanford-crfm/helm)
  - Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda.
  - Key Word: Calibration and Uncertainty; Robustness; Fairness; Bias and Stereotype; Toxicity.
  - <details><summary>Digest</summary> Language models (LMs) are becoming the foundation for almost all major language technologies, but their capabilities, limitations, and risks are not well understood. We present Holistic Evaluation of Language Models (HELM) to improve the transparency of language models.

- DC-Check: A Data-Centric AI checklist to guide the development of reliable machine learning systems. [[paper]](https://arxiv.org/abs/2211.05764)
  - Nabeel Seedat, Fergus Imrie, Mihaela van der Schaar.
  - Key Word: Data-centric AI; ML Pipelines, Reliable ML.
  - <details><summary>Digest</summary> We propose DC-Check, an actionable checklist-style framework to elicit data-centric considerations at different stages of the ML pipeline: Data, Training, Testing, and Deployment. This data-centric lens on development aims to promote thoughtfulness and transparency prior to system development. Additionally, we highlight specific data-centric AI challenges and research opportunities. 

- Surgical Fine-Tuning Improves Adaptation to Distribution Shifts. [[paper]](https://arxiv.org/abs/2210.11466)
  - Yoonho Lee, Annie S. Chen, Fahim Tajwar, Ananya Kumar, Huaxiu Yao, Percy Liang, Chelsea Finn.
  - Key Word: Fine-Tuning; Distribution Shift.
  - <details><summary>Digest</summary> A common approach to transfer learning under distribution shift is to fine-tune the last few layers of a pre-trained model, preserving learned features while also adapting to the new task. This paper shows that in such settings, selectively fine-tuning a subset of layers (which we term surgical fine-tuning) matches or outperforms commonly used fine-tuning approaches. Moreover, the type of distribution shift influences which subset is more effective to tune: for example, for image corruptions, fine-tuning only the first few layers works best. 

- Scaling up Trustless DNN Inference with Zero-Knowledge Proofs. [[paper]](https://arxiv.org/abs/2210.08674)
  - Daniel Kang, Tatsunori Hashimoto, Ion Stoica, Yi Sun.
  - Key Word: Zero-Knowledge Proof; Inference Validation.
  - <details><summary>Digest</summary> We present the first practical ImageNet-scale method to verify ML model inference non-interactively, i.e., after the inference has been done. To do so, we leverage recent developments in ZK-SNARKs (zero-knowledge succinct non-interactive argument of knowledge), a form of zero-knowledge proofs. ZK-SNARKs allows us to verify ML model execution non-interactively and with only standard cryptographic hardness assumptions. 

- DeepMed: Semiparametric Causal Mediation Analysis with Debiased Deep Learning. [[paper]](https://arxiv.org/abs/2210.04389)
  - Siqi Xu, Lin Liu, Zhonghua Liu. *NeurIPS 2022*
  - Key Word: Causal Mediation Analysis; Fairness.
  - <details><summary>Digest</summary> Causal mediation analysis can unpack the black box of causality and is therefore a powerful tool for disentangling causal pathways in biomedical and social sciences, and also for evaluating machine learning fairness. To reduce bias for estimating Natural Direct and Indirect Effects in mediation analysis, we propose a new method called DeepMed that uses deep neural networks (DNNs) to cross-fit the infinite-dimensional nuisance functions in the efficient influence functions. 

- FedMT: Federated Learning with Mixed-type Labels. [[paper]](https://arxiv.org/abs/2210.02042)
  - Qiong Zhang, Aline Talhouk, Gang Niu, Xiaoxiao Li.
  - Key Word: Federated Learning; Neural Tangent Kernel.
  - <details><summary>Digest</summary> We consider an important yet under-explored setting of FL, namely FL with mixed-type labels where different labeling criteria can be employed by various centers, leading to inter-center label space differences and challenging existing FL methods designed for the classical setting. To effectively and efficiently train models with mixed-type labels, we propose a theory-guided and model-agnostic approach that can make use of the underlying correspondence between those label spaces and can be easily combined with various FL methods such as FedAvg. 

- Data Budgeting for Machine Learning. [[paper]](https://arxiv.org/abs/2210.00987)
  - Xinyi Zhao, Weixin Liang, James Zou.
  - Key Word: Benchmark; Data Budgeting.
  - <details><summary>Digest</summary> We study the data budgeting problem and formulate it as two sub-problems: predicting (1) what is the saturating performance if given enough data, and (2) how many data points are needed to reach near the saturating performance. Different from traditional dataset-independent methods like PowerLaw, we proposed a learning method to solve data budgeting problems. 

- I Speak, You Verify: Toward Trustworthy Neural Program Synthesis. [[paper]](https://arxiv.org/abs/2210.00848)
  - Darren Key, Wen-Ding Li, Kevin Ellis.
  - Key Word: Program Synthesis; Natural Language to Code; Large Language Models.
  - <details><summary>Digest</summary> We develop an approach for improving the trustworthiness and overall accuracy of program synthesizers based on large language models for source code. Given a natural language description of a programming problem, our method samples both candidate programs as well as candidate predicates specifying how the program should behave. 

- Causal Knowledge Transfer from Task Affinity. [[paper]](https://arxiv.org/abs/2210.00380)
  - Ahmed Aloui, Juncheng Dong, Cat P. Le, Vahid Tarokh.
  - Key Word: Causal Inference; Transfer Learning; Task Similarity.
  - <details><summary>Digest</summary> We focus on transferring the causal knowledge acquired in prior experiments to new scenarios for which only limited data is available. To this end, we first observe that the absolute values of ITEs are invariant under the action of the symmetric group on the labels of treatments. Given this invariance, we propose a symmetrized task distance for calculating the similarity of a target scenario with those encountered before. The aforementioned task distance is then used to transfer causal knowledge from the closest of all the available previously learned tasks to the target scenario. 

- SoK: On the Impossible Security of Very Large Foundation Models. [[paper]](https://arxiv.org/abs/2209.15259)
  - El-Mahdi El-Mhamdi, Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Lê-Nguyên Hoang, Rafael Pinot, John Stephan.
  - Key Word: Foundation Models; Security; Privacy.
  - <details><summary>Digest</summary> We identify several key features of today's foundation model learning problem which, given the current understanding in adversarial machine learning, suggest incompatibility of high accuracy with both security and privacy. We begin by observing that high accuracy seems to require (1) very high-dimensional models and (2) huge amounts of data that can only be procured through user-generated datasets. Moreover, such data is fundamentally heterogeneous, as users generally have very specific (easily identifiable) data-generating habits. 

- Totems: Physical Objects for Verifying Visual Integrity. [[paper]](https://arxiv.org/abs/2209.13032)
  - Jingwei Ma, Lucy Chai, Minyoung Huh, Tongzhou Wang, Ser-Nam Lim, Phillip Isola, Antonio Torralba. *ECCV 2022*
  - Key Word: Digitial Signatures; Detecting Image Manipulations.
  - <details><summary>Digest</summary> We introduce a new approach to image forensics: placing physical refractive objects, which we call totems, into a scene so as to protect any photograph taken of that scene. Totems bend and redirect light rays, thus providing multiple, albeit distorted, views of the scene within a single image. A defender can use these distorted totem pixels to detect if an image has been manipulated. Our approach unscrambles the light rays passing through the totems by estimating their positions in the scene and using their known geometric and material properties.

- Interventional Causal Representation Learning. [[paper]](https://arxiv.org/abs/2209.11924)
  - Kartik Ahuja, Yixin Wang, Divyat Mahajan, Yoshua Bengio.
  - Key Word: Interventional Data; Causal Representation Learning.
  - <details><summary>Digest</summary> We explore the role of interventional data for identifiable representation learning in this work. We study the identifiability of latent causal factors with and without interventional data, under minimal distributional assumptions on the latents. We prove that, if the true latent variables map to the observed high-dimensional data via a polynomial function, then representation learning via minimizing the standard reconstruction loss of autoencoders identifies the true latents up to affine transformation. 

- NashAE: Disentangling Representations through Adversarial Covariance Minimization. [[paper]](https://arxiv.org/abs/2209.10677) [[code]](https://github.com/ericyeats/nashae-beamsynthesis)
  - Eric Yeats, Frank Liu, David Womble, Hai Li. *ECCV 2022*
  - Key Word: Disentangled Representation.
  - <details><summary>Digest</summary> We present a self-supervised method to disentangle factors of variation in high-dimensional data that does not rely on prior knowledge of the underlying variation profile (e.g., no assumptions on the number or distribution of the individual latent variables to be extracted). In this method which we call NashAE, high-dimensional feature disentanglement is accomplished in the low-dimensional latent space of a standard autoencoder (AE) by promoting the discrepancy between each encoding element and information of the element recovered from all other encoding elements.

- A Survey of Deep Causal Model. [[paper]](https://arxiv.org/abs/2209.08860)
  - Zongyu Li, Zhenfeng Zhu. 
  - Key Word: Causality; Survey.
  - <details><summary>Digest</summary> This paper focuses on the survey of the deep causal models, and its core contributions are as follows: 1) we provide relevant metrics under multiple treatments and continuous-dose treatment; 2) we incorporate a comprehensive overview of deep causal models from both temporal development and method classification perspectives; 3) we assist a detailed and comprehensive classification and analysis of relevant datasets and source code.

- Introspective Learning : A Two-Stage Approach for Inference in Neural Networks. [[paper]](https://arxiv.org/abs/2209.08425) [[code]](https://github.com/olivesgatech/introspective-learning)
  - Mohit Prabhushankar, Ghassan AlRegib. *NeurIPS 2022*
  - Key Word: Active Learning; Out-of-Distribution Detection; Uncertainty Estimation; Image Quality Assessment.
  - <details><summary>Digest</summary> We advocate for two stages in a neural network's decision making process. The first is the existing feed-forward inference framework where patterns in given data are sensed and associated with previously learned patterns. The second stage is a slower reflection stage where we ask the network to reflect on its feed-forward decision by considering and evaluating all available choices. Together, we term the two stages as introspective learning. We use gradients of trained neural networks as a measurement of this reflection. A simple three-layered Multi Layer Perceptron is used as the second stage that predicts based on all extracted gradient features.

- Making Intelligence: Ethics, IQ, and ML Benchmarks. [[paper]](https://arxiv.org/abs/2209.00692)
  - Borhane Blili-Hamelin, Leif Hancox-Li.
  - Key Word: Ethics.
  - <details><summary>Digest</summary> The ML community recognizes the importance of anticipating and mitigating the potential negative impacts of benchmark research. In this position paper, we argue that more attention needs to be paid to areas of ethical risk that lie at the technical and scientific core of ML benchmarks. We identify overlooked structural similarities between human IQ and ML benchmarks. Human intelligence and ML benchmarks share similarities in setting standards for describing, evaluating and comparing performance on tasks relevant to intelligence. This enables us to unlock lessons from feminist philosophy of science scholarship that need to be considered by the ML benchmark community. Finally, we outline practical recommendations for benchmark research ethics and ethics review.

- Fundamentals of Task-Agnostic Data Valuation. [[paper]](https://arxiv.org/abs/2208.12354)
  - Mohammad Mohammadi Amiri, Frederic Berdoz, Ramesh Raskar.
  - Key Word: Task-Agnostic Data Valuation.
  - <details><summary>Digest</summary> We focus on task-agnostic data valuation without any validation requirements. The data buyer has access to a limited amount of data (which could be publicly available) and seeks more data samples from a data seller. We formulate the problem as estimating the differences in the statistical properties of the data at the seller with respect to the baseline data available at the buyer.

- Language Models (Mostly) Know What They Know. [[paper]](https://arxiv.org/abs/2207.05221)
  - Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, Jared Kaplan.
  - Key Word: Language Models; Calibration.
  - <details><summary>Digest</summary> We study whether language models can evaluate the validity of their own claims and predict which questions they will be able to answer correctly. We first show that larger models are well-calibrated on diverse multiple choice and true/false questions when they are provided in the right format. Thus we can approach self-evaluation on open-ended sampling tasks by asking models to first propose answers, and then to evaluate the probability "P(True)" that their answers are correct. We find encouraging performance, calibration, and scaling for P(True) on a diverse array of tasks.

- Repairing Neural Networks by Leaving the Right Past Behind. [[paper]](https://arxiv.org/abs/2207.04806)
  - Ryutaro Tanno, Melanie F. Pradier, Aditya Nori, Yingzhen Li.
  - Key Word: Bayesian Continual Unlearning; Model Repairment.
  - <details><summary>Digest</summary> This work draws on the Bayesian view of continual learning, and develops a generic framework for both, identifying training examples that have given rise to the target failure, and fixing the model through erasing information about them. This framework naturally allows leveraging recent advances in continual learning to this new problem of model repairment, while subsuming the existing works on influence functions and data deletion as specific instances. Experimentally, the proposed approach outperforms the baselines for both identification of detrimental training data and fixing model failures in a generalisable manner.

- Mechanisms that Incentivize Data Sharing in Federated Learning. [[paper]](https://arxiv.org/abs/2207.04557)
  - Sai Praneeth Karimireddy, Wenshuo Guo, Michael I. Jordan.
  - Key Word: Data Maximization Incentivization; Federated Learning; Contract Theory.
  - <details><summary>Digest</summary> Federated learning is typically considered a beneficial technology which allows multiple agents to collaborate with each other, improve the accuracy of their models, and solve problems which are otherwise too data-intensive / expensive to be solved individually. However, under the expectation that other agents will share their data, rational agents may be tempted to engage in detrimental behavior such as free-riding where they contribute no data but still enjoy an improved model. In this work, we propose a framework to analyze the behavior of such rational data generators.

- On the Need and Applicability of Causality for Fair Machine Learning. [[paper]](https://arxiv.org/abs/2207.04053)
  - Rūta Binkytė, Sami Zhioua.
  - Key Word: Causality; Fairness.
  - <details><summary>Digest</summary> Causal reasoning has an indispensable role in how humans make sense of the world and come to decisions in everyday life. While 20th century science was reserved from making causal claims as too strong and not achievable, the 21st century is marked by the return of causality encouraged by the mathematization of causal notions and the introduction of the non-deterministic concept of cause. Besides its common use cases in epidemiology, political, and social sciences, causality turns out to be crucial in evaluating the fairness of automated decisions, both in a legal and everyday sense. We provide arguments and examples of why causality is particularly important for fairness evaluation.

- Robustness of Epinets against Distributional Shifts. [[paper]](https://arxiv.org/abs/2207.00137)
  - Xiuyuan Lu, Ian Osband, Seyed Mohammad Asghari, Sven Gowal, Vikranth Dwaracherla, Zheng Wen, Benjamin Van Roy.
  - Key Word: Epinets; Uncertainty; Distribution Shifts.
  - <details><summary>Digest</summary> Recent work introduced the epinet as a new approach to uncertainty modeling in deep learning. An epinet is a small neural network added to traditional neural networks, which, together, can produce predictive distributions. In particular, using an epinet can greatly improve the quality of joint predictions across multiple inputs, a measure of how well a neural network knows what it does not know. In this paper, we examine whether epinets can offer similar advantages under distributional shifts. We find that, across ImageNet-A/O/C, epinets generally improve robustness metrics.

- Causal Machine Learning: A Survey and Open Problems. [[paper]](https://arxiv.org/abs/2206.15475)
  - Jean Kaddour, Aengus Lynch, Qi Liu, Matt J. Kusner, Ricardo Silva.
  - Key Word: Causality; Survey.
  - <details><summary>Digest</summary> Causal Machine Learning (CausalML) is an umbrella term for machine learning methods that formalize the data-generation process as a structural causal model (SCM). This allows one to reason about the effects of changes to this process (i.e., interventions) and what would have happened in hindsight (i.e., counterfactuals). We categorize work in \causalml into five groups according to the problems they tackle: (1) causal supervised learning, (2) causal generative modeling, (3) causal explanations, (4) causal fairness, (5) causal reinforcement learning.

- Can Foundation Models Talk Causality? [[paper]](https://arxiv.org/abs/2206.10591)
  - Moritz Willig, Matej Zečević, Devendra Singh Dhami, Kristian Kersting.
  - Key Word: Foundation Models; Causality.
  - <details><summary>Digest</summary> Foundation models are subject to an ongoing heated debate, leaving open the question of progress towards AGI and dividing the community into two camps: the ones who see the arguably impressive results as evidence to the scaling hypothesis, and the others who are worried about the lack of interpretability and reasoning capabilities. By investigating to which extent causal representations might be captured by these large scale language models, we make a humble efforts towards resolving the ongoing philosophical conflicts.

- X-Risk Analysis for AI Research. [[paper]](https://arxiv.org/abs/2206.05862)
  - Dan Hendrycks, Mantas Mazeika.
  - Key Word: AI Risk.
  - <details><summary>Digest</summary> Artificial intelligence (AI) has the potential to greatly improve society, but as with any powerful technology, it comes with heightened risks and responsibilities. Current AI research lacks a systematic discussion of how to manage long-tail risks from AI systems, including speculative long-term risks. Keeping in mind the potential benefits of AI, there is some concern that building ever more intelligent and powerful AI systems could eventually result in systems that are more powerful than us; some say this is like playing with fire and speculate that this could create existential risks (x-risks). To add precision and ground these discussions, we provide a guide for how to analyze AI x-risk.

- BaCaDI: Bayesian Causal Discovery with Unknown Interventions. [[paper]](https://arxiv.org/abs/2206.01665)
  - Alexander Hägele, Jonas Rothfuss, Lars Lorch, Vignesh Ram Somnath, Bernhard Schölkopf, Andreas Krause.
  - Key Word: Causal Discovery.
  - <details><summary>Digest</summary> Learning causal structures from observation and experimentation is a central task in many domains. For example, in biology, recent advances allow us to obtain single-cell expression data under multiple interventions such as drugs or gene knockouts. However, a key challenge is that often the targets of the interventions are uncertain or unknown. Thus, standard causal discovery methods can no longer be used. To fill this gap, we propose a Bayesian framework (BaCaDI) for discovering the causal structure that underlies data generated under various unknown experimental/interventional conditions.

- Differentiable Invariant Causal Discovery. [[paper]](https://arxiv.org/abs/2205.15638)
  - Yu Wang, An Zhang, Xiang Wang, Xiangnan He, Tat-Seng Chua.
  - Key Word: Causal Discovery.
  - <details><summary>Digest</summary> This paper proposes Differentiable Invariant Causal Discovery (DICD), utilizing the multi-environment information based on a differentiable framework to avoid learning spurious edges and wrong causal directions. Specifically, DICD aims to discover the environment-invariant causation while removing the environment-dependent correlation. We further formulate the constraint that enforces the target structure equation model to maintain optimal across the environments.  

- AI and Ethics -- Operationalising Responsible AI. [[paper]](https://arxiv.org/abs/2105.08867)
  - Liming Zhu, Xiwei Xu, Qinghua Lu, Guido Governatori, Jon Whittle.
  - Key Word: Survey; Ethics; Responsibility.
  - <details><summary>Digest</summary> In the last few years, AI continues demonstrating its positive impact on society while sometimes with ethically questionable consequences. Building and maintaining public trust in AI has been identified as the key to successful and sustainable innovation. This chapter discusses the challenges related to operationalizing ethical AI principles and presents an integrated view that covers high-level ethical AI principles, the general notion of trust/trustworthiness, and product/process support in the context of responsible AI, which helps improve both trust and trustworthiness of AI for a wider set of stakeholders.

- State of AI Ethics Report (Volume 6, February 2022). [[paper]](https://arxiv.org/abs/2202.07435)
  - Abhishek Gupta, Connor Wright, Marianna Bergamaschi Ganapini, Masa Sweidan, Renjie Butalid.
  - Key Word: Report; Ethics.
  - <details><summary>Digest</summary> This report from the Montreal AI Ethics Institute (MAIEI) covers the most salient progress in research and reporting over the second half of 2021 in the field of AI ethics. Particular emphasis is placed on an "Analysis of the AI Ecosystem", "Privacy", "Bias", "Social Media and Problematic Information", "AI Design and Governance", "Laws and Regulations", "Trends", and other areas covered in the "Outside the Boxes" section. The two AI spotlights feature application pieces on "Constructing and Deconstructing Gender with AI-Generated Art" as well as "Will an Artificial Intellichef be Cooking Your Next Meal at a Michelin Star Restaurant?".

- Optimal transport for causal discovery. [[paper]](https://arxiv.org/abs/2201.09366)
  - Ruibo Tu, Kun Zhang, Hedvig Kjellström, Cheng Zhang. *ICLR 2022*
  - Key Word: Causal Discovery; Optimal Transport.
  - <details><summary>Digest</summary> To determine causal relationships between two variables, approaches based on Functional Causal Models (FCMs) have been proposed by properly restricting model classes; however, the performance is sensitive to the model assumptions, which makes it difficult to use. In this paper, we provide a novel dynamical-system view of FCMs and propose a new framework for identifying causal direction in the bivariate case. We first show the connection between FCMs and optimal transport, and then study optimal transport under the constraints of FCMs.

</details>

<!-- ### Others: 2021 -->

<details>
<summary>Others: 2021</summary>

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

</details>

<!-- ### Others: 2020 -->

<details>
<summary>Others: 2020</summary>

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

</details>

<!-- ### Others: 2019 -->

<details>
<summary>Others: 2019</summary>

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

</details>

<!-- ### Others: 2018 -->

<details>
<summary>Others: 2018</summary>

- Towards Reverse-Engineering Black-Box Neural Networks. [[paper]](https://arxiv.org/abs/1711.01768) [[code]](https://github.com/coallaoh/WhitenBlackBox)
  - Seong Joon Oh, Max Augustin, Bernt Schiele, Mario Fritz. *ICLR 2018*
  - Key Word: Model Extraction Attacks; Membership inference Attacks.
  - <details><summary>Digest</summary> This work shows that such attributes of neural networks can be exposed from a sequence of queries. This has multiple implications. On the one hand, our work exposes the vulnerability of black-box neural networks to different types of attacks -- we show that the revealed internal information helps generate more effective adversarial examples against the black box model.  

</details>

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
