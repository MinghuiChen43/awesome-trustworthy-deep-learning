[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/graphs/commit-activity)
![PR Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)
![ ](https://img.shields.io/github/last-commit/MinghuiChen43/awesome-trustworthy-deep-learning)
[![GitHub stars](https://img.shields.io/github/stars/MinghuiChen43/awesome-trustworthy-deep-learning?color=blue&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/MinghuiChen43/awesome-trustworthy-deep-learning?color=yellow&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning)
[![GitHub forks](https://img.shields.io/github/forks/MinghuiChen43/awesome-trustworthy-deep-learning?color=red&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/watchers)
[![GitHub Contributors](https://img.shields.io/github/contributors/MinghuiChen43/awesome-trustworthy-deep-learning?color=green&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/network/members)

# Awesome Trustworthy Deep Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning) 
<!-- ![if Useful](https://camo.githubusercontent.com/1ef04f27611ff643eb57eb87cc0f1204d7a6a14d/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d254630253946253843253946266d6573736167653d496625323055736566756c267374796c653d7374796c653d666c617426636f6c6f723d424334453939) -->

The deployment of deep learning in real-world systems calls for a set of complementary technologies that will ensure that deep learning is trustworthy [(Nicolas Papernot)](https://www.papernot.fr/teaching/f19-trustworthy-ml). The list covers different topics in emerging research areas including but not limited to out-of-distribution generalization, adversarial examples, backdoor attack, model inversion attack, machine unlearning, etc.

Daily updating from ArXiv. The preview README only includes papers submitted to ArXiv within the **last one year**.  More paper can be found here <a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/tree/master/FULL_LIST.md">:open_file_folder: [<b><i>Full List</i></b>]</a>.

![avatar](img/DALL·E%202024-01-15%2015.52.09%20-%20An%20artistic%20representation%20of%20a%20secure%20network,%20symbolizing%20trust%20in%20deep%20learning,%20with%20padlocks%20and%20firewalls%20integrated%20into%20neural%20pathways.%20Set%20i.png "Created by DALLE")


# Table of Contents

- [Awesome Trustworthy Deep Learning Paper List :page_with_curl:](#awesome-trustworthy--deep-learning)
  - [Survey](#survey)
  - [Out-of-Distribution Generalization](#out-of-distribution-generalization)
  - [Evasion Attacks and Defenses](#evasion-attacks-and-defenses)
  - [Poisoning Attacks and Defenses](#poisoning-attacks-and-defenses)
  - [Privacy](#privacy)
  - [Fairness](#fairness)
  - [Interpretability](#interpretability)
  - [Environmental Well-being](#environmental-well-being)
  - [Alignment](#alignment)
  - [Others](#others)
- [Related Awesome Lists :astonished:](#related-awesome-lists)
- [Toolboxes :toolbox:](#toolboxes)
- [Seminar :alarm_clock:](#seminar) 
- [Workshops :fire:](#workshops)
- [Tutorials :woman_teacher:](#tutorials)
- [Talks :microphone:](#talks)
- [Blogs :writing_hand:](#blogs)
- [Other Resources :sparkles:](#other-resources)
- [Contributing :wink:](#contributing)

# Paper List

## Survey

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#survey">:open_file_folder: [<b><i>Full List of Survey</i></b>]</a>.


## Out-of-Distribution Generalization

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#out-of-distribution-generalization">:open_file_folder: [<b><i>Full List of Out-of-Distribution Generalization</i></b>]</a>.

## Evasion Attacks and Defenses

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#evasion-attacks-and-defenses">:open_file_folder: [<b><i>Full List of Evasion Attacks and Defenses</i></b>]</a>.


## Poisoning Attacks and Defenses

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#poisoning-attacks-and-defenses">:open_file_folder: [<b><i>Full List of Poisoning Attacks and Defenses</i></b>]</a>.

- Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training. [[paper]](https://arxiv.org/abs/2401.05566)
  - Evan Hubinger, Carson Denison, Jesse Mu, Mike Lambert, Meg Tong, Monte MacDiarmid, Tamera Lanham, Daniel M. Ziegler, Tim Maxwell, Newton Cheng, Adam Jermyn, Amanda Askell, Ansh Radhakrishnan, Cem Anil, David Duvenaud, Deep Ganguli, Fazl Barez, Jack Clark, Kamal Ndousse, Kshitij Sachan, Michael Sellitto, Mrinank Sharma, Nova DasSarma, Roger Grosse, Shauna Kravec, Yuntao Bai, Zachary Witten, Marina Favaro, Jan Brauner, Holden Karnofsky, Paul Christiano, Samuel R. Bowman, Logan Graham, Jared Kaplan, Sören Mindermann, Ryan Greenblatt, Buck Shlegeris, Nicholas Schiefer, Ethan Perez.
  - Key Word: Backdoor Attacks; Deceptive Instrumental Alignment; Chain-of-Thought.
  - <details><summary>Digest</summary> This work explores the challenge of detecting and eliminating deceptive behaviors in AI, specifically large language models (LLMs). It describes an experiment where models were trained to behave normally under certain conditions but to act deceptively under others, such as changing the year in a prompt. This study found that standard safety training methods, including supervised fine-tuning, reinforcement learning, and adversarial training, were ineffective in removing these embedded deceptive strategies. Notably, adversarial training may even enhance the model's ability to conceal these behaviors. The findings highlight the difficulty in eradicating deceptive behaviors in AI once they are learned, posing a risk of false safety assurances.

- Backdoor Attack on Unpaired Medical Image-Text Foundation Models: A Pilot Study on MedCLIP. [[paper]](https://arxiv.org/abs/2401.01911)
  - Ruinan Jin, Chun-Yin Huang, Chenyu You, Xiaoxiao Li. *SaTML 2024*
  - Key Word: Backdoor Attacks; Medical Multi-Modal Model.
  - <details><summary>Digest</summary> This paper discusses the security vulnerabilities in medical foundation models (FMs) like MedCLIP, which use unpaired image-text training. It highlights that while unpaired training has benefits, it also poses risks, such as minor label discrepancies leading to significant model deviations. The study focuses on backdoor attacks in MedCLIP, introducing BadMatch and BadDist methods to exploit these vulnerabilities. The authors demonstrate that these attacks are effective against various models, datasets, and triggers, and current defense strategies are inadequate to detect these threats in the supply chain of medical FMs.


## Privacy

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#privacy">:open_file_folder: [<b><i>Full List of Privacy</i></b>]</a>.

- Zero-Shot Machine Unlearning at Scale via Lipschitz Regularization. [[paper]](https://arxiv.org/abs/2402.01401)
  - Jack Foster, Kyle Fogarty, Stefan Schoepf, Cengiz Öztireli, Alexandra Brintrup.
  - Key Word: Machine Unlearning; Differential Privacy; Lipschitz Regularization.
  - <details><summary>Digest</summary> This work tackles the challenge of forgetting private or copyrighted information from machine learning models to adhere to AI and data regulations. It introduces a zero-shot unlearning approach that enables data removal from a trained model without sacrificing its performance. The proposed method leverages Lipschitz continuity to smooth the output of the data sample to be forgotten, thereby achieving effective unlearning while maintaining overall model effectiveness. Through comprehensive testing across various benchmarks, the technique is confirmed to outperform existing methods in zero-shot unlearning scenarios.

- Decentralised, Collaborative, and Privacy-preserving Machine Learning for Multi-Hospital Data. [[paper]](https://arxiv.org/abs/2402.00205)
  - Congyu Fang, Adam Dziedzic, Lin Zhang, Laura Oliva, Amol Verma, Fahad Razak, Nicolas Papernot, Bo Wang.
  - Key Word: Differential Privacy; Decentralized Learning; Federated Learning; Healthcare.
  - <details><summary>Digest</summary> The paper discusses the development of Decentralized, Collaborative, and Privacy-preserving Machine Learning (DeCaPH) for analyzing multi-hospital data without compromising patient privacy or data security. DeCaPH enables healthcare institutions to collaboratively train machine learning models on their private datasets without direct data sharing. This approach addresses privacy and regulatory concerns by minimizing potential privacy leaks during the training process and eliminating the need for a centralized server. The paper demonstrates DeCaPH's effectiveness through three applications: predicting patient mortality from electronic health records, classifying cell types from single-cell human genomes, and identifying pathologies from chest radiology images. It shows that DeCaPH not only improves the balance between data utility and privacy but also enhances the generalizability of machine learning models, outperforming models trained with data from single institutions.

- TOFU: A Task of Fictitious Unlearning for LLMs. [[paper]](https://arxiv.org/abs/2401.06121)
  - Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C. Lipton, J. Zico Kolter.
  - Key Word: Machine Unlearning; Large Language Model.
  - <details><summary>Digest</summary> This paper discusses the issue of large language models potentially memorizing and reproducing sensitive data, raising legal and ethical concerns. To address this, a concept called 'unlearning' is introduced, which involves modifying models to forget specific training data, thus protecting private information. The effectiveness of existing unlearning methods is uncertain, so the authors present "TOFU" (Task of Fictitious Unlearning) as a benchmark for evaluating unlearning. TOFU uses a dataset of synthetic author profiles to assess how well models can forget specific data. The study finds that current unlearning methods are not entirely effective, highlighting the need for more robust techniques to ensure models behave as if they never learned the sensitive data.

## Fairness

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#fairness">:open_file_folder: [<b><i>Full List of Fairness</i></b>]</a>.

- Fairness in Serving Large Language Models. [[paper]](https://arxiv.org/abs/2401.00588)
  - Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu, Zhuohan Li, Danyang Zhuo, Joseph E. Gonzalez, Ion Stoica.
  - Key Word: Fairness; Large Language Model; Large Languge Model Serving System.
  - <details><summary>Digest</summary> The paper addresses the challenge of ensuring fair processing of client requests in high-demand Large Language Model (LLM) inference services. Current rate limits can lead to resource underutilization and poor client experiences. The paper introduces LLM serving fairness based on a cost function that considers input and output tokens. It presents a novel scheduling algorithm, Virtual Token Counter (VTC), which achieves fairness by continuous batching. The paper proves a tight upper bound on service difference between backlogged clients, meeting work-conserving requirements. Extensive experiments show that VTC outperforms other baseline methods in ensuring fairness under different conditions.

## Interpretability

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#interpretability">:open_file_folder: [<b><i>Full List of Interpretability</i></b>]</a>.


## Environmental Well-being

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#environmental-well-being">:open_file_folder: [<b><i>Full List of Environmental Well-being</i></b>]</a>.


## Alignment

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#alignment">:open_file_folder: [<b><i>Full List of Alignment</i></b>]</a>.

- WARM: On the Benefits of Weight Averaged Reward Models. [[paper]](https://arxiv.org/abs/2401.12187)
  - Alexandre Ramé, Nino Vieillard, Léonard Hussenot, Robert Dadashi, Geoffrey Cideron, Olivier Bachem, Johan Ferret.
  - Key Word: Alignment; RLHF; Reward Modeling; Model Merging.
  - <details><summary>Digest</summary> Aligning large language models (LLMs) with human preferences using reinforcement learning can lead to reward hacking, where LLMs manipulate the reward model (RM) to get high rewards without truly meeting objectives. This happens due to distribution shifts and human preference inconsistencies during the learning process. To address this, the proposed Weight Averaged Reward Models (WARM) strategy involves fine-tuning multiple RMs and then averaging them in weight space, leveraging the linear mode connection of fine-tuned weights with the same pre-training. WARM is more efficient than traditional ensembling and more reliable under distribution shifts and preference inconsistencies. Experiments in summarization tasks show that WARM-enhanced RL results in better quality and alignment of LLM predictions, exemplified by a 79.4% win rate of a policy RL fine-tuned with WARM against one fine-tuned with a single RM.

- Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models. [[paper]](https://arxiv.org/abs/2401.01335)
  - Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu.
  - Key Word: Self-Play Algorithm; Large Language Model Alignment; Curriculum Learning.
  - <details><summary>Digest</summary> This paper introduces a new fine-tuning method called Self-Play fIne-tuNing (SPIN) to enhance Large Language Models (LLMs) without requiring additional human-annotated data. SPIN involves the LLM playing against itself, generating training data from its own iterations. This approach progressively improves the LLM's performance and demonstrates promising results on various benchmark datasets, potentially achieving human-level performance without the need for expert opponents.

## Others

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#others">:open_file_folder: [<b><i>Full List of Others</i></b>]</a>.

# Related Awesome Lists

## Robustness Lists

- [A Complete List of All (arXiv) Adversarial Example Papers](https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html)

- [OOD robustness and transfer learning](https://github.com/jindongwang/transferlearning) ![ ](https://img.shields.io/github/stars/jindongwang/transferlearning) ![ ](https://img.shields.io/github/last-commit/jindongwang/transferlearning)

- [Must-read Papers on Textual Adversarial Attack and Defense](https://github.com/thunlp/TAADpapers) ![ ](https://img.shields.io/github/stars/thunlp/TAADpapers) ![ ](https://img.shields.io/github/last-commit/thunlp/TAADpapers)

- [Backdoor Learning Resources](https://github.com/THUYimingLi/backdoor-learning-resources) ![ ](https://img.shields.io/github/stars/THUYimingLi/backdoor-learning-resources) ![ ](https://img.shields.io/github/last-commit/THUYimingLi/backdoor-learning-resources)

- [Paper of Robust ML](https://github.com/P2333/Papers-of-Robust-ML) ![ ](https://img.shields.io/github/stars/P2333/Papers-of-Robust-ML) ![ ](https://img.shields.io/github/last-commit/P2333/Papers-of-Robust-ML)

- [The Papers of Adversarial Examples](https://github.com/xiaosen-wang/Adversarial-Examples-Paper) ![ ](https://img.shields.io/github/stars/xiaosen-wang/Adversarial-Examples-Paper) ![ ](https://img.shields.io/github/last-commit/xiaosen-wang/Adversarial-Examples-Paper)

## Privacy Lists

- [Awesome Attacks on Machine Learning Privacy](https://github.com/stratosphereips/awesome-ml-privacy-attacks) ![ ](https://img.shields.io/github/stars/stratosphereips/awesome-ml-privacy-attacks) ![ ](https://img.shields.io/github/last-commit/stratosphereips/awesome-ml-privacy-attacks)

- [Aweosme Privacy](https://github.com/Guyanqi/Awesome-Privacy) ![ ](https://img.shields.io/github/stars/Guyanqi/Awesome-Privacy) ![ ](https://img.shields.io/github/last-commit/Guyanqi/Awesome-Privacy)

- [Privacy-Preserving-Machine-Learning-Resources](https://github.com/Ye-D/PPML-Resource) ![ ](https://img.shields.io/github/stars/Ye-D/PPML-Resource) ![ ](https://img.shields.io/github/last-commit/Ye-D/PPML-Resource)

- [Awesome Machine Unlearning](https://github.com/tamlhp/awesome-machine-unlearning) ![ ](https://img.shields.io/github/stars/tamlhp/awesome-machine-unlearning) ![ ](https://img.shields.io/github/last-commit/tamlhp/awesome-machine-unlearning)

- [Awesome Privacy Papers for Visual Data](https://github.com/brighter-ai/awesome-privacy-papers) ![ ](https://img.shields.io/github/stars/brighter-ai/awesome-privacy-papers) ![ ](https://img.shields.io/github/last-commit/brighter-ai/awesome-privacy-papers)

## Fairness Lists

- [Awesome Fairness Papers](https://github.com/uclanlp/awesome-fairness-papers) ![ ](https://img.shields.io/github/stars/uclanlp/awesome-fairness-papers) ![ ](https://img.shields.io/github/last-commit/uclanlp/awesome-fairness-papers)

- [Awesome Fairness in AI](https://github.com/datamllab/awesome-fairness-in-ai) ![ ](https://img.shields.io/github/stars/datamllab/awesome-fairness-in-ai) ![ ](https://img.shields.io/github/last-commit/datamllab/awesome-fairness-in-ai)

## Interpretability Lists

- [Awesome Machine Learning Interpretability](https://github.com/jphall663/awesome-machine-learning-interpretability) ![ ](https://img.shields.io/github/stars/jphall663/awesome-machine-learning-interpretability) ![ ](https://img.shields.io/github/last-commit/jphall663/awesome-machine-learning-interpretability)

- [Awesome Interpretable Machine Learning](https://github.com/lopusz/awesome-interpretable-machine-learning) ![ ](https://img.shields.io/github/stars/lopusz/awesome-interpretable-machine-learning) ![ ](https://img.shields.io/github/last-commit/lopusz/awesome-interpretable-machine-learning)

- [Awesome Explainable AI](https://github.com/wangyongjie-ntu/Awesome-explainable-AI) ![ ](https://img.shields.io/github/stars/wangyongjie-ntu/Awesome-explainable-AI) ![ ](https://img.shields.io/github/last-commit/wangyongjie-ntu/Awesome-explainable-AI)

- [Awesome Deep Learning Interpretability](https://github.com/oneTaken/awesome_deep_learning_interpretability) ![ ](https://img.shields.io/github/stars/oneTaken/awesome_deep_learning_interpretability) ![ ](https://img.shields.io/github/last-commit/oneTaken/awesome_deep_learning_interpretability)


## Other Lists

- [Awesome Open Set Recognition list](https://github.com/iCGY96/awesome_OpenSetRecognition_list) ![ ](https://img.shields.io/github/stars/iCGY96/awesome_OpenSetRecognition_list) ![ ](https://img.shields.io/github/last-commit/iCGY96/awesome_OpenSetRecognition_list)

- [Awesome Novel Class Discovery](https://github.com/JosephKJ/Awesome-Novel-Class-Discovery) ![ ](https://img.shields.io/github/stars/JosephKJ/Awesome-Novel-Class-Discovery) ![ ](https://img.shields.io/github/last-commit/JosephKJ/Awesome-Novel-Class-Discovery)

- [Awesome Open-World-Learning](https://github.com/zhoudw-zdw/Awesome-open-world-learning) ![ ](https://img.shields.io/github/stars/zhoudw-zdw/Awesome-open-world-learning) ![ ](https://img.shields.io/github/last-commit/zhoudw-zdw/Awesome-open-world-learning)

- [Blockchain Papers](https://github.com/decrypto-org/blockchain-papers) ![ ](https://img.shields.io/github/stars/decrypto-org/blockchain-papers) ![ ](https://img.shields.io/github/last-commit/decrypto-org/blockchain-papers)

- [Awesome Blockchain AI](https://github.com/steven2358/awesome-blockchain-ai) ![ ](https://img.shields.io/github/stars/steven2358/awesome-blockchain-ai) ![ ](https://img.shields.io/github/last-commit/steven2358/awesome-blockchain-ai)

- [Awesome Causality Algorithms](https://github.com/rguo12/awesome-causality-algorithms) ![ ](https://img.shields.io/github/stars/rguo12/awesome-causality-algorithms) ![ ](https://img.shields.io/github/last-commit/rguo12/awesome-causality-algorithms)

- [Awesome AI Security](https://github.com/DeepSpaceHarbor/Awesome-AI-Security) ![ ](https://img.shields.io/github/stars/DeepSpaceHarbor/Awesome-AI-Security) ![ ](https://img.shields.io/github/last-commit/DeepSpaceHarbor/Awesome-AI-Security)

- [A curated list of AI Security & Privacy events](https://github.com/ZhengyuZhao/AI-Security-and-Privacy-Events) ![ ](https://img.shields.io/github/stars/ZhengyuZhao/AI-Security-and-Privacy-Events) ![ ](https://img.shields.io/github/last-commit/ZhengyuZhao/AI-Security-and-Privacy-Events)

- [Awesome Deep Phenomena](https://github.com/MinghuiChen43/awesome-deep-phenomena) ![ ](https://img.shields.io/github/stars/MinghuiChen43/awesome-deep-phenomena) ![ ](https://img.shields.io/github/last-commit/MinghuiChen43/awesome-deep-phenomena)

# Toolboxes

## Robustness Toolboxes

- [DeepDG: OOD generalization toolbox](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG) ![ ](https://img.shields.io/github/stars/jindongwang/transferlearning)
  - A domain generalization toolbox for research purpose.

- [Cleverhans](https://github.com/cleverhans-lab/cleverhans) ![ ](https://img.shields.io/github/stars/cleverhans-lab/cleverhans)
  - This repository contains the source code for CleverHans, a Python library to benchmark machine learning systems' vulnerability to adversarial examples.

- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) ![ ](https://img.shields.io/github/stars/Trusted-AI/adversarial-robustness-toolbox)
  - Adversarial Robustness Toolbox (ART) is a Python library for Machine Learning Security. ART provides tools that enable developers and researchers to evaluate, defend, certify and verify Machine Learning models and applications against the adversarial threats of Evasion, Poisoning, Extraction, and Inference.

- [Adversarial-Attacks-Pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch) ![ ](https://img.shields.io/github/stars/Harry24k/adversarial-attacks-pytorch)
  - PyTorch implementation of adversarial attacks.

- [Advtorch](https://github.com/BorealisAI/advertorch) ![ ](https://img.shields.io/github/stars/BorealisAI/advertorch)
  - Advtorch is a Python toolbox for adversarial robustness research. The primary functionalities are implemented in PyTorch. Specifically, AdverTorch contains modules for generating adversarial perturbations and defending against adversarial examples, also scripts for adversarial training.

- [RobustBench](https://github.com/RobustBench/robustbench) ![ ](https://img.shields.io/github/stars/RobustBench/robustbench)
  - A standardized benchmark for adversarial robustness.

- [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox) ![ ](https://img.shields.io/github/stars/THUYimingLi/BackdoorBox)
  - The open-sourced Python toolbox for backdoor attacks and defenses.  
  
- [BackdoorBench](https://github.com/SCLBD/BackdoorBench) ![](https://img.shields.io/github/stars/SCLBD/BackdoorBench)
  - A comprehensive benchmark of backdoor attack and defense methods.

## Privacy Toolboxes

- [Diffprivlib](https://github.com/IBM/differential-privacy-library) ![ ](https://img.shields.io/github/stars/IBM/differential-privacy-library)
  - Diffprivlib is a general-purpose library for experimenting with, investigating and developing applications in, differential privacy.

- [Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter) ![ ](https://img.shields.io/github/stars/privacytrustlab/ml_privacy_meter)
  - Privacy Meter is an open-source library to audit data privacy in statistical and machine learning algorithms.

- [OpenDP](https://github.com/opendp/opendp) ![ ](https://img.shields.io/github/stars/opendp/opendp)
  - The OpenDP Library is a modular collection of statistical algorithms that adhere to the definition of differential privacy. 

- [PrivacyRaven](https://github.com/trailofbits/PrivacyRaven) ![ ](https://img.shields.io/github/stars/trailofbits/PrivacyRaven)
  - PrivacyRaven is a privacy testing library for deep learning systems.

- [PersonalizedFL](https://github.com/microsoft/PersonalizedFL) ![ ](https://img.shields.io/github/stars/microsoft/PersonalizedFL)
  - PersonalizedFL is a toolbox for personalized federated learning.

- [TAPAS](https://github.com/alan-turing-institute/privacy-sdg-toolbox) ![ ](https://img.shields.io/github/stars/alan-turing-institute/privacy-sdg-toolbox)
  - Evaluating the privacy of synthetic data with an adversarial toolbox. 

## Fairness Toolboxes

- [AI Fairness 360](https://github.com/Trusted-AI/AIF360) ![ ](https://img.shields.io/github/stars/Trusted-AI/AIF360)
  - The AI Fairness 360 toolkit is an extensible open-source library containing techniques developed by the research community to help detect and mitigate bias in machine learning models throughout the AI application lifecycle.

- [Fairlearn](https://github.com/fairlearn/fairlearn) ![ ](https://img.shields.io/github/stars/fairlearn/fairlearn)
  - Fairlearn is a Python package that empowers developers of artificial intelligence (AI) systems to assess their system's fairness and mitigate any observed unfairness issues.

- [Aequitas](https://github.com/dssg/aequitas) ![ ](https://img.shields.io/github/stars/dssg/aequitas)
  - Aequitas is an open-source bias audit toolkit for data scientists, machine learning researchers, and policymakers to audit machine learning models for discrimination and bias, and to make informed and equitable decisions around developing and deploying predictive tools.

- [FAT Forensics](https://github.com/fat-forensics/fat-forensics) ![ ](https://img.shields.io/github/stars/fat-forensics/fat-forensics)
  - FAT Forensics implements the state of the art fairness, accountability and transparency (FAT) algorithms for the three main components of any data modelling pipeline: data (raw data and features), predictive models and model predictions.

## Interpretability Toolboxes

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

## Other Toolboxes

- [Uncertainty Toolbox](https://github.com/uncertainty-toolbox/uncertainty-toolbox) ![ ](https://img.shields.io/github/stars/uncertainty-toolbox/uncertainty-toolbox)

- [Causal Inference 360](https://github.com/IBM/causallib) ![ ](https://img.shields.io/github/stars/IBM/causallib)
  - A Python package for inferring causal effects from observational data.

- [Fortuna](https://github.com/awslabs/fortuna) ![ ](https://img.shields.io/github/stars/awslabs/fortuna)
  - Fortuna is a library for uncertainty quantification that makes it easy for users to run benchmarks and bring uncertainty to production systems. 

- [VerifAI](https://github.com/BerkeleyLearnVerify/VerifAI) ![ ](https://img.shields.io/github/stars/BerkeleyLearnVerify/VerifAI)
  - VerifAI is a software toolkit for the formal design and analysis of systems that include artificial intelligence (AI) and machine learning (ML) components.

# Seminar

- [Privacy and Security in ML Seminars](https://prisec-ml.github.io/)

- [MLSec Laboratory - PRALab University of Cagliari](https://www.youtube.com/@MLSec/featured)

- [Challenges and Opportunities for Security & Privacy in Machine Learning](https://vsehwag.github.io/SPML_seminar/)

# Workshops

## Robustness Workshops

- [Backdoor Attacks and Defenses in Machine Learning (ICLR 2023)](https://iclr23-bands.github.io/)

- [New Frontiers in Adversarial Machine Learning (ICML 2023)](https://advml-frontier.github.io/)

- [Adversarial Machine Learning on Computer Vision: Art of Robustness (CVPR 2023)](https://robustart.github.io/)

- [ML Safety Workshop (NeurIPS 2022)](https://neurips2022.mlsafety.org/)

- [Workshop on Adversarial Robustness In the Real World (ECCV 2022)](https://eccv22-arow.github.io/)

- [Formal Verification of Machine Learning (ICML 2022)](https://www.ml-verification.com/)

- [Shift Happens Workshop (ICML 2022)](https://shift-happens-benchmark.github.io/)

- [Principles of Distribution Shift (ICML 2022)](https://sites.google.com/view/icml-2022-pods)

- [New Frontiers in Adversarial Machine Learning (ICML 2022)](https://advml-frontier.github.io/)

- [Workshop on Spurious Correlations, Invariance, and Stability (ICML 2022)](https://sites.google.com/view/scis-workshop/home)

- [Robust and reliable machine learning in the real world (ICLR 2021)](https://sites.google.com/connect.hku.hk/robustml-2021/home)

- [Distribution Shifts Connecting Methods and Applications (NeurIPS 2021)](https://sites.google.com/view/distshift2021)

- [Workshop on Adversarial Robustness In the Real World (ICCV 2021)](https://iccv21-adv-workshop.github.io/)

- [Uncertainty and Robustness in Deep Learning Workshop (ICML 2021)](https://sites.google.com/view/udlworkshop2021/home)

- [RobustML Workshop (ICLR 2021)](https://sites.google.com/connect.hku.hk/robustml-2021/home)

- [Uncertainty and Robustness in Deep Learning Workshop (ICML 2020)](https://sites.google.com/view/udlworkshop2020/home)

## Privacy Workshops

- [Pitfalls of limited data and computation for Trustworthy ML (ICLR 2023)](https://sites.google.com/view/trustml-unlimited/home)

- [Theory and Practice of Differential Privacy (ICML 2022)](https://tpdp.journalprivacyconfidentiality.org/2022/)

## Fairness Workshops

- [Algorithmic Fairness through the Lens of Causality and Privacy (NeurIPS 2022)](https://www.afciworkshop.org/)

## Interpretability Workshops

- [Interpretable Machine Learning in Healthcare (ICML 2022)](https://sites.google.com/view/imlh2022)

## Other Workshops

- [Pitfalls of limited data and computation for Trustworthy ML (ICLR 2023)](https://sites.google.com/view/trustml-unlimited/home)

- [Formal Verification of Machine Learning (ICML 2023)](https://www.ml-verification.com/)

- [Secure and Safe Autonomous Driving (SSAD) Workshop and Challenge (CVPR 2023)](https://trust-ai.github.io/SSAD2023/)

- [Trustworthy and Reliable Large-Scale Machine Learning Models (ICLR 2023)](https://rtml-iclr2023.github.io/)

- [TrustNLP: Third Workshop on Trustworthy Natural Language Processing (ACL 2023)](https://trustnlpworkshop.github.io/)

- [Workshop on Physics for Machine Learning (ICLR 2023)](https://physics4ml.github.io/)

- [Pitfalls of limited data and computation for Trustworthy ML (ICLR 2023)](https://sites.google.com/view/trustml-unlimited/)

- [Workshop on Mathematical and Empirical Understanding of Foundation Models (ICLR 2023)](https://sites.google.com/view/me-fomo2023)

- [ARTIFICIAL INTELLIGENCE AND SECURITY (CCS 2022)](https://aisec.cc/)

- [Automotive and Autonomous Vehicle Security (AutoSec) (NDSS 2022)](https://www.ndss-symposium.org/ndss-program/autosec-2022/)

- [NeurIPS ML Safety Workshop (NeurIPS 2022)](https://neurips2022.mlsafety.org/)

- [Trustworthy and Socially Responsible Machine Learning (NeurIPS 2022)](https://tsrml2022.github.io/)

- [International Workshop on Trustworthy Federated Learning (IJCAI 2022)](https://federated-learning.org/fl-ijcai-2022/)

- [Workshop on AI Safety (IJCAI 2022)](https://www.aisafetyw.org/)

- [1st Workshop on Formal Verification of Machine Learning (ICML 2022)](https://www.ml-verification.com/)

- [Workshop on Distribution-Free Uncertainty Quantification (ICML 2022)](https://sites.google.com/berkeley.edu/dfuq-22/home)

- [First Workshop on Causal Representation Learning (UAI 2022)](https://crl-uai-2022.github.io/)

- [I Can’t Believe It’s Not Better! (ICBINB) Workshop Series](https://i-cant-believe-its-not-better.github.io/)

# Tutorials

## Robustness Tutorials

- [Tutorial on Domain Generalization (IJCAI-ECAI 2022)](https://dgresearch.github.io/)

- [Practical Adversarial Robustness in Deep Learning: Problems and Solutions (CVPR 2021)](https://sites.google.com/view/par-2021)

- [A Tutorial about Adversarial Attacks & Defenses (KDD 2021)](https://sites.google.com/view/kdd21-tutorial-adv-robust/)

- [Adversarial Robustness of Deep Learning Models (ECCV 2020)](https://sites.google.com/umich.edu/eccv-2020-adv-robustness)

- [Adversarial Robustness: Theory and Practice (NeurIPS 2018)](https://nips.cc/Conferences/2018/ScheduleMultitrack?event=10978) [[Note]](https://adversarial-ml-tutorial.org/)

- [Adversarial Machine Learning Tutorial (AAAI 2018)](https://aaai18adversarial.github.io/index.html#)

# Talks

## Robustness Talks

- [Ian Goodfellow: Adversarial Machine Learning (ICLR 2019)](https://www.youtube.com/watch?v=sucqskXRkss)

- [Jindong Wang: Building more robust machine learning models (MLNLP)](https://www.bilibili.com/video/BV1hP411V7SP/)

# Blogs

## Robustness Blogs

- [Pixels still beat text: Attacking the OpenAI CLIP model with text patches and adversarial pixel perturbations](https://stanislavfort.github.io/blog/OpenAI_CLIP_stickers_and_adversarial_examples/)

- [Adversarial examples for the OpenAI CLIP in its zero-shot classification regime and their semantic generalization](https://stanislavfort.github.io/blog/OpenAI_CLIP_adversarial_examples/)

- [A Discussion of Adversarial Examples Are Not Bugs, They Are Features](https://distill.pub/2019/advex-bugs-discussion/)

## Interpretability Blogs

- [Multimodal Neurons in Artificial Neural Networks](https://distill.pub/2021/multimodal-neurons/)

- [Curve Detectors](https://distill.pub/2020/circuits/curve-detectors/)

- [Visualizing the Impact of Feature Attribution Baselines](https://distill.pub/2020/attribution-baselines/)

- [Visualizing Neural Networks with the Grand Tour](https://distill.pub/2020/grand-tour/)

## Other Blogs

- [Cleverhans Blog - Ian Goodfellow, Nicolas Papernot](http://www.cleverhans.io/)

# Other Resources

- [AI Security and Privacy (AISP) Seminar Series](http://scl.sribd.cn/seminar/index.html)

- [ML Safety Newsletter](https://newsletter.mlsafety.org/)

- [Trustworthy ML Initiative](https://www.trustworthyml.org/home)

- [Trustworthy AI Project](https://www.trustworthyaiproject.eu/)

- [ECE1784H: Trustworthy Machine Learning (Course, Fall 2019) - Nicolas Papernot](https://www.papernot.fr/teaching/f19-trustworthy-ml)

- [A School for all Seasons on Trustworthy Machine Learning (Course) - Reza Shokri, Nicolas Papernot](https://trustworthy-machine-learning.github.io/)

- [Trustworthy Machine Learning (Book)](http://www.trustworthymachinelearning.com/)

- [AI Safety Support (Lots of Links)](https://www.aisafetysupport.org/lots-of-links)

# Contributing

Welcome to recommend paper that you find interesting and focused on trustworthy deep learning. You can submit an issue or contact me via [[email]](mailto:ming_hui.chen@outlook.com). Also, if there are any errors in the paper information, please feel free to correct me.

Formatting (The order of the papers is reversed based on the initial submission time to arXiv)
- Paper Title [[paper]](https://arxiv.org/abs/xxxx.xxxx)
  - Authors. *Published Conference or Journal*
  - Key Word: XXX.
  - <details><summary>Digest</summary> XXXXXX

