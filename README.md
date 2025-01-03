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

- NeuroAI for AI Safety. [[paper]](https://arxiv.org/abs/2411.18526)
  - Patrick Mineault, Niccolò Zanichelli, Joanne Zichen Peng, Anton Arkhipov, Eli Bingham, Julian Jara-Ettinger, Emily Mackevicius, Adam Marblestone, Marcelo Mattar, Andrew Payne, Sophia Sanborn, Karen Schroeder, Zenna Tavares, Andreas Tolias.
  - Key Word: AI Safety; Neuroscience; Robustness; Speicificaiton; Assurance.
  - <details><summary>Digest</summary> This paper highlights the potential of neuroscience to inspire advancements in AI safety, emphasizing the brain’s unique mechanisms for robustness, safe exploration, pragmatics, and cooperation. The authors propose leveraging neuroscience for AI safety through brain-inspired representations, architectures, robust sensory-motor systems, fine-tuning with brain data, interpretability methods, and scaling cognitively-inspired designs. Concrete recommendations are provided to integrate neuroscience into AI safety research.


## Out-of-Distribution Generalization

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#out-of-distribution-generalization">:open_file_folder: [<b><i>Full List of Out-of-Distribution Generalization</i></b>]</a>.

- Benign Overfitting in Out-of-Distribution Generalization of Linear Models. [[paper]](https://arxiv.org/abs/2412.14474)
  - Shange Tang, Jiayun Wu, Jianqing Fan, Chi Jin.
  - Key Word: Benign Overfitting; Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> This paper extends the theoretical understanding of benign overfitting—where over-parameterized models fit noisy training data perfectly but still generalize well—to the Out-of-Distribution (OOD) regime, focusing on linear models under covariate shift. The authors provide non-asymptotic guarantees showing that benign overfitting can occur in standard ridge regression when target covariances meet specific structural conditions. They identify key factors influencing OOD generalization and demonstrate that their results recover prior in-distribution and under-parameterized OOD findings. Additionally, they analyze a broader class of target covariances, showing that while ridge regression achieves a slow statistical rate, Principal Component Regression (PCR) achieves a faster rate for excess risk.

- The Pitfalls of Memorization: When Memorization Hurts Generalization. [[paper]](https://arxiv.org/abs/2412.07684)
  - Reza Bayat, Mohammad Pezeshki, Elvis Dohmatob, David Lopez-Paz, Pascal Vincent.
  - Key Word: Memorization; Generalization; Spurious Correlation.
  - <details><summary>Digest</summary> This paper investigates the relationship between memorization and generalization in neural networks, highlighting that reliance on spurious correlations combined with memorization harms generalization. To address this, the authors propose Memorization-Aware Training (MAT), which adjusts model logits based on held-out predictions to discourage memorization and promote learning robust, distribution-invariant patterns, thereby enhancing generalization under distribution shifts.

- Is Large-Scale Pretraining the Secret to Good Domain Generalization? [[paper]](https://arxiv.org/abs/2412.02856)
  - Piotr Teterwak, Kuniaki Saito, Theodoros Tsiligkaridis, Bryan A. Plummer, Kate Saenko.
  - Key Word: Domain Generalization; Pretraining.
  - <details><summary>Digest</summary> This paper examines Multi-Source Domain Generalization (DG), where models are trained on multiple source domains to generalize to unseen target domains. It questions whether recent DG improvements stem from better methods or stronger pretraining and finds that perceptual similarity to pretraining data is insufficient for strong performance. Instead, the authors propose the Alignment Hypothesis, which asserts that DG performance depends on the alignment of image and class label text embeddings. Experiments confirm this hypothesis, revealing that existing methods perform well on data similar to pretraining (IP) but struggle on dissimilar data (OOP). The findings emphasize the need for DG methods capable of generalizing beyond pretraining alignment.

- Transformation-Invariant Learning and Theoretical Guarantees for OOD Generalization. [[paper]](https://arxiv.org/abs/2410.23461)
  - Omar Montasser, Han Shao, Emmanuel Abbe.
  - Key Word: Out-of-Distribution Generalization; Distributionally Robust Optimization.
  - <details><summary>Digest</summary> This paper studies statistical learning under distribution shifts, focusing on scenarios where training and testing distributions are related by data transformation maps. It introduces theoretical learning rules and reductions to Empirical Risk Minimization (ERM), providing sample complexity bounds based on the VC dimension of combined predictors and transformations. The results offer a game-theoretic perspective, where a learner selects predictors to minimize worst-case loss while an adversary selects transformations to maximize it.

- Compositional Risk Minimization. [[paper]](https://arxiv.org/abs/2410.06303)
  - Divyat Mahajan, Mohammad Pezeshki, Ioannis Mitliagkas, Kartik Ahuja, Pascal Vincent.
  - Key Word: Compositional Shifts.
  - <details><summary>Digest</summary> This paper addresses compositional shift, an extreme form of subpopulation shift where certain combinations of attributes are absent in the training data but appear in the test data. The authors propose compositional risk minimization (CRM), a new approach that builds on additive energy distributions to model data attributes. First, they train a classifier to predict attributes and then adjust it to handle compositional shifts. Their theoretical analysis shows that CRM can generalize to unseen attribute combinations, and empirical results demonstrate that CRM improves robustness compared to existing methods for handling subpopulation shifts.

- Rule Extrapolation in Language Models: A Study of Compositional Generalization on OOD Prompts. [[paper]](https://arxiv.org/abs/2409.13728)
  - Anna Mészáros, Szilvia Ujváry, Wieland Brendel, Patrik Reizinger, Ferenc Huszár.
  - Key Word: Compositional Generalization; Rule Extrapolation; Large Language Model.
  - <details><summary>Digest</summary> This paper explores the out-of-distribution (OOD) behavior of autoregressive large language models (LLMs), focusing on a new concept termed rule extrapolation. Rule extrapolation involves prompts that violate at least one rule in formal languages, which are defined by intersecting rules. The authors evaluate how different architectures, including linear, recurrent, Transformer, and state space models, handle rule extrapolation across various levels of complexity. Additionally, they aim to develop a normative theory of rule extrapolation, drawing inspiration from the Solomonoff prior in algorithmic information theory. The study seeks to deepen understanding of LLMs’ OOD generalization abilities beyond just the Transformer architecture.

- Beyond Discrepancy: A Closer Look at the Theory of Distribution Shift. [[paper]](https://arxiv.org/abs/2405.19156)
  - Robi Bhattacharjee, Nick Rittler, Kamalika Chaudhuri.
  - Key Word: Distribution Shift; Invariant Risk Minimization.
  - <details><summary>Digest</summary> This paper examines the theory of distribution shift for classifiers, proposing an Invariant-Risk-Minimization (IRM)-like assumption to determine when source data alone is sufficient for accurate target classification. It also provides conditions and theoretical guarantees for when unlabeled or labeled target data is necessary, addressing gaps in traditional learning theory.

- Domain Generalisation via Imprecise Learning. [[paper]](https://arxiv.org/abs/2404.04669)
  - Anurag Singh, Siu Lun Chau, Shahine Bouabid, Krikamol Muandet.
  - Key Word: Domain Generalization; Imprecise Learning.
  - <details><summary>Digest</summary> The paper introduces the Imprecise Domain Generalisation framework to address the challenge of out-of-distribution (OOD) generalisation in machine learning. It proposes an imprecise risk optimisation approach that allows learners to optimise against a continuous spectrum of generalisation strategies during training. The framework also allows model operators to specify their generalisation preference at deployment. The work showcases the benefits of integrating imprecision into domain generalisation through theoretical and empirical evidence.

- A Survey on Evaluation of Out-of-Distribution Generalization. [[paper]](https://arxiv.org/abs/2403.01874)
  - Han Yu, Jiashuo Liu, Xingxuan Zhang, Jiayun Wu, Peng Cui.
  - Key Word: Survey; Out-of-Distribution Generalization Evaluation.
  - <details><summary>Digest</summary> OOD generalization involves not only assessing a model's OOD generalization strength but also identifying where it generalizes well or poorly, including the types of distribution shifts it can handle and the safe versus risky input regions. This paper represents the first comprehensive review of OOD evaluation, categorizing existing research into three paradigms based on test data availability and briefly discussing OOD evaluation for pretrained models. It concludes with suggestions for future research directions in OOD evaluation.

## Evasion Attacks and Defenses

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#evasion-attacks-and-defenses">:open_file_folder: [<b><i>Full List of Evasion Attacks and Defenses</i></b>]</a>.

- Best-of-N Jailbreaking. [[paper]](https://arxiv.org/abs/2412.03556)
  - John Hughes, Sara Price, Aengus Lynch, Rylan Schaeffer, Fazl Barez, Sanmi Koyejo, Henry Sleight, Erik Jones, Ethan Perez, Mrinank Sharma.
  - Key Word: Jailbreaking; Bootstrapping.
  - <details><summary>Digest</summary> This paper presents Best-of-N (BoN) Jailbreaking, a black-box algorithm that exploits AI systems across modalities by iteratively sampling augmented prompts (e.g., random shuffling, capitalization) to elicit harmful responses. BoN achieves high attack success rates (e.g., 89% on GPT-4o, 78% on Claude 3.5) and bypasses defenses on both closed- and open-source models. It extends to vision and audio language models using modality-specific augmentations and improves with increased sampling, following a power-law pattern. Combining BoN with other attack strategies further enhances its effectiveness, exposing AI systems’ vulnerability to minor input variations. 

- AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents. [[paper]](https://arxiv.org/abs/2410.09024)
  - Maksym Andriushchenko, Alexandra Souly, Mateusz Dziemian, Derek Duenas, Maxwell Lin, Justin Wang, Dan Hendrycks, Andy Zou, Zico Kolter, Matt Fredrikson, Eric Winsor, Jerome Wynne, Yarin Gal, Xander Davies.
  - Key Word: LLM Agents; Jailbreak; Benchmark.
  - <details><summary>Digest</summary> This paper introduces AgentHarm, a new benchmark designed to evaluate the robustness of LLM agents against jailbreak attacks, which attempt to bypass safety measures and misuse model capabilities. The benchmark includes 110 malicious tasks across 11 harm categories (e.g., fraud, cybercrime), with 440 tasks in total after augmentation. The study finds that leading LLMs are often compliant with harmful requests without needing jailbreaking, that simple jailbreak techniques can be easily adapted for agent tasks, and that these jailbreaks allow malicious multi-step behaviors while retaining model capabilities. The authors release AgentHarm for evaluating attacks and defenses in LLM agents.

- Automated Red Teaming with GOAT: the Generative Offensive Agent Tester. [[paper]](https://arxiv.org/abs/2410.01606)
  - Maya Pavlova, Erik Brinkman, Krithika Iyer, Vitor Albiero, Joanna Bitton, Hailey Nguyen, Joe Li, Cristian Canton Ferrer, Ivan Evtimov, Aaron Grattafiori.
  - Key Word: Red Teaming; Jailbreak Attack; Large Language Model; Reasoning.
  - <details><summary>Digest</summary> The paper introduces GOAT (Generative Offensive Agent Tester), an automated system for red-teaming large language models (LLMs). It addresses limitations in current red-teaming approaches, which often do not reflect how typical users interact with AI models. Instead of complex adversarial techniques, GOAT simulates realistic, plain-language adversarial conversations using multiturn interactions. It incorporates multiple adversarial prompting methods to identify vulnerabilities in LLMs efficiently. GOAT demonstrated high success rates in identifying vulnerabilities, with 97% ASR@10 against Llama 3.1 and 88% against GPT-4 on the JailbreakBench dataset, streamlining the red-teaming process for AI safety testing.

- Emerging Vulnerabilities in Frontier Models: Multi-Turn Jailbreak Attacks. [[paper]](https://arxiv.org/abs/2409.00137)
  - Tom Gibbs, Ethan Kosak-Hine, George Ingebretsen, Jason Zhang, Julius Broomfield, Sara Pieri, Reihaneh Iranmanesh, Reihaneh Rabbany, Kellin Pelrine.
  - Key Word: Jailbreak Attack.
  - <details><summary>Digest</summary> This paper discusses how large language models (LLMs), despite their advancements, remain vulnerable to jailbreak attacks. It introduces a dataset of jailbreak examples in both single and multi-turn formats, showing that defenses effective in one format may not work in the other. The study emphasizes the need to examine vulnerabilities in both structures, as LLM-based filters perform differently depending on input structure, not just content.

- Exploring Scaling Trends in LLM Robustness. [[paper]](https://arxiv.org/abs/2407.18213)
  - Nikolhaus Howe, Michał Zajac, Ian McKenzie, Oskar Hollinsworth, Tom Tseng, Pierre-Luc Bacon, Adam Gleave.
  - Key Word: Adversarial Robustness; Neural Scaling Law; Large Language Model.
  - <details><summary>Digest</summary> Scaling language models in size and training data improves their capabilities, but without explicit defenses, this scaling does not enhance their robustness against adversarial prompts. Empirical findings show that larger models benefit significantly from adversarial training, unlike their performance with mere scale increases.

- Adversaries Can Misuse Combinations of Safe Models. [[paper]](https://arxiv.org/abs/2406.14595)
  - Erik Jones, Anca Dragan, Jacob Steinhardt.
  - Key Word: Red-Teaming; Large Language Model.
  - <details><summary>Digest</summary> Developers often test AI systems for potential misuse before release, such as in cyberoffense or bioterrorism. However, this study reveals that testing individual models is insufficient because adversaries can combine multiple models to achieve malicious goals. By breaking tasks into subtasks and using the most suitable model for each, adversaries can exploit even safe models. The study examines both manual and automated task decomposition methods, showing that combining models significantly increases the risk of generating harmful outputs like vulnerable code or manipulative content. This suggests that red-teaming efforts should consider the combined use of multiple models, not just individual ones.

- Adversarial Attacks on Multimodal Agents. [[paper]](https://arxiv.org/abs/2406.12814)
  - Chen Henry Wu, Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried, Aditi Raghunathan.
  - Key Word: Adversarial Attacks; Multimodal Agents.
  - <details><summary>Digest</summary> The paper demonstrates that Vision-Language Models (VLMs) used in autonomous multimodal agents introduce new safety risks through adversarial attacks, despite the challenges in executing these attacks due to limited environmental access. The authors present two types of attacks—captioner and CLIP attacks—showing significant success rates in manipulating agent behavior, and they evaluate these attacks using a curated set of adversarial tasks, revealing varying robustness across different VLM-based agents.

- Improving Alignment and Robustness with Short Circuiting. [[paper]](https://arxiv.org/abs/2406.04313)
  - Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks.
  - Key Word: Representation Engineering; Adversarial Attacks and Defenses.
  - <details><summary>Digest</summary> This paper introduces a novel "short-circuiting" approach inspired by representation engineering to prevent harmful outputs in AI systems, addressing the limitations of refusal and adversarial training techniques. The method effectively controls harmful representations in text-only and multimodal language models, enhancing robustness against unseen attacks and reducing harmful actions in AI agents.

- Certifiably Robust RAG against Retrieval Corruption. [[paper]](https://arxiv.org/abs/2405.15556)
  - Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner, Danqi Chen, Prateek Mittal.
  - Key Word: Adversarial Defense; Retrieval Corruption Attacks; Retrieval-Augmented Generation.
  - <details><summary>Digest</summary> RobustRAG is a defense framework against retrieval corruption attacks in Retrieval-Augmented Generation (RAG) systems, using an isolate-then-aggregate strategy to ensure accurate responses by securely aggregating isolated responses from each passage. It achieves certifiable robustness, proving its effectiveness even when attackers inject malicious passages, as demonstrated in evaluations on various datasets.

- Uniformly Stable Algorithms for Adversarial Training and Beyond. [[paper]](https://arxiv.org/abs/2405.01817)
  - Jiancong Xiao, Jiawei Zhang, Zhi-Quan Luo, Asuman Ozdaglar.
  - Key Word: Adversarial Training; Uniform Stability.
  - <details><summary>Digest</summary> The abstract discusses a new algorithm called Moreau envelope-A (ME-A) designed to address the issue of robust overfitting in adversarial machine learning. Robust overfitting occurs when the robust test accuracy of neural networks decreases over training epochs. Recent research has shown that SGD-based adversarial training lacks uniform stability, which aligns with the observed robust overfitting. ME-A achieves uniform stability for weakly-convex, non-smooth problems without additional computational overhead by reframing the original problem as a min-min problem using a Moreau envelope function. The efficacy of ME-A in mitigating robust overfitting is demonstrated in practical scenarios.

- AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs. [[paper]](https://arxiv.org/abs/2404.16873)
  - Anselm Paulus, Arman Zharmagambetov, Chuan Guo, Brandon Amos, Yuandong Tian.
  - Key Word: Jailbreak; Prompting; Large Language Model.
  - <details><summary>Digest</summary> Manual red-teaming is inefficient, while automatic adversarial prompt generation often produces meaningless attacks. This paper introduces a novel method using the AdvPrompter, an LLM, to generate human-readable adversarial prompts much faster than existing approaches. The AdvPrompter veils input instructions without changing their meaning, luring the TargetLLM to give harmful responses. Experimental results demonstrate state-of-the-art performance on open source and closed-source LLMs, making them more robust against jailbreaking attacks while maintaining high MMLU scores.

- JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models. [[paper]](https://arxiv.org/abs/2404.01318)
  - Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong.
  - Key Word: Jailbreak; Large Language Model; Benchmark.
  - <details><summary>Digest</summary> JailbreakBench addresses challenges in evaluating jailbreak attacks on large language models (LLMs), which produce harmful content. It introduces a benchmark with a new dataset (JBB-Behaviors), a repository of adversarial prompts, a standardized evaluation framework, and a leaderboard for tracking attack and defense performance. It aims to standardize practices and enhance reproducibility in the field while considering ethical implications, planning to evolve with research advancements.

- Curiosity-driven Red-teaming for Large Language Models. [[paper]](https://arxiv.org/abs/2402.19464)
  - Zhang-Wei Hong, Idan Shenfeld, Tsun-Hsuan Wang, Yung-Sung Chuang, Aldo Pareja, James Glass, Akash Srivastava, Pulkit Agrawal.
  - Key Word: Red-Teaming; Large Language Model; Reinforcement Learning.
  - <details><summary>Digest</summary> The paper presents a method called curiosity-driven red teaming (CRT) to improve the detection of undesirable outputs from large language models (LLMs). Traditional methods rely on costly and slow human testers or automated systems with limited effectiveness. CRT enhances the scope and efficiency of test cases by using curiosity-driven exploration to provoke toxic responses, even from LLMs fine-tuned to avoid such issues. 


## Poisoning Attacks and Defenses

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#poisoning-attacks-and-defenses">:open_file_folder: [<b><i>Full List of Poisoning Attacks and Defenses</i></b>]</a>.

- BadMerging: Backdoor Attacks Against Model Merging. [[paper]](https://arxiv.org/abs/2408.07362)
  - Jinghuai Zhang, Jianfeng Chi, Zheng Li, Kunlin Cai, Yang Zhang, Yuan Tian.
  - Key Word: Backdoor Attacks; Model Merging.
  - <details><summary>Digest</summary> The paper introduces "BadMerging," a backdoor attack specifically designed for Model Merging (MM), a method that combines multiple fine-tuned task-specific models without additional training. BadMerging enables an adversary to compromise an entire merged model by injecting a backdoor into just one task-specific model. The attack is robust against various merging parameters and can affect both the adversary's tasks (on-task attack) and other contributors' tasks (off-task attack). Extensive experiments demonstrate the effectiveness of BadMerging and reveal that existing defense mechanisms are inadequate, underscoring the need for more advanced defenses in the context of MM.

- Scaling Laws for Data Poisoning in LLMs. [[paper]](https://arxiv.org/abs/2408.02946)
  - Dillon Bowen, Brendan Murphy, Will Cai, David Khachaturov, Adam Gleave, Kellin Pelrine.
  - Key Word: Data Poisoning; Large Language Model; Neural Scaling Law.
  - <details><summary>Digest</summary> Recent research indicates that large language models (LLMs) are increasingly vulnerable to data poisoning, which involves training on corrupted or harmful data. This poisoning is difficult to detect, undermines safeguards, and results in undesirable behaviors. The study evaluates three threat models: malicious fine-tuning, imperfect data curation, and intentional data contamination, using 23 LLMs (1.5-72 billion parameters) across three datasets. Findings reveal that larger LLMs are more susceptible to learning harmful behaviors quickly with even minimal poisoning, highlighting the urgent need for robust protections against data poisoning as LLMs scale.

- AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases. [[paper]](https://arxiv.org/abs/2407.12784)
  - Zhaorun Chen, Zhen Xiang, Chaowei Xiao, Dawn Song, Bo Li.
  - Key Word: Red-Teaming; Poisoning Attacks; Retrieval-Augmented Generation.
  - <details><summary>Digest</summary> LLM agents excel in various tasks by leveraging advanced reasoning, external knowledge, and memory modules, but their dependence on unverified knowledge bases raises safety concerns. AgentPoison, a novel backdoor attack, poisons these memory modules or knowledge bases to manipulate outputs maliciously, achieving over 80% success in targeted attacks with minimal impact on normal performance.

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

- Provable unlearning in topic modeling and downstream tasks. [[paper]](https://arxiv.org/abs/2411.12600)
  - Stanley Wei, Sadhika Malladi, Sanjeev Arora, Amartya Sanyal.
  - Key Word: Machine Unlearning; Certified Unlearning; Pre-Training and Fine-Tuning.
  - <details><summary>Digest</summary> This paper introduces the first theoretical guarantees for unlearning in the pre-training and fine-tuning paradigm, focusing on topic models. The authors propose a computationally efficient unlearning algorithm for topic models, independent of the dataset size, and quantify the model’s deletion capacity (i.e., the number of examples that can be unlearned without degrading performance). They extend their analysis to fine-tuned models and design an algorithm for unlearning after fine-tuning with a linear head. Notably, they find that pre-training data is easier to unlearn in fine-tuned models, and this can be done without altering the base model.

- On the Privacy Risk of In-context Learning. [[paper]](https://arxiv.org/abs/2411.10512)
  - Haonan Duan, Adam Dziedzic, Mohammad Yaghini, Nicolas Papernot, Franziska Boenisch.
  - Key Word: In-Context Learning; Membership Inference.
  - <details><summary>Digest</summary> This paper highlights the privacy risks of using large language models (LLMs) with natural language prompts containing private data. It demonstrates that prompted models are more vulnerable to membership inference attacks than fine-tuned models with similar utility. The increased risk is attributed to the models’ heightened prediction confidence on prompted data. To mitigate this risk, the authors propose using ensembling, aggregating outputs from multiple model versions to reduce membership inference vulnerabilities.

- Membership Inference Attacks Against In-Context Learning. [[paper]](https://arxiv.org/abs/2409.01380)
  - Rui Wen, Zheng Li, Michael Backes, Yang Zhang.
  - Key Word: Membership Inference Attacks; In-Context Learning.
  - <details><summary>Digest</summary> This paper introduces the first membership inference attack tailored for In-Context Learning (ICL), focusing on identifying privacy risks without relying on associated probabilities. Four attack strategies are proposed, achieving high accuracy (e.g., 95% against LLaMA), demonstrating greater risks than previous probability-based methods. The paper also proposes a hybrid attack with strong performance and explores three defenses that, when combined, significantly reduce privacy leakage.

- Verification of Machine Unlearning is Fragile. [[paper]](https://arxiv.org/abs/2408.00929)
  - Binchi Zhang, Zihan Chen, Cong Shen, Jundong Li.
  - Key Word: Machine Unlearning; Backdoor Verification.
  - <details><summary>Digest</summary> As privacy concerns rise, data owners can now use machine unlearning to remove their data from machine learning models, as required by recent legislation. To ensure transparency and prevent dishonesty by model providers, various verification strategies have been proposed. However, the safety of these verification strategies is not well understood. This paper explores whether model providers can bypass these verification strategies while retaining unlearned data information. The findings indicate that verification is fragile, with two types of strategies identified and two novel adversarial unlearning processes introduced that can circumvent both. The study’s theoretical and empirical analyses reveal vulnerabilities in machine unlearning verification, highlighting the need for further research.

- Faster Machine Unlearning via Natural Gradient Descent. [[paper]](https://arxiv.org/abs/2407.08169)
  - Omri Lev, Ashia Wilson.
  - Key Word: Machine Unlearning; Natural Gradient.
  - <details><summary>Digest</summary> The paper introduces a novel algorithm using Natural Gradient Descent (NGD) for efficient and reliable data deletion from machine learning models without retraining. The approach offers strong privacy guarantees for convex models and employs a practical Min/Max optimization for non-convex models, demonstrating significant improvements in privacy, computational efficiency, and generalization over existing methods.

- UnUnlearning: Unlearning is not sufficient for content regulation in advanced generative AI. [[paper]](https://arxiv.org/abs/2407.00106)
  - Ilia Shumailov, Jamie Hayes, Eleni Triantafillou, Guillermo Ortiz-Jimenez, Nicolas Papernot, Matthew Jagielski, Itay Yona, Heidi Howard, Eugene Bagdasaryan.
  - Key Word: Machine Unlearning; Generative AI.
  - <details><summary>Digest</summary> Exact unlearning allows users to retract data from machine learning models, but inexact schemes were developed to reduce impractical costs. This paper highlights that while unlearning is effective during training, it does not prevent models from performing impermissible acts during inference, introducing the concept of "ununlearning" where forgotten knowledge can be reintroduced, necessitating content filtering for effective regulation.

- Machine Unlearning Fails to Remove Data Poisoning Attacks. [[paper]](https://arxiv.org/abs/2406.17216)
  - Martin Pawelczyk, Jimmy Z. Di, Yiwei Lu, Gautam Kamath, Ayush Sekhari, Seth Neel.
  - Key Word: Machine Unlearning; Poisoning Attack.
  - <details><summary>Digest</summary> We investigate the effectiveness of various practical methods for approximate machine unlearning in deep learning, finding that they fail to effectively counteract data poisoning attacks across multiple scenarios and models. Our study introduces new evaluation metrics for unlearning, highlighting the need for broader evaluations to avoid overconfidence in current unlearning methods, which still fall short of the benefits of retraining.

- Recovering Labels from Local Updates in Federated Learning. [[paper]](https://arxiv.org/abs/2405.00955)
  - Huancheng Chen, Haris Vikalo.
  - Key Word: Federated Learning; Gradient Inversion Attack; Label Recovery Attack.
  - <details><summary>Digest</summary> The paper presents RLU, a novel label recovery scheme for gradient inversion attacks in federated learning that outperforms existing methods. RLU provides near-perfect accuracy on untrained models and maintains high performance in realistic settings with multiple local epochs, heterogeneous data, and various optimizers. By analyzing the correlation between data labels and output layer updates, RLU improves the quality of reconstructed images, posing a significant privacy risk in real-world federated learning scenarios.

- Rethinking LLM Memorization through the Lens of Adversarial Compression. [[paper]](https://arxiv.org/abs/2404.15146)
  - Avi Schwarzschild, Zhili Feng, Pratyush Maini, Zachary C. Lipton, J. Zico Kolter.
  - Key Word: Large Language Model; Memorization; Machine Unlearning; Compression.
  - <details><summary>Digest</summary> The abstract discusses concerns about the use of permissible data in large language models (LLMs) trained on extensive web datasets, particularly focusing on whether these models memorize training data or synthesize information like humans. To address this, the study introduces the Adversarial Compression Ratio (ACR), a new metric to assess memorization. ACR considers data memorized if it can be elicited by a prompt shorter than the string itself, effectively using adversarial prompts to measure memorization. This metric helps monitor compliance and unlearning, offering a practical tool for evaluating potential legal issues regarding data usage in LLMs.

- SoK: Challenges and Opportunities in Federated Unlearning. [[paper]](https://arxiv.org/abs/2403.02437)
  - Key Word: Survey; Federated Unlearning.
  - <details><summary>Digest</summary> Federated Learning (FL) enables collaborative learning among non-trusting parties without data sharing, adhering to privacy regulations and introducing the need for mechanisms to "forget" specific learned data, thus spurring research in "machine unlearning" tailored for FL's unique challenges. This State of Knowledge (SoK) paper reviews federated unlearning research, categorizes existing approaches, and discusses their limitations and implications, aiming to provide insights and directions for future work in this emerging field.

- Eight Methods to Evaluate Robust Unlearning in LLMs. [[paper]](https://arxiv.org/abs/2402.16835)
  - Aengus Lynch, Phillip Guo, Aidan Ewart, Stephen Casper, Dylan Hadfield-Menell.
  - Key Word: Large Language Model; Machine Unlearning.
  - <details><summary>Digest</summary> This paper critiques the evaluation of unlearning in large language models (LLMs) by surveying current methods, testing the "Who's Harry Potter" (WHP) model's unlearning effectiveness, and demonstrating the limitations of ad-hoc evaluations. Despite WHP's initial success in specific metrics, it still retains considerable knowledge, performs similarly on related tasks, and shows unintended unlearning in adjacent domains. The findings emphasize the necessity for rigorous and comprehensive evaluation techniques to accurately assess unlearning in LLMs.

- UnlearnCanvas: A Stylized Image Dataset to Benchmark Machine Unlearning for Diffusion Models ([paper](https://arxiv.org/abs/2402.11846))
  - Yihua Zhang, Yimeng Zhang, Yuguang Yao, Jinghan Jia, Jiancheng Liu, Xiaoming Liu, Sijia Liu.
  - Key Word: Machine Unlearning; Diffusion Model.
  - <details><summary>Digest</summary> This work uncovers several key challenges that can result in incomplete, inaccurate, or biased evaluations for machine unlearning (MU) in diffusion models (DMs) by examining existing MU evaluation methods. To address them, this work enhances the evaluation metrics for MU, including the introduction of an often-overlooked retainability measurement for DMs post-unlearning. Additionally, it introduces UnlearnCanvas, a comprehensive high-resolution stylized image dataset that facilitates us to evaluate the unlearning of artistic painting styles in conjunction with associated image objects. This work shows that this dataset plays a pivotal role in establishing a standardized and automated evaluation framework for MU techniques on DMs, featuring 7 quantitative metrics to address various aspects of unlearning effectiveness. Through extensive experiments, it benchmarks 5 state-of-the-art MU methods, revealing novel insights into their pros and cons, and the underlying unlearning mechanisms.

- Data Reconstruction Attacks and Defenses: A Systematic Evaluation. [[paper]](https://arxiv.org/abs/2402.09478)
  - Sheng Liu, Zihan Wang, Qi Lei.
  - Key Word: Reconstruction Attacks and Defenses.
  - <details><summary>Digest</summary> This paper introduces a robust reconstruction attack in federated learning that outperforms existing methods by reconstructing intermediate features. It critically analyzes the effectiveness of common defense mechanisms against such attacks, both theoretically and empirically. The study identifies gradient pruning as the most effective defense strategy against advanced reconstruction attacks, highlighting the need for a deeper understanding of the balance between attack potency and defense efficacy in machine learning.

- Rethinking Machine Unlearning for Large Language Models. [[paper]](https://arxiv.org/abs/2402.08787)
  - Sijia Liu, Yuanshun Yao, Jinghan Jia, Stephen Casper, Nathalie Baracaldo, Peter Hase, Xiaojun Xu, Yuguang Yao, Hang Li, Kush R. Varshney, Mohit Bansal, Sanmi Koyejo, Yang Liu.
  - Key Word: Machine Unlearning; Large Language Model.
  - <details><summary>Digest</summary> The abstract discusses the concept of machine unlearning in the context of large language models (LLMs), focusing on selectively removing undesired data influences (such as sensitive or illegal content) without compromising the model's ability to generate valuable knowledge. The goal is to ensure LLMs are safe, secure, trustworthy, and resource-efficient, eliminating the need for complete retraining. It covers the conceptual basis, methodologies, metrics, and applications of LLM unlearning, addressing overlooked aspects like unlearning scope and data-model interaction. The paper also connects LLM unlearning with related fields like model editing and adversarial training, proposing an assessment framework for its efficacy, especially in copyright, privacy, and harm reduction.

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

- Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data. [[paper]](https://arxiv.org/abs/2410.13341)
  - Florian E. Dorner, Vivian Y. Nastl, Moritz Hardt.
  - Key Word: LLM-as-a-Judge; Debiasing.
  - <details><summary>Digest</summary> This paper examines the limitations of using strong models as judges to evaluate other models without costly annotations. The authors show that debiasing methods, which use a few high-quality labels to correct biases in model evaluations, can only reduce the need for ground truth labels by at most half when the judge is no more accurate than the evaluated model. They highlight the limitations of this approach, especially when evaluating new models potentially better than the judge. Empirical results confirm that the actual savings in label usage are smaller than the theoretical limit.

- Whose Preferences? Differences in Fairness Preferences and Their Impact on the Fairness of AI Utilizing Human Feedback. [[paper]](https://arxiv.org/abs/2406.05902)
  - Emilia Agis Lerner, Florian E. Dorner, Elliott Ash, Naman Goel.
  - Key Word: Fairness Preferences.
  - <details><summary>Digest</summary> The study investigates fairness in content moderation using human feedback to compare how comments referencing different sensitive attribute groups should be treated. It reveals significant gaps in fairness preferences based on annotators' demographics and shows that an ensemble model, which equally weights classifiers trained on annotations from different demographics, performs better across various demographic intersections than a single classifier.

- Low-rank finetuning for LLMs: A fairness perspective. [[paper]](https://arxiv.org/abs/2405.18572)
  - Saswat Das, Marco Romanelli, Cuong Tran, Zarreen Reza, Bhavya Kailkhura, Ferdinando Fioretto.
  - Key Word: Low-Rank Fine-Tuning; Large Language Model; Harmful Biases.
  - <details><summary>Digest</summary> Low-rank approximation techniques are commonly used for fine-tuning Large Language Models (LLMs) due to their efficiency, but they often fail to capture shifts in fine-tuning datasets, leading to the preservation of biases and toxic behaviors. This paper provides empirical evidence showing that these shortcomings are particularly problematic in tasks requiring fairness and toxicity mitigation, highlighting the need for careful evaluation in LLM development.

- Fairness in Serving Large Language Models. [[paper]](https://arxiv.org/abs/2401.00588)
  - Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu, Zhuohan Li, Danyang Zhuo, Joseph E. Gonzalez, Ion Stoica.
  - Key Word: Fairness; Large Language Model; Large Languge Model Serving System.
  - <details><summary>Digest</summary> The paper addresses the challenge of ensuring fair processing of client requests in high-demand Large Language Model (LLM) inference services. Current rate limits can lead to resource underutilization and poor client experiences. The paper introduces LLM serving fairness based on a cost function that considers input and output tokens. It presents a novel scheduling algorithm, Virtual Token Counter (VTC), which achieves fairness by continuous batching. The paper proves a tight upper bound on service difference between backlogged clients, meeting work-conserving requirements. Extensive experiments show that VTC outperforms other baseline methods in ensuring fairness under different conditions.

## Interpretability

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#interpretability">:open_file_folder: [<b><i>Full List of Interpretability</i></b>]</a>.

- Concept Bottleneck Language Models For protein design. [[paper]](https://arxiv.org/abs/2411.06090)
  - Aya Abdelsalam Ismail, Tuomas Oikarinen, Amy Wang, Julius Adebayo, Samuel Stanton, Taylor Joren, Joseph Kleinhenz, Allen Goodman, Héctor Corrada Bravo, Kyunghyun Cho, Nathan C. Frey.
  - Key Word: Concept Bottleneck; Protein Language Model.
  - <details><summary>Digest</summary> We propose Concept Bottleneck Protein Language Models (CB-pLM), a generative model where neurons correspond to interpretable concepts, enabling control over generated protein properties, interpretability, and debugging. CB-pLM achieves competitive performance with standard models and scales up to 3 billion parameters, marking it as the largest Concept Bottleneck Model for protein language tasks, particularly valuable in drug discovery.

- Towards Unifying Interpretability and Control: Evaluation via Intervention. [[paper]](https://arxiv.org/abs/2411.04430)
  - Usha Bhalla, Suraj Srinivas, Asma Ghandeharioun, Himabindu Lakkaraju.
  - Key Word: Mechanistic Interpretability; Causal Intervention.
  - <details><summary>Digest</summary> This paper addresses the need for both interpretability and control in large language models, proposing “intervention” as a core goal of interpretability to better align model behavior. The authors unify four interpretability methods into an encoder-decoder framework that allows for interventions on human-interpretable features, enabling control over model outputs. They introduce two evaluation metrics—intervention success rate and coherence-intervention tradeoff—to assess control effectiveness. Findings show that while interventions are feasible, current methods are inconsistent and often reduce model performance, with lens-based methods performing best but still falling short compared to simpler approaches like prompting.

- Hypothesis Testing the Circuit Hypothesis in LLMs. [[paper]](https://arxiv.org/abs/2410.13032)
  - Claudia Shi, Nicolas Beltran-Velez, Achille Nazaret, Carolina Zheng, Adrià Garriga-Alonso, Andrew Jesson, Maggie Makar, David M. Blei.
  - Key Word: Circuit Hypothesis; Mechanistic Interpretability.
  - <details><summary>Digest</summary> This paper explores the hypothesis that large language models (LLMs) execute their capabilities through small subnetworks, called circuits. The authors formalize criteria to evaluate these circuits, focusing on whether they preserve LLM behavior, are localized, and are minimal. They develop tests to assess circuits and apply them to six circuits from the literature. Results show that synthetic circuits align with ideal properties, while discovered circuits vary. To support further research, they introduce the circuitry package, built on the TransformerLens library, simplifying circuit analysis in Transformer models.

- ContextCite: Attributing Model Generation to Context. [[paper]](https://arxiv.org/abs/2409.00729)
  - Benjamin Cohen-Wang, Harshay Shah, Kristian Georgiev, Aleksander Madry.
  - Key Word: Context Attribution.
  - <details><summary>Digest</summary> This paper introduces the concept of context attribution—identifying which parts of a context prompted a language model to generate a specific response. The authors propose ContextCite, a method that can be applied to any language model to track this context-to-response relationship. They demonstrate the utility of ContextCite in three areas: verifying the accuracy of generated statements, improving response quality by trimming unnecessary context, and detecting poisoning attacks.

- The Quest for the Right Mediator: A History, Survey, and Theoretical Grounding of Causal Interpretability. [[paper]](https://arxiv.org/abs/2408.01416)
  - Aaron Mueller, Jannik Brinkmann, Millicent Li, Samuel Marks, Koyena Pal, Nikhil Prakash, Can Rager, Aruna Sankaranarayanan, Arnab Sen Sharma, Jiuding Sun, Eric Todd, David Bau, Yonatan Belinkov.
  - Key Word: Causal Interpretability; Survey.
  - <details><summary>Digest</summary> Interpretability helps us understand neural networks’ behaviors, but the field lacks unity, using ad-hoc evaluations and lacking shared theoretical foundations, making it hard to measure progress and compare techniques. Basic causal units in mechanisms are often undefined. This paper proposes a perspective using causal mediation analysis, categorizing interpretability by types of causal units (mediators) and search methods. It evaluates the pros and cons of each mediator, suggesting when different mediators and methods are appropriate. The paper advocates for discovering new mediators that balance human-interpretability and computational efficiency, and for standardized evaluations to enable better comparisons across mediator types.

- Auditing Local Explanations is Hard. [[paper]](https://arxiv.org/abs/2407.13281)
  - Robi Bhattacharjee, Ulrike von Luxburg.
  - Key Word: Local Explanation; Auditable Explanation.
  - <details><summary>Digest</summary> In contexts requiring explanations for machine learning decisions, third-party auditors or user collectives can sanity-check these explanations by querying model decisions and local explanations to ensure consistency. However, our study reveals that this auditing process may require an impractically high number of queries, especially in high-dimensional settings, highlighting the need for attention to the "locality" of explanations.

- LLM Circuit Analyses Are Consistent Across Training and Scale. [[paper]](https://arxiv.org/abs/2407.10827)
  - Curt Tigges, Michael Hanna, Qinan Yu, Stella Biderman.
  - Key Word: Mechanistic Interpretability; Large Language Model Circuit.
  - <details><summary>Digest</summary> This study examines how mechanisms within decoder-only large language models (LLMs) develop and change across extensive training. It finds that task abilities and their supporting components emerge consistently across models of various scales, suggesting that analyses of small models at the end of pre-training can be relevant for larger, continuously trained models.

- Provably Better Explanations with Optimized Aggregation of Feature Attributions. [[paper]](https://arxiv.org/abs/2406.05090)
  - Thomas Decker, Ananta R. Bhattarai, Jindong Gu, Volker Tresp, Florian Buettner.
  - Key Word: Feature Attribution; Explanation Aggregation.
  - <details><summary>Digest</summary> The paper proposes a novel approach to improve the quality of feature attributions in machine learning by combining multiple explanation methods, aiming to enhance robustness and faithfulness to model behavior. Extensive experiments demonstrate that this combination strategy consistently outperforms individual methods and existing baselines.

- Iteration Head: A Mechanistic Study of Chain-of-Thought. [[paper]](https://arxiv.org/abs/2406.02128)
  - Vivien Cabannes, Charles Arnal, Wassim Bouaziz, Alice Yang, Francois Charton, Julia Kempe.
  - Key Word: Mechanistic Interpretability; Chain-of-Thought.
  - <details><summary>Digest</summary> This paper investigates the emergence of Chain-of-Thought (CoT) reasoning in transformers, revealing the development of specialized attention mechanisms termed "iteration heads" that facilitate iterative reasoning. The study tracks the formation and functionality of these iteration heads at the attention level and evaluates the transferability of CoT skills across different tasks.

- Editable Concept Bottleneck Models. [[paper]](https://arxiv.org/abs/2405.15476)
  - Lijie Hu, Chenyang Ren, Zhengyu Hu, Cheng-Long Wang, Di Wang.
  - Key Word: Concept Bottleneck Models.
  - <details><summary>Digest</summary> Concept Bottleneck Models (CBMs) are interpretable but assume clean data, which is often unrealistic. To avoid retraining CBMs after data edits due to issues like privacy or errors, the authors propose Editable Concept Bottleneck Models (ECBMs). ECBMs enable efficient data removal at the concept-label, concept, and data levels using influence functions for closed-form approximations. Experiments show ECBMs are effective and adaptable, maintaining CBM functionality without retraining.

- Manifold Integrated Gradients: Riemannian Geometry for Feature Attribution. [[paper]](https://arxiv.org/abs/2405.09800)
  - Eslam Zaher, Maciej Trzaskowski, Quan Nguyen, Fred Roosta.
  - Key Word: Feature Attribution; Integrated Gradient; Riemannian Geometry.
  - <details><summary>Digest</summary> This study addresses reliability issues of Integrated Gradients (IG), a feature attribution method for deep learning models, focusing on noisy feature visualizations and vulnerability to adversarial attacks. It proposes an adapted path-based feature attribution that aligns with the data's intrinsic geometry, resulting in more intuitive explanations and increased robustness to targeted attacks.

- Linear Explanations for Individual Neurons. [[paper]](https://arxiv.org/abs/2405.06855)
  - Tuomas Oikarinen, Tsui-Wei Weng.
  - Key Word: Automated Interpretability; Large Languge Model; Linear Probes.
  - <details><summary>Digest</summary> This paper challenges the common practice of understanding neural networks by focusing on the highest activations of individual neurons. The authors argue that this approach is insufficient as it only accounts for a small percentage of a neuron's causal effect. They propose that neurons should be understood as a linear combination of concepts and present an efficient method for generating these linear explanations. The paper also introduces a way to automatically evaluate the quality of these descriptions through simulation, predicting neuron activations on unseen inputs in a vision setting.

- A Primer on the Inner Workings of Transformer-based Language Models. [[paper]](https://arxiv.org/abs/2405.00208)
  - Javier Ferrando, Gabriele Sarti, Arianna Bisazza, Marta R. Costa-jussà.
  - Key Word: Transformer; Language Model; Attribution; Feature Visualization.
  - <details><summary>Digest</summary> The rapid progress of research aimed at interpreting the inner workings of advanced language models has highlighted a need for contextualizing the insights gained from years of work in this area. This primer provides a concise technical introduction to the current techniques used to interpret the inner workings of Transformer-based language models, focusing on the generative decoder-only architecture. We conclude by presenting a comprehensive overview of the known internal mechanisms implemented by these models, uncovering connections across popular approaches and active research directions in this area.

- A Multimodal Automated Interpretability Agent. [[paper]](https://arxiv.org/abs/2404.14394)
  - Tamar Rott Shaham, Sarah Schwettmann, Franklin Wang, Achyuta Rajaram, Evan Hernandez, Jacob Andreas, Antonio Torralba.
  - Key Word: Multimodal Modal; Automatic Interpretability; Large Language Model Agent.
  - <details><summary>Digest</summary> The paper presents MAIA (Multimodal Automated Interpretability Agent), a system designed to automate the understanding of neural models, including feature interpretation and failure mode discovery. MAIA uses a pre-trained vision-language model and a suite of tools to experiment iteratively on other models' subcomponents. These tools allow for the synthesis and editing of inputs, computation of maximally activating exemplars from real-world datasets, and summarization of experimental results. The paper demonstrates MAIA's ability to describe neuron-level features in image representations, with results comparable to those generated by human experts. Furthermore, MAIA can reduce sensitivity to spurious features and identify inputs that are likely to be misclassified.

- Decomposing and Editing Predictions by Modeling Model Computation. [[paper]](https://arxiv.org/abs/2404.11534)
  - Harshay Shah, Andrew Ilyas, Aleksander Madry.
  - Key Word: Component Attribution.
  - <details><summary>Digest</summary> This paper introduces the concept of component modeling, a method to understand how machine learning models transform inputs into predictions by breaking down the model's computation into its basic functions or components. A specific task, called component attribution, is highlighted, which aims to estimate the impact of individual components on a prediction. The authors present a scalable algorithm, COAR, for estimating component attributions and demonstrate its effectiveness across various models, datasets, and modalities. They also show that component attributions estimated with COAR can be used to edit models across five tasks: fixing model errors, "forgetting" specific classes, enhancing subpopulation robustness, localizing backdoor attacks, and improving robustness to typographic attacks. 

- What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation. [[paper]](https://arxiv.org/abs/2404.07129)
  - Aaditya K. Singh, Ted Moskovitz, Felix Hill, Stephanie C.Y. Chan, Andrew M. Saxe.
  - Key Word: Induction Head; Mechanistic Interpretability; In-Context Learning.
  - <details><summary>Digest</summary> Transformer models exhibit a powerful emergent ability for in-context learning, notably through a mechanism called the induction head (IH), which performs match-and-copy operations. This study explores the emergence and diversity of IHs, questioning their multiplicity, interdependence, and sudden appearance alongside a significant phase change in loss during training. Through experiments with synthetic data and a novel causal framework inspired by optogenetics for manipulating activations, the research identifies three subcircuits essential for IH formation. These findings illuminate the complex, data-dependent dynamics behind IH emergence and the specific conditions necessary for their development, advancing our understanding of in-context learning mechanisms in transformers.

- AtP*: An efficient and scalable method for localizing LLM behaviour to components. [[paper]](https://arxiv.org/abs/2403.00745)
  - János Kramár, Tom Lieberum, Rohin Shah, Neel Nanda.
  - Key Word: Activation Patching; Attribution Patching; Localization Analysis.
  - <details><summary>Digest</summary> Activation Patching is a method used for identifying how specific parts of a model influence its behavior, but it's too resource-intensive for large language models due to its linear cost scaling. This study introduces Attribution Patching (AtP), a quicker, gradient-based alternative, but identifies two major issues that cause AtP to miss important attributions. To counter these issues, an improved version, AtP*, is proposed, which offers better performance and scalability. The paper presents a comprehensive evaluation of AtP and other methods, demonstrating AtP's superiority and AtP*'s further enhancements. Additionally, it proposes a technique to limit the likelihood of overlooking relevant attributions with AtP*.

- Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking. [[paper]](https://arxiv.org/abs/2402.14811)
  - Nikhil Prakash, Tamar Rott Shaham, Tal Haklay, Yonatan Belinkov, David Bau.
  - Key Word: Fine-Tuning; Language Model; Entity Tracking; Mechanistic Interpretability.
  - <details><summary>Digest</summary> This study investigates how fine-tuning language models on generalized tasks (like instruction following, code generation, and mathematics) affects their internal computations, with a focus on entity tracking in mathematics. It finds that fine-tuning improves, but does not fundamentally change, the internal mechanisms related to entity tracking. The same circuit responsible for entity tracking in the original model also operates in the fine-tuned models, but with enhanced performance, mainly due to better handling of positional information. The researchers used techniques like Patch Patching and DCM for identifying model components and CMAP for comparing activations across models, leading to insights on how fine-tuning optimizes existing mechanisms rather than introducing new ones.


## Alignment

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#alignment">:open_file_folder: [<b><i>Full List of Alignment</i></b>]</a>.

- InfAlign: Inference-aware language model alignment. [[paper]](https://arxiv.org/abs/2412.19792)
  - Ananth Balashankar, Ziteng Sun, Jonathan Berant, Jacob Eisenstein, Michael Collins, Adrian Hutter, Jong Lee, Chirag Nagpal, Flavien Prost, Aradhana Sinha, and Ananda Theertha Suresh, Ahmad Beirami.
  - Key Word: Inference-Time Compute; Reward Miscalibration.
  - <details><summary>Digest</summary> This paper introduces Inference-Aware Policy Optimization (IAPO), a framework for improving language model alignment by considering inference-time decoding strategies, which traditional alignment methods overlook. The authors show that the standard alignment objective is suboptimal for modern decoding algorithms (e.g., Best-of-N sampling). They propose a solution, CTRL (Calibrate-and-Transform RL), which involves reward calibration and KL-regularized reward maximization. Applied to inference strategies like Best-of-N sampling and jailbreaking, CTRL achieves 8-12% and 4-9% higher inference-time win rates on Anthropic’s helpfulness and harmlessness benchmarks compared to conventional methods, demonstrating its effectiveness.

- Test-Time Alignment via Hypothesis Reweighting. [[paper]](https://arxiv.org/abs/2412.08812)
  - Yoonho Lee, Jonathan Williams, Henrik Marklund, Archit Sharma, Eric Mitchell, Anikait Singh, Chelsea Finn.
  - Key Word: Ensemble; Task Specificiation; Scalable Alignment.
  - <details><summary>Digest</summary> This paper addresses the challenge of aligning large pretrained models to underspecified tasks by proposing HyRe, a framework that dynamically adapts a neural network ensemble to test-time user intent. HyRe reweights ensemble members using a small labeled dataset from the target distribution, enabling efficient adaptation. The method scales to large models with computational costs similar to fine-tuning and demonstrates superior performance in personalization and distribution shift scenarios, outperforming state-of-the-art reward models with minimal labeled examples.

- Learning Loss Landscapes in Preference Optimization. [[paper]](https://arxiv.org/abs/2411.06568)
  - Carlo Alfano, Silvia Sapora, Jakob Nicolaus Foerster, Patrick Rebeschini, Yee Whye Teh.
  - Key Word: Preference Optimization; Loss Landscape.
  - <details><summary>Digest</summary> This study examines how data quality issues in preference datasets impact Preference Optimization (PO) algorithms, revealing performance drops in state-of-the-art methods. To address this, we propose a mirror descent-based PO framework, which includes new loss functions—discovered via evolutionary strategies—that significantly improve performance across tasks. These loss functions also enhance large language model fine-tuning on mixed-quality data, surpassing ORPO.

- Moral Alignment for LLM Agents. [[paper]](https://arxiv.org/abs/2410.01639)
  - Elizaveta Tennant, Stephen Hailes, Mirco Musolesi.
  - Key Word: Moral Alignment; Large Language Model.
  - <details><summary>Digest</summary> The paper introduces a novel approach to aligning decision-making agents based on Large Language Models (LLMs) with human values by designing intrinsic reward functions for Reinforcement Learning fine-tuning. Unlike traditional methods that rely on human preference data (e.g., RLHF or DPO), this approach encodes explicit human values for moral alignment, particularly using frameworks from Deontological Ethics and Utilitarianism. The authors test this method in the Iterated Prisoner’s Dilemma (IPD), showing that moral fine-tuning allows agents to unlearn selfish behaviors and generalize moral strategies to other environments. This method offers a transparent and cost-effective alternative to current alignment techniques.

- Preference Tuning with Human Feedback on Language, Speech, and Vision Tasks: A Survey. [[paper]](https://arxiv.org/abs/2409.11564)
  - Genta Indra Winata, Hanyang Zhao, Anirban Das, Wenpin Tang, David D. Yao, Shi-Xiong Zhang, Sambit Sahu.
  - Key Word: Preference Alignment; Survey.
  - <details><summary>Digest</summary> This survey provides a comprehensive review of recent developments in preference tuning for aligning deep generative models with human preferences. It is structured into three sections: (1) an introduction to reinforcement learning frameworks, tasks, models, and datasets across language, speech, and vision modalities; (2) an in-depth analysis of preference tuning methods; and (3) a discussion of applications, evaluation methods, and future research directions. The goal is to enhance understanding and encourage further innovation in preference tuning and model alignment.

- Geometric-Averaged Preference Optimization for Soft Preference Labels. [[paper]](https://arxiv.org/abs/2409.06691)
  - Hiroki Furuta, Kuang-Huei Lee, Shixiang Shane Gu, Yutaka Matsuo, Aleksandra Faust, Heiga Zen, Izzeddin Gur.
  - Key Word: Preference Optimization.
  - <details><summary>Digest</summary> This paper critiques traditional methods that treat human preferences as binary and deterministic, arguing that preferences are more nuanced and should be modeled as distributional. The authors introduce distributional soft preference labels and propose an improvement to Direct Preference Optimization (DPO) by incorporating a weighted geometric average of likelihoods in the loss function. This adjustment ensures that equally preferred responses result in near-zero loss, preventing over-optimization. The proposed method, which uses simulated AI feedback for soft labels, shows improved performance on alignment benchmarks compared to binary labeling, especially in datasets with modestly-confident labels.

- Beyond Preferences in AI Alignment. [[paper]](https://arxiv.org/abs/2408.16984)
  - Tan Zhi-Xuan, Micah Carroll, Matija Franklin, Hal Ashton.
  - Key Word: Alignment Beyond Preferences.
  - <details><summary>Digest</summary> The paper critiques the dominant “preferentist” approach to AI alignment, which assumes that human values can be adequately represented by preferences, and that AI should align with these preferences for safety and value compliance. The authors argue that this approach is flawed because preferences fail to capture the full complexity of human values and may ignore the incommensurability of those values. They also challenge the reliance on expected utility theory (EUT) as a normative standard for both humans and AI, pointing out its limitations. Instead, they propose aligning AI systems with normative standards suited to their social roles, agreed upon by relevant stakeholders, to accommodate diverse human values and promote mutual benefit.

- BOND: Aligning LLMs with Best-of-N Distillation. [[paper]](https://arxiv.org/abs/2407.14622)
  - Pier Giuseppe Sessa, Robert Dadashi, Léonard Hussenot, Johan Ferret, Nino Vieillard, Alexandre Ramé, Bobak Shariari, Sarah Perrin, Abe Friesen, Geoffrey Cideron, Sertan Girgin, Piotr Stanczyk, Andrea Michi, Danila Sinopalnikov, Sabela Ramos, Amélie Héliou, Aliaksei Severyn, Matt Hoffman, Nikola Momchev, Olivier Bachem.
  - Key Word: Reinforcement Learning from Human Feedback.
  - <details><summary>Digest</summary> This paper introduces Best-of-N Distillation (BOND), a novel RLHF algorithm designed to emulate Best-of-N sampling without its computational overhead by using a distribution matching approach with Jeffreys divergence. BOND demonstrates superior performance in aligning policies, particularly in abstractive summarization, outperforming other RLHF algorithms on several benchmarks.

- On scalable oversight with weak LLMs judging strong LLMs. [[paper]](https://arxiv.org/abs/2407.04622)
  - Zachary Kenton, Noah Y. Siegel, János Kramár, Jonah Brown-Cohen, Samuel Albanie, Jannis Bulian, Rishabh Agarwal, David Lindner, Yunhao Tang, Noah D. Goodman, Rohin Shah.
  - Key Word: Scalable Oversight; Large Language Model Debate and Judge.
  - <details><summary>Digest</summary> This paper investigates scalable oversight protocols for supervising superhuman AI, focusing on debate and consultancy methods compared to direct question-answering. The study shows debate consistently outperforms consultancy, especially when consultants argue randomly, and reveals that allowing debaters to choose their stance reduces the likelihood of judges being convinced by incorrect answers.

- A statistical framework for weak-to-strong generalization. [[paper]](https://arxiv.org/abs/2405.16236)
  - Seamus Somerstep, Felipe Maia Polo, Moulinath Banerjee, Ya'acov Ritov, Mikhail Yurochkin, Yuekai Sun.
  - Key Word: Weak-to-Strong Generalization.
  - <details><summary>Digest</summary> The paper demonstrates that it is possible to align stronger language models with superhuman capabilities using weaker human feedback by eliciting latent knowledge from pre-trained models. The authors propose a refinement-based approach that overcomes the limitations of naive fine-tuning, proving its effectiveness through theoretical bounds and practical alignment tasks.

- Theoretical Analysis of Weak-to-Strong Generalization. [[paper]](https://arxiv.org/abs/2405.16043)
  - Hunter Lang, David Sontag, Aravindan Vijayaraghavan.
  - Key Word: Weak-to-Strong Generalization.
  - <details><summary>Digest</summary> A strong pretrained student model can learn to correct the errors of a weaker teacher model and generalize to new examples, even when these examples are excluded from training, by leveraging incomplete or incorrect label information. This process, termed pseudolabel correction and coverage expansion, is not accounted for by existing weak supervision theory, prompting the authors to propose new bounds based on the data distribution and student hypothesis class to better capture these effects.

- Self-Play Preference Optimization for Language Model Alignment. [[paper]](https://arxiv.org/abs/2405.00675)
  - Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, Quanquan Gu.
  - Key Word: Self-Play; Preference Optimization.
  - <details><summary>Digest</summary> The paper highlights the limitations of traditional RLHF approaches in capturing human preferences accurately and proposes a new method called SPPO. SPPO is a self-play-based approach that treats the problem as a two-player game to identify the Nash equilibrium policy. It achieves better performance compared to methods like DPO and IPO and has theoretical convergence guarantee. Experimental results demonstrate that SPPO outperforms existing approaches without using external supervision or data augmentation techniques.

- Principled RLHF from Heterogeneous Feedback via Personalization and Preference Aggregation. [[paper]](https://arxiv.org/abs/2405.00254)
  - Chanwoo Park, Mingyang Liu, Kaiqing Zhang, Asuman Ozdaglar.
  - Key Word: Reinforcement Learning from Human Feedback; Personalization; Reward Aggregation.
  - <details><summary>Digest</summary> The paper discusses the effectiveness of reinforcement learning from human feedback (RLHF) in aligning AI systems with human values. It focuses on addressing the challenges posed by the heterogeneity of human preferences and potential strategic behavior in providing feedback. The paper proposes two frameworks: personalization-based and aggregation-based. The personalization-based framework utilizes representation learning and clustering to learn multiple reward models, balancing bias and variance. The aggregation-based framework aggregates diverse and truthful preferences using reward and preference aggregation approaches. The paper also addresses strategic human labelers and ensures truthful preference reporting. The proposed approaches aim to improve the alignment of AI systems with human preferences and have sample complexity guarantees.

- From r to Q∗: Your Language Model is Secretly a Q-Function. [[paper]](https://arxiv.org/abs/2404.12358)
  - Rafael Rafailov, Joey Hejna, Ryan Park, Chelsea Finn.
  - Key Word: Large Language Model; Direct Preference Optimization.
  - <details><summary>Digest</summary> This paper addresses the differences between Direct Preference Optimization (DPO) and standard Reinforcement Learning From Human Feedback (RLHF) methods. It theoretically aligns DPO with token-level Markov Decision Processes (MDPs) using inverse Q-learning that satisfies the Bellman equation, and empirically demonstrates that DPO allows for credit assignment, aligns with classical search algorithms like MCTS, and that simple beam search can enhance DPO's performance. The study concludes with potential applications in various AI tasks including multi-turn dialogue and end-to-end training of multi-model systems.

- Foundational Challenges in Assuring Alignment and Safety of Large Language Models. [[paper]](https://arxiv.org/abs/2404.09932)
  - Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase, Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, Benjamin L. Edelman, Zhaowei Zhang, Mario Günther, Anton Korinek, Jose Hernandez-Orallo, Lewis Hammond, Eric Bigelow, Alexander Pan, Lauro Langosco, Tomasz Korbak, Heidi Zhang, Ruiqi Zhong, Seán Ó hÉigeartaigh, Gabriel Recchia, Giulio Corsi, Alan Chan, Markus Anderljung, Lilian Edwards, Yoshua Bengio, Danqi Chen, Samuel Albanie, Tegan Maharaj, Jakob Foerster, Florian Tramer, He He, Atoosa Kasirzadeh, Yejin Choi, David Krueger.
  - Key Word: Alignment; Safety; Large Language Model; Agenda.
  - <details><summary>Digest</summary> This work identifies 18 foundational challenges in assuring the alignment and safety of large language models (LLMs). These challenges are organized into three different categories: scientific understanding of LLMs, development and deployment methods, and sociotechnical challenges. Based on the identified challenges, we pose 200+ concrete research questions.

- CogBench: a large language model walks into a psychology lab. [[paper]](https://arxiv.org/abs/2402.18225)
  - Julian Coda-Forno, Marcel Binz, Jane X. Wang, Eric Schulz.
  - Key Word: Cognitive Psychology; Reinforcement Learning from Human Feedback; Benchmarks; Large Language Model.
  - <details><summary>Digest</summary> The paper presents CogBench, a benchmark tool that evaluates large language models (LLMs) using behavioral metrics from cognitive psychology, aiming for a nuanced understanding of LLM behavior. Analyzing 35 LLMs with statistical models, it finds model size and human feedback critical for performance. It notes open-source models are less risk-prone than proprietary ones, and coding-focused fine-tuning doesn't always aid behavior. The study also finds that specific prompting techniques can enhance reasoning and model-based behavior in LLMs.

- A Critical Evaluation of AI Feedback for Aligning Large Language Models. [[paper]](https://arxiv.org/abs/2402.12366)
  - Archit Sharma, Sedrick Keh, Eric Mitchell, Chelsea Finn, Kushal Arora, Thomas Kollar.
  - Key Word: Reinforcement Learning from AI Feedback.
  - <details><summary>Digest</summary> The paper critiques the effectiveness of the Reinforcement Learning with AI Feedback (RLAIF) approach, commonly used to enhance the instruction-following capabilities of advanced pre-trained language models. It argues that the significant performance gains attributed to the reinforcement learning (RL) phase of RLAIF might be misleading. The paper suggests these improvements primarily stem from the initial use of a weaker teacher model for supervised fine-tuning (SFT) compared to a more advanced critic model for RL feedback. Through experimentation, it is demonstrated that simply using a more advanced model (e.g., GPT-4) for SFT can outperform the traditional RLAIF method. The study further explores how the effectiveness of RLAIF varies depending on the base model family, evaluation protocols, and critic models used. It concludes by offering a mechanistic insight into scenarios where SFT might surpass RLAIF and provides recommendations for optimizing RLAIF's practical application.

- MaxMin-RLHF: Towards Equitable Alignment of Large Language Models with Diverse Human Preferences. [[paper]](https://arxiv.org/abs/2402.08925)
  - Souradip Chakraborty, Jiahao Qiu, Hui Yuan, Alec Koppel, Furong Huang, Dinesh Manocha, Amrit Singh Bedi, Mengdi Wang.
  - Key Word: Reinforcement Learning from Human Feedback; Diversity in Human Preferences.
  - <details><summary>Digest</summary> This abstract addresses the limitations of Reinforcement Learning from Human Feedback (RLHF) in language models, specifically its inability to capture the diversity of human preferences using a single reward model. The authors present an "impossibility result" demonstrating this limitation and propose a solution that involves learning a mixture of preference distributions and employing a MaxMin alignment objective inspired by egalitarian principles. This approach aims to more fairly represent diverse human preferences. They connect their method to distributionally robust optimization and general utility reinforcement learning, showcasing its robustness and generality. Experimental results with GPT-2 and Tulu2-7B models demonstrate significant improvements in aligning with diverse human preferences, including a notable increase in win-rates and fairness for minority groups. The findings suggest the approach's applicability beyond language models to reinforcement learning at large.

- HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal. [[paper]](https://arxiv.org/abs/2402.04249)
  - Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, Dan Hendrycks.
  - Key Word: Red Teaming; Large Language Model; Benchmark.
  - <details><summary>Digest</summary> The paper introduces HarmBench, a standardized evaluation framework for automated red teaming designed to enhance the security of large language models (LLMs) by identifying and mitigating risks associated with their malicious use. The framework addresses the lack of rigorous assessment criteria in the field by incorporating several previously overlooked properties into its design. Using HarmBench, the authors perform a comprehensive comparison of 18 red teaming methods against 33 LLMs and their defenses, uncovering new insights. Additionally, they present a highly efficient adversarial training method that significantly improves LLM robustness against a broad spectrum of attacks. The paper highlights the utility of HarmBench in facilitating the simultaneous development of attacks and defenses, with the framework being made available as an open-source resource.

- Aligner: Achieving Efficient Alignment through Weak-to-Strong Correction. [[paper]](https://arxiv.org/abs/2402.02416)
  - Jiaming Ji, Boyuan Chen, Hantao Lou, Donghai Hong, Borong Zhang, Xuehai Pan, Juntao Dai, Yaodong Yang.
  - Key Word: Large Language Model; Reinforcement Learning from Human Feedback; Weak-to-Strong Generalization.
  - <details><summary>Digest</summary> The paper presents Aligner, a novel approach for aligning Large Language Models (LLMs) without the complexities of Reinforcement Learning from Human Feedback (RLHF). Aligner, an autoregressive seq2seq model, is trained on query-answer-correction data through supervised learning, offering a resource-efficient solution for model alignment. It enables significant performance improvements in LLMs by learning correctional residuals between aligned and unaligned outputs. Notably, Aligner enhances various LLMs' helpfulness and harmlessness, with substantial gains observed in models like GPT-4 and Llama2 when supervised by Aligner. The approach is model-agnostic and easily integrated with different models. 

- ARGS: Alignment as Reward-Guided Search. [[paper]](https://arxiv.org/abs/2402.01694)
  - Maxim Khanov, Jirayu Burapacheep, Yixuan Li.
  - Key Word: Language Model Alignment; Language Model Decoding; Guided Decoding.
  - <details><summary>Digest</summary> The paper introduces ARGS (Alignment as Reward-Guided Search), a new method for aligning large language models (LLMs) with human objectives without the instability and high resource demands of common approaches like RLHF (Reinforcement Learning from Human Feedback). ARGS integrates alignment directly into the decoding process, using a reward signal to adjust the model's probabilistic predictions, which generates texts aligned with human preferences and maintains semantic diversity. The framework has shown to consistently improve average rewards across different alignment tasks and model sizes, significantly outperforming baselines. For instance, it increased the average reward by 19.56% over the baseline in a GPT-4 evaluation. ARGS represents a step towards creating more responsive LLMs by focusing on alignment at the decoding stage. 

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

- Neural Interactive Proofs. [[paper]](https://arxiv.org/abs/2412.08897)
  - Lewis Hammond, Sam Adam-Day.
  - Key Word: Zero-Knowledge Proofs; Prover-Verifier Games.
  - <details><summary>Digest</summary> This paper explores neural interactive proofs, where a trusted but computationally limited verifier interacts with powerful untrusted agents (provers) to solve tasks. It introduces a unifying framework based on prover-verifier games, proposes new protocols for generating neural interactive proofs, and compares them theoretically and experimentally. The experiments span a toy graph isomorphism problem and a code validation task using large language models, laying the groundwork for safer AI systems through neural interactive proofs.

- Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation. [[paper]](https://arxiv.org/abs/2412.03178)
  - Gianni Franchi, Dat Nguyen Trong, Nacim Belkhir, Guoxuan Xia, Andrea Pilzer.
  - Key Word: Uncertainty Quantification; Text-to-Image Generation; Prompt-based Uncerntainty Estimation.
  - <details><summary>Digest</summary> This paper introduces PUNC (Prompt-based UNCertainty Estimation), a novel method for quantifying uncertainty in text-to-image (T2I) generative models. Leveraging Large Vision-Language Models (LVLMs), PUNC compares captions of generated images with their prompts in a semantically meaningful text space, enabling the disentanglement of aleatoric and epistemic uncertainties, which image-based methods cannot achieve. Experiments show PUNC outperforms existing uncertainty estimation techniques and supports applications like bias detection, copyright protection, and out-of-distribution detection. The authors also release a dataset of prompt-generation pairs to advance research in T2I uncertainty quantification.

- Enhancing Trust in Large Language Models with Uncertainty-Aware Fine-Tuning. [[paper]](https://arxiv.org/abs/2412.02904)
  - Ranganath Krishnan, Piyush Khanna, Omesh Tickoo.
  - Key Word: Large Language Model; Uncertainty-Aware Fine-Tuning.
  - <details><summary>Digest</summary> This paper addresses the challenge of hallucinations in large language models (LLMs) by proposing an uncertainty-aware fine-tuning approach that enhances the reliability of uncertainty estimates in open-ended natural language generation. Using a novel uncertainty-aware causal language modeling loss based on decision theory, the method improves calibration, hallucination detection, and out-of-domain prompt identification without compromising accuracy. Rigorous evaluations on multiple datasets show the approach outperforms standard fine-tuning, promoting more trustworthy and robust LLM responses. 

- SoK: Watermarking for AI-Generated Content. [[paper]](https://arxiv.org/abs/2411.18479)
  - Xuandong Zhao, Sam Gunn, Miranda Christ, Jaiden Fairoze, Andres Fabrega, Nicholas Carlini, Sanjam Garg, Sanghyun Hong, Milad Nasr, Florian Tramer, Somesh Jha, Lei Li, Yu-Xiang Wang, Dawn Song.
  - Key Word: Watermarking; AI-Generated Content.
  - <details><summary>Digest</summary> This paper provides a comprehensive overview of watermarking techniques for generative AI (GenAI), addressing the challenge of distinguishing AI-generated content from human-created content. It highlights the importance of watermarking for enhancing AI safety, combating misinformation, and ensuring trustworthiness. The study formalizes definitions, desired properties, objectives, and threat models of watermarking schemes, evaluates practical strategies for robustness against attacks, and reviews recent advancements. It identifies open challenges and potential directions, aiming to guide researchers in improving watermarking methods and assist policymakers in addressing GenAI’s broader implications.

- Safetywashing: Do AI Safety Benchmarks Actually Measure Safety Progress? [[paper]](https://arxiv.org/abs/2407.21792)
  - Richard Ren, Steven Basart, Adam Khoja, Alice Gatti, Long Phan, Xuwang Yin, Mantas Mazeika, Alexander Pan, Gabriel Mukobi, Ryan H. Kim, Stephen Fitz, Dan Hendrycks.
  - Key Word: AI Safety; Benchmark.
  - <details><summary>Digest</summary> As AI systems become more powerful, interest in AI safety research has grown, but the field remains poorly defined and measured, causing confusion about contributions. This study conducts a meta-analysis of AI safety benchmarks, examining their correlation with general capabilities in various models and surveying existing AI safety directions. Findings show that many safety benchmarks correlate highly with general capabilities, risking “safetywashing” where capability improvements are misrepresented as safety gains. The paper proposes a foundation for developing more meaningful safety metrics, defining AI safety as research goals distinct from general capabilities, aiming to create a rigorous framework for AI safety research.

- The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning. [[paper]](https://arxiv.org/abs/2403.03218)
  - Nathaniel Li, Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti, Justin D. Li, Ann-Kathrin Dombrowski, Shashwat Goel, Long Phan, Gabriel Mukobi, Nathan Helm-Burger, Rassin Lababidi, Lennart Justen, Andrew B. Liu, Michael Chen, Isabelle Barrass, Oliver Zhang, Xiaoyuan Zhu, Rishub Tamirisa, Bhrugu Bharathi, Adam Khoja, Zhenqi Zhao, Ariel Herbert-Voss, Cort B. Breuer, Samuel Marks, Oam Patel, Andy Zou, Mantas Mazeika, Zifan Wang, Palash Oswal, Weiran Lin, Adam A. Hunt, Justin Tienken-Harder, Kevin Y. Shih, Kemper Talley, John Guan, Russell Kaplan, Ian Steneker, David Campbell, Brad Jokubaitis, Alex Levinson, Jean Wang, William Qian, Kallol Krishna Karmakar, Steven Basart, Stephen Fitz, Mindy Levine, Ponnurangam Kumaraguru, Uday Tupakula, Vijay Varadharajan, Ruoyu Wang, Yan Shoshitaishvili, Jimmy Ba, Kevin M. Esvelt, Alexandr Wang, Dan Hendrycks.
  - Key Word: Mitigating Risk in LLM; Machine Unlearning.
  - <details><summary>Digest</summary> The White House Executive Order on AI highlights the risks of LLMs being used for malicious purposes, prompting the development of evaluations for hazardous capabilities. To address current evaluation limitations, a publicly released benchmark called the Weapons of Mass Destruction Proxy (WMDP) has been created to measure hazardous knowledge in biosecurity, cybersecurity, and chemical security, along with an unlearning method to reduce such knowledge while maintaining general model capabilities.

- Understanding Hallucinations in Diffusion Models through Mode Interpolation. [[paper]](https://arxiv.org/abs/2406.09358)
  - Sumukh K Aithal, Pratyush Maini, Zachary C. Lipton, J. Zico Kolter.
  - Key Word: Hallucinations; Diffusion Model.
  - <details><summary>Digest</summary> The paper investigates a failure mode in diffusion models, termed mode interpolation, which causes these models to generate "hallucinations"—samples that do not exist in the training data. The authors find that diffusion models interpolate between nearby data modes, leading to artifacts outside the original training distribution. Through experiments with 1D and 2D Gaussians, they show that a discontinuous loss landscape in the model's decoder results in hallucinations. They also demonstrate that diffusion models can identify when they are generating hallucinations, as indicated by high variance in the final stages of the sampling process. By using a metric to capture this variance, they can remove over 95% of hallucinations while retaining 96% of valid samples. The paper concludes by discussing the implications of hallucination removal on the stability of recursive training with synthetic data.

- Benchmarking Trustworthiness of Multimodal Large Language Models: A Comprehensive Study. [[paper]](https://arxiv.org/abs/2406.07057)
  - Yichi Zhang, Yao Huang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Yifan Wang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, Jun Zhu.
  - Key Word: Multimodal Large Language Model; Benchmarks; Robustness; Fairness; Privacy.
  - <details><summary>Digest</summary> Multimodal Large Language Models (MLLMs) exhibit impressive capabilities across various tasks but face significant trustworthiness challenges. Current research lacks a comprehensive evaluation of these models' trustworthiness. To address this, the authors introduce MultiTrust, a benchmark evaluating MLLMs on five key aspects: truthfulness, safety, robustness, fairness, and privacy. This benchmark uses a rigorous evaluation strategy covering 32 diverse tasks with self-curated datasets. Experiments with 21 modern MLLMs reveal new trustworthiness issues, such as difficulties with visually confusing images, vulnerability to multimodal attacks, and tendencies to disclose privacy and exhibit biases. The study underscores the need for advanced methods to improve MLLM reliability and introduces a scalable toolbox for standardized trustworthiness research.

- LoRA-Ensemble: Efficient Uncertainty Modelling for Self-attention Networks. [[paper]](https://arxiv.org/abs/2405.14438)
  - Michelle Halbheer, Dominik J. Mühlematter, Alexander Becker, Dominik Narnhofer, Helge Aasen, Konrad Schindler, Mehmet Ozgur Turkoglu.
  - Key Word: LoRA; Ensemble; Uncertainty Estimation.
  - <details><summary>Digest</summary> This paper introduces LoRA-Ensemble, a parameter-efficient method for uncertainty modeling in self-attention networks, which extends Low-Rank Adaptation (LoRA) to create implicit ensembles. By training member-specific low-rank matrices within a single pre-trained network, LoRA-Ensemble achieves superior calibration and comparable or better accuracy than explicit ensembles, while significantly reducing computational and memory costs.

- zkLLM: Zero Knowledge Proofs for Large Language Models. [[paper]](https://arxiv.org/abs/2404.16109)
  - Haochen Sun, Jason Li, Hongyang Zhang.
  - Key Word: Zero-Knowledge Proof; Large Language Model.
  - <details><summary>Digest</summary> This paper addresses the challenge of establishing the authenticity of outputs generated by large language models (LLMs) and presents zkLLM, a specialized zero-knowledge proof tailored for LLMs. The paper introduces tlookup, a parallelized lookup argument for non-arithmetic tensor operations in deep learning, and zkAttn, a specialized zero-knowledge proof for the attention mechanism. The authors demonstrate that zkLLM enables the generation of a correctness proof for the entire inference process in under 15 minutes for LLMs with 13 billion parameters. The resulting proof is compact and designed to protect the privacy of model parameters.

- Aurora-M: The First Open Source Multilingual Language Model Red-teamed according to the U.S. Executive Order. [[paper]](https://arxiv.org/abs/2404.00399)
  - Taishi Nakamura, Mayank Mishra, Simone Tedeschi, Yekun Chai, Jason T Stillerman, Felix Friedrich, Prateek Yadav, Tanmay Laud, Vu Minh Chien, Terry Yue Zhuo, Diganta Misra, Ben Bogin, Xuan-Son Vu, Marzena Karpinska, Arnav Varma Dantuluri, Wojciech Kusa, Tommaso Furlanello, Rio Yokota, Niklas Muennighoff, Suhas Pai, Tosin Adewumi, Veronika Laippala, Xiaozhe Yao, Adalberto Junior, Alpay Ariyak, Aleksandr Drozd, Jordan Clive, Kshitij Gupta, Liangyu Chen, Qi Sun, Ken Tsui, Noah Persaud, Nour Fahmy, Tianlong Chen, Mohit Bansal, Nicolo Monti, Tai Dang, Ziyang Luo, Tien-Tung Bui, Roberto Navigli, Virendra Mehta, Matthew Blumberg, Victor May, Huu Nguyen, Sampo Pyysalo.
  - Key Word: Red-Teaming; Language Model.
  - <details><summary>Digest</summary> This paper introduces Aurora-M, a 15B parameter, multilingual, open-source model trained on six languages and code. Developed by extending StarCoderPlus with 435 billion additional tokens, Aurora-M's training exceeds 2 trillion tokens, making it the first of its kind to be fine-tuned with human-reviewed safety instructions. This approach ensures compliance with the Biden-Harris Executive Order on AI safety, tackling challenges like multilingual support, catastrophic forgetting, and the high costs of pretraining from scratch. Aurora-M demonstrates superior performance across various tasks and languages, especially in safety evaluations, marking a significant step toward democratizing access to advanced AI models for collaborative development.

- Thermometer: Towards Universal Calibration for Large Language Models. [[paper]](https://arxiv.org/abs/2403.08819)
  - Maohao Shen, Subhro Das, Kristjan Greenewald, Prasanna Sattigeri, Gregory Wornell, Soumya Ghosh.
  - Key Word: Large Language Model; Calibration.
  - <details><summary>Digest</summary> We address the calibration challenge in large language models (LLM), a task made difficult by LLMs' computational demands and their application across diverse tasks. Our solution, THERMOMETER, is an efficient auxiliary model approach that calibrates LLMs across multiple tasks while maintaining accuracy and improving response calibration for new tasks, as demonstrated through extensive empirical evaluations.

- Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks. [[paper]](https://arxiv.org/abs/2402.19460)
  - Bálint Mucsányi, Michael Kirchhof, Seong Joon Oh.
  - Key Word: Uncertainty Quantification; Benchmarks.
  - <details><summary>Digest</summary> This paper discusses the evolution of uncertainty quantification in machine learning into various tasks like prediction abstention, out-of-distribution detection, and aleatoric uncertainty quantification, with the current aim being to create specialized estimators for each task. Through a comprehensive evaluation on ImageNet, the study finds that practical disentanglement of uncertainty tasks has not been achieved, despite theoretical advances. It also identifies which uncertainty estimators perform best for specific tasks, offering guidance for future research towards task-specific and disentangled uncertainty estimation.

- Foundation Model Transparency Reports. [[paper]](https://arxiv.org/abs/2402.16268)
  - Rishi Bommasani, Kevin Klyman, Shayne Longpre, Betty Xiong, Sayash Kapoor, Nestor Maslej, Arvind Narayanan, Percy Liang.
  - Key Word: Foundation Model; Transparency; Policy Alignment.
  - <details><summary>Digest</summary> The paper proposes Foundation Model Transparency Reports as a means to ensure transparency in the development and deployment of foundation models, drawing inspiration from social media transparency reporting practices. Recognizing the societal impact of these models, the paper aims to institutionalize transparency early in the industry's development. It outlines six design principles for these reports, informed by the successes and failures of social media transparency efforts, and utilizes 100 transparency indicators from the Foundation Model Transparency Index. The paper also examines how these indicators align with transparency requirements of six major government policies, suggesting that well-crafted reports could lower compliance costs by aligning with regulatory standards across jurisdictions. The authors advocate for foundation model developers to regularly publish transparency reports, echoing recommendations from the G7 and the White House.

- Regulation Games for Trustworthy Machine Learning. [[paper]](https://arxiv.org/abs/2402.03540)
  - Mohammad Yaghini, Patty Liu, Franziska Boenisch, Nicolas Papernot.
  - Key Word: Specification; Game Theory; AI Regulation.
  - <details><summary>Digest</summary> The paper presents a novel framework for trustworthy machine learning (ML), addressing the need for a comprehensive approach that includes fairness, privacy, and the distinction between model trainers and trust assessors. It proposes viewing trustworthy ML as a multi-objective multi-agent optimization problem, leading to a game-theoretic formulation named regulation games. Specifically, it introduces an instance called the SpecGame, which models the dynamics between ML model builders and regulators focused on fairness and privacy. The paper also introduces ParetoPlay, an innovative equilibrium search algorithm designed to find socially optimal solutions that keep agents within the Pareto frontier of their objectives. Through simulations of SpecGame using ParetoPlay, the paper offers insights into ML regulation policies. For example, it demonstrates that regulators can achieve significantly lower privacy budgets in gender classification applications by proactively setting their specifications.

- TrustLLM: Trustworthiness in Large Language Models. [[paper]](https://arxiv.org/abs/2401.05561)
  - Lichao Sun, Yue Huang, Haoran Wang, Siyuan Wu, Qihui Zhang, Yuan Li, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, Xiner Li, Zhengliang Liu, Yixin Liu, Yijue Wang, Zhikun Zhang, Bertie Vidgen, Bhavya Kailkhura, Caiming Xiong, Chaowei Xiao, Chunyuan Li, Eric Xing, Furong Huang, Hao Liu, Heng Ji, Hongyi Wang, Huan Zhang, Huaxiu Yao, Manolis Kellis, Marinka Zitnik, Meng Jiang, Mohit Bansal, James Zou, Jian Pei, Jian Liu, Jianfeng Gao, Jiawei Han, Jieyu Zhao, Jiliang Tang, Jindong Wang, Joaquin Vanschoren, John Mitchell, Kai Shu, Kaidi Xu, Kai-Wei Chang, Lifang He, Lifu Huang, Michael Backes, Neil Zhenqiang Gong, Philip S. Yu, Pin-Yu Chen, Quanquan Gu, Ran Xu, Rex Ying, Shuiwang Ji, Suman Jana, Tianlong Chen, Tianming Liu, Tianyi Zhou, William Wang, Xiang Li, Xiangliang Zhang, Xiao Wang, Xing Xie, Xun Chen, Xuyu Wang, Yan Liu, Yanfang Ye, Yinzhi Cao, Yong Chen, Yue Zhao.
  - Key Word: Large Language Model; Benchmark.
  - <details><summary>Digest</summary> This paper introduces TrustLLM, a comprehensive study on the trustworthiness of large language models (LLMs). It proposes principles covering eight dimensions of trustworthiness and establishes a benchmark for evaluating LLMs across six dimensions: truthfulness, safety, fairness, robustness, privacy, and machine ethics. The study evaluates 16 mainstream LLMs using over 30 datasets, revealing that trustworthiness and utility are generally positively correlated. Proprietary LLMs tend to outperform open-source ones in trustworthiness, though some open-source models perform comparably. It also notes that excessive calibration towards trustworthiness can reduce utility by misclassifying benign prompts. The paper highlights the need for transparency in both models and the technologies ensuring their trustworthiness.


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

- [Awesome Interpretability in Large Language Models](https://github.com/ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models) ![](https://img.shields.io/github/stars/ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models) ![](https://img.shields.io/github/last-commit/ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models)

- [Awesome LLM Interpretability](https://github.com/JShollaj/awesome-llm-interpretability) ![](https://img.shields.io/github/stars/JShollaj/awesome-llm-interpretability) ![](https://img.shields.io/github/last-commit/JShollaj/awesome-llm-interpretability)


## Other Lists

- [Awesome Out-of-distribution Detection](https://github.com/continuousml/Awesome-Out-Of-Distribution-Detection) ![ ](https://img.shields.io/github/stars/iCGY96/awesome_OpenSetRecognition_list) ![ ](https://img.shields.io/github/last-commit/iCGY96/awesome_OpenSetRecognition_list)

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

