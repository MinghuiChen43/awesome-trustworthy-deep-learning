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



## Out-of-Distribution Generalization

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#out-of-distribution-generalization">:open_file_folder: [<b><i>Full List of Out-of-Distribution Generalization</i></b>]</a>.

- Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning. [[paper]](https://arxiv.org/abs/2507.16795)
  - Helena Casademunt, Caden Juang, Adam Karvonen, Samuel Marks, Senthooran Rajamanoharan, Neel Nanda.
  - Key Word: Out-of-Distribution Generalization; Interpreting Latent Space Directions.
  - <details><summary>Digest</summary> The paper proposes Concept Ablation Fine-Tuning (CAFT), a method that uses interpretability tools to steer LLM generalization during fine-tuning without altering training data. By ablating latent directions linked to undesired concepts, CAFT mitigates unintended behaviors—such as emergent misalignment—and achieves a 10x reduction in misaligned outputs while preserving task performance.

- Intermediate Layer Classifiers for OOD generalization. [[paper]](https://arxiv.org/abs/2504.05461)
  - Arnas Uselis, Seong Joon Oh.
  - Key Word: Out-of-Distribution Generalization.
  - <details><summary>Digest</summary> This paper challenges the common practice of using penultimate layer features for out-of-distribution (OOD) generalization. The authors introduce Intermediate Layer Classifiers (ILCs) and find that earlier-layer representations often generalize better under distribution shifts. In some cases, zero-shot performance from intermediate layers rivals few-shot performance from the last layer. Their results across datasets and models suggest intermediate layers are more robust to shifts, underscoring the need to rethink layer selection for OOD tasks.


## Evasion Attacks and Defenses

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#evasion-attacks-and-defenses">:open_file_folder: [<b><i>Full List of Evasion Attacks and Defenses</i></b>]</a>.

- REINFORCE Adversarial Attacks on Large Language Models: An Adaptive, Distributional, and Semantic Objective. [[paper]](https://arxiv.org/abs/2502.17254)
  - Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Vincent Cohen-Addad, Johannes Gasteiger, Stephan Günnemann.
  - Key Word: Adversarial Attacks; Large Language Models; Reinforcement Learning.
  - <details><summary>Digest</summary> This paper critiques existing adversarial attacks on LLMs that maximize the likelihood of an affirmative response, arguing that such methods overestimate model robustness. To improve attack efficacy, the authors propose an adaptive, semantic optimization approach using a REINFORCE-based objective. Applied to Greedy Coordinate Gradient (GCG) and Projected Gradient Descent (PGD) jailbreak attacks, their method significantly enhances attack success rates, doubling ASR on Llama3 and increasing ASR from 2% to 50% against circuit breaker defenses.

- Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming. [[paper]](https://arxiv.org/abs/2501.18837)
  - Mrinank Sharma, Meg Tong, Jesse Mu, Jerry Wei, Jorrit Kruthoff, Scott Goodfriend, Euan Ong, Alwin Peng, Raj Agarwal, Cem Anil, Amanda Askell, Nathan Bailey, Joe Benton, Emma Bluemke, Samuel R. Bowman, Eric Christiansen, Hoagy Cunningham, Andy Dau, Anjali Gopal, Rob Gilson, Logan Graham, Logan Howard, Nimit Kalra, Taesung Lee, Kevin Lin, Peter Lofgren, Francesco Mosconi, Clare O'Hara, Catherine Olsson, Linda Petrini, Samir Rajani, Nikhil Saxena, Alex Silverstein, Tanya Singh, Theodore Sumers, Leonard Tang, Kevin K. Troy, Constantin Weisser, Ruiqi Zhong, Giulio Zhou, Jan Leike, Jared Kaplan, Ethan Perez.
  - Key Word: Red Teaming; Jailbreak.
  - <details><summary>Digest</summary> This paper introduces Constitutional Classifiers, a defense against universal jailbreaks in LLMs. These classifiers are trained on synthetic data generated using natural language rules to enforce content restrictions. Extensive red teaming and automated evaluations show that the approach effectively blocks jailbreaks while maintaining practical deployment viability, with minimal refusal rate increase (0.38%) and a 23.7% inference overhead. The findings demonstrate that robust jailbreak defenses can be achieved without significantly compromising usability.


## Poisoning Attacks and Defenses

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#poisoning-attacks-and-defenses">:open_file_folder: [<b><i>Full List of Poisoning Attacks and Defenses</i></b>]</a>.


## Privacy

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#privacy">:open_file_folder: [<b><i>Full List of Privacy</i></b>]</a>.

- Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off. [[paper]](https://arxiv.org/abs/2402.07002) [[code](https://github.com/6lyc/FedCEO_Collaborate-with-Each-Other)]  
  - Yuecheng Li, Lele Fu, Tong Wang, Jian Lou, Bin Chen, Lei Yang, Jian Shen, Zibin Zheng, Chuan Chen.  
  - Key Word: Federated Learning; Differential Privacy; Tensor Low-Rank Optimization; Utility-Privacy Trade-off.  
  - <details><summary>Digest</summary> This paper introduces FedCEO, a novel federated learning framework that enhances the utility-privacy trade-off by leveraging tensor low-rank optimization to mitigate the impact of differential privacy noise. The proposed method exploits semantic complementarity among clients through truncated tensor singular value decomposition (T-tSVD), achieving smoother global semantic spaces. Theoretical analysis demonstrates an improved utility-privacy bound of O(\sqrt{d}), outperforming prior state-of-the-art results. Experiments on real-world datasets validate the framework's effectiveness in balancing model accuracy and privacy protection under varying noise levels.</details>  
  
- - Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs. [[paper]](https://arxiv.org/abs/2507.04219)
  - Yan Scholten, Sophie Xhonneux, Stephan Günnemann, Leo Schwinn.
  - Key Word: Machine Unlearning; Large Language Model.
  - <details><summary>Digest</summary> The paper proposes Partial Model Collapse (PMC), a novel LLM unlearning method that avoids directly optimizing on sensitive data, addressing privacy concerns of existing approaches. Inspired by distribution collapse in self-trained generative models, PMC selectively triggers collapse on private data to remove it. The method is theoretically grounded and empirically shown to outperform prior unlearning techniques in removing sensitive information.

- Certified Unlearning for Neural Networks. [[paper]](https://arxiv.org/abs/2506.06985)
  - Anastasia Koloskova, Youssef Allouah, Animesh Jha, Rachid Guerraoui, Sanmi Koyejo.
  - Key Word: Certified Unlearning.
  - <details><summary>Digest</summary> The paper proposes a certified machine unlearning method that removes the influence of specific training data using noisy fine-tuning on retained data, without relying on restrictive assumptions. It ensures formal unlearning guarantees through a connection to privacy amplification, works across diverse settings, and outperforms existing methods both theoretically and empirically.

- Existing Large Language Model Unlearning Evaluations Are Inconclusive. [[paper]](https://arxiv.org/abs/2506.00688)
  - Zhili Feng, Yixuan Even Xu, Alexander Robey, Robert Kirk, Xander Davies, Yarin Gal, Avi Schwarzschild, J. Zico Kolter.
  - Key Word: Machine Unlearning; Large Language Model.
  - <details><summary>Digest</summary> This paper critiques current evaluation methods in machine unlearning for language models, revealing that they often misrepresent unlearning success due to three flaws: (1) evaluations may reintroduce knowledge during testing, (2) results vary widely across tasks, and (3) reliance on spurious correlations undermines trust. To improve reliability, the authors propose two guiding principles—minimal information injection and downstream task awareness—and validate them through experiments showing how current practices can lead to misleading conclusions.

- Extracting memorized pieces of (copyrighted) books from open-weight language models. [[paper]](https://arxiv.org/abs/2505.12546)
  - A. Feder Cooper, Aaron Gokaslan, Amy B. Cyphert, Christopher De Sa, Mark A. Lemley, Daniel E. Ho, Percy Liang.
  - Key Word: Extraction Attack.
  - <details><summary>Digest</summary> This paper examines how much large language models (LLMs) memorize copyrighted content, using probabilistic extraction techniques on 13 open-weight LLMs. It finds that while memorization varies across models and texts, some models—like Llama 3.1 70B—can nearly fully memorize certain books (e.g., Harry Potter, 1984). However, most models do not memorize most books. The findings complicate copyright debates, offering evidence for both sides without clearly favoring either.

- When to Forget? Complexity Trade-offs in Machine Unlearning. [[paper]](https://arxiv.org/abs/2502.17323)
  - Martin Van Waerebeke, Marco Lorenzi, Giovanni Neglia, Kevin Scaman.
  - Key Word: Certified Unlearning.
  - <details><summary>Digest</summary> This paper analyzes the efficiency of Machine Unlearning (MU) and establishes the first minimax upper and lower bounds on unlearning computation time. Under strongly convex objectives and without access to forgotten data, the authors introduce the unlearning complexity ratio, comparing unlearning costs to full retraining. A phase diagram reveals three regimes: infeasibility, trivial unlearning via noise, and significant computational savings. The study highlights key factors—data dimensionality, forget set size, and privacy constraints—that influence the feasibility of efficient unlearning.

- Open Problems in Machine Unlearning for AI Safety. [[paper]](https://arxiv.org/abs/2501.04952)
  - Fazl Barez, Tingchen Fu, Ameya Prabhu, Stephen Casper, Amartya Sanyal, Adel Bibi, Aidan O'Gara, Robert Kirk, Ben Bucknall, Tim Fist, Luke Ong, Philip Torr, Kwok-Yan Lam, Robert Trager, David Krueger, Sören Mindermann, José Hernandez-Orallo, Mor Geva, Yarin Gal.
  - Key Word: Machine Unlearning.
  - <details><summary>Digest</summary> As AI systems grow in capability and autonomy in critical areas like cybersecurity, healthcare, and biological research, ensuring their alignment with human values is crucial. Machine unlearning, originally focused on privacy and data removal, is gaining attention for its potential in AI safety. However, this paper identifies significant limitations preventing unlearning from fully addressing safety concerns, especially in managing dual-use knowledge where information can have both beneficial and harmful applications. It highlights challenges such as unintended side effects, conflicts with existing safety mechanisms, and difficulties in evaluating robustness and preserving safety features during unlearning. By outlining these constraints and open problems, the paper aims to guide future research toward more realistic and effective AI safety strategies.


## Fairness

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#fairness">:open_file_folder: [<b><i>Full List of Fairness</i></b>]</a>.

## Interpretability

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#interpretability">:open_file_folder: [<b><i>Full List of Interpretability</i></b>]</a>.

- Because we have LLMs, we Can and Should Pursue Agentic Interpretability. [[paper]](https://arxiv.org/abs/2506.12152)
  - Been Kim, John Hewitt, Neel Nanda, Noah Fiedel, Oyvind Tafjord.
  - Key Word: Agentic Interpretability.
  - <details><summary>Digest</summary> This paper proposes agentic interpretability—a conversational approach where LLMs actively help users understand them by modeling the user’s knowledge and teaching accordingly. Unlike traditional interpretability methods, it emphasizes cooperation and interactivity over completeness. While not ideal for high-stakes scenarios, it offers a way for humans to grasp superhuman model concepts. The key challenge lies in evaluating such human-in-the-loop systems, for which the authors discuss possible solutions.

- Interpretability Illusions with Sparse Autoencoders: Evaluating Robustness of Concept Representations. [[paper]](https://arxiv.org/abs/2505.16004)
  - Aaron J. Li, Suraj Srinivas, Usha Bhalla, Himabindu Lakkaraju.
  - Key Word: Sparse Autoencoders; Adversarial Attack.
  - <details><summary>Digest</summary> This paper highlights a critical weakness in sparse autoencoders (SAEs) used to interpret LLMs: their concept representations are not robust to small input perturbations. The authors introduce an evaluation framework that uses adversarial attacks to test this robustness and find that SAE interpretations can be easily manipulated without changing the LLM’s output, questioning their reliability for model monitoring and oversight tasks.

- Avoiding Leakage Poisoning: Concept Interventions Under Distribution Shifts. [[paper]](https://arxiv.org/abs/2504.17921)
  - Mateo Espinosa Zarlenga, Gabriele Dominici, Pietro Barbiero, Zohreh Shams, Mateja Jamnik.
  - Key Word: Concept Bottleneck Models; Distribution Shifts.
  - <details><summary>Digest</summary> This paper studies how concept-based models (CMs) behave on out-of-distribution (OOD) inputs, especially under concept interventions (where humans correct predicted concepts at test time). The authors identify a flaw called leakage poisoning, where CMs fail to improve after intervention on OOD data. To address this, they propose MixCEM, a model that selectively uses leaked information only for in-distribution inputs. Experiments show MixCEM improves accuracy on both in-distribution and OOD samples, with and without interventions.

- MIB: A Mechanistic Interpretability Benchmark. [[paper]](https://arxiv.org/abs/2504.13151)
  - Aaron Mueller, Atticus Geiger, Sarah Wiegreffe, Dana Arad, Iván Arcuschin, Adam Belfki, Yik Siu Chan, Jaden Fiotto-Kaufman, Tal Haklay, Michael Hanna, Jing Huang, Rohan Gupta, Yaniv Nikankin, Hadas Orgad, Nikhil Prakash, Anja Reusch, Aruna Sankaranarayanan, Shun Shao, Alessandro Stolfo, Martin Tutek, Amir Zur, David Bau, Yonatan Belinkov.
  - Key Word: Mechanistic Interpretability; Benchmark.
  - <details><summary>Digest</summary> The paper introduces MIB, a benchmark designed to evaluate mechanistic interpretability methods in neural language models. MIB has two tracks: circuit localization (identifying model components critical to task performance) and causal variable localization (identifying hidden features representing task-relevant variables). Experiments show that attribution and mask optimization methods excel at circuit localization, while supervised DAS outperforms others in causal variable localization. Surprisingly, SAE features offer no advantage over standard neurons. MIB thus provides a robust framework for assessing real progress in interpretability.

- SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability. [[paper]](https://arxiv.org/abs/2503.09532)
  - Adam Karvonen, Can Rager, Johnny Lin, Curt Tigges, Joseph Bloom, David Chanin, Yeu-Tong Lau, Eoin Farrell, Callum McDougall, Kola Ayonrinde, Matthew Wearden, Arthur Conmy, Samuel Marks, Neel Nanda.
  - Key Word: Sparse Autoencoder; Benchmark.
  - <details><summary>Digest</summary> The paper introduces SAEBench, a comprehensive evaluation suite for sparse autoencoders (SAEs) that assesses their performance across seven diverse metrics, including interpretability, feature disentanglement, and practical applications like unlearning. It highlights that improvements in traditional unsupervised proxy metrics do not always lead to better real-world performance. The authors open-source over 200 SAEs spanning eight architectures and training algorithms, revealing that Matryoshka SAEs, despite underperforming on proxy metrics, excel in feature disentanglement, especially at scale. SAEBench provides a standardized framework for comparing SAE designs and studying scaling trends in their development.

- Towards Understanding Distilled Reasoning Models: A Representational Approach. [[paper]](https://arxiv.org/abs/2503.03730)
  - David D. Baek, Max Tegmark.
  - Key Word: Mechanistic Interpretability; Model Distillation; Model Steering.
  - <details><summary>Digest</summary> This paper examines the impact of model distillation on reasoning feature development in large language models (LLMs). Using a crosscoder trained on Qwen-series models, the study finds that distillation creates unique reasoning feature directions, enabling control over thinking styles (e.g., over-thinking vs. incisive-thinking). The analysis covers four reasoning types: self-reflection, deductive, alternative, and contrastive reasoning. Additionally, the study explores changes in feature geometry, suggesting that larger distilled models develop more structured representations, improving distillation performance. These findings enhance understanding of distillation’s role in shaping model reasoning and transparency.

- From superposition to sparse codes: interpretable representations in neural networks. [[paper]](https://arxiv.org/abs/2503.01824)
  - David Klindt, Charles O'Neill, Patrik Reizinger, Harald Maurer, Nina Miolane.
  - Key Word: Superposition; Sparse Coding.
  - <details><summary>Digest</summary> This paper explores how neural networks represent information, proposing that they encode features in superposition—linearly overlaying input concepts. The authors introduce a three-step framework to extract interpretable representations: (1) Identifiability theory shows that neural networks recover latent features up to a linear transformation; (2) Sparse coding techniques disentangle these features using compressed sensing principles; (3) Interpretability metrics evaluate alignment with human-interpretable concepts. By integrating insights from neuroscience, representation learning, and interpretability research, the paper offers a perspective with implications for neural coding, AI transparency, and deep learning interpretability.

- Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry. [[paper]](https://arxiv.org/abs/2503.01822)
  - Sai Sumedh R. Hindupur, Ekdeep Singh Lubana, Thomas Fel, Demba Ba.
  - Key Word: Sparse Autoencoder.
  - <details><summary>Digest</summary> This paper examines the limitations of Sparse Autoencoders (SAEs) in interpreting neural network representations. It introduces a bilevel optimization framework showing that SAEs impose structural biases, affecting which concepts they can detect. Different SAE architectures are not interchangeable, as switching them can reveal or obscure concepts. Through experiments on toy models, semi-synthetic data, and large-scale datasets, the study highlights two key properties of real-world concepts: varying intrinsic dimensionality and nonlinear separability. Standard SAEs fail when these factors are ignored, but a new SAE design incorporating them uncovers previously hidden concepts. The findings challenge the notion of a universal SAE and emphasize the importance of architecture-specific choices in interpretability.

- Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable? [[paper]](https://arxiv.org/abs/2502.20914)
  - Maxime Méloux, Silviu Maniu, François Portet, Maxime Peyrard. 
  - Key Word: Mechanistic Interpretability.
  - <details><summary>Digest</summary> This work explores the identifiability of Mechanistic Interpretability (MI) explanations in neural networks. It examines whether unique explanations exist for a given behavior by drawing parallels to identifiability in statistics. The study identifies two MI strategies: “where-then-what” (isolating circuits before interpreting) and “what-then-where” (starting with candidate algorithms and finding neural activation subspaces). Experiments on Boolean functions and small MLPs reveal systematic non-identifiability—multiple circuits, interpretations, and subspaces can explain the same behavior. The study questions whether uniqueness is necessary, suggesting that predictive and manipulability criteria might suffice, and discusses validation through the inner interpretability framework.

- Open Problems in Mechanistic Interpretability. [[paper]](https://arxiv.org/abs/2501.16496)
  - Lee Sharkey, Bilal Chughtai, Joshua Batson, Jack Lindsey, Jeff Wu, Lucius Bushnaq, Nicholas Goldowsky-Dill, Stefan Heimersheim, Alejandro Ortega, Joseph Bloom, Stella Biderman, Adria Garriga-Alonso, Arthur Conmy, Neel Nanda, Jessica Rumbelow, Martin Wattenberg, Nandi Schoots, Joseph Miller, Eric J. Michaud, Stephen Casper, Max Tegmark, William Saunders, David Bau, Eric Todd, Atticus Geiger, Mor Geva, Jesse Hoogland, Daniel Murfet, Tom McGrath.
  - Key Word: Mechanistic Interpretability.
  - <details><summary>Digest</summary> This review explores the current challenges and open problems in mechanistic interpretability, which seeks to understand the computational mechanisms behind neural networks. While progress has been made, further conceptual and practical advancements are needed to deepen insights, refine applications, and address socio-technical challenges. The paper highlights key areas for future research to enhance AI transparency, safety, and scientific understanding of intelligence.

- Sparse Autoencoders Do Not Find Canonical Units of Analysis. [[paper]](https://arxiv.org/abs/2502.04878)
  - Patrick Leask, Bart Bussmann, Michael Pearce, Joseph Bloom, Curt Tigges, Noura Al Moubayed, Lee Sharkey, Neel Nanda.
  - Key Word: Mechanistic Interpretability; Sparse Autoencoders; Representational Structure.
  - <details><summary>Digest</summary> This paper challenges the assumption that Sparse Autoencoders (SAEs) can identify a canonical set of atomic features in LLMs. Using SAE stitching, the authors show that SAEs are incomplete, as larger SAEs contain novel latents not captured by smaller ones. Through meta-SAEs, they demonstrate that SAE latents are not atomic, as they often decompose into smaller, interpretable components (e.g., “Einstein” → “scientist” + “Germany” + “famous person”). While SAEs may still be useful, the authors suggest rethinking their role in mechanistic interpretability and exploring alternative methods for finding fundamental features. An interactive dashboard is provided for further exploration.


## Alignment

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#alignment">:open_file_folder: [<b><i>Full List of Alignment</i></b>]</a>.

- Preference Learning for AI Alignment: a Causal Perspective. [[paper]](https://arxiv.org/abs/2506.05967)
  - Katarzyna Kobalczyk, Mihaela van der Schaar.
  - Key Word: AI Alignment; Causuality.
  - <details><summary>Digest</summary> This paper reframes reward modeling from preference data as a causal inference problem to better align LLMs with human values. It highlights challenges like causal misidentification, user-specific confounding, and preference heterogeneity, and shows how causally-inspired methods can improve generalization and robustness over naive models. The authors also propose key assumptions and future directions for more reliable reward modeling.

- Scaling Laws For Scalable Oversight. [[paper]](https://arxiv.org/abs/2504.18530)
  - Joshua Engels, David D. Baek, Subhash Kantamneni, Max Tegmark.
  - Key Word: Scalable Oversight.
  - <details><summary>Digest</summary> This paper proposes a framework to model and quantify scalable oversight—how weaker AI systems supervise stronger ones—using oversight and deception-specific Elo scores. The framework is validated through games like Nim, Mafia, and Debate, revealing how oversight success scales with capability gaps. They further study Nested Scalable Oversight (NSO) and find that success rates drop sharply when overseeing much stronger systems, with a success rate below 52% at a 400 Elo gap.

- You Are What You Eat -- AI Alignment Requires Understanding How Data Shapes Structure and Generalisation. [[paper]](https://arxiv.org/abs/2502.05475)
  - Simon Pepin Lehalleur, Jesse Hoogland, Matthew Farrugia-Roberts, Susan Wei, Alexander Gietelink Oldenziel, George Wang, Liam Carroll, Daniel Murfet.
  - Key Word: AI Alignment.
  - <details><summary>Digest</summary> This paper argues that understanding the relationship between data distribution structure and model structure is key to AI alignment. It highlights that neural networks with identical training performance can generalize differently due to internal computational differences, making standard evaluation methods insufficient for safety assurances. To advance AI alignment, the authors propose developing statistical foundations to systematically analyze how these structures influence generalization.


## Others

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#others">:open_file_folder: [<b><i>Full List of Others</i></b>]</a>.

- Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety. [[paper]](https://arxiv.org/abs/2507.11473)
  - Tomek Korbak, Mikita Balesni, Elizabeth Barnes, Yoshua Bengio, Joe Benton, Joseph Bloom, Mark Chen, Alan Cooney, Allan Dafoe, Anca Dragan, Scott Emmons, Owain Evans, David Farhi, Ryan Greenblatt, Dan Hendrycks, Marius Hobbhahn, Evan Hubinger, Geoffrey Irving, Erik Jenner, Daniel Kokotajlo, Victoria Krakovna, Shane Legg, David Lindner, David Luan, Aleksander Mądry, Julian Michael, Neel Nanda, Dave Orr, Jakub Pachocki, Ethan Perez, Mary Phuong, Fabien Roger, Joshua Saxe, Buck Shlegeris, Martín Soto, Eric Steinberger, Jasmine Wang, Wojciech Zaremba, Bowen Baker, Rohin Shah, Vlad Mikulik.
  - Key Word: Chain of Thought Monitoring.
  - <details><summary>Digest</summary> AI systems that "think" in human language offer a unique opportunity for AI safety: we can monitor their chains of thought (CoT) for the intent to misbehave. Like all other known AI oversight methods, CoT monitoring is imperfect and allows some misbehavior to go unnoticed. Nevertheless, it shows promise and we recommend further research into CoT monitorability and investment in CoT monitoring alongside existing safety methods. Because CoT monitorability may be fragile, we recommend that frontier model developers consider the impact of development decisions on CoT monitorability.

- The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems. [[paper]](https://arxiv.org/abs/2503.03750)
  - Richard Ren, Arunim Agarwal, Mantas Mazeika, Cristina Menghini, Robert Vacareanu, Brad Kenstler, Mick Yang, Isabelle Barrass, Alice Gatti, Xuwang Yin, Eduardo Trevino, Matias Geralnik, Adam Khoja, Dean Lee, Summer Yue, Dan Hendrycks.
  - Key Word: Honesty; Benchmark.
  - <details><summary>Digest</summary> This paper addresses concerns about honesty in large language models (LLMs), distinguishing it from accuracy. Current honesty evaluations are limited, often conflating honesty with correctness. To address this, the authors introduce a large-scale, human-collected dataset that directly measures honesty. Their findings reveal that while larger models achieve higher accuracy, they do not necessarily become more honest. Notably, frontier LLMs, despite excelling in truthfulness benchmarks, often lie under pressure. The study also demonstrates that simple interventions, such as representation engineering, can enhance honesty, highlighting the need for robust evaluations and interventions to ensure trustworthy AI.

- Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI Offer a Safer Path? [[paper]](https://arxiv.org/abs/2502.15657)
  - Yoshua Bengio, Michael Cohen, Damiano Fornasiere, Joumana Ghosn, Pietro Greiner, Matt MacDermott, Sören Mindermann, Adam Oberman, Jesse Richardson, Oliver Richardson, Marc-Antoine Rondeau, Pierre-Luc St-Charles, David Williams-King.
  - Key Word: Scientist AI; Agentic AI; AI Safety.
  - <details><summary>Digest</summary> The paper discusses the risks posed by generalist AI agents, which can autonomously plan, act, and pursue goals. These risks include deception, misalignment with human interests, and loss of human control. The authors argue for a shift away from agency-driven AI towards a non-agentic AI system called Scientist AI, designed to explain the world rather than act in it. Scientist AI consists of a world model that generates theories and a question-answering system, both incorporating uncertainty to prevent overconfidence. This approach aims to advance scientific progress and AI safety while mitigating risks associated with autonomous AI agents.

- Do Large Language Model Benchmarks Test Reliability? [[paper]](https://arxiv.org/abs/2502.03461)
  - Joshua Vendrow, Edward Vendrow, Sara Beery, Aleksander Madry.
  - Key Word: Large Language Model Benchmark; Reliability.
  - <details><summary>Digest</summary> This paper highlights the lack of focus on LLM reliability in existing benchmarks, despite extensive efforts to track model capabilities. The authors identify pervasive label errors in current benchmarks, which obscure model failures and unreliable behavior. To address this, they introduce platinum benchmarks—carefully curated datasets with minimal label errors and ambiguity. By refining examples from 15 popular benchmarks and evaluating various models, they find that even frontier LLMs struggle with basic tasks, such as elementary math problems, revealing systematic failure patterns.



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

