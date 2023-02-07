[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/graphs/commit-activity)
![ ](https://img.shields.io/github/last-commit/MinghuiChen43/awesome-trustworthy-deep-learning)
[![GitHub stars](https://img.shields.io/github/stars/MinghuiChen43/awesome-trustworthy-deep-learning?color=blue&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/MinghuiChen43/awesome-trustworthy-deep-learning?color=yellow&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning)
[![GitHub forks](https://img.shields.io/github/forks/MinghuiChen43/awesome-trustworthy-deep-learning?color=red&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/watchers)
[![GitHub Contributors](https://img.shields.io/github/contributors/MinghuiChen43/awesome-trustworthy-deep-learning?color=green&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/network/members)

# Awesome Trustworthy Deep Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning) ![if Useful](https://camo.githubusercontent.com/1ef04f27611ff643eb57eb87cc0f1204d7a6a14d/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f6c6162656c3d254630253946253843253946266d6573736167653d496625323055736566756c267374796c653d7374796c653d666c617426636f6c6f723d424334453939)

The deployment of deep learning in real-world systems calls for a set of complementary technologies that will ensure that deep learning is trustworthy [(Nicolas Papernot)](https://www.papernot.fr/teaching/f19-trustworthy-ml). The list covers different topics in emerging research areas including but not limited to out-of-distribution generalization, adversarial examples, backdoor attack, model inversion attack, machine unlearning, etc.

Daily updating from ArXiv. The preview README only includes papers submitted to ArXiv within the **last one year**.  More paper can be found here <a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/tree/master/FULL_LIST.md">:open_file_folder: [<b><i>Full List</i></b>]</a>.

# Table of Contents

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

# Paper List

## Survey

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#survey">:open_file_folder: [<b><i>Full List of Survey</i></b>]</a>.

## Out-of-Distribution Generalization

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#out-of-distribution-generalization">:open_file_folder: [<b><i>Full List of Out-of-Distribution Generalization</i></b>]</a>.

- Effective Robustness against Natural Distribution Shifts for Models with Different Training Data. [[paper]](https://arxiv.org/abs/2302.01381)
  - Zhouxing Shi, Nicholas Carlini, Ananth Balashankar, Ludwig Schmidt, Cho-Jui Hsieh, Alex Beutel, Yao Qin.
  - Key Word: Effective Robustness; Natural Distribution Shifts.
  - <details><summary>Digest</summary> We propose a new effective robustness evaluation metric to compare the effective robustness of models trained on different data distributions. To do this we control for the accuracy on multiple ID test sets that cover the training distributions for all the evaluated models. Our new evaluation metric provides a better estimate of the effectiveness robustness and explains the surprising effective robustness gains of zero-shot CLIP-like models exhibited when considering only one ID dataset, while the gains diminish under our evaluation.

- Free Lunch for Domain Adversarial Training: Environment Label Smoothing. [[paper]](https://arxiv.org/abs/2302.00194)
  - YiFan Zhang, Xue Wang, Jian Liang, Zhang Zhang, Liang Wang, Rong Jin, Tieniu Tan.
  - Key Word: Domain Adversarial Training; Label Smoothing.
  - <details><summary>Digest</summary> We proposed Environment Label Smoothing (ELS), which encourages the discriminator to output soft probability, which thus reduces the confidence of the discriminator and alleviates the impact of noisy environment labels. We demonstrate, both experimentally and theoretically, that ELS can improve training stability, local convergence, and robustness to noisy environment labels.

- FedFA: Federated Feature Augmentation. [[paper]](https://arxiv.org/abs/2301.12995) [[code]](https://github.com/tfzhou/FedFA)
  - Tianfei Zhou, Ender Konukoglu. *ICLR 2023*
  - Key Word: Federated Learning; Feature Augmentation; Feature Shifts.
  - <details><summary>Digest</summary> The primary goal of this paper is to develop a robust federated learning algorithm to address feature shift in clients' samples, which can be caused by various factors, e.g., acquisition differences in medical imaging. To reach this goal, we propose FedFA to tackle federated learning from a distinct perspective of federated feature augmentation. FedFA is based on a major insight that each client's data distribution can be characterized by statistics (i.e., mean and standard deviation) of latent features; and it is likely to manipulate these local statistics globally, i.e., based on information in the entire federation, to let clients have a better sense of the underlying distribution and therefore alleviate local data bias.

- Alignment with human representations supports robust few-shot learning. [[paper]](https://arxiv.org/abs/2301.11990)
  - Ilia Sucholutsky, Thomas L. Griffiths. 
  - Key Word: Representational Alignment; Few-Shot Learning; Domain Shifts; Adversarial Robustness.
  - <details><summary>Digest</summary> Should we care whether AI systems have representations of the world that are similar to those of humans? We provide an information-theoretic analysis that suggests that there should be a U-shaped relationship between the degree of representational alignment with humans and performance on few-shot learning tasks. We confirm this prediction empirically, finding such a relationship in an analysis of the performance of 491 computer vision models.

- DEJA VU: Continual Model Generalization For Unseen Domains. [[paper]](https://arxiv.org/abs/2301.10418) [[code]](https://github.com/dawnliu35/dejavu-ratp)
  - Chenxi Liu, Lixu Wang, Lingjuan Lyu, Chen Sun, Xiao Wang, Qi Zhu.
  - Key Word: Domain Generalization; Domain Adaptation.
  - <details><summary>Digest</summary> Existing DG works are ineffective for continually changing domains due to severe catastrophic forgetting of learned knowledge. To overcome these limitations of DA and DG in handling the Unfamiliar Period during continual domain shift, we propose RaTP, a framework that focuses on improving models' target domain generalization (TDG) capability, while also achieving effective target domain adaptation (TDA) capability right after training on certain domains and forgetting alleviation (FA) capability on past domains. 

- ManyDG: Many-domain Generalization for Healthcare Applications. [[paper]](https://arxiv.org/abs/2301.08834) [[code]](https://github.com/ycq091044/ManyDG)
  - Chaoqi Yang, M Brandon Westover, Jimeng Sun. *ICLR 2023*
  - Key Word: Domain Generalization; Healthcare.
  - <details><summary>Digest</summary> In healthcare applications, most existing domain generalization methods assume a small number of domains. In this paper, considering the diversity of patient covariates, we propose a new setting by treating each patient as a separate domain (leading to many domains). We develop a new domain generalization method ManyDG, that can scale to such many-domain problems. Our method identifies the patient domain covariates by mutual reconstruction and removes them via an orthogonal projection step.

## Evasion Attacks and Defenses

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#evasion-attacks-and-defenses">:open_file_folder: [<b><i>Full List of Evasion Attacks and Defenses</i></b>]</a>.

- Exploring and Exploiting Decision Boundary Dynamics for Adversarial Robustness. [[paper]](https://arxiv.org/abs/2302.03015)
  - Yuancheng Xu, Yanchao Sun, Micah Goldblum, Tom Goldstein, Furong Huang. *ICLR 2023*
  - Key Word: Adversarial Robustness; Decision Boundary Analysis; Margin Maximization.
  - <details><summary>Digest</summary> The robustness of a deep classifier can be characterized by its margins: the decision boundary's distances to natural data points. However, it is unclear whether existing robust training methods effectively increase the margin for each vulnerable point during training. To understand this, we propose a continuous-time framework for quantifying the relative speed of the decision boundary with respect to each individual point.

- Defensive ML: Defending Architectural Side-channels with Adversarial Obfuscation. [[paper]](https://arxiv.org/abs/2302.01474)
  - Hyoungwook Nam, Raghavendra Pradyumna Pothukuchi, Bo Li, Nam Sung Kim, Josep Torrellas.
  - Key Word: Microarchitectural Side-channel Attacks; Adversarial Attacks.
  - <details><summary>Digest</summary> Side-channel attacks that use machine learning (ML) for signal analysis have become prominent threats to computer security, as ML models easily find patterns in signals. To address this problem, this paper explores using Adversarial Machine Learning (AML) methods as a defense at the computer architecture layer to obfuscate side channels. We call this approach Defensive ML, and the generator to obfuscate signals, defender. 

- On the Robustness of Randomized Ensembles to Adversarial Perturbations. [[paper]](https://arxiv.org/abs/2302.01375)
  - Hassan Dbouk, Naresh R. Shanbhag.
  - Key Word: Ensemble Adversarial Defenses; Randomized Adversarial Defenses.
  - <details><summary>Digest</summary> We first demystify RECs (Randomized ensemble classifiers) as we derive fundamental results regarding their theoretical limits, necessary and sufficient conditions for them to be useful, and more. Leveraging this new understanding, we propose a new boosting algorithm (BARRE) for training robust RECs, and empirically demonstrate its effectiveness at defending against strong ℓ∞ norm-bounded adversaries across various network architectures and datasets.

- Are Defenses for Graph Neural Networks Robust? [[paper]](https://arxiv.org/abs/2301.13694) [[code]](https://github.com/LoadingByte/are-gnn-defenses-robust)
  - Felix Mujkanovic, Simon Geisler, Stephan Günnemann, Aleksandar Bojchevski. *NeurIPS 2022*
  - Key Word: Graph Neural Networks; Adversarial Defenses.
  - <details><summary>Digest</summary> The standard methodology has a serious flaw - virtually all of the defenses are evaluated against non-adaptive attacks leading to overly optimistic robustness estimates. We perform a thorough robustness analysis of 7 of the most popular defenses spanning the entire spectrum of strategies, i.e., aimed at improving the graph, the architecture, or the training. The results are sobering - most defenses show no or only marginal improvement compared to an undefended baseline. 

- Benchmarking Robustness to Adversarial Image Obfuscations. [[paper]](https://arxiv.org/abs/2301.12993) [[code]](https://github.com/deepmind/image_obfuscation_benchmark)
  - Florian Stimberg, Ayan Chakrabarti, Chun-Ta Lu, Hussein Hazimeh, Otilia Stretcu, Wei Qiao, Yintao Liu, Merve Kaya, Cyrus Rashtchian, Ariel Fuxman, Mehmet Tek, Sven Gowal.
  - Key Word: Natural Distribution Shifts; Adversarial Robustness.
  - <details><summary>Digest</summary> We invite researchers to tackle this specific issue and present a new image benchmark. This benchmark, based on ImageNet, simulates the type of obfuscations created by malicious actors. It goes beyond ImageNet-C and ImageNet-C¯ by proposing general, drastic, adversarial modifications that preserve the original content intent. It aims to tackle a more common adversarial threat than the one considered by ℓp-norm bounded adversaries. 

- Uncovering Adversarial Risks of Test-Time Adaptation. [[paper]](https://arxiv.org/abs/2301.12576)
  - Tong Wu, Feiran Jia, Xiangyu Qi, Jiachen T. Wang, Vikash Sehwag, Saeed Mahloujifar, Prateek Mittal.
  - Key Word: Adversarial Attacks; Test-Time Adaptation.
  - <details><summary>Digest</summary> We propose Distribution Invading Attack (DIA), which injects a small fraction of malicious data into the test batch. DIA causes models using TTA to misclassify benign and unperturbed test data, providing an entirely new capability for adversaries that is infeasible in canonical machine learning pipelines. 

- Improving the Accuracy-Robustness Trade-off of Classifiers via Adaptive Smoothing. [[paper]](https://arxiv.org/abs/2301.12554) [[code]](https://github.com/Bai-YT/AdaptiveSmoothing)
  - Yatong Bai, Brendon G. Anderson, Aerin Kim, Somayeh Sojoudi.
  - Key Word: Adversarial Robustness; Locally Biased Smoothing.
  - <details><summary>Digest</summary> While it is shown in the literature that simultaneously accurate and robust classifiers exist for common datasets, previous methods that improve the adversarial robustness of classifiers often manifest an accuracy-robustness trade-off. We build upon recent advancements in data-driven ``locally biased smoothing'' to develop classifiers that treat benign and adversarial test data differently. Specifically, we tailor the smoothing operation to the usage of a robust neural network as the source of robustness. 

- Data Augmentation Alone Can Improve Adversarial Training. [[paper]](https://arxiv.org/abs/2301.09879) [[code]](https://github.com/treelli/da-alone-improves-at)
  - Lin Li, Michael Spratling. *ICLR 2023*
  - Key Word: Data Augmentation; Adversarial Training.
  - <details><summary>Digest</summary> This work proves that, contrary to previous findings, data augmentation alone can significantly boost accuracy and robustness in adversarial training. We find that the hardness and the diversity of data augmentation are important factors in combating robust overfitting. 

- On the Robustness of AlphaFold: A COVID-19 Case Study. [[paper]](https://arxiv.org/abs/2301.04093)
  - Ismail Alkhouri, Sumit Jha, Andre Beckus, George Atia, Alvaro Velasquez, Rickard Ewetz, Arvind Ramanathan, Susmit Jha.
  - Key Word: Adversarial Robustness; AlphaFold.
  - <details><summary>Digest</summary> We demonstrate that AlphaFold does not exhibit such robustness despite its high accuracy. This raises the challenge of detecting and quantifying the extent to which these predicted protein structures can be trusted. To measure the robustness of the predicted structures, we utilize (i) the root-mean-square deviation (RMSD) and (ii) the Global Distance Test (GDT) similarity measure between the predicted structure of the original sequence and the structure of its adversarially perturbed version.

- On adversarial robustness and the use of Wasserstein ascent-descent dynamics to enforce it. [[paper]](https://arxiv.org/abs/2301.03662)
  - Camilo Garcia Trillos, Nicolas Garcia Trillos.
  - Key Word: Adversarial Robustness; Wasserstein Gradient Flow; Mean-Field Limit.
  - <details><summary>Digest</summary> We propose iterative algorithms to solve adversarial problems in a variety of supervised learning settings of interest. Our algorithms, which can be interpreted as suitable ascent-descent dynamics in Wasserstein spaces, take the form of a system of interacting particles. These interacting particle dynamics are shown to converge toward appropriate mean-field limit equations in certain large number of particles regimes. 

- REaaS: Enabling Adversarially Robust Downstream Classifiers via Robust Encoder as a Service. [[paper]](https://arxiv.org/abs/2301.02905)
  - Wenjie Qu, Jinyuan Jia, Neil Zhenqiang Gong. *NDSS 2023*
  - Key Word: Adversarial Robustness.
  - <details><summary>Digest</summary> In this work, we show that, via providing two APIs, a cloud server 1) makes it possible for a client to certify robustness of its downstream classifier against adversarial perturbations using any certification method and 2) makes it orders of magnitude more communication efficient and more computation efficient to certify robustness using smoothed classifier based certification.

## Poisoning Attacks and Defenses

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#poisoning-attacks-and-defenses">:open_file_folder: [<b><i>Full List of Poisoning Attacks and Defenses</i></b>]</a>.

- Run-Off Election: Improved Provable Defense against Data Poisoning Attacks. [[paper]](https://arxiv.org/abs/2302.02300)
  - Keivan Rezaei, Kiarash Banihashem, Atoosa Chegini, Soheil Feizi.
  - Key Word: Poisoning Defenses; Deep Partition Aggregation; Finite Aggregation.
  - <details><summary>Digest</summary> We show that merely considering the majority vote in ensemble defenses is wasteful as it does not effectively utilize available information in the logits layers of the base models. Instead, we propose Run-Off Election (ROE), a novel aggregation method based on a two-round election across the base models: In the first round, models vote for their preferred class and then a second, Run-Off election is held between the top two classes in the first round. 

- Backdoor Attacks Against Dataset Distillation. [[paper]](https://arxiv.org/abs/2301.01197) [[code]](https://github.com/liuyugeng/baadd)
  - Yugeng Liu, Zheng Li, Michael Backes, Yun Shen, Yang Zhang. *NDSS 2023*
  - Key Word: Backdoor Attacks; Dataset Distillation.
  - <details><summary>Digest</summary> This study performs the first backdoor attack against the models trained on the data distilled by dataset distillation models in the image domain. Concretely, we inject triggers into the synthetic data during the distillation procedure rather than during the model training stage, where all previous attacks are performed. 

## Privacy

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#privacy">:open_file_folder: [<b><i>Full List of Privacy</i></b>]</a>.

- From Robustness to Privacy and Back. [[paper]](https://arxiv.org/abs/2302.01855)
  - Hilal Asi, Jonathan Ullman, Lydia Zakynthinou.
  - Key Word: Differential Privacy; Adversarial Robustness.
  - <details><summary>Digest</summary> We study the relationship between two desiderata of algorithms in statistical inference and machine learning: differential privacy and robustness to adversarial data corruptions. Their conceptual similarity was first observed by Dwork and Lei (STOC 2009), who observed that private algorithms satisfy robustness, and gave a general method for converting robust algorithms to private ones. However, all general methods for transforming robust algorithms into private ones lead to suboptimal error rates. Our work gives the first black-box transformation that converts any adversarially robust algorithm into one that satisfies pure differential privacy. 

- Dataset Distillation Fixes Dataset Reconstruction Attacks. [[paper]](https://arxiv.org/abs/2302.01428)
  - Noel Loo, Ramin Hasani, Mathias Lechner, Daniela Rus.
  - Key Word: Dataset Distillation; Reconstruction Attacks.
  - <details><summary>Digest</summary> We first build a stronger version of the dataset reconstruction attack and show how it can provably recover its entire training set in the infinite width regime. We then empirically study the characteristics of this attack on two-layer networks and reveal that its success heavily depends on deviations from the frozen infinite-width Neural Tangent Kernel limit. More importantly, we formally show for the first time that dataset reconstruction attacks are a variation of dataset distillation. 

- Personalized Privacy Auditing and Optimization at Test Time. [[paper]](https://arxiv.org/abs/2302.00077)
  - Cuong Tran, Ferdinando Fioretto.
  - Key Word: Differential Privacy; Feature Selection; Active Learning; Test-Time Privacy Leakage.
  - <details><summary>Digest</summary> This paper introduced the concept of information leakage at test time whose goal is to minimize the number of features that individuals need to disclose during model inference while maintaining accurate predictions from the model. The motivations of this notion are grounded in the privacy risks imposed by the adoption of learning models in consequential domains, by the significant efforts required by organizations to verify the accuracy of the released information, and align with the data minimization principle outlined in the GDPR

## Fairness

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#fairness">:open_file_folder: [<b><i>Full List of Fairness</i></b>]</a>.

- Out of Context: Investigating the Bias and Fairness Concerns of "Artificial Intelligence as a Service". [[paper]](https://arxiv.org/abs/2302.01448)
  - Kornel Lewicki, Michelle Seng Ah Lee, Jennifer Cobbe, Jatinder Singh.
  - Key Word: Algorithmic Fairness; AI as a Service; AI Regulation.
  - <details><summary>Digest</summary> This paper argues that the context-sensitive nature of fairness is often incompatible with AIaaS' 'one-size-fits-all' approach, leading to issues and tensions. Specifically, we review and systematise the AIaaS space by proposing a taxonomy of AI services based on the levels of autonomy afforded to the user. 

- Hyper-parameter Tuning for Fair Classification without Sensitive Attribute Access. [[paper]](https://arxiv.org/abs/2302.01385)
  - Akshaj Kumar Veldanda, Ivan Brugere, Sanghamitra Dutta, Alan Mishler, Siddharth Garg.
  - Key Word: Fair Classification.
  - <details><summary>Digest</summary> We propose Antigone, a framework to train fair classifiers without access to sensitive attributes on either training or validation data. Instead, we generate pseudo sensitive attributes on the validation data by training a biased classifier and using the classifier's incorrectly (correctly) labeled examples as proxies for minority (majority) groups.

- Debiasing Vision-Language Models via Biased Prompts. [[paper]](https://arxiv.org/abs/2302.00070)
  - Ching-Yao Chuang, Varun Jampani, Yuanzhen Li, Antonio Torralba, Stefanie Jegelka.
  - Key Word: Debiasing; Prompting.
  - <details><summary>Digest</summary> We propose a general approach for debiasing vision-language foundation models by projecting out biased directions in the text embedding. In particular, we show that debiasing only the text embedding with a calibrated projection matrix suffices to yield robust classifiers and fair generative models. 

- Superhuman Fairness. [[paper]](https://arxiv.org/abs/2301.13420)
  - Omid Memarrast, Linh Vu, Brian Ziebart.
  - Key Word: Fairness; Imitation Learning.
  - <details><summary>Digest</summary> Most fairness approaches optimize a specified trade-off between performance measure(s) (e.g., accuracy, log loss, or AUC) and fairness metric(s) (e.g., demographic parity, equalized odds). This begs the question: are the right performance-fairness trade-offs being specified? We instead re-cast fair machine learning as an imitation learning task by introducing superhuman fairness, which seeks to simultaneously outperform human decisions on multiple predictive performance and fairness measures. 

- Fairness and Accuracy under Domain Generalization. [[paper]](https://arxiv.org/abs/2301.13323) [[code]](https://github.com/pth1993/FATDM)
  - Thai-Hoang Pham, Xueru Zhang, Ping Zhang. *ICLR 2023*
  - Key Word: Fairness; Domain Generalization.
  - <details><summary>Digest</summary> We study the transfer of both fairness and accuracy under domain generalization where the data at test time may be sampled from never-before-seen domains. We first develop theoretical bounds on the unfairness and expected loss at deployment, and then derive sufficient conditions under which fairness and accuracy can be perfectly transferred via invariant representation learning. Guided by this, we design a learning algorithm such that fair ML models learned with training data still have high fairness and accuracy when deployment environments change.

- On Fairness of Medical Image Classification with Multiple Sensitive Attributes via Learning Orthogonal Representations. [[paper]](https://arxiv.org/abs/2301.01481) [[code]](https://github.com/vengdeng/fcro)
  - Wenlong Deng, Yuan Zhong, Qi Dou, Xiaoxiao Li. *IPMI 2023*
  - Key Word: Fair Classification; Medical Imaging.
  - <details><summary>Digest</summary> We propose a novel method for fair representation learning with respect to multi-sensitive attributes. We pursue the independence between target and multi-sensitive representations by achieving orthogonality in the representation space. Concretely, we enforce the column space orthogonality by keeping target information on the complement of a low-rank sensitive space. 


## Interpretability

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#interpretability">:open_file_folder: [<b><i>Full List of Interpretability</i></b>]</a>.

- Selective Explanations: Leveraging Human Input to Align Explainable AI. [[paper]](https://arxiv.org/abs/2301.09656)
  - Vivian Lai, Yiming Zhang, Chacha Chen, Q. Vera Liao, Chenhao Tan.
  - Key Word: Explainable AI; Learning from Human Input.
  - <details><summary>Digest</summary> We attempt to close these gaps by making AI explanations selective -- a fundamental property of human explanations -- by selectively presenting a subset from a large set of model reasons based on what aligns with the recipient's preferences. We propose a general framework for generating selective explanations by leveraging human input on a small sample. 

## Open-World Learning

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#open-world-learning">:open_file_folder: [<b><i>Full List of Open-World Learning</i></b>]</a>.

- Vocabulary-informed Zero-shot and Open-set Learning. [[paper]](https://arxiv.org/abs/2301.00998) [[code]](https://github.com/xiaomeiyy/wmm-voc)
  - Yanwei Fu, Xiaomei Wang, Hanze Dong, Yu-Gang Jiang, Meng Wang, Xiangyang Xue, Leonid Sigal. *TPAMI*
  - Key Word: Vocabulary-Informed Learning; Generalized Zero-Shot Learning, Open-set Recognition.
  - <details><summary>Digest</summary> Zero-shot learning is one way of addressing these challenges, but it has only been shown to work with limited sized class vocabularies and typically requires separation between supervised and unsupervised classes, allowing former to inform the latter but not vice versa. We propose the notion of vocabulary-informed learning to alleviate the above mentioned challenges and address problems of supervised, zero-shot, generalized zero-shot and open set recognition using a unified framework.

## Environmental Well-being

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#environmental-well-being">:open_file_folder: [<b><i>Full List of Environmental Well-being</i></b>]</a>.

- A Green(er) World for A.I. [[paper]](https://arxiv.org/abs/2301.11581)
  - Dan Zhao, Nathan C. Frey, Joseph McDonald, Matthew Hubbell, David Bestor, Michael Jones, Andrew Prout, Vijay Gadepally, Siddharth Samsi.
  - Key Word: Green AI; Sustainable AI; Energy Efficiency.
  - <details><summary>Digest</summary> We outline our outlook for Green A.I. -- a more sustainable, energy-efficient and energy-aware ecosystem for developing A.I. across the research, computing, and practitioner communities alike -- and the steps required to arrive there. 

- A Systematic Review of Green AI. [[paper]](https://arxiv.org/abs/2301.11047) [[code]](https://github.com/luiscruz/slr-green-ai)
  - Roberto Verdecchia, June Sallou, Luís Cruz.
  - Key Word: Green AI; Hyperparameter Tuning; Model Benchmarking; Deployment; Model Comparison.
  - <details><summary>Digest</summary> We present a systematic review of the Green AI literature. From the analysis of 98 primary studies, different patterns emerge. The topic experienced a considerable growth from 2020 onward. Most studies consider monitoring AI model footprint, tuning hyperparameters to improve model sustainability, or benchmarking models. 

## Interactions with Blockchain

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#interactions-with-blockchain">:open_file_folder: [<b><i>Full List of Interactions with Blockchain</i></b>]</a>.

## Others

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#others">:open_file_folder: [<b><i>Full List of Others</i></b>]</a>.

- Causal Triplet: An Open Challenge for Intervention-centric Causal Representation Learning. [[paper]](https://arxiv.org/abs/2301.05169) [[code]](https://sites.google.com/view/causaltriplet?pli=1)
  - Yuejiang Liu, Alexandre Alahi, Chris Russell, Max Horn, Dominik Zietlow, Bernhard Schölkopf, Francesco Locatello.
  - Key Word: Causal Representation Learning; Benchmark.
  - <details><summary>Digest</summary> We present Causal Triplet, a causal representation learning benchmark featuring not only visually more complex scenes, but also two crucial desiderata commonly overlooked in previous works: (i) an actionable counterfactual setting, where only certain object-level variables allow for counterfactual observations whereas others do not; (ii) an interventional downstream task with an emphasis on out-of-distribution robustness from the independent causal mechanisms principle.

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

## Open-World Lists

- [Awesome Open Set Recognition list](https://github.com/iCGY96/awesome_OpenSetRecognition_list) ![ ](https://img.shields.io/github/stars/iCGY96/awesome_OpenSetRecognition_list) ![ ](https://img.shields.io/github/last-commit/iCGY96/awesome_OpenSetRecognition_list)

- [Awesome Novel Class Discovery](https://github.com/JosephKJ/Awesome-Novel-Class-Discovery) ![ ](https://img.shields.io/github/stars/JosephKJ/Awesome-Novel-Class-Discovery) ![ ](https://img.shields.io/github/last-commit/JosephKJ/Awesome-Novel-Class-Discovery)

- [Awesome Open-World-Learning](https://github.com/zhoudw-zdw/Awesome-open-world-learning) ![ ](https://img.shields.io/github/stars/zhoudw-zdw/Awesome-open-world-learning) ![ ](https://img.shields.io/github/last-commit/zhoudw-zdw/Awesome-open-world-learning)

## Blockchain Lists

- [Blockchain Papers](https://github.com/decrypto-org/blockchain-papers) ![ ](https://img.shields.io/github/stars/decrypto-org/blockchain-papers) ![ ](https://img.shields.io/github/last-commit/decrypto-org/blockchain-papers)

- [Awesome Blockchain AI](https://github.com/steven2358/awesome-blockchain-ai) ![ ](https://img.shields.io/github/stars/steven2358/awesome-blockchain-ai) ![ ](https://img.shields.io/github/last-commit/steven2358/awesome-blockchain-ai)

## Other Lists

- [Awesome Causality Algorithms](https://github.com/rguo12/awesome-causality-algorithms) ![ ](https://img.shields.io/github/stars/rguo12/awesome-causality-algorithms) ![ ](https://img.shields.io/github/last-commit/rguo12/awesome-causality-algorithms)

- [Awesome AI Security](https://github.com/DeepSpaceHarbor/Awesome-AI-Security) ![ ](https://img.shields.io/github/stars/DeepSpaceHarbor/Awesome-AI-Security) ![ ](https://img.shields.io/github/last-commit/DeepSpaceHarbor/Awesome-AI-Security)

- [A curated list of AI Security & Privacy events](https://github.com/ZhengyuZhao/AI-Security-and-Privacy-Events) ![ ](https://img.shields.io/github/stars/ZhengyuZhao/AI-Security-and-Privacy-Events) ![ ](https://img.shields.io/github/last-commit/ZhengyuZhao/AI-Security-and-Privacy-Events)

- [Awesome Deep Phenomena](https://github.com/MinghuiChen43/awesome-deep-phenomena) ![ ](https://img.shields.io/github/stars/MinghuiChen43/awesome-deep-phenomena) ![ ](https://img.shields.io/github/last-commit/MinghuiChen43/awesome-deep-phenomena)

# Toolboxes

## Robustness Toolboxes

- [Cleverhans](https://github.com/cleverhans-lab/cleverhans) ![ ](https://img.shields.io/github/stars/cleverhans-lab/cleverhans)
  - This repository contains the source code for CleverHans, a Python library to benchmark machine learning systems' vulnerability to adversarial examples.

- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) ![ ](https://img.shields.io/github/stars/Trusted-AI/adversarial-robustness-toolbox)
  - Adversarial Robustness Toolbox (ART) is a Python library for Machine Learning Security. ART provides tools that enable developers and researchers to evaluate, defend, certify and verify Machine Learning models and applications against the adversarial threats of Evasion, Poisoning, Extraction, and Inference.

- [Advtorch](https://github.com/BorealisAI/advertorch) ![ ](https://img.shields.io/github/stars/BorealisAI/advertorch)
  - Advtorch is a Python toolbox for adversarial robustness research. The primary functionalities are implemented in PyTorch. Specifically, AdverTorch contains modules for generating adversarial perturbations and defending against adversarial examples, also scripts for adversarial training.
  
- [RobustBench](https://github.com/RobustBench/robustbench) ![ ](https://img.shields.io/github/stars/RobustBench/robustbench)
  - A standardized benchmark for adversarial robustness.

- [DeepDG: OOD generalization toolbox](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG)
  - A domain generalization toolbox for research purpose.

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

- [VerifAI](https://github.com/BerkeleyLearnVerify/VerifAI) ![ ](https://img.shields.io/github/stars/BerkeleyLearnVerify/VerifAI)
  - VerifAI is a software toolkit for the formal design and analysis of systems that include artificial intelligence (AI) and machine learning (ML) components.

# Workshops

## Robustness Workshops

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

- [Uncertainty and Robustness in Deep Learning Workshop (ICML 2020)](https://sites.google.com/view/udlworkshop2020/home)

## Privacy Workshops

- [Theory and Practice of Differential Privacy (ICML 2022)](https://tpdp.journalprivacyconfidentiality.org/2022/)

## Fairness Workshops

- [Algorithmic Fairness through the Lens of Causality and Privacy (NeurIPS 2022)](https://www.afciworkshop.org/)

## Interpretability Workshops

- [Interpretable Machine Learning in Healthcare (ICML 2022)](https://sites.google.com/view/imlh2022)

## Other Workshops

- [Workshop on Physics for Machine Learning (ICLR 2023)](https://physics4ml.github.io/)

- [Pitfalls of limited data and computation for Trustworthy ML (ICLR 2023)](https://sites.google.com/view/trustml-unlimited/)

- [Workshop on Mathematical and Empirical Understanding of Foundation Models (ICLR 2023)](https://sites.google.com/view/me-fomo2023)

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

# Contributing

TBD
