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

- A Survey of Trustworthy Federated Learning with Perspectives on Security, Robustness, and Privacy. [[paper]](https://arxiv.org/abs/2302.10637)
  - Yifei Zhang, Dun Zeng, Jinglong Luo, Zenglin Xu, Irwin King.
  - Key Word: Federated Learning; Robustness; Privacy.
  - <details><summary>Digest</summary> We propose a comprehensive roadmap for developing trustworthy FL systems and summarize existing efforts from three key aspects: security, robustness, and privacy. We outline the threats that pose vulnerabilities to trustworthy federated learning across different stages of development, including data processing, model training, and deployment.

- Adversarial Machine Learning: A Systematic Survey of Backdoor Attack, Weight Attack and Adversarial Example. [[paper]](https://arxiv.org/abs/2302.09457)
  - Baoyuan Wu, Li Liu, Zihao Zhu, Qingshan Liu, Zhaofeng He, Siwei Lyu.
  - Key Word: Backdoor Attack; Deployment-Time Adversarial Attack; Inference-Time Adversarial Attack.
  - <details><summary>Digest</summary> In this work, we aim to provide a unified perspective to the AML community to systematically review the overall progress of this field. We firstly provide a general definition about AML, and then propose a unified mathematical framework to covering existing attack paradigms. According to the proposed unified framework, we can not only clearly figure out the connections and differences among these paradigms, but also systematically categorize and review existing works in each paradigm.

## Out-of-Distribution Generalization

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#out-of-distribution-generalization">:open_file_folder: [<b><i>Full List of Out-of-Distribution Generalization</i></b>]</a>.

- Domain Generalization via Nuclear Norm Regularization. [[paper]](https://arxiv.org/abs/2303.07527)
  - Zhenmei Shi, Yifei Ming, Ying Fan, Frederic Sala, Yingyu Liang.
  - Key Word: Domain Generalization; Nuclear Norm Regularization; Low-Rank Regularization.
  - <details><summary>Digest</summary> We propose a simple and effective regularization method based on the nuclear norm of the learned features for domain generalization. Intuitively, the proposed regularizer mitigates the impacts of environmental features and encourages learning domain-invariant features. Theoretically, we provide insights into why nuclear norm regularization is more effective compared to ERM and alternative regularization methods. 

- Statistical Learning under Heterogenous Distribution Shift. [[paper]](https://arxiv.org/abs/2302.13934)
  - Max Simchowitz, Anurag Ajay, Pulkit Agrawal, Akshay Krishnamurthy.
  - Key Word: Heterogenous Covariate Shifts; Statistical Learning Theory.
  - <details><summary>Digest</summary> This paper studies the prediction of a target z from a pair of random variables (x,y), where the ground-truth predictor is additive E[z∣x,y]=f⋆(x)+g⋆(y). We study the performance of empirical risk minimization (ERM) over functions f+g, f∈F and g∈G, fit on a given training distribution, but evaluated on a test distribution which exhibits covariate shift. 

- Robust Weight Signatures: Gaining Robustness as Easy as Patching Weights? [[paper]](https://arxiv.org/abs/2302.12480)
  - Ruisi Cai, Zhenyu Zhang, Zhangyang Wang.
  - Key Word: Corruption Robustness; Task Vector.
  - <details><summary>Digest</summary> Our work is dedicated to investigating how natural corruption “robustness” is encoded in weights and how to disentangle/transfer them. We introduce “Robust Weight Signature”(RWS), which nontrivially generalizes the prior wisdom in model weight interpolation and arithmetic, to analyzing standard/robust models, with both methodological innovations and new key findings. RWSs lead to a powerful in-situ model patching framework to easily achieve on-demand robustness towards a wide range of corruptions.

- Change is Hard: A Closer Look at Subpopulation Shift. [[paper]](https://arxiv.org/abs/2302.12254) [[code]](https://github.com/YyzHarry/SubpopBench)
  - Yuzhe Yang, Haoran Zhang, Dina Katabi, Marzyeh Ghassemi.
  - Key Word: Subpopulation Shift; Benchmark.
  - <details><summary>Digest</summary> We provide a fine-grained analysis of subpopulation shift. We first propose a unified framework that dissects and explains common shifts in subgroups. We then establish a comprehensive benchmark of 20 state-of-the-art algorithms evaluated on 12 real-world datasets in vision, language, and healthcare domains. With results obtained from training over 10,000 models, we reveal intriguing observations for future progress in this space. 

- On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective. [[paper]](https://arxiv.org/abs/2302.12095) [[code]](https://github.com/microsoft/robustlearn)
  - Jindong Wang, Xixu Hu, Wenxin Hou, Hao Chen, Runkai Zheng, Yidong Wang, Linyi Yang, Haojun Huang, Wei Ye, Xiubo Geng, Binxin Jiao, Yue Zhang, Xing Xie.
  - Key Word: Adversarial Robustness; Out-of-Distribution Generalization; ChatGPT.
  - <details><summary>Digest</summary> ChatGPT is a recent chatbot service released by OpenAI and is receiving increasing attention over the past few months. While evaluations of various aspects of ChatGPT have been done, its robustness, i.e., the performance when facing unexpected inputs, is still unclear to the public. Robustness is of particular concern in responsible AI, especially for safety-critical applications. In this paper, we conduct a thorough evaluation of the robustness of ChatGPT from the adversarial and out-of-distribution (OOD) perspective. 

- Out-of-Domain Robustness via Targeted Augmentations. [[paper]](https://arxiv.org/abs/2302.11861) [[code]](https://github.com/i-gao/targeted-augs)
  - Irena Gao, Shiori Sagawa, Pang Wei Koh, Tatsunori Hashimoto, Percy Liang.
  - Key Word: Out-of-Distribution Generalization; Data Augmentation.
  - <details><summary>Digest</summary> We study principles for designing data augmentations for out-of-domain (OOD) generalization. In particular, we focus on real-world scenarios in which some domain-dependent features are robust, i.e., some features that vary across domains are predictive OOD. For example, in the wildlife monitoring application above, image backgrounds vary across camera locations but indicate habitat type, which helps predict the species of photographed animals. 

- PerAda: Parameter-Efficient and Generalizable Federated Learning Personalization with Guarantees. [[paper]](https://arxiv.org/abs/2302.06637)
  - Chulin Xie, De-An Huang, Wenda Chu, Daguang Xu, Chaowei Xiao, Bo Li, Anima Anandkumar.
  - Key Word: Personalized Federated Learning; Knowledge Distillation.
  - <details><summary>Digest</summary> We propose PerAda, a parameter-efficient pFL framework that reduces communication and computational costs and exhibits superior generalization performance, especially under test-time distribution shifts. PerAda reduces the costs by leveraging the power of pretrained models and only updates and communicates a small number of additional parameters from adapters. PerAda has good generalization since it regularizes each client's personalized adapter with a global adapter, while the global adapter uses knowledge distillation to aggregate generalized information from all clients. 

- Domain Generalization by Functional Regression. [[paper]](https://arxiv.org/abs/2302.04724)
  - Markus Holzleitner, Sergei V. Pereverzyev, Werner Zellinger.
  - Key Word: Domain Generalization; Function-to-Function Regression.
  - <details><summary>Digest</summary> We study domain generalization as a problem of functional regression. Our concept leads to a new algorithm for learning a linear operator from marginal distributions of inputs to the corresponding conditional distributions of outputs given inputs. 

- Federated Minimax Optimization with Client Heterogeneity. [[paper]](https://arxiv.org/abs/2302.04249)
  - Pranay Sharma, Rohan Panda, Gauri Joshi.
  - Key Word: Heterogeneous Federated Minimax Optimization.
  - <details><summary>Digest</summary> We propose a general federated minimax optimization framework that subsumes such settings and several existing methods like Local SGDA. We show that naive aggregation of heterogeneous local progress results in optimizing a mismatched objective function -- a phenomenon previously observed in standard federated minimization. To fix this problem, we propose normalizing the client updates by the number of local steps undertaken between successive communication rounds. 

- Leveraging Domain Relations for Domain Generalization. [[paper]](https://arxiv.org/abs/2302.02609)
  - Huaxiu Yao, Xinyu Yang, Xinyi Pan, Shengchao Liu, Pang Wei Koh, Chelsea Finn.
  - Key Word: Domain Generalization; Ensemble Learning.
  - <details><summary>Digest</summary> We focus on domain shifts, which occur when the model is applied to new domains that are different from the ones it was trained on, and propose a new approach called D^3G. Unlike previous approaches that aim to learn a single model that is domain invariant, D^3G learns domain-specific models by leveraging the relations among different domains. 

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

- No more Reviewer #2: Subverting Automatic Paper-Reviewer Assignment using Adversarial Learning. [[paper]](https://arxiv.org/abs/2303.14443)
  - Thorsten Eisenhofer, Erwin Quiring, Jonas Möller, Doreen Riepel, Thorsten Holz, Konrad Rieck. *USENIX Security Symposium 2023*
  - Key Word: Adversarial Attacks; Paper-Reviewer Assignment.
  - <details><summary>Digest</summary> The number of papers submitted to academic conferences is steadily rising in many scientific disciplines. To handle this growth, systems for automatic paper-reviewer assignments are increasingly used during the reviewing process. These systems use statistical topic models to characterize the content of submissions and automate the assignment to reviewers. In this paper, we show that this automation can be manipulated using adversarial learning. We propose an attack that adapts a given paper so that it misleads the assignment and selects its own reviewers. Our attack is based on a novel optimization strategy that alternates between the feature space and problem space to realize unobtrusive changes to the paper.

- Generalist: Decoupling Natural and Robust Generalization. [[paper]](https://arxiv.org/abs/2303.13813) [[code]](https://github.com/PKU-ML/Generalist)
  - Hongjun Wang, Yisen Wang. *CVPR 2023*
  - Key Word: Adversarial Training; Weight Averaging.
  - <details><summary>Digest</summary> We decouple the natural generalization and the robust generalization from joint training and formulate different training strategies for each one. Specifically, instead of minimizing a global loss on the expectation over these two generalization errors, we propose a bi-expert framework called Generalist where we simultaneously train base learners with task-aware strategies so that they can specialize in their own fields.

- Certifiable (Multi)Robustness Against Patch Attacks Using ERM. [[paper]](https://arxiv.org/abs/2303.08944)
  - Saba Ahmadi, Avrim Blum, Omar Montasser, Kevin Stangl.
  - Key Word: Certifiable Adversarial Robustness; Adversarial Patch Defenses.
  - <details><summary>Digest</summary> In the non-realizable setting when no predictor is perfectly correct on all two-mask operations on all images, we exhibit an example where ERM fails. To overcome this challenge, we propose a different algorithm that provably learns a predictor robust to all two-mask operations using an ERM oracle, based on prior work by Feige et al. [2015]. 

- Certified Robust Neural Networks: Generalization and Corruption Resistance. [[paper]](https://arxiv.org/abs/2303.02251) [[code]](https://github.com/RyanLucas3/HR_Neural_Networks)
  - Amine Bennouna, Ryan Lucas, Bart Van Parys.
  - Key Word: Certified Adversarial Robustness; Robust Overfitting.
  - <details><summary>Digest</summary> We provide here theoretical evidence for this peculiar ``robust overfitting'' phenomenon. Subsequently, we advance a novel loss function which we show both theoretically as well as empirically to enjoy a certified level of robustness against data evasion and poisoning attacks while ensuring guaranteed generalization. We indicate through careful numerical experiments that our resulting holistic robust (HR) training procedure yields SOTA performance in terms of adversarial error loss.

- Rethinking the Effect of Data Augmentation in Adversarial Contrastive Learning. [[paper]](https://arxiv.org/abs/2303.01289) [[code]](https://github.com/PKU-ML/DYNACL)
  - Rundong Luo, Yifei Wang, Yisen Wang.
  - Key Word: Data Augmentation; Adversarial Training; Contrastive Learning.
  - <details><summary>Digest</summary> We revisit existing self-AT methods and discover an inherent dilemma that affects self-AT robustness: either strong or weak data augmentations are harmful to self-AT, and a medium strength is insufficient to bridge the gap. To resolve this dilemma, we propose a simple remedy named DYNACL (Dynamic Adversarial Contrastive Learning). In particular, we propose an augmentation schedule that gradually anneals from a strong augmentation to a weak one to benefit from both extreme cases.

- On the Hardness of Robustness Transfer: A Perspective from Rademacher Complexity over Symmetric Difference Hypothesis Space. [[paper]](https://arxiv.org/abs/2302.12351)
  - Yuyang Deng, Nidham Gazagnadou, Junyuan Hong, Mehrdad Mahdavi, Lingjuan Lyu.
  - Key Word: Robustness Transfer; Rademacher Complexity; Domain Adaptation.
  - <details><summary>Digest</summary> Recent studies demonstrated that the adversarially robust learning under ℓ∞ attack is harder to generalize to different domains than standard domain adaptation. How to transfer robustness across different domains has been a key question in domain adaptation field. To investigate the fundamental difficulty behind adversarially robust domain adaptation (or robustness transfer), we propose to analyze a key complexity measure that controls the cross-domain generalization: the adversarial Rademacher complexity over {symmetric difference hypothesis space HΔH.

- Boosting Adversarial Transferability using Dynamic Cues. [[paper]](https://arxiv.org/abs/2302.12252) [[code]](https://github.com/Muzammal-Naseer/DCViT-AT)
  - Muzammal Naseer, Ahmad Mahmood, Salman Khan, Fahad Khan. *ICLR 2023*
  - Key Word: Adversarial Transferability; Prompting.
  - <details><summary>Digest</summary> We induce dynamic cues within the image models without sacrificing their original performance on images. To this end, we optimize temporal prompts through frozen image models to capture motion dynamics. Our temporal prompts are the result of a learnable transformation that allows optimizing for temporal gradients during an adversarial attack to fool the motion dynamics.

- MultiRobustBench: Benchmarking Robustness Against Multiple Attacks. [[paper]](https://arxiv.org/abs/2302.10980)
  - Sihui Dai, Saeed Mahloujifar, Chong Xiang, Vikash Sehwag, Pin-Yu Chen, Prateek Mittal.
  - Key Word: Adversarial Robustness; Benchmark.
  - <details><summary>Digest</summary> We present the first leaderboard, MultiRobustBench, for benchmarking multiattack evaluation which captures performance across attack types and attack strengths. We evaluate the performance of 16 defended models for robustness against a set of 9 different attack types, including Lp-based threat models, spatial transformations, and color changes, at 20 different attack strengths (180 attacks total).

- Seasoning Model Soups for Robustness to Adversarial and Natural Distribution Shifts. [[paper]](https://arxiv.org/abs/2302.10164)
  - Francesco Croce, Sylvestre-Alvise Rebuffi, Evan Shelhamer, Sven Gowal.
  - Key Word: Adversarial Robustness; Natural Distribution Shifts; Weight Averaging.
  - <details><summary>Digest</summary> We describe how to obtain adversarially-robust model soups (i.e., linear combinations of parameters) that smoothly trade-off robustness to different ℓp-norm bounded adversaries. We demonstrate that such soups allow us to control the type and level of robustness, and can achieve robustness to all threats without jointly training on all of them.

- Randomization for adversarial robustness: the Good, the Bad and the Ugly. [[paper]](https://arxiv.org/abs/2302.07221)
  - Lucas Gnecco-Heredia, Yann Chevaleyre, Benjamin Negrevergne, Laurent Meunier.
  - Key Word: Randomized Classifiers; Adversarial Robustness.
  - <details><summary>Digest</summary> In this work we show that in the binary classification setting, for any randomized classifier, there is always a deterministic classifier with better adversarial risk. In other words, randomization is not necessary for robustness. In many common randomization schemes, the deterministic classifiers with better risk are explicitly described: For example, we show that ensembles of classifiers are more robust than mixtures of classifiers, and randomized smoothing is more robust than input noise injection.

- Raising the Cost of Malicious AI-Powered Image Editing. [[paper]](https://arxiv.org/abs/2302.06588) [[code]](https://github.com/madrylab/photoguard)
  - Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew Ilyas, Aleksander Madry.
  - Key Word: Adversarial Attack Latent Diffusion Models.
  - <details><summary>Digest</summary> We present an approach to mitigating the risks of malicious image editing posed by large diffusion models. The key idea is to immunize images so as to make them resistant to manipulation by these models. This immunization relies on injection of imperceptible adversarial perturbations designed to disrupt the operation of the targeted diffusion models, forcing them to generate unrealistic images. 

- Making Substitute Models More Bayesian Can Enhance Transferability of Adversarial Examples. [[paper]](https://arxiv.org/abs/2302.05086) [[code]](https://github.com/qizhangli/morebayesian-attack)
  - Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen. *ICLR 2023*
  - Key Word: Transferable Adversarial Attacks; Bayesian Neural Network.
  - <details><summary>Digest</summary> The transferability of adversarial examples across deep neural networks (DNNs) is the crux of many black-box attacks. Many prior efforts have been devoted to improving the transferability via increasing the diversity in inputs of some substitute models. In this paper, by contrast, we opt for the diversity in substitute models and advocate to attack a Bayesian model for achieving desirable transferability. Deriving from the Bayesian formulation, we develop a principled strategy for possible finetuning, which can be combined with many off-the-shelf Gaussian posterior approximations over DNN parameters. 

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

- TrojDiff: Trojan Attacks on Diffusion Models with Diverse Targets. [[paper]](https://arxiv.org/abs/2303.05762) [[code]](https://github.com/chenweixin107/TrojDiff)
  - Weixin Chen, Dawn Song, Bo Li.
  - Key Word: Trojan Attacks; Diffusion Models. *CVPR 2023*
  - <details><summary>Digest</summary> We aim to explore the vulnerabilities of diffusion models under potential training data manipulations and try to answer: How hard is it to perform Trojan attacks on well-trained diffusion models? What are the adversarial targets that such Trojan attacks can achieve? To answer these questions, we propose an effective Trojan attack against diffusion models, TrojDiff, which optimizes the Trojan diffusion and generative processes during training.

- CleanCLIP: Mitigating Data Poisoning Attacks in Multimodal Contrastive Learning. [[paper]](https://arxiv.org/abs/2303.03323)
  - Hritik Bansal, Nishad Singhi, Yu Yang, Fan Yin, Aditya Grover, Kai-Wei Chang.
  - Key Word: Poisoning Defenses; Multimodal Contrastive Learning.
  - <details><summary>Digest</summary> We propose CleanCLIP, a finetuning framework that weakens the learned spurious associations introduced by backdoor attacks by re-aligning the representations for individual modalities independently. CleanCLIP can be employed for both unsupervised finetuning on paired image-text data and for supervised finetuning on labeled image data. 

- Poisoning Web-Scale Training Datasets is Practical. [[paper]](https://arxiv.org/abs/2302.10149)
  - Nicholas Carlini, Matthew Jagielski, Christopher A. Choquette-Choo, Daniel Paleka, Will Pearce, Hyrum Anderson, Andreas Terzis, Kurt Thomas, Florian Tramèr.
  - Key Word: Split-View Data Poisoning Attacks; Frontrunning Data Poisoning; Integrity Verification; Timing-Based Defenses.
  - <details><summary>Digest</summary> We introduce two new dataset poisoning attacks that intentionally introduce malicious examples to a model's performance. Our attacks are immediately practical and could, today, poison 10 popular datasets. 

- Temporal Robustness against Data Poisoning. [[paper]](https://arxiv.org/abs/2302.03684)
  - Wenxiao Wang, Soheil Feizi.
  - Key Word: Poisoning Defenses; Temporal Modeling.
  - <details><summary>Digest</summary> Existing defenses are essentially vulnerable in practice when poisoning more samples remains a feasible option for attackers. To address this issue, we leverage timestamps denoting the birth dates of data, which are often available but neglected in the past. Benefiting from these timestamps, we propose a temporal threat model of data poisoning and derive two novel metrics, earliness and duration, which respectively measure how long an attack started in advance and how long an attack lasted. 

- Run-Off Election: Improved Provable Defense against Data Poisoning Attacks. [[paper]](https://arxiv.org/abs/2302.02300)
  - Keivan Rezaei, Kiarash Banihashem, Atoosa Chegini, Soheil Feizi.
  - Key Word: Poisoning Defenses; Deep Partition Aggregation; Finite Aggregation.
  - <details><summary>Digest</summary> We show that merely considering the majority vote in ensemble defenses is wasteful as it does not effectively utilize available information in the logits layers of the base models. Instead, we propose Run-Off Election (ROE), a novel aggregation method based on a two-round election across the base models: In the first round, models vote for their preferred class and then a second, Run-Off election is held between the top two classes in the first round. 

- Backdoor Attacks Against Dataset Distillation. [[paper]](https://arxiv.org/abs/2301.01197) [[code]](https://github.com/liuyugeng/baadd)
  - Yugeng Liu, Zheng Li, Michael Backes, Yun Shen, Yang Zhang. *NDSS 2023*
  - Key Word: Backdoor Attacks; Dataset Distillation.
  - <details><summary>Digest</summary> This study performs the first backdoor attack against the models trained on the data distilled by dataset distillation models in the image domain. Concretely, we inject triggers into the synthetic data during the distillation procedure rather than during the model training stage, where all previous attacks are performed. 

- UNICORN: A Unified Backdoor Trigger Inversion Framework. [[paper]](https://openreview.net/forum?id=Mj7K4lglGyj) [[code]](https://github.com/RU-System-Software-and-Security/UNICORN)
  - Zhenting Wang, Kai Mei, Juan Zhai, Shiqing Ma. *ICLR 2023*
  - Key Word: Backdoor attacks; Backdoor defenses.
  - <details><summary>Digest</summary> Trigger inversion is an effective way of identifying backdoor models and understanding embedded adversarial behaviors. However existing methods cannot generalize to various types of triggers. This work formally defines and analyzes the triggers injected in different spaces and the inversion problem. Then, it proposes a unified framework to invert backdoor triggers based on the formalization of triggers and the identified inner behaviors of backdoor models from our analysis.


## Privacy

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#privacy">:open_file_folder: [<b><i>Full List of Privacy</i></b>]</a>.

- A Recipe for Watermarking Diffusion Models. [[paper]](https://arxiv.org/abs/2303.10137) [[code]](https://github.com/yunqing-me/WatermarkDM)
  - Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Ngai-Man Cheung, Min Lin.
  - Key Word: Diffusion Models; Watermarking; Ownership Verification.
  - <details><summary>Digest</summary> Watermarking has been a proven solution for copyright protection and content monitoring, but it is underexplored in the DMs literature. Specifically, DMs generate samples from longer tracks and may have newly designed multimodal structures, necessitating the modification of conventional watermarking pipelines. To this end, we conduct comprehensive analyses and derive a recipe for efficiently watermarking state-of-the-art DMs (e.g., Stable Diffusion), via training from scratch or finetuning. 

- CUDA: Convolution-based Unlearnable Datasets. [[paper]](https://arxiv.org/abs/2303.04278)
  - Vinu Sankar Sadasivan, Mahdi Soltanolkotabi, Soheil Feizi.
  - Key Word: Privacy; Adversarial Attacks; Poisoning Attacks.
  - <details><summary>Digest</summary> We propose a novel, model-free, Convolution-based Unlearnable DAtaset (CUDA) generation technique. CUDA is generated using controlled class-wise convolutions with filters that are randomly generated via a private key. CUDA encourages the network to learn the relation between filters and labels rather than informative features for classifying the clean data. We develop some theoretical analysis demonstrating that CUDA can successfully poison Gaussian mixture data by reducing the clean data performance of the optimal Bayes classifier. 

- Can Membership Inferencing be Refuted? [[paper]](https://arxiv.org/abs/2303.03648)
  - Zhifeng Kong, Amrita Roy Chowdhury, Kamalika Chaudhuri.
  - Key Word: Membership Inference Attacks; Proof-Of-Learning; Machine Unlearning.
  - <details><summary>Digest</summary> We study the reliability of membership inference attacks in practice. Specifically, we show that a model owner can plausibly refute the result of a membership inference test on a data point x by constructing a proof of repudiation that proves that the model was trained without x. 

- Why Is Public Pretraining Necessary for Private Model Training? [[paper]](https://arxiv.org/abs/2302.09483)
  - Key Word: Pretraining; Differential Privacy; Differentially Private Stochastic Convex Optimization.
  - <details><summary>Digest</summary> In the privacy-utility tradeoff of a model trained on benchmark language and vision tasks, remarkable improvements have been widely reported with the use of pretraining on publicly available data. This is in part due to the benefits of transfer learning, which is the standard motivation for pretraining in non-private settings. However, the stark contrast in the improvement achieved through pretraining under privacy compared to non-private settings suggests that there may be a deeper, distinct cause driving these gains. To explain this phenomenon, we hypothesize that the non-convex loss landscape of a model training necessitates an optimization algorithm to go through two phases.

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

- Overcoming Bias in Pretrained Models by Manipulating the Finetuning Dataset. [[paper]](https://arxiv.org/abs/2303.06167)
  - Angelina Wang, Olga Russakovsky.
  - Key Word: Fine-Tuning Datasets; Fairness.
  - <details><summary>Digest</summary>  we investigate bias when conceptualized as both spurious correlations between the target task and a sensitive attribute as well as underrepresentation of a particular group in the dataset. Under both notions of bias, we find that (1) models finetuned on top of pretrained models can indeed inherit their biases, but (2) this bias can be corrected for through relatively minor interventions to the finetuning dataset, and often with a negligible impact to performance. Our findings imply that careful curation of the finetuning dataset is important for reducing biases on a downstream task, and doing so can even compensate for bias in the pretrained model.

- Robustness Implies Fairness in Casual Algorithmic Recourse. [[paper]](https://arxiv.org/abs/2302.03465)
  - Ahmad-Reza Ehyaei, Amir-Hossein Karimi, Bernhard Schölkopf, Setareh Maghsudi.
  - Key Word: Fairness; Adversarial Robustness; Structural Causal Model.
  - <details><summary>Digest</summary> This study explores the concept of individual fairness and adversarial robustness in causal algorithmic recourse and addresses the challenge of achieving both. To resolve the challenges, we propose a new framework for defining adversarially robust recourse. The new setting views the protected feature as a pseudometric and demonstrates that individual fairness is a special case of adversarial robustness. 

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

- Extending class activation mapping using Gaussian receptive field. [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S1077314223000437)
  - Bum Jun Kim, Gyogwon Koo, Hyeyeon Choi, Sang Woo Kim.
  - Key Word: Class Activation Mapping.
  - <details><summary>Digest</summary> Focusing on class activation mapping (CAM)-based saliency methods, we discuss two problems with the existing studies. First, we introduce conservativeness, a property that prevents redundancy and deficiency in saliency map and ensures that the saliency map is on the same scale as the prediction score. We identify that existing CAM studies do not satisfy the conservativeness and derive a new CAM equation with the improved theoretical property. Second, we discuss the common practice of using bilinear upsampling as problematic. We propose Gaussian upsampling, an improved upsampling method that reflects deep neural networks’ properties. Based on these two options, we propose Extended-CAM, an advanced CAM-based visualization method.

- Deephys: Deep Electrophysiology, Debugging Neural Networks under Distribution Shifts. [[paper]](https://arxiv.org/abs/2303.11912)
  - Anirban Sarkar, Matthew Groth, Ian Mason, Tomotake Sasaki, Xavier Boix.
  - Key Word: Individual Neurons; Distribution Shifts.
  - <details><summary>Digest</summary> We introduce a tool to visualize and understand such failures. We draw inspiration from concepts from neural electrophysiology, which are based on inspecting the internal functioning of a neural networks by analyzing the feature tuning and invariances of individual units. Deep Electrophysiology, in short Deephys, provides insights of the DNN's failures in out-of-distribution scenarios by comparative visualization of the neural activity in in-distribution and out-of-distribution datasets. 

- Finding the right XAI method -- A Guide for the Evaluation and Ranking of Explainable AI Methods in Climate Science. [[paper]](https://arxiv.org/abs/2303.00652)
  - Philine Bommer, Marlene Kretschmer, Anna Hedström, Dilyara Bareeva, Marina M.-C. Höhne.
  - Key Word: Explainable AI; Climate Science.
  - <details><summary>Digest</summary> We introduce XAI evaluation in the context of climate research and assess different desired explanation properties, namely, robustness, faithfulness, randomization, complexity, and localization. To this end we build upon previous work and train a multi-layer perceptron (MLP) and a convolutional neural network (CNN) to predict the decade based on annual-mean temperature maps. 

- Benchmarking Interpretability Tools for Deep Neural Networks. [[paper]](https://arxiv.org/abs/2302.10894)
  - Stephen Casper, Yuxiao Li, Jiawei Li, Tong Bu, Kevin Zhang, Dylan Hadfield-Menell.
  - Key Word: Interpretability; Benchmark.
  - <details><summary>Digest</summary> Inspired by how benchmarks tend to guide progress in AI, we make three contributions. First, we propose trojan rediscovery as a benchmarking task to evaluate how useful interpretability tools are for generating engineering-relevant insights. Second, we design two such approaches for benchmarking: one for feature attribution methods and one for feature synthesis methods. Third, we apply our benchmarks to evaluate 16 feature attribution/saliency methods and 9 feature synthesis methods.

- The Meta-Evaluation Problem in Explainable AI: Identifying Reliable Estimators with MetaQuantus. [[paper]](https://arxiv.org/abs/2302.07265)
  - Key Word: Explainable AI; Meta-Consistency.
  - <details><summary>Digest</summary> In this paper, to identify the most reliable evaluation method in a given explainability context, we propose MetaQuantus -- a simple yet powerful framework that meta-evaluates two complementary performance characteristics of an evaluation method: its resilience to noise and reactivity to randomness. We demonstrate the effectiveness of our framework through a series of experiments, targeting various open questions in XAI, such as the selection of explanation methods and optimisation of hyperparameters of a given metric.

- Efficient XAI Techniques: A Taxonomic Survey. [[paper]](https://arxiv.org/abs/2302.03225)
  - Yu-Neng Chuang, Guanchu Wang, Fan Yang, Zirui Liu, Xuanting Cai, Mengnan Du, Xia Hu.
  - Key Word: Explainable AI; Feature Attribution Explanation; Counterfactual Explanation.
  - <details><summary>Digest</summary> In this paper we provide a review of efficient XAI. Specifically, we categorize existing techniques of XAI acceleration into efficient non-amortized and efficient amortized methods. The efficient non-amortized methods focus on data-centric or model-centric acceleration upon each individual instance. In contrast, amortized methods focus on learning a unified distribution of model explanations, following the predictive, generative, or reinforcement frameworks, to rapidly derive multiple model explanations.

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

- Green Federated Learning. [[paper]](https://arxiv.org/abs/2303.14604)
  - Ashkan Yousefpour, Shen Guo, Ashish Shenoy, Sayan Ghosh, Pierre Stock, Kiwan Maeng, Schalk-Willem Krüger, Michael Rabbat, Carole-Jean Wu, Ilya Mironov.
  - Key Word: Sustainability; Green AI; Federated Learning.
  - <details><summary>Digest</summary> We propose the concept of Green FL, which involves optimizing FL parameters and making design choices to minimize carbon emissions consistent with competitive performance and training time. The contributions of this work are two-fold. First, we adopt a data-driven approach to quantify the carbon emissions of FL by directly measuring real-world at-scale FL tasks running on millions of phones. Second, we present challenges, guidelines, and lessons learned from studying the trade-off between energy efficiency, performance, and time-to-train in a production FL system. Our findings offer valuable insights into how FL can reduce its carbon footprint, and they provide a foundation for future research in the area of Green AI.

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

- Proof-of-Contribution-Based Design for Collaborative Machine Learning on Blockchain. [[paper]](https://arxiv.org/abs/2302.14031)
  - Baturalp Buyukates, Chaoyang He, Shanshan Han, Zhiyong Fang, Yupeng Zhang, Jieyi Long, Ali Farahanchi, Salman Avestimehr.
  - Key Word: Blockchain; Federated Learning; Data Market; Zero-Knowledge Proof.
  - <details><summary>Digest</summary> Our goal is to design a data marketplace for such decentralized collaborative/federated learning applications that simultaneously provides i) proof-of-contribution based reward allocation so that the trainers are compensated based on their contributions to the trained model; ii) privacy-preserving decentralized model training by avoiding any data movement from data owners; iii) robustness against malicious parties (e.g., trainers aiming to poison the model); iv) verifiability in the sense that the integrity, i.e., correctness, of all computations in the data market protocol including contribution assessment and outlier detection are verifiable through zero-knowledge proofs; and v) efficient and universal design. We propose a blockchain-based marketplace design to achieve all five objectives mentioned above. 

## Others

<a href="https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/blob/master/FULL_LIST.md#others">:open_file_folder: [<b><i>Full List of Others</i></b>]</a>.

- Foundation Models and Fair Use. [[paper]](https://arxiv.org/abs/2303.15715)
  - Peter Henderson, Xuechen Li, Dan Jurafsky, Tatsunori Hashimoto, Mark A. Lemley, Percy Liang.
  - Key Word: Foundation Models; Fair Use Doctrine.
  - <details><summary>Digest</summary> In the United States and several other countries, copyrighted content may be used to build foundation models without incurring liability due to the fair use doctrine. However, there is a caveat: If the model produces output that is similar to copyrighted data, particularly in scenarios that affect the market of that data, fair use may no longer apply to the output of the model. In this work, we emphasize that fair use is not guaranteed, and additional work may be necessary to keep model development and deployment squarely in the realm of fair use. 

- XAIR: A Framework of Explainable AI in Augmented Reality. [[paper]](https://arxiv.org/abs/2303.16292)
  - Xuhai Xu, Mengjie Yu, Tanya R. Jonker, Kashyap Todi, Feiyu Lu, Xun Qian, João Marcelo Evangelista Belo, Tianyi Wang, Michelle Li, Aran Mun, Te-Yen Wu, Junxiao Shen, Ting Zhang, Narine Kokhlikyan, Fulton Wang, Paul Sorenson, Sophie Kahyun Kim, Hrvoje Benke.
  - Key Word: Explainable AI; Augmented Reality.
  - <details><summary>Digest</summary> The framework was based on a multi-disciplinary literature review of XAI and HCI research, a large-scale survey probing 500+ end-users' preferences for AR-based explanations, and three workshops with 12 experts collecting their insights about XAI design in AR. XAIR's utility and effectiveness was verified via a study with 10 designers and another study with 12 end-users. 

- Natural Selection Favors AIs over Humans. [[paper]](https://arxiv.org/abs/2303.16200)
  - Dan Hendrycks.
  - Key Word: AI Evoluation; Darwinian Forces; Natural Selection.
  - <details><summary>Digest</summary> We argue that natural selection creates incentives for AI agents to act against human interests. Our argument relies on two observations. Firstly, natural selection may be a dominant force in AI development. Competition and power-seeking may dampen the effects of safety measures, leaving more “natural” forces to select the surviving AI agents. Secondly, evolution by natural selection tends to give rise to selfish behavior. While evolution can result in cooperative behavior in some situations (for example in ants), we will argue that AI development is not such a situation. From these two premises, it seems likely that the most influential AI agents will be selfish.

- Causal Deep Learning. [[paper]](https://arxiv.org/abs/2303.02186)
  - Jeroen Berrevoets, Krzysztof Kacprzyk, Zhaozhi Qian, Mihaela van der Schaar.
  - Key Word: Causality.
  - <details><summary>Digest</summary> The framework which we propose for causal deep learning spans three dimensions: (1) a structural dimension, which allows incomplete causal knowledge rather than assuming either full or no causal knowledge; (2) a parametric dimension, which encompasses parametric forms which are typically ignored; and finally, (3) a temporal dimension, which explicitly allows for situations which capture exposure times or temporal structure. 

- Provable Copyright Protection for Generative Models. [[paper]](https://arxiv.org/abs/2302.10870)
  - Nikhil Vyas, Sham Kakade, Boaz Barak.
  - Key Word: Copyright Protection; Privacy; Memorization.
  - <details><summary>Digest</summary> There is a growing concern that learned conditional generative models may output samples that are substantially similar to some copyrighted data C that was in their training set. We give a formal definition of near access-freeness (NAF) and prove bounds on the probability that a model satisfying this definition outputs a sample similar to C, even if C is included in its training set. Roughly speaking, a generative model p is k-NAF if for every potentially copyrighted data C, the output of p diverges by at most k-bits from the output of a model q that did not access C at all.

- Dataset Interfaces: Diagnosing Model Failures Using Controllable Counterfactual Generation. [[paper]](https://arxiv.org/abs/2302.07865) [[code]](https://github.com/MadryLab/dataset-interfaces)
  - Joshua Vendrow, Saachi Jain, Logan Engstrom, Aleksander Madry.
  - Key Word: Model Debugging; Distribution Shifts; Counterfactual Generation; Diffusion Model.
  - <details><summary>Digest</summary> We introduce dataset interfaces: a framework which allows users to scalably synthesize such counterfactual examples from a given dataset. Specifically, we represent each class from the input dataset as a custom token within the text space of a text-to-image diffusion model. 

- Dataset Distillation with Convexified Implicit Gradients. [[paper]](https://arxiv.org/abs/2302.06755)
  - Noel Loo, Ramin Hasani, Mathias Lechner, Daniela Rus.
  - Key Word: Dataset Distillation; Neural Tangent Kernel.
  - <details><summary>Digest</summary> We propose a new dataset distillation algorithm using reparameterization and convexification of implicit gradients (RCIG), that substantially improves the state-of-the-art. To this end, we first formulate dataset distillation as a bi-level optimization problem. Then, we show how implicit gradients can be effectively used to compute meta-gradient updates. We further equip the algorithm with a convexified approximation that corresponds to learning on top of a frozen finite-width neural tangent kernel. 

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

# Workshops

## Robustness Workshops

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

- [Backdoor Attacks and Defenses in Machine Learning (ICLR 2023)](https://iclr23-bands.github.io/)

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

# Contributing

TBD
