[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/graphs/commit-activity)
![ ](https://img.shields.io/github/last-commit/MinghuiChen43/awesome-trustworthy-deep-learning)
![visitor badge](https://visitor-badge-reloaded.herokuapp.com/badge?page_id=MinghuiChen43/awesome-trustworthy-deep-learning&text=views&color=crimson)
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

- A Comprehensive Survey on Trustworthy Recommender Systems. [[paper]](https://arxiv.org/abs/2209.10117)
  - Wenqi Fan, Xiangyu Zhao, Xiao Chen, Jingran Su, Jingtong Gao, Lin Wang, Qidong Liu, Yiqi Wang, Han Xu, Lei Chen, Qing Li.
  - Key Word: Recommender Systems; Survey.
  - <details><summary>Digest</summary> We provide a comprehensive overview of Trustworthy Recommender systems (TRec) with a specific focus on six of the most important aspects; namely, Safety & Robustness, Nondiscrimination & Fairness, Explainability, Privacy, Environmental Well-being, and Accountability & Auditability. For each aspect, we summarize the recent related technologies and discuss potential research directions to help achieve trustworthy recommender systems in the future.

- Non-Imaging Medical Data Synthesis for Trustworthy AI: A Comprehensive Survey. [[paper]](https://arxiv.org/abs/2209.09239)
  - Xiaodan Xing, Huanjun Wu, Lichao Wang, Iain Stenson, May Yong, Javier Del Ser, Simon Walsh, Guang Yang.
  - Key Word: Non-Imaging Medical Data Generation; Healthcare; Survey.
  - <details><summary>Digest</summary> State-of-the-art data synthesis algorithms, especially deep learning algorithms, focus more on imaging data while neglecting the synthesis of non-imaging healthcare data, including clinical measurements, medical signals and waveforms, and electronic healthcare records (EHRs). Thus, in this paper, we will review the synthesis algorithms, particularly for non-imaging medical data, with the aim of providing trustworthy AI in this domain. This tutorial-styled review paper will provide comprehensive descriptions of non-imaging medical data synthesis on aspects including algorithms, evaluations, limitations and future research directions.

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

## Out-of-Distribution Generalization

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

- Invariant Structure Learning for Better Generalization and Causal Explainability. [[paper]](https://arxiv.org/abs/2206.06469)
  - Yunhao Ge, Sercan Ö. Arik, Jinsung Yoon, Ao Xu, Laurent Itti, Tomas Pfister.
  - Key Word: Causal Structure Discovery; Explainability; Invariant Learning.
  - <details><summary>Digest</summary> Learning the causal structure behind data is invaluable for improving generalization and obtaining high-quality explanations. We propose a novel framework, Invariant Structure Learning (ISL), that is designed to improve causal structure discovery by utilizing generalization as an indication. ISL splits the data into different environments, and learns a structure that is invariant to the target across different environments by imposing a consistency constraint. An aggregation mechanism then selects the optimal classifier based on a graph structure that reflects the causal mechanisms in the data more accurately compared to the structures learnt from individual environments.

- Causal Balancing for Domain Generalization. [[paper]](https://arxiv.org/abs/2206.05263)
  - Xinyi Wang, Michael Saxon, Jiachen Li, Hongyang Zhang, Kun Zhang, William Yang Wang.
  - Key Word: Invariant Learning; Causal Semantic Generative Model.
  - <details><summary>Digest</summary> While current domain generalization methods usually focus on enforcing certain invariance properties across different domains by new loss function designs, we propose a balanced mini-batch sampling strategy to reduce the domain-specific spurious correlations in the observed training distributions. More specifically, we propose a two-phased method that 1) identifies the source of spurious correlations, and 2) builds balanced mini-batches free from spurious correlations by matching on the identified source.

- GSmooth: Certified Robustness against Semantic Transformations via Generalized Randomized Smoothing. 
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

## Evasion Attacks and Defenses

- Implicit Bias of Adversarial Training for Deep Neural Networks. [[paper]](https://openreview.net/forum?id=l8It-0lE5e7)
  - Bochen Lv, Zhanxing Zhu. *ICLR 2022*
  - Key Word: Adversarial Training.
  - <details><summary>Digest</summary> We provide theoretical understandings of the implicit bias imposed by adversarial training for homogeneous deep neural networks without any explicit regularization. In particular, for deep linear networks adversarially trained by gradient descent on a linearly separable dataset, we prove that the direction of the product of weight matrices converges to the direction of the max-margin solution of the original dataset.

- Reducing Excessive Margin to Achieve a Better Accuracy vs. Robustness Trade-off. [[paper]](https://openreview.net/forum?id=Azh9QBQ4tR7) [[code]](https://github.com/imrahulr/hat)
  - Rahul Rade, Seyed-Mohsen Moosavi-Dezfooli. *ICLR 2022*
  - Key Word: Adversarial Training.
  - <details><summary>Digest</summary> We closely examine the changes induced in the decision boundary of a deep network during adversarial training. We find that adversarial training leads to unwarranted increase in the margin along certain adversarial directions, thereby hurting accuracy. Motivated by this observation, we present a novel algorithm, called Helper-based Adversarial Training (HAT), to reduce this effect by incorporating additional wrongly labelled examples during training.

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

## Poisoning Attacks and Defenses

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

## Privacy

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
  </details>

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

## Fairness

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

## Interpretability

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

## Open-World Learning

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

## Environmental Well-being

- Measuring the Carbon Intensity of AI in Cloud Instances. [[paper]](https://arxiv.org/abs/2206.05229)
  - Jesse Dodge, Taylor Prewitt, Remi Tachet Des Combes, Erika Odmark, Roy Schwartz, Emma Strubell, Alexandra Sasha Luccioni, Noah A. Smith, Nicole DeCario, Will Buchanan. *FAccT 2022*
  - Key Word: Carbon Emissions; Cloud.
  - <details><summary>Digest</summary> We provide a framework for measuring software carbon intensity, and propose to measure operational carbon emissions by using location-based and time-specific marginal emissions data per energy unit. We provide measurements of operational software carbon intensity for a set of modern models for natural language processing and computer vision, and a wide range of model sizes, including pretraining of a 6.1 billion parameter language model.

- The Carbon Footprint of Machine Learning Training Will Plateau, Then Shrink. [[paper]](https://arxiv.org/abs/2204.05149)
  - David Patterson, Joseph Gonzalez, Urs Hölzle, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, Jeff Dean.
  - Key Word: Carbon Footprint.
  - <details><summary>Digest</summary> We recommend that ML papers include emissions explicitly to foster competition on more than just model quality. Estimates of emissions in papers that omitted them have been off 100x-100,000x, so publishing emissions has the added benefit of ensuring accurate accounting. Given the importance of climate change, we must get the numbers right to make certain that we work on its biggest challenges.

## Interactions with Blockchain

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

## Others

- Advances, challenges and opportunities in creating data for trustworthy AI. [[paper]](https://www.nature.com/articles/s42256-022-00516-1)
  - Weixin Liang, Girmaw Abebe Tadesse, Daniel Ho, Fei-Fei Li, Matei Zaharia, Ce Zhang, James Zou. *Nature Machine Intelligence*
  - Key Word: Trustworthy AI; Data Design; Data Sculpting; Data Strategies; Data Policy.
  - <details><summary>Digest</summary> Automated AI model builders that are publicly available can now achieve top performance in many applications. In contrast, the design and sculpting of the data used to develop AI often rely on bespoke manual work, and they critically affect the trustworthiness of the model. This Perspective discusses key considerations for each stage of the data-for-AI pipeline—starting from data design to data sculpting (for example, cleaning, valuation and annotation) and data evaluation—to make AI more reliable.

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

# Related Awesome Lists

## Robustness Lists

- [A Complete List of All (arXiv) Adversarial Example Papers](https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html)

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

## Privacy Toolboxes

- [Diffprivlib](https://github.com/IBM/differential-privacy-library) ![ ](https://img.shields.io/github/stars/IBM/differential-privacy-library)
  - Diffprivlib is a general-purpose library for experimenting with, investigating and developing applications in, differential privacy.

- [Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter) ![ ](https://img.shields.io/github/stars/privacytrustlab/ml_privacy_meter)
  - Privacy Meter is an open-source library to audit data privacy in statistical and machine learning algorithms.

- [PrivacyRaven](https://github.com/trailofbits/PrivacyRaven) ![ ](https://img.shields.io/github/stars/trailofbits/PrivacyRaven)
  - PrivacyRaven is a privacy testing library for deep learning systems.

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

- [Causal Inference 360](https://github.com/IBM/causallib) ![ ](https://img.shields.io/github/stars/IBM/causallib)
  - A Python package for inferring causal effects from observational data.

# Workshops

## Robustness Workshops

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

## Interpretability Workshops

- [Interpretable Machine Learning in Healthcare (ICML 2022)](https://sites.google.com/view/imlh2022)

## Other Workshops

- [International Workshop on Trustworthy Federated Learning (IJCAI 2022)](https://federated-learning.org/fl-ijcai-2022/)

- [Workshop on AI Safety (IJCAI 2022)](https://www.aisafetyw.org/)

- [1st Workshop on Formal Verification of Machine Learning (ICML 2022)](https://www.ml-verification.com/)

- [Workshop on Distribution-Free Uncertainty Quantification (ICML 2022)](https://sites.google.com/berkeley.edu/dfuq-22/home)

- [First Workshop on Causal Representation Learning (UAI 2022)](https://crl-uai-2022.github.io/)

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
