---
title: "DoH Deception: Testing the Robustness of ML-Based Tunnel Detection"
date: 2024-09-16T12:00:00-03:00
tags: ["doh", "dns", "adversarial-machine-learning", "mlsec", "network-security", "research"]
ShowToc: true
math: true
---

High accuracy is not the same thing as robustness.[1]

That was the question behind my master's research on DNS over HTTPS (DoH) tunnel detection. Several machine-learning classifiers reported excellent performance when evaluated on conventional train/test splits. But an attacker does not need to draw another sample from the same dataset distribution. They can change how a tunnel behaves, observe the detector's output, and search for a traffic pattern that still carries their data while the classifier labels the flow as benign.[1]

The result was **DoH Deception: Evading ML-Based Tunnel Detection Models with Real-world Adversarial Examples**, published at SBSeg 2024 with André A. Osti, Lourenço A. P. Júnior, and Júlio C. Estrella.[1] The work was part of my master's research at USP in collaboration with ITA.

The conference timeline is worth making explicit: the BSides Las Vegas and DEF CON 32 Red Team Village presentations were both in **2024**, before SBSeg 2024, which took place from September 16 to 19, 2024. The slides and demonstrations for both talks are available in the public project repository.[2][3][4]

![DoH Deception methodology: adapting ZOO attacks to DoH flow features](https://raw.githubusercontent.com/e-valente/doh-deception/main/img/methodology.png)

## The problem: encrypted DNS makes the observation point statistical

DoH protects DNS queries by transporting them through HTTPS. That is good for privacy, but it also means that many detection systems cannot inspect individual DNS labels or payload bytes at the network boundary. Instead, they classify encrypted flows using derived statistics: flow byte counts and rates, packet-length distributions, packet timing, and request/response timing.[1]

Our experiments used 28 traffic features extracted by DoHLyzer. They included sent and received bytes, packet-length statistics, packet-time statistics, and request/response-time statistics. The study trained four representative classifier architectures: Gradient Boosting, XGBoost, Balanced Stacked, and Random Forest. Their clean-data accuracy ranged from 91.92% to 99.93%.[1]

Those figures are useful, but incomplete. A classifier can perform well on a static benchmark while failing when an adversary deliberately searches its decision boundary. The relevant question is not only whether the model recognizes known tunnel traffic; it is whether a feasible tunnel can be adjusted enough to cross into the benign region.

## From adversarial examples to network flows

For a classifier $F_{\mathrm{model}}$, a standard adversarial-example formulation starts with an input feature vector $x$ and its predicted label $y$:

$$
\begin{aligned}
y &= F_{\mathrm{model}}(x), \\
x_{\mathrm{adv}} &= x + \delta, \\
F_{\mathrm{model}}(x_{\mathrm{adv}}) &= y_t.
\end{aligned}
$$

Here, $\delta$ is a perturbation and $y_t$ is the desired target class. In this work, the target was benign: a flow that began as a tunnel should be classified as ordinary DoH traffic.[1]

The starting point for the optimization was the Carlini-and-Wagner-style objective used by Zeroth Order Optimization (ZOO):

$$
J(x, x_{\mathrm{adv}}, y_t, c) = \lVert x - x_{\mathrm{adv}} \rVert_2^2 + c \cdot \max\left(0, \max_{i \ne y_t} f_i(x_{\mathrm{adv}}) - f_{y_t}(x_{\mathrm{adv}}) + \kappa\right).
$$

The first term penalizes distance from the original traffic features. The second rewards the target misclassification. The constant $c$ is adjusted through binary search to trade off low distortion against attack success; $\kappa$ controls the desired confidence margin.[1]

ZOO is particularly relevant to a defensive setting because it is a black-box method. It does not require classifier gradients, architecture details, or model weights. It estimates the useful direction through model queries and finite differences. That makes it a more realistic stress test for a deployed detector than an attack that assumes access to its internals.[1]

## Feature space is not the problem space

Applying an image-oriented adversarial method directly to network features produces an immediate problem. A vector may fool the classifier while describing traffic that cannot exist: negative packet times, impossible packet statistics, or changes that a tunnel implementation cannot actually create.[1]

The research therefore moved through three increasingly constrained versions of the attack.

### 1. ZOO-DoH

The first adaptation replaced image pixels with DoH flow features and retained the ZOO objective. It demonstrated that the classifiers could be evaded, but it allowed every feature to move freely. The result was useful as a vulnerability signal, not as a realistic traffic recipe: some generated values could not be reproduced by a real tunnel.[1]

### 2. Constrained ZOO-DoH

The second adaptation bounded every feature to a range observed for the available tunnel tools. For each feature $j$, the feasible interval came from the 10th and 90th percentiles of observed values:

$$
q_{0.10,j} \le x_{\mathrm{adv},j} \le q_{0.90,j}.
$$

The optimization objective stayed the same, but candidate vectors were clipped to those tool-derived limits. This removed plainly impossible values and turned the attack from an unconstrained feature-space exercise into a more realistic approximation of the problem space.[1]

### 3. Target ZOO-DoH

The final version added the most important constraint: only features that a tunnel operator could plausibly influence were allowed to change. Let $M$ be the selected set of feature indices. The attack enforced:

$$
\begin{aligned}
x_{\mathrm{adv},j} &= x_j && \text{for every } j \notin M, \\
q_{0.10,j} &\le x_{\mathrm{adv},j} \le q_{0.90,j}
&& \text{for every } j \in M.
\end{aligned}
$$

In the paper's main experiment, $M$ was the twelve time-dependent features F13–F24, covering packet-time and request/response-time statistics. The remaining features stayed fixed. This distinction is the difference between asking whether a classifier can be fooled mathematically and asking whether a tunnel implementation can reproduce the proposed behavior.[1]

![Target ZOO-DoH demonstration: real tunnel traffic, feature extraction, adversarial optimization, and model evaluation](https://raw.githubusercontent.com/e-valente/doh-deception/main/img/demo_target_zoo.png)

## What the experiments showed

The unconstrained version evaded every target architecture in the study, but it did so by proposing values that did not always make sense for network traffic. Adding feature ranges lowered the success rate, as expected, and exposed an important result: some constraints are not merely implementation details. They change whether an adversarial example can exist in the real system.

The Target ZOO-DoH experiment was stricter still. It limited the attack to the time-dependent features that could be reproduced through the tunnel setup. The paper reports that this most constrained version generated adversarial examples for five of the six public DoH tunnel tools considered. Results varied by tool and classifier: the constrained experiments did not converge for TCP-over-DNS, and the Balanced Stacked architecture resisted the targeted configuration in the reported experiments.[1]

We also reproduced generated behavior with the `dnstt` tunnel tool and a TCP proxy, rather than stopping at altered CSV rows. That step was central to the research. The claim was not that arbitrary feature perturbations are meaningful; it was that at least some adversarial examples could be translated back into observable traffic produced by a real tunnel environment.[1][2]

## Why this matters for detection engineering

This work was not an argument that DoH detection is futile. It was an argument that model evaluation needs to include an adversarial feasibility layer.

A useful robustness program for network classifiers should ask at least four questions:

1. Which input features can an attacker realistically influence?
2. What ranges and relationships must those features satisfy to remain physically and protocol feasible?
3. Can the proposed adversarial example be reproduced by the tool or protocol implementation?
4. Does the detector remain effective after retraining or testing against those feasible perturbations?

The distinction generalizes beyond DoH. Network-security models often consume aggregated flow features because the raw payload is encrypted, unavailable, or too expensive to inspect. That makes feature-level attacks tempting, but a defense should not assess them only in normalized vector space. The real boundary is the system that generates the traffic.[1]

## From SBSeg to BSides and DEF CON

The research paper gave the method and experimental evidence a permanent academic record. The BSides Las Vegas and DEF CON 32 Red Team Village versions focused more on reproducibility and demonstration: the baseline tunnel setup, black-box attack workflow, constrained feature selection, and the gap between a synthetic adversarial vector and a flow that can actually be produced.[2][3][4]

The public repository contains the notebooks, the feature list, the demonstration material, and the presentations. It should be used for authorized research, validation, and defensive model testing. The repository also makes an important limitation explicit: outcomes depend on the network conditions under which data is collected, including latency, bandwidth, tunnel configuration, and the DoH resolver used in the experiment.[2]

For me, DoH Deception remains a useful lesson from the master's work: model accuracy tells us how a classifier behaves on a benchmark. Robustness requires testing how the surrounding system behaves when an adversary can manipulate the measurable parts of a real protocol.[1]

## Sources

[1] [DoH Deception: Evading ML-Based Tunnel Detection Models with Real-world Adversarial Examples, SBSeg 2024](https://sol.sbc.org.br/index.php/sbseg/article/view/30032)

[2] [DoH Deception: source code, experiments, and demonstrations](https://github.com/e-valente/doh-deception)

[3] [BSides Las Vegas 2024 presentation: DoH Deception](https://github.com/e-valente/doh-deception/blob/main/BsidesLV_2024_presentation-DoHDecepetion.pdf)

[4] [DEF CON 32 Red Team Village presentation: DoH Deception](https://github.com/e-valente/doh-deception/blob/main/Defcon_RTV_2024_presentation_small.pdf)
