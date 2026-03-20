# Replicating the DeepSeek R1 “Aha Moment” using Reinforcement Learning.

The DeepSeek team introduced their model, DeepSeek-R1-Zero, which was trained using pure reinforcement learning where their is no supervised fine-tuning, no human-curated chain-of-thought data, no hand-crafted reasoning examples. Just a language model, with a reward signal, and the right incentives. During training, the model spontaneously developed the ability to stop, reconsider, and correct its own reasoning. And this phenomenon was called the “Aha Moment”. 

## Understanding Group Relative Policy Optimization (GRPO)

DeepSeek-R1 was trained using a reinforcement learning algorithm called GRPO. GRPO is a variant of policy optimization that is specifi- cally designed for language model training and is simpler and more memory-efficient than the traditional PPO (Proximal Policy Optimization) approach.  
GRPO is more effective than PPO because of some reasons; PPO requires four models which are;  
(1) The policy model (the model being trained)  
(2) A reference model (frozen copy of the original model, necessary for KL divergence)  
(3) A reward model (evaluates how good the outputs is)  
(4) A value model / critic (calculates expected future reward)

GRPO solves the limitation of PPO by eliminating the need for the value model by computing baseline from the group of samples itself. This reduces memory requirements significantly necessary for training on GPUs.

## How GRPO Works
The GRPO algorithm operates in four steps for each training iteration:
1. Sampling. For each prompt in the batch, generate multiple completions (e.g., G = 4 completions per prompt) using the current policy model.
2. Reward Scoring. Score each completion using the reward function(s). For instance, using rule-based rewards based on format correctness and answer accuracy or using learned reward model.
3. Advantage Calculation. For each group of G completions from the same prompt:
   Compute the group mean μ and group standard deviation σ of the rewards.
4. The advantage of each completion is:

$$
A_i = \frac{r_i - \mu}{\sigma}
$$

where $r_i$ is the reward score.

5. Policy Update. Update the policy to maximize the GRPO objective:

$$
J_{\mathrm{GRPO}} =
\mathbb{E} \Bigg[
\frac{1}{G} \sum_{i=1}^{G}
\min \Big(
\frac{\pi_\theta(o_i | q)}{\pi_{\theta_{\mathrm{old}}}(o_i | q)} A_i,\;
\mathrm{clip}\Big(
\frac{\pi_\theta(o_i | q)}{\pi_{\theta_{\mathrm{old}}}(o_i | q)},
1 - \varepsilon,\; 1 + \varepsilon
\Big) A_i
\Big)
- \beta \, D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})
\Bigg]
$$

where $\varepsilon$ is the clipping ratio, $\beta$ is the KL divergence coefficient, $\pi_\theta$ is the current policy, and $\pi_{\text{ref}}$ is the reference policy.
