# Replicating the DeepSeek R1 “Aha Moment” using Reinforcement Learning.

The DeepSeek team introduced their model, DeepSeek-R1-Zero, which was trained using pure reinforcement learning where their is no supervised fine-tuning, no human-curated chain-of-thought data, no hand-crafted reasoning examples. Just a language model, with a reward signal, and the right incentives. During training, the model spontaneously developed the ability to stop, reconsider, and correct its own reasoning. And this phenomenon was called the “Aha Moment”. 

## Understanding Group Relative Policy Optimization (GRPO)

DeepSeek-R1 was trained using a reinforcement learning algorithm called GRPO. GRPO is a variant of policy optimization that is specifically designed for language model training and is simpler and more memory-efficient than the traditional PPO (Proximal Policy Optimization) approach.  
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

5. Policy Update. Update the policy to maximize the GRPO objective based on GRPO formula:

## The Countdown Game: A Reasoning Testbed
The experiment uses the Countdown Game as the task for training the model to reason.
This is a numerical puzzle where:
You are given a set of numbers (e.g., [3, 7, 12, 50]).
You are given a target number (e.g., 28).
You must combine the given numbers using basic arithmetic (+, −, ×, ÷) to reach the target.
Each number may be used exactly once.

For instance: given numbers [3, 7, 12, 50] and target 28, a valid solution is: 50-12-7-3 = 28
(each used number appears exactly once).

The Countdown Game is an ideal example for studying emergent reasoning because:
1. It has verifiable answers that can be check if the equation is mathematically correct or not.
2. It requires multi-step reasoning. The model must try different combinations to arrive at the right answer.
3. It rewards systematic exploration than guessing.
4. It ensures that the model cannot memorize solutions because different combination of numbers is needed forcing the model to learn strategies and not answers.

## Reward Functions
The training uses two rule-based reward functions (no learned reward model needed)

Format Reward:
Checks whether the model’s output follows the required format: Returns 1.0 if the format is correct, 0.0 otherwise. This teaches the model to separate its thinking from its answer (a critical structural pattern for reasoning).

Accuracy Reward:
Checks whether the equation in the answer tags is mathematically correct by doing:
1. Extracts the equation from the answer tags.
2. Verifies that each given number is used exactly once.
3. Evaluates the equation and checks if it equals the target (within tolerance of 10^−5).
4. Returns 1.0 for correct, 0.0 for incorrect.

Note: The reward funtion does not include the following: 
reward for good reasoning, using the word wait, self-correction. This is done so the model can discover the aha moment.

## Experiments
All the experiments was done on Runpod using NVIDIA H100 80 GB SXM GPU.

For the experiment Qwen2.5-3B-Instruct was used because it offers a strong balance between reasoning ability, stability, and compute efficiency. It is large enough to learn multi-step reasoning patterns needed for GRPO and gives a clean, structured starting policy for GRPO to improve rather than learn from scratch, but still small enough to train on limited GPU resources. A smaller model usually struggles with multi-step reasoning and collapses under RL reward noise. 

## Understanding the Key Parameters
learning_rate = 5e-7: Very small learning rate for stability
num_generations = 2: Number of completions generated per prompt for GRPO’s group comparison. More generations = better advantage estimates but more compute.
beta = 0.001: KL divergence penalty weight. Keeps the policy close to the reference model.
max_completion_length = 1024: Maximum tokens the model can generate per completion. Must be long enough for the model to “think” and produce an answer.
max_steps = 20: Total training steps. The aha moment typically begins around step 100–200. Increasing the max_steps leads to more compute resources.
save_steps = 1: Checkpoints are saved every 1 step, to inspect the model behavior at any point during training. 

## Further training

To train the model further say max_steps = 500 which is where the aha moment occur it is advised to use LLM parallelism techniques to ensure distributed training. Training for max_steps = 500 on a single GPU will take days. 
One of the recommended way for the parallelism is using DeepSpeed and ensure ZeRO Stage 3 where the model parameters, gradients, and optimizer states are shards across all training GPUs.

Say 4 GPUs, 3 of them will be use for training computation (forward pass, backward pass, optimizer step). And the remaining GPU is use by vLLM for generation. Where during each GRPO step, the training GPUs send prompts to the vLLM server, which generates the G completions in parallel using optimized inference. This separation is efficient because generation (autoregressive, memory-bandwidth bound) and training (compute-bound with gradient computation) have very different hardware utilization profiles.

## Expected behavior if train for about 450 steps
Early training (step ∼25–50): The model likely produces garbled or incorrectly formatted output. 
Format learning (step ∼50–100): The model uses <think> and <answer> tags correctly but its reasoning is primitive or incorrect.
Verbal reasoning (step ∼100–150): The model begins to explain its thinking in natural language, trying different combinations of numbers.
The aha moment (step ∼150–250): The model exhibits self-correction.
Systematic reasoning (step ∼300–450): The model methodically tries combinations and evaluates them, demonstrating a programmatic approach.

