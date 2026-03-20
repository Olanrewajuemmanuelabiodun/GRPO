# Replicating the DeepSeek R1 “Aha Moment” using Reinforcement Learning.

The DeepSeek team introduced their model, DeepSeek-R1-Zero, which was trained using pure reinforcement learning where their is no supervised fine-tuning, no human-curated chain-of-thought data, no hand-crafted reasoning examples. Just a language model, with a reward signal, and the right incentives. During training, the model spontaneously developed the ability to stop, reconsider, and correct its own reasoning. And this phenomenon was called the “Aha Moment”. 

## Understanding Group Relative Policy Optimization (GRPO)

DeepSeek-R1 was trained using a reinforcement learning algorithm called GRPO. GRPO is a variant of policy optimization that is specifi- cally designed for language model training and is simpler and more memory-efficient than the traditional PPO (Proximal Policy Optimization) approach.  
GRPO is more effective than PPO because of some reasons; PPO requires four models which are;  
(1) The policy model (the model being trained)  
(2) A reference model (frozen copy of the original model, necessary for KL divergence)  
(3) A reward model (evaluates how good the outputs is)  
(4) A value model / critic (calculates expected future reward)
