# NVIDIA Nemotron Model Reasoning Challenge

## Background

The [NVIDIA Nemotron Model Reasoning Challenge][nemotron-challenge-page] is an open online challenge in which competitors are required to fine-tune **Nemotron-3-Nano-30B** (hereafter referred to as the "Nemotron model"), a large language model, so that it can better reason about and answer a series of mathematical problems.

More specifically, participants need to fine-tune the Nemotron model with a **LoRA adapter** on the given training examples, each of which consists of a _unique ID_, a _prompt_, and an _answer_. The prompt is a mathematical problem that requires a certain level of reasoning ability to solve. The submission should contain only the trained LoRA adapter for the Nemotron model and a file with some metadata about it. During evaluation, the Nemotron model is loaded with the LoRA adapter using the [vLLM engine][vllm]. For each test case, the model is prompted to generate a response and instructed to place its final answer within a `\boxed{}` LaTeX command. The provided answer is graded as correct if it matches the ground truth either exactly as a string or within a relative numerical tolerance of $10^{-2}$.

The vLLM parameters for the evaluation are specified. The significant parameters worth noting are as follows:

1. `max_lora_rank = 32`: The maximum rank of the LoRA adapter is 32.
2. `max_tokens = 7680; max_model_len = 8192`: The maximum number of tokens that the model can generate for an answer is 7680; the maximum model length (context window) is 8192.
3. `temperature = 0.0; top_p = 1.0`: This means that the sampler always selects the word with the highest probability in the distribution as the next generated token.

This challenge **started on March 16, 2026**, and **ended on June 22, 2026**.

### The Nemotron-3-Nano-30B Model

### Dataset Analysis

[nemotron-challenge-page]: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
[vllm]: https://docs.vllm.ai/en/stable/



### SVD Denoising and Truncation Principle for LoRA Weights
The core idea of LoRA (Low-Rank Adaptation) is to implicitly decompose the weight update matrix $\Delta W$ into the product of two low-rank matrices
$$\Delta W = B \times A$$
where $A \in \mathbb{R}^{r \times d_{in}}$, $B \in \mathbb{R}^{d_{out} \times r}$, and the rank $r \ll \min(d_{in}, d_{out})$.

After fine-tuning, we can apply **QR Decomposition** and **SVD (Singular Value Decomposition)** to truncate the low-rank LoRA weights. This process effectively filters out overfitting noise introduced during training and further compresses the model size.

#### QR Decomposition
Directly performing SVD on the full weight update matrix $\Delta W \in \mathbb{R}^{d_{out} \times d_{in}}$ is computationally prohibitive when dealing with massive network dimensions (e.g., $d > 4096$). To bypass this, we first apply QR decomposition to the low-rank matrices $A^T$ and $B$ individually:
$A^T=Q_A \cdot R_A$, and $B=Q_B\cdot R_B$, where $Q_A\in \mathbb{R}^{d_{in} \times r}$ and $Q_B\in \mathbb{R}^{d_{out} \times r}$ are orthogonal matrices, which means $Q_A^TQ_A=I$ and $Q_B^TQ_B=I$.

Also, we can define the matrix $M$ as
$$M = R_B \cdot R_A^T\in \mathbb{r\times r}$$.
#### SVD Decomposition
Since the dimension of $M$ is only $r \times r$ (where $r$ is typically small, like 8, 16, or 32), performing SVD on it requires negligible computational effort:
$$M = U_m \cdot S \cdot V_m^T,$$ where $U_m \in \mathbb{R}^{r \times r}$ and $V_m^T \in \mathbb{R}^{r \times r}$ are orthogonal matrices, and $S = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_r)$ is a diagonal matrix containing the singular values sorted in descending order ($\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r$).

Substituting the SVD of $M$ back into our equation, the full weight update can be expressed as:
$$\Delta W = Q_B \cdot U_m \cdot S \cdot V_m^T \cdot Q_A^T$$

To eliminate weak features that represent training noise, we keep only the top $k$ most significant singular values (setting a truncation threshold $k < r$) and zero out the remaining components:
$$S_{new} = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_k, 0, \dots, 0)$$

The denoised weight update $\Delta W_{new}$ can then be smoothly reconstructed as:
$$\Delta W_{new} = Q_B \cdot U_m \cdot \sqrt{S_{new}} \cdot \sqrt{S_{new}} \cdot V_m^T \cdot Q_A^T$$

To maintain the native LoRA two-matrix product structure, we distribute the diagonal matrix $\sqrt{S_{new}}$ evenly to both sides. This yields the newly denoised pair of LoRA matrices:

New Matrix $B_{new}$:
  $$B_{new} = Q_B \cdot U_m \cdot \sqrt{S_{new}} \in \mathbb{R}^{d_{out} \times r}$$
New Matrix $A_{new}$:
  $$A_{new} = \sqrt{S_{new}} \cdot V_m^T \cdot Q_A^T \in \mathbb{R}^{r \times d_{in}}$$
