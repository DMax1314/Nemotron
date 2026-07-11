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

#### Synthetic Data Generation
Goal & Why: In order to improve the reasoning ability of solving `SYMBOL` puzzle. We try to use generative AI to enhance the puzzle. 
Challenge: 1. Choose the right LLM model 2. The speed of generating the answer of `SYMBOL` puzzle, each puzzle takes 30 min to generate the answer.
Process:
We try claude, codex, deepseek, GPT pro to solve the same `SYMBOL` puzzle, then find that GPT pro has the highest accuracy rate.
We generate the synthetic data for the puzzle type `SYMBOL` using GPT pro. 

## Synthetic Data Generation

To improve the model’s reasoning ability on `SYMBOL` puzzles, we use generative AI to expand the existing dataset with high-quality synthetic examples. A key challenge in this process was selecting the most suitable large language model. To make an informed choice, we evaluated Claude, Codex, DeepSeek, and GPT Pro on the same set of SYMBOL puzzles and compared their solution accuracy. GPT Pro achieved the highest accuracy among the models tested and was therefore selected for synthetic data generation. Specifically, GPT-5.5 Pro was used to generate the final dataset. Using the original puzzles as seed data, we instruct the model to generate new puzzles that follow the same style, structure, difficulty, and answer format while avoiding duplicates and ambiguous transformation rules. Each generated sample contains a unique identifier, a self-contained puzzle prompt, a concise final answer, and a complete, externally readable explanation of the reasoning process. The generation prompt also requires the model to validate the consistency between each answer and its reasoning, ensure diversity across transformation-rule types, and produce a correctly formatted CSV containing exactly 1,000 high-quality synthetic puzzles. The exact prompt used for generation is shown below:

> You are given an attached CSV file containing seed puzzles. The CSV has columns such as `id`, `prompt`, and `answer`. The puzzles are symbolic transformation puzzles: each prompt provides several example input-output equations, and the solver must infer the hidden transformation rule and apply it to a new input.
>
> Your task is to generate exactly 1,000 new synthetic puzzles that are similar to the attached seed puzzles in style, format, difficulty, and answer type.
>
> Create a new CSV file:
>
> `id`, `prompt`, `answer`, `process`
>
> **Requirements**
>
> **1. Study the attached seed puzzles carefully.**
>
> - Match their general style and structure.
> - Use symbolic strings, punctuation, digits, or other compact character sequences similar to the seed data.
> - Each puzzle should contain several example transformations followed by a final query such as: “Now, determine the result for: ...”
> - Do not copy any original puzzle exactly.
> - Do not create duplicate or near-duplicate generated puzzles.
>
> **2. Generate exactly 1,000 new puzzle rows.**
>
> - Each puzzle must be self-contained.
> - Each puzzle must have one clear, intended solution.
> - The transformation rule should be inferable from the examples.
> - The answer should be concise, usually a short symbol, digit, or character sequence, consistent with the seed data.
>
> **3. Fill the `id` column.**
>
> - Use a unique ID for each generated row.
> - The ID may follow the same style as the seed file, such as an 8-character lowercase hexadecimal string.
> - No two generated rows may have the same ID.
>
> **4. Fill the `prompt` column.**
>
> - Write the full puzzle prompt.
> - Preserve the style of the seed puzzles, including the “Alice's Wonderland” framing if that appears in the attached data.
> - Include enough examples for the transformation rule to be deduced.
> - Make sure all special characters are properly escaped in the CSV.
>
> **5. Fill the `answer` column.**
>
> - Provide only the final answer for the queried input.
> - Do not include explanation, punctuation outside the answer, or extra commentary.
>
> **6. Fill the `process` column.**
>
> - The `process` column should contain a clear, logically sound, externally readable explanation of how the final answer is derived.
> - Explain the transformation rule inferred from the examples.
> - Show how the rule applies to the queried input.
> - Include relevant observations, deductions, checks, and any intermediate reasoning needed to verify the answer.
> - The reasoning should be coherent and complete enough for a reader to verify the solution.
> - Do not include vague statements such as “by pattern recognition” without explaining the pattern.
>
> **7. If a puzzle cannot be solved with confidence:**
>
> - Leave both the `answer` and `process` fields blank for that row.
> - Do not guess.
> - Do not fabricate reasoning.
> - Ideally, discard unsolved generated puzzles and replace them with solvable ones so that the final CSV still contains exactly 1,000 high-quality rows.
>
> **8. Quality constraints:**
>
> - Avoid overly trivial puzzles where the answer is copied directly from one example.
> - Avoid ambiguous rules where multiple answers could fit the examples.
> - Vary the transformation rules across the dataset.
> - Include a mix of rule types, such as character substitution, deletion, insertion, reversal, position-based selection, adjacent-pair mapping, digit arithmetic, symbol mapping, and multi-step transformations.
> - Ensure the examples in each puzzle are sufficient to support the intended rule.
> - Verify that each `answer` matches the rule described in `process`.
>
> **9. CSV output requirements:**
>
> - Output only a valid CSV file.
> - Do not include Markdown, explanations, summaries, or commentary outside the CSV.
> - Column order must be exactly:
>
>   `id,prompt,answer,process`
>
> - Properly quote and escape commas, quotation marks, backslashes, apostrophes, and line breaks inside CSV fields.
> - The final CSV must contain exactly 1,000 data rows plus the header row.
>
> **Before finalizing, internally validate the generated CSV:**
>
> - Count exactly 1,000 generated rows.
> - Confirm every ID is unique.
> - Confirm every prompt is unique.
> - Confirm every solved row has both `answer` and `process` filled.
> - Confirm no row has `answer` filled while `process` is blank, or `process` filled while `answer` is blank.
> - Confirm each `process` explains the same rule used to produce the `answer`.
> - Confirm the output is valid CSV and can be parsed without errors.



### On-policy Distillation



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
