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

Nemotron-3-Nano-30B-A3B-BF16 is a large language model (LLM) trained from scratch by NVIDIA, and designed as a unified model for both reasoning and non-reasoning tasks. It responds to user queries and tasks by first generating a reasoning trace and then concluding with a final response.

According to the [model description on Hugging Face][nemotron-model]:

> The model employs a hybrid Mixture-of-Experts (MoE) architecture, consisting of 23 Mamba-2 and MoE layers, along with 6 Attention layers. Each MoE layer includes 128 experts plus 1 shared expert, with 6 experts activated per token. The model has 3.5B active parameters and 30B parameters in total.

The Nemotron model is not a conventional stack of 52 Transformer blocks. NVIDIA counts each residual operation (i.e., Mamba, attention, or MoE) as an individual "layer".

The exact arrangement shown in NVIDIA's architecture figure is

```
MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
```

Here, `M` is a [Mamba-2][mamba-2] layer, `E` is a [MoE][moe] layer, and `A` is a grouped-query [self-attention][self-attention].

The model activates six of the 128 routed experts in each MoE layer, along with one shared expert that is always active. As a result, roughly 5.42% of the stored parameters are activated for each token.

| Component                    | Total Stored | Active per token |
|------------------------------|--------------|------------------|
| 23 MoE layers                | 29.842B      | 1.844B           |
| 23 Mamba-2 layers            | 0.891B       | 0.891B           |
| 6 attention layers           | 0.140B       | 0.140B           |
| Output vocabulary projection | 0.352B       | 0.352B           |
| Input token embeddings       | 0.352B       | 0.352B           |

Because the model uses BF16 to store paraemters, each parameter occupies two bytes. The weight-storage is therefore:

$$
30 \text{B} \times 2 \text{bytes} = 60 \text{GB}.
$$

And the active parameters is

$$
3.5 \text{B} \times 2 \text{bytes} = 7 \text{GB}.
$$

#### Mixture of Experts (MoE)

[reserved: explanation of MoE and how the one in Nemotorn differs]

#### SVD Denoising and Truncation Principle for LoRA Weights

The core idea of LoRA (Low-Rank Adaptation) is to implicitly decompose the weight update matrix $\Delta W$ into the product of two low-rank matrices

$$\Delta W = B \times A$$

where $A \in \mathbb{R}^{r \times d_{in}}$, $B \in \mathbb{R}^{d_{out} \times r}$, and the rank $r \ll \min(d_{in}, d_{out})$.

After fine-tuning, we can apply **QR Decomposition** and **SVD (Singular Value Decomposition)** to truncate the low-rank LoRA weights. This process effectively filters out overfitting noise introduced during training and further compresses the model size.

##### QR Decomposition

Directly performing SVD on the full weight update matrix $\Delta W \in \mathbb{R}^{d_{out} \times d_{in}}$ is computationally prohibitive when dealing with massive network dimensions (e.g., $d > 4096$). To bypass this, we first apply QR decomposition to the low-rank matrices $A^T$ and $B$ individually: $A^T=Q_A \cdot R_A$, and $B=Q_B\cdot R_B$, where $Q_A\in \mathbb{R}^{d_{in} \times r}$ and $Q_B\in \mathbb{R}^{d_{out} \times r}$ are orthogonal matrices, which means $Q_A^TQ_A=I$ and $Q_B^TQ_B=I$.

Also, we can define the matrix $M$ as

$$
M = R_B \cdot R_A^T\in \mathbb{r\times r}
$$

##### SVD Decomposition

Since the dimension of $M$ is only $r \times r$ (where $r$ is typically small, like 8, 16, or 32), performing SVD on it requires negligible computational effort:
$$M = U_m \cdot S \cdot V_m^T,$$ where $U_m \in \mathbb{R}^{r \times r}$ and $V_m^T \in \mathbb{R}^{r \times r}$ are orthogonal matrices, and $S = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_r)$ is a diagonal matrix containing the singular values sorted in descending order ($\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r$).

Substituting the SVD of $M$ back into our equation, the full weight update can be expressed as:

$$
\Delta W = Q_B \cdot U_m \cdot S \cdot V_m^T \cdot Q_A^T
$$

To eliminate weak features that represent training noise, we keep only the top $k$ most significant singular values (setting a truncation threshold $k < r$) and zero out the remaining components:

$$
S_{new} = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_k, 0, \dots, 0)
$$

The denoised weight update $\Delta W_{new}$ can then be smoothly reconstructed as:

$$
\Delta W_{new} = Q_B \cdot U_m \cdot \sqrt{S_{new}} \cdot \sqrt{S_{new}} \cdot V_m^T \cdot Q_A^T
$$

To maintain the native LoRA two-matrix product structure, we distribute the diagonal matrix $\sqrt{S_{new}}$ evenly to both sides. This yields the newly denoised pair of LoRA matrices:

New Matrix $B_{new}$:

$$
B_{new} = Q_B \cdot U_m \cdot \sqrt{S_{new}} \in \mathbb{R}^{d_{out} \times r}
$$

New Matrix $A_{new}$:

$$
A_{new} = \sqrt{S_{new}} \cdot V_m^T \cdot Q_A^T \in \mathbb{R}^{r \times d_{in}}
$$

### Dataset Analysis

In this challenge, we are provided with a training set (a `train.csv` file) and a test set (a `test.csv` file). The columns of the training set are `id`, `prompt`, and `answer`, while the test set has only `id` and `prompt`. The training set has 9500 examples. In this document, an **example** refers to the combination of a prompt and its corresponding answer. Because the test set has only prompts without answers, we didn't use it in this project.

The `prompt` of every example is a math problem. To avoid any confusion, we refer to a `prompt` value as a **puzzle** in this document. We found that there are six types of puzzles in the training set. The type of the puzzle can be inferred from keywords in its first sentence. The following is the implementation we used to infer the puzzle type:

```python
def infer_puzzle_type(prompt: str) -> PuzzleType:
    first_line = prompt.split('\n', 1)[0]
    if 'bit' in first_line:
        return PUZZLE_BIT
    if 'gravitational' in first_line:
        return PUZZLE_GRAVITY
    if 'unit' in first_line:
        return PUZZLE_UNIT
    if 'numeral' in first_line:
        return PUZZLE_ROMAN
    if 'encryption' in first_line:
        return PUZZLE_CIPHER
    if 'equations' in first_line:
        return PUZZLE_SYMBOL
    return PUZZLE_OTHER
```

In this section, we will discuss the six puzzle types. It is worth noting that these puzzle types were not mentioned by the host, and there is no consistent naming for them across the community.

#### 1. Type `Bit`

The first puzzle of type `Bit` in the training set is as follows:

```
In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
01010001 -> 11011101
00001001 -> 01101101
00010101 -> 01010101
11111111 -> 10000001
10011101 -> 01000101
00111011 -> 00001001
10111101 -> 00000101
00100110 -> 10110011

Now, determine the output for: 00110100
```

We are asked to find a combination of bit operations that transforms the given example inputs into their corresponding outputs, and then to determine the output for a new input. The answer to this puzzle is `10010111`.

This puzzle is easy to solve by brute force. There are ten bit operations to consider (both bit shifts and rotations come in two directions). However, we don't know how many times these operations are applied. In addition, bit shifts and rotations on 8-bit values take a parameter indicating the number of positions to shift or rotate, which makes the brute-force search even harder.

This is the _second-hardest puzzle type_ for Nemotron to solve.

#### 2. Type `Gravity`

The first puzzle of type `Gravity` in the training set is as follows:

```
In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:
For t = 1.37s, distance = 14.92 m
For t = 4.27s, distance = 144.96 m
For t = 3.28s, distance = 85.54 m
For t = 3.67s, distance = 107.09 m
For t = 1.78s, distance = 25.19 m
Now, determine the falling distance for t = 4.41s given d = 0.5*g*t^2.
```

In this type of puzzle, we first find the gravitational constant. The formula is provided, so the constant can be found by $g = 2d/t^2$, using any pair of time and distance from the examples. Then, we simply plug $g$ and the given $t$ into the formula to find the distance $d$. The answer to this puzzle is `154.62`.

This is easy for a human or a rule-based system to solve. However, medium-scale large language models are not good at decimal multiplication and division. Therefore, we may need to give some guidance to help it reach the correct answer.

#### 3. Type `Unit`

The first puzzle of type `Unit` in the training set is as follows:

```
In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:
10.08 m becomes 6.69
17.83 m becomes 11.83
35.85 m becomes 23.79
17.06 m becomes 11.32
31.54 m becomes 20.93
Now, convert the following measurement: 25.09 m
```

In this type of puzzle, we just need to find the conversion factor by dividing the first number by the second number in any given example. Then, we can obtain the answer by dividing the given number by the ratio. The answer to this puzzle is `16.65`.

This is easy for a human or a rule-based system to solve. For a medium-scale large language model, the trickiest part here is the decimal division.

#### 4. Type `Roman`

The first puzzle of type `Roman` in the training set is as follows:

```
In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:
11 -> XI
15 -> XV
94 -> XCIV
19 -> XIX
Now, write the number 38 in the Wonderland numeral system.
```

We are required to convert an Arabic numeral into a Roman numeral. The answer to this puzzle is `XXXVIII`.

The interesting thing is that if we know the conversion rule, we don't need the provided examples. The same goes for a large language model. This is the easiest type of puzzle among all.

#### 5. Type `Cipher`

The first puzzle of type `Cipher` in the training set is as follows:

```
In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley
pqrsfv pqorzg wvgwpo trgbjo -> dragon dreams inside castle
gbcpovb tqorbog bxo zrswtrj pffq -> student creates the magical door
bxo sfjpov pqrsfv dfjjfig -> the golden dragon follows
nqwvtogg qorpg bxo zegboqwfcg gotqob -> princess reads the mysterious secret
Now, decrypt the following text: trb wzrswvog hffk
```

Observing the example encryptions, it is noticeable that the lengths of the words don't change after encryption. Also, `ucoov` maps to `queen`, and `bxo` maps to `the` twice. Our conjecture is that there exists a mapping that converts each character during encryption. The answer to this puzzle is `cat imagines book`.

This is extremely easy for human and rule-based system to solve. Unsurprisingly, this is an easy type of puzzle for Nemotron.

#### 6. Type `Symbol`

```
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
`!*[{ = '"[`
\'*'> = ![@
\'-!` = \\
`!*\& = '@'{
Now, determine the result for: [[-!'
```

In this type of puzzle, we need to find the transformation rules of some characters and some binary operations. After observing the shape of more of these types of puzzles, the `+`, `-`, `*`, `/` on the left-hand side of the equation are binary operators, but they may not perform as they are. For example, a + operator can do subtraction. The answer to this puzzle is `@&`.

The hardest part of this type of puzzles is guessing the digits mapped by the symbols that are not operators. It turns out that we can only do brute force to solve them, and it takes some time for rule-based solver to find the answer. A not fine-tuned Nemotron model can only solve one to two puzzles in the 1555 Symbol puzzles.

Without any question, this is the hardest type of puzzle among all.

## Experiments

### Rule-based Reasoning Content Generation (RRCG)

At first, we considered using better reasoning models (e.g., ChatGPT, Gemini, and Claude) to generate correct reasoning content for all 9,500 examples in the training set. However, we would have had to automate the process using API calls, which would have cost us an arm and a leg. Even worse, after asking Claude to solve some puzzles, we found that it sometimes provided incorrect answers, which meant that the corresponding reasoning content was also incorrect.

We found that all six types of puzzles can be solved step by step. Types such as `Gravity` and `Unit` require very few reasoning steps, while others require more. Types such as `Bit` and `Symbol` require brute force, resulting in many reasoning steps. Regardless, we can create a rule-based solver that solves each puzzle and produces the correct answer. This process requires no LLM and is deterministic.

The rule-based reasoning content generator (RRCG) first solves a given puzzle using an algorithm. For each step, it generates a corresponding "reasoning statement" that describes the step. Finally, the RRCG joins all the reasoning statements together and returns them as the reasoning content.

In this project, you can reproduce this process using the following command:

```bash
source env.sh && nemotron rrcg generate 00fdc0be
```

Here, `00fdc0be` is the ID of the following puzzle in the training set:

```
In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
01101111 -> 10111111
01001110 -> 00111111
01111101 -> 11111110
10111111 -> 11111111
01101011 -> 10111101
00010101 -> 11011110
11011111 -> 11111111

Now, determine the output for: 10101111
```

The answer to this puzzle is `11111111`. RRCG solves the puzzle correctly and generates the following reasoning content:

```
Treat x as an eight-bit word. Shifts fill vacant positions with 0, rotations wrap around, and NOT flips every bit.

Keep the examples in their given order. The required output vector is: 10111111 | 00111111 | 11111110 | 11111111 | 10111101 | 11011110 | 11111111.

First enumerate every one-word operand (x, every nonzero shift and rotation, then NOT of each of those 29 words):
x: 01101111 | 01001110 | 01111101 | 10111111 | 01101011 | 00010101 | 11011111
SHL1(x): 11011110 | 10011100 | 11111010 | 01111110 | 11010110 | 00101010 | 10111110
SHR1(x): 00110111 | 00100111 | 00111110 | 01011111 | 00110101 | 00001010 | 01101111
ROL1(x): 11011110 | 10011100 | 11111010 | 01111111 | 11010110 | 00101010 | 10111111
ROR1(x): 10110111 | 00100111 | 10111110 | 11011111 | 10110101 | 10001010 | 11101111
SHL2(x): 10111100 | 00111000 | 11110100 | 11111100 | 10101100 | 01010100 | 01111100
SHR2(x): 00011011 | 00010011 | 00011111 | 00101111 | 00011010 | 00000101 | 00110111
ROL2(x): 10111101 | 00111001 | 11110101 | 11111110 | 10101101 | 01010100 | 01111111
ROR2(x): 11011011 | 10010011 | 01011111 | 11101111 | 11011010 | 01000101 | 11110111
SHL3(x): 01111000 | 01110000 | 11101000 | 11111000 | 01011000 | 10101000 | 11111000
SHR3(x): 00001101 | 00001001 | 00001111 | 00010111 | 00001101 | 00000010 | 00011011
ROL3(x): 01111011 | 01110010 | 11101011 | 11111101 | 01011011 | 10101000 | 11111110
ROR3(x): 11101101 | 11001001 | 10101111 | 11110111 | 01101101 | 10100010 | 11111011
SHL4(x): 11110000 | 11100000 | 11010000 | 11110000 | 10110000 | 01010000 | 11110000
SHR4(x): 00000110 | 00000100 | 00000111 | 00001011 | 00000110 | 00000001 | 00001101
ROL4(x): 11110110 | 11100100 | 11010111 | 11111011 | 10110110 | 01010001 | 11111101
ROR4(x): 11110110 | 11100100 | 11010111 | 11111011 | 10110110 | 01010001 | 11111101
SHL5(x): 11100000 | 11000000 | 10100000 | 11100000 | 01100000 | 10100000 | 11100000
SHR5(x): 00000011 | 00000010 | 00000011 | 00000101 | 00000011 | 00000000 | 00000110
ROL5(x): 11101101 | 11001001 | 10101111 | 11110111 | 01101101 | 10100010 | 11111011
ROR5(x): 01111011 | 01110010 | 11101011 | 11111101 | 01011011 | 10101000 | 11111110
SHL6(x): 11000000 | 10000000 | 01000000 | 11000000 | 11000000 | 01000000 | 11000000
SHR6(x): 00000001 | 00000001 | 00000001 | 00000010 | 00000001 | 00000000 | 00000011
ROL6(x): 11011011 | 10010011 | 01011111 | 11101111 | 11011010 | 01000101 | 11110111
ROR6(x): 10111101 | 00111001 | 11110101 | 11111110 | 10101101 | 01010100 | 01111111
SHL7(x): 10000000 | 00000000 | 10000000 | 10000000 | 10000000 | 10000000 | 10000000
SHR7(x): 00000000 | 00000000 | 00000000 | 00000001 | 00000000 | 00000000 | 00000001
ROL7(x): 10110111 | 00100111 | 10111110 | 11011111 | 10110101 | 10001010 | 11101111
ROR7(x): 11011110 | 10011100 | 11111010 | 01111111 | 11010110 | 00101010 | 10111111
NOT(x): 10010000 | 10110001 | 10000010 | 01000000 | 10010100 | 11101010 | 00100000
NOT(SHL1(x)): 00100001 | 01100011 | 00000101 | 10000001 | 00101001 | 11010101 | 01000001
NOT(SHR1(x)): 11001000 | 11011000 | 11000001 | 10100000 | 11001010 | 11110101 | 10010000
NOT(ROL1(x)): 00100001 | 01100011 | 00000101 | 10000000 | 00101001 | 11010101 | 01000000
NOT(ROR1(x)): 01001000 | 11011000 | 01000001 | 00100000 | 01001010 | 01110101 | 00010000
NOT(SHL2(x)): 01000011 | 11000111 | 00001011 | 00000011 | 01010011 | 10101011 | 10000011
NOT(SHR2(x)): 11100100 | 11101100 | 11100000 | 11010000 | 11100101 | 11111010 | 11001000
NOT(ROL2(x)): 01000010 | 11000110 | 00001010 | 00000001 | 01010010 | 10101011 | 10000000
NOT(ROR2(x)): 00100100 | 01101100 | 10100000 | 00010000 | 00100101 | 10111010 | 00001000
NOT(SHL3(x)): 10000111 | 10001111 | 00010111 | 00000111 | 10100111 | 01010111 | 00000111
NOT(SHR3(x)): 11110010 | 11110110 | 11110000 | 11101000 | 11110010 | 11111101 | 11100100
NOT(ROL3(x)): 10000100 | 10001101 | 00010100 | 00000010 | 10100100 | 01010111 | 00000001
NOT(ROR3(x)): 00010010 | 00110110 | 01010000 | 00001000 | 10010010 | 01011101 | 00000100
NOT(SHL4(x)): 00001111 | 00011111 | 00101111 | 00001111 | 01001111 | 10101111 | 00001111
NOT(SHR4(x)): 11111001 | 11111011 | 11111000 | 11110100 | 11111001 | 11111110 | 11110010
NOT(ROL4(x)): 00001001 | 00011011 | 00101000 | 00000100 | 01001001 | 10101110 | 00000010
NOT(ROR4(x)): 00001001 | 00011011 | 00101000 | 00000100 | 01001001 | 10101110 | 00000010
NOT(SHL5(x)): 00011111 | 00111111 | 01011111 | 00011111 | 10011111 | 01011111 | 00011111
NOT(SHR5(x)): 11111100 | 11111101 | 11111100 | 11111010 | 11111100 | 11111111 | 11111001
NOT(ROL5(x)): 00010010 | 00110110 | 01010000 | 00001000 | 10010010 | 01011101 | 00000100
NOT(ROR5(x)): 10000100 | 10001101 | 00010100 | 00000010 | 10100100 | 01010111 | 00000001
NOT(SHL6(x)): 00111111 | 01111111 | 10111111 | 00111111 | 00111111 | 10111111 | 00111111
NOT(SHR6(x)): 11111110 | 11111110 | 11111110 | 11111101 | 11111110 | 11111111 | 11111100
NOT(ROL6(x)): 00100100 | 01101100 | 10100000 | 00010000 | 00100101 | 10111010 | 00001000
NOT(ROR6(x)): 01000010 | 11000110 | 00001010 | 00000001 | 01010010 | 10101011 | 10000000
NOT(SHL7(x)): 01111111 | 11111111 | 01111111 | 01111111 | 01111111 | 01111111 | 01111111
NOT(SHR7(x)): 11111111 | 11111111 | 11111111 | 11111110 | 11111111 | 11111111 | 11111110
NOT(ROL7(x)): 01001000 | 11011000 | 01000001 | 00100000 | 01001010 | 01110101 | 00010000
NOT(ROR7(x)): 00100001 | 01100011 | 00000101 | 10000000 | 00101001 | 11010101 | 01000000

Unary loop: compare all 58 vectors with the target. Exact matches: none.
Binary loop: for each named operation, evaluate all 58 × 58 ordered operand pairs. Nonlisted pairs differ from the target vector.
The loop order is left operand, right operand, then XOR, AND, OR, AND-NOT, OR-NOT, XOR-NOT.
XOR: 3364 pairs tested; no exact match
AND: 3364 pairs tested; no exact match
OR: 3364 pairs tested; OR(ROR1(x), SHL2(x)), OR(SHL2(x), ROR1(x)), OR(SHL2(x), ROL7(x)), OR(ROL7(x), SHL2(x))
AND-NOT: 3364 pairs tested; no exact match
OR-NOT: 3364 pairs tested; OR-NOT(ROR1(x), NOT(SHL2(x))), OR-NOT(SHL2(x), NOT(ROR1(x))), OR-NOT(SHL2(x), NOT(ROL7(x))), OR-NOT(ROL7(x), NOT(SHL2(x)))
XOR-NOT: 3364 pairs tested; no exact match

The first exact formula in that loop order is:
output = OR(ROR1(x), SHL2(x)).

Verification against the supplied examples:
01101111 -> 10111111; expected 10111111: match.
01001110 -> 00111111; expected 00111111: match.
01111101 -> 11111110; expected 11111110: match.
10111111 -> 11111111; expected 11111111: match.
01101011 -> 10111101; expected 10111101: match.
00010101 -> 11011110; expected 11011110: match.
11011111 -> 11111111; expected 11111111: match.

Question input: 10101111.
Evaluate the formula from its innermost terms:
ROR1(x) = 11010111
SHL2(x) = 10111100
OR(ROR1(x), SHL2(x)) = 11111111

Therefore, the answer is \boxed{11111111}.
```

The RRCG can solve all the puzzles in the trianing set correctly. We can verify it with the following unit test:

```bash
source env.sh && python src/nemotron/rrcg/generators.test.py -v
```

> [!IMPORTANT]
>
> A small portion of "Symbol" puzzles do not have an answer because they are underdetermined. For these puzzles, the RRCG will not produce any reasoning content.

#### Supervised Fine-Tuning

With all the generated reasoning content, we can feed them to the Nemotron model and train a LoRA adapter to imporve its reasoning ability to sovlve these types of puzzles.

For each training example, we constructed a supervised record containing:

1. `prompt`: the original puzzle, followed by the instruction to place the final answer inside a `\boxed{}` command.
2. `completion`: the reasoning content generated by the RRCG, followed by the verified final answer in the required format.

Each record had the following form:

```
User:
<original puzzle>

Please put your final answer inside `\\boxed{}`.
For example: `\\boxed{your answer}`

Assistant reasoning:
<RRCG-generated reasoning content>

Assistant final response:
Therefore, the answer is \boxed{<answer>}.
```

Let $x$ denote the formatted prompt and let $y = (y_1, y_2, \ldots, y_T)$ denote the target completion containing both the reasoning trace and final answer. SFT minimizes the negative log-likelihood of the target completion:

$$
-\sum_{t=1}^{T} \log p_{\theta_0, \phi} \left( y_t \mid x, y_{<t} \right)
$$

where $\theta_0$ represents the frozen parameters of the original Nemotron model and $\phi$ represents the trainable LoRA parameters. The prompt tokens were excluded from the loss so that training focused on predicting the assistant's reasoning and answer rather than reproducing the puzzle itself.

We trained only the LoRA parameters and saved the resulting adapter rather than producing a complete copy of the foundation model.

The resulting LoRA adapter achieved a score of only 0.81, which was lower than expected and insufficient to compete for a medal in this challenge. Therefore, we plan to explore other methods, such as on-policy distillation and synthetic data generation, to further improve the model’s performance.

#### The 0.86 Adapter

We were not the only team to come up with the RRCG approach. In fact, many other teams used similar approaches and achieved higher scores, although they referred to the approach by different names. By the midpoint cutoff date (April 9, 2026), some teams had achieved a score of 0.86 and released their adapter (hereafter referred to as the "0.86 adapter") on Kaggle. Competitors could then download the adapter, re-upload it, and achieve the same score.

Because the LoRA adapter produced using our RRCG approach did not achieve a score of 0.86, in the following experiments, we continued improving our adapter based on the 0.86 adapter.

### Self-Distillation

Self-distillation is a type of knowledge distillation in which the teacher and the student are the same model. Some recent research shows that iterative self-distillation can improve a model's reasoning ability [[1], [2]]. However, research also shows that self-distillation can shorten responses while degrading performance on mathematical
reasoning \[[3]\].

We attempted to improve the Nemotron model's accuracy by building on the RRCG adapter. We created an [automation script](experiments/self-distillation/main.py) that implements self-distillation, along with a companion [Bash launcher](experiments/self-distillation/slurm.sh) for running it. The automation script performs the following steps in sequence:

1. Parse the input arguments and build a configuration object.
2. Download the Nemotron model from Kaggle if it doesn't already exist.
3. Download the [RRCG adapter][0.86-adapter] from Kaggle if it doesn't already exist.
4. Load the training dataset into memory.
5. Create a vLLM instance with the RRCG LoRA adapter.
6. Iterate over the prompts in the training set and generate rollouts (LLM completions) for each prompt. More specifically, we append the following instruction to each prompt:
   ```
   Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
   ```
   and have the Nemotron model generate 3 outputs at each of three different temperatures (`1.2`, `1.5`, and `1.8`). The rollout results are saved so that the program can resume after an interruption.
7. Load all rollouts into memory. Print the rollout accuracy broken down by puzzle type and by temperature.
8. Build the self-distillation dataset. Specifically, the script iterates over all per-prompt results. For each per-prompt result, it collects the booleans indicating whether each rollout is correct. If all rollouts are correct or all are incorrect, we skip the prompt without creating a self-distillation record. This is inspired by DAPO \[[4]\], where such cases are considered to carry little preference signal, and the model cannot learn much from them. If there is at least one correct rollout, we check whether the majority rollout is correct. If so, we check whether any other rollouts give the correct answer. If they do, we rule out all the majority rollouts and adopt the correct rollout with the shortest rollout.

Here's some additional explanation for step 8: For puzzles that the Nemotron model can easily solve, it may generate nine rationales that all arrive at the correct answer. In this case, even if we select the best rationales for Nemotron to learn from, the model won't improve much. For puzzles that Nemotron struggles with, it generates nine rationales that all arrive at the wrong answer. In this case, there is nothing for Nemotron to learn from, so we have to discard these puzzles. For the remaining puzzles, the rollouts are a mix of correct and incorrect ones. Among the correct rationales, we pick the shorter ones and add them to the final self-distillation dataset.

During the challenge, we found that the "Symbol" and "Bit" puzzle types require brute-force enumeration, so the model needs to try many combinations and therefore generate many tokens. If the model produces too many filler tokens (e.g., "we can see that," "it is not hard to find"), it may fail to reach the correct answer within the generation limit, which is 7,680 tokens in this challenge. For this reason, we encourage the fine-tuned model to generate shorter completions.

Over the course of experimenting with the script, we found that "majority rollouts" usually don't exist: when the temperature is above 1 and the reasoning content is long, identical completions rarely appear.

Since we didn't have enough compute, we only used this script to generate rollouts for the "Symbol" and "Bit" puzzles.

Unfortunately, we ended up with very few self-distillation records. Increasing the temperature didn't lead to higher accuracy on hard puzzles; instead, it got questions wrong that it originally could have gotten right, decreasing the accuracy. Also, only 13 of the total 3149 puzzles had at least one correct rationale and one incorrect rationale. This indicates that simply increasing the temperature cannot magically make the model generate more correct rationales for hard puzzles. But from this experiment, we collected the IDs of prompts that the fine-tuned Nemotron model still couldn't solve, which were used in the [on-policy distillation experiment](#on-policy-distillation-experiment).

> [!INFO]
> The results of the self-distillation experiment can be found on [Hugging Face][self-distillation-results].

### Synthetic Data Generation

To improve the model’s reasoning ability on `Symbol` puzzles, we use generative AI to expand the existing dataset with high-quality synthetic examples. A key challenge in this process was selecting the most suitable large language model. To make an informed choice, we evaluated Claude, Codex, DeepSeek, and GPT Pro on the same set of SYMBOL puzzles and compared their solution accuracy. GPT Pro achieved the highest accuracy among the models tested and was therefore selected for synthetic data generation. Specifically, GPT-5.5 Pro was used to generate the final dataset. Using the original puzzles as seed data, we instruct the model to generate new puzzles that follow the same style, structure, difficulty, and answer format while avoiding duplicates and ambiguous transformation rules. Each generated sample contains a unique identifier, a self-contained puzzle prompt, a concise final answer, and a complete, externally readable explanation of the reasoning process. The generation prompt also requires the model to validate the consistency between each answer and its reasoning, ensure diversity across transformation-rule types, and produce a correctly formatted CSV containing exactly 1,000 high-quality synthetic puzzles. The exact prompt used for generation is shown below:

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
> - Include a mix of rule types, such as character substitution, deletion, insertion, reversal, position-based selection, adjacent-pair mapping, digit arithmetic, symbol mapping, and multistep transformations.
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

On-policy distillation (OPD) is knowledge distillation in which the training examples come from the student's own rollouts rather than the teacher's. In OPD, there is a student model and a teacher model: the student generates a completion (a rollout) for a given prompt, the teacher grades it, and the student learns from the teacher's feedback (or signal). Some common variants, categorized by the form of the teacher's signal, are as follows:

- **Full token distributions**: The teacher provides a distribution over the vocabulary for every token in the completion sequence. This method applies only when the student and teacher models use the same tokenizer.
- **Corrected trajectories**: The teacher rewrites the student's rollout. The student then trains with cross-entropy on the corrected version.
- **Scalar reward or preference**: The teacher assigns a scalar representing the reward or preference for each completion. Reinforcement learning is then used to fine-tune the student based on that scalar.
- **Sequence-level selection**: The student samples N complete rollouts. The teacher scores them and keeps only those above a threshold. The student is then trained with cross-entropy on the survivors, new rollouts are sampled from the updated student, and the process repeats.

Theoretically speaking, the self-distillation mentioned above is a special case of OPD in which the student and the teacher are the same model. In this experiment, we use ChatGPT 5 (Pro thinking) as the teacher and have it provide corrected trajectories for puzzles that the Nemotron model cannot solve. Here is an example prompt we gave to the ChatGPT 5 (Pro thinking).

> The attached CSV file has two columns. For each row, read the prompt, which is a puzzle, and deduce the solution. You should never write or run scripts to solve the puzzle by brute force. You should reason through it. Generate a CSV file with two additional columns: process and answer. The process column should contain the reasoning used to produce the final answer. If you cannot solve a puzzle, leave both columns blank. Ensure that the reasoning process is logically sound and coherent. It should include the complete reasoning process, including all trials you did. Normally, it should not be shorter than 200 words.

In this experiment, we collected 10 prompts that the Nemotron model (with the RRCG adapter) failed to solve and stored them in a CSV file. We then had ChatGPT 5 read each original completion, keep the correct parts, revise the incorrect parts, and generate a corrected trajectory, which was then used to fine-tune the Nemotron model. The correct trajectories are saved in [data_public/processed/correct_trajectories.csv](data_public/processed/correct_trajectories.csv).

Unfortunately, although the trajectories provided by the teacher model arrived at correct answers, they didn't improve the accuracy of the Nemotron model. This may be due to an insufficient number of training samples.

## CLI Tool and Tests in This Project

To use the CLI tool or run the test cases in this project, you must first install Python 3.12.11 or later and Poetry 2.4.1 or later. The project supports only Linux and macOS.

First, run the following command from the project’s root directory:

```bash
# Install dependencies
poetry install --no-root

# Set up the environment; must be executed in every shell session
source env.sh
```

Then, run the following command to view all available commands:

```bash
nemotron --help
```

The following command runs the test case for RRCG:

```bash
source env.sh && python src/nemotron/rrcg/generators.test.py -v
```

## What We Learned

We’ve learned a lot from this project. Almost all of the highest-scoring teams combined synthetic data generation, rule-based solvers with deterministic reasoning generation (similar to our RRCG), and SFT. The main difference was that they generated more synthetic data, including puzzles of similar types, and built better reasoning generators.

While running our self-distillation script on a GPU server managed by Slurm, we found that it was important to use compatible versions of Python, CUDA, vLLM, and other environment dependencies. If the versions were incompatible or the environment was misconfigured, the script would not run. Because none of the three of us had much experience with AI infrastructure, we spent a significant amount of time debugging issues on the cloud GPU server.

To improve a mid-sized large language model’s ability to reason through specific types of problems, on-policy distillation, especially self-distillation, may not be the best approach. Our three subsequent experiments did not further improve the adapter’s score of 0.86. For problems that can be solved step by step, the aforementioned RRCG may still be the most effective approach currently available. However, improving the quality of the generated reasoning content remains a highly complex challenge.

<!-- TO DOCUMENT WRITERS: Please keep these references at the end of the document. -->

[nemotron-model]: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
[nemotron-challenge-page]: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
[vllm]: https://docs.vllm.ai/en/stable/
[1]: https://arxiv.org/abs/2601.18734
[2]: https://arxiv.org/abs/2605.12400
[3]: https://arxiv.org/abs/2605.28791
[0.86-adapter]: https://www.kaggle.com/datasets/leegongman/0-86-adapter
[4]: https://arxiv.org/abs/2503.14476
[self-distillation-results]: https://huggingface.co/datasets/FindAJobJMR/rollout-results/
[mamba-2]: https://arxiv.org/abs/2405.21060
[MoE]: https://arxiv.org/abs/2401.06066
[self-attention]: https://arxiv.org/abs/1706.03762
