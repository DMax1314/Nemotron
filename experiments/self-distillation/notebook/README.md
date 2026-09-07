### 📌 Overview

This notebook is built upon the **NVIDIA-Nemotron-3-Nano-30B-A3B-BF16** base model, leveraging a hybrid corpus of pre-training data and verified **On-Policy Data (OPD)** for **Self-Distillation** and **Continual Learning**. By initializing from a high-scoring pre-trained adapter and integrating aggressive VRAM optimizations, this pipeline further enhances complex reasoning capabilities within minimal training steps.

---

### 📊 Dataset & Loss Masking

* **Base Corpus:** 7,830 pre-tokenized foundational training examples.
* **On-Policy Data (OPD):** 8,416 verified self-distilled samples featuring complete Chain-of-Thought (CoT) reasoning paths.
* **Total Dataset:** 16,246 combined training examples.
* **Targeted Loss Masking:** Prompt tokens are masked with a `0.0` weight, ensuring cross-entropy loss is computed exclusively on generated CoT and final answer tokens (`1.0` weight).

---

### 🛠️ Core Technical Optimizations

* **Unsloth & Cut Cross Entropy (CCE):** Accelerated using the Unsloth framework and integrated with CCE to dramatically reduce VRAM footprint during training.
* **Mamba Fast Path Patching:** Manually enabled and patched the fast-path execution kernel for Mamba layers to boost execution throughput.
* **FP32 LoRA Precision:** Explicitly cast LoRA trainable parameters to FP32 precision to guarantee numerical stability during gradient updates.