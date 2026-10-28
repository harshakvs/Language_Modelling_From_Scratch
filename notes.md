Basics

Tokenization, Model Architecture and Training
Resource Accounting:Memory and Compute Characteristics of a model
Model parameters must be moved form memory(HBM) to the comput (SMs)
Understand whether we are compute bound or memory bound

Kernels

Kernel is a function that runs on GPU
when using pytorch, each primitive operation launches a standard kernel
can write custom kernels to make GPU's go brrr
Write kernels in CUDA/Triton/CUTLASS

Parallelism

What if we have 1024 GPUS?
Shard memory (Parameters, activations,gradients and optimizer states) across GPUs
How to split computation: data,tensor,sequence, parallelism

Inference

Goal: Use the models
Infernce is also needed for reinforcement learning, tet time compute, evaluation
Two Phases: Prefill and Decode
Prefill:Tokens are given adn can process all at once(Compute Bound)
Decode: Need to generate one tooken at a time(memory-bound)
Methods to speed up decoding: Fused kernels, Continuous Batching, Speculative Decoding or use a cheaper model

Scaling

How do we scale up?
If you have 1e25 FLOPs of compute, what hyperparameters would you use to train a good model?
Scaling Recipe? FLOPs to hyperparameters
Run expermients to compute the loss at various smaller scales
For a scaling law to predict the loss of the scaling recipe at the target scale
Scaling laws dont happen automatically
Given a FLOP's budget (C = 6ND) use a bigger model or tain more tokens (D)?
TLDR D=20N
70B parameter model should be trained on ~1.4T tokens
Doesnt take into account inference costs

Data

Question: What capabilties do we want the model to have?
Multilingual? Good at conversation? Agentic coding capabilities?

Evaluation

What is the purpose of evaluation?
Internal: Guide model development
External: Measure absolute quality of a real use case
Advanced use cases: GPQA,HLE, SWE-Bench, Terminal-Bench
LM's are general purpose, require a diverse set of evalutations.

Data curation

Fairness and might have to license data
Raw data is HTML, PDF, Directories and require processing

Data processing

Transformation: convert HTML/PDF to test
Filtering: Keep high quality data, remove hamful content
Deduplciation: Save compute, avoid memorization; use Bloom filters or MinHash
Data mixing
Rewriting/Synthetic Data

Types of data: 
Pretraining data:Large and diverse
Mid-training data: high quality, including long-context
Post-training data: supervised fine-tuning (conversations, agentic traces with tool calling)

Alignment:
Weak Supervision?
Basic Template:
1) generate response from the model
2) Score responses with a human, verifier or LM Judge
3) Update h model to prefer better responses

Algorithms:
- Proximal Policy Optimization(PPO)
- Direct Policy Optimization(DPO)
- Group Relative Preference Optimzation (GRPO)

RL algortihms are unstable and hard to tune
Constantly trading off systems efficienct and on policyness.

Its all about efficiency:
Resource: Data + Hardware ( Compute, Memory , Communication Bandwidth)
How do we train the best model given a fixed set of resources?

Today we are compute constrained so desgin decisions will reflect squeezing the most out of given Systems, Tokenization, Architecture of the Model, Data filtering, Scaling laws


Tokenization

Intro to tokenization

Unicode strings are encoded into tokens and decodes token back to strings
Tokenizer is a class that implements the encode and decode methods

A word and its preceding space are part of the same token
A word at the begineeing adn in the middle are represented differently

simple tokenizer like gpt5_tokenizer
gives you indices and you can encode and decode your string using the class

Compression Ratio: Number of bytes per token

Larger the compression ratio , the shorter the sequence( good since attention is quadratic in seq length)
One could increase compression ratio by incresing vocab size( number of possible token vlaue increases) leading to sparsity

Simple tokenizer like Character Tokenizer is just using ord
as the string are in unicode which are reporesrested by numbers 
use ord and char as encoder decoder
Now there are 150K unicode characters, this is very large vocab and the compression ratio will be lower for the same sentence

Or we can use byte tokenizer
convert to sequence of bytes
vocab size? 256
Byte sequence are long and compression ratio is bad and limited vocab size

Word tokenizer

Basic NLP, Map each chunk to an integer
compression ratio is good but vocab size is huge
many words are rare and the model wont learn much about them
this doesnt prvide a fixed vocab size
new words we havent seen during training get a special UNK token which is ugly and can mess wiht calculations

BPE Tokenizer (Byte Pair Tokenizer)
Byte Pair Encoding

BPE algorithm for data compression.
Basic idea: Train the tokenizer on raw text to construct a vocabulary tailored to the data.
Intuition: Common sequences of bytes are represented by a single token, rare sequences are represented by many tokens

Sketch: start with each byte as a token, and successively merge the most common pair of adjacent tokens.

Motivation Question:
How long would it take to train a 70B parameter model on 15T tokens on 1024 H100s?
total_flops = 6*number of parameters*number of tokens
Each GPU has its own spec sheet and its speed
and mfu

Question: How long would it take to train a 70B parameter model on 15T tokens on 1024 H100s?
total_flops = 6 * 70e9 * 15e12  
h100_flop_per_sec = 1979e12 / 2
mfu = 0.5
flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24  
days = total_flops / flops_per_day  

Question: What's the largest model that can you can train on 8 H100s using AdamW?
h100_bytes = 80e9  
bytes_per_parameter = 2 + 2 + (4 + 4)  # parameters (2), gradients (2), optimizer state (4 + 4) 
num_parameters = (h100_bytes * 8) / bytes_per_parameter  
Caveat: activations are not accounted for (depends on batch size and sequence length), so this is an upper bound.
This is a rough back-of-the-envelope calculation.

Takeaway is resource accounting and get a sense of how resources are spent

Tensors are the basic building block for storing everything:
data, parameters, gradients, optimizer state, activations

Each tensor has a rank, which is the number of dimensions.
x = torch.zeros(4)        # rank 1 tensor (vector) 
x = torch.zeros(4, 8)     # rank 2 tensor (matrix) 
x = torch.zeros(4, 8, 2)  # rank 3 tensor 
In Transformers, will see tensors of rank 4:
B = 32   # Batch size
S = 16   # Sequence length
H = 16   # Number of heads
D = 64   # Hidden dimension per head
x = torch.zeros(B, S, H, D)

A concrete everyday example is video. A video is (frames, height, width, color):

one pixel = 3 numbers (RGB) → vector
one row of pixels → matrix
one frame (image) → rank 3
many frames → rank 4


Tensors Memory 
Almost everything are stored as floating points.

float32  has 32 bits
fp32 or single precision is the default
Traditionally, in scietific computing float32 is the baseline; you can use double precision in some cases.

By default tesnor is float32
Float is 4 bytes 

Eg x = torch.zeros(4,8)
is 4*8*4 is 128 bytes

One matrix in the feedforward layer of GPT-3:
assert get_memory_usage(torch.empty(12288 * 4, 12288)) == 2304 * 1024 * 1024  # 2.3 GB 

Since we are interested in efficiency, we try to reduce storage

Now we can do fp16, 16 bit floating point, less size and less processing time

fp16-float16

we cant really represent huge or small numbers
beacuse exponent is just 5 when comaprked to 8 in fp32

However, the dynamic range (especially for small numbers) isn't great.
x = torch.tensor([1e-8], dtype=torch.float16)  
assert x == 0  # Underflow!
If this happens when you train, you can get instability.

bf16 was developed to address this issue

bfloat16 uses the same memory as float 16 but same dynamic range as flaot32

basically moving and increasing exponent bits from 5 to 8
The only catch is that the resolution is worse, but this matters less for deep learning.
x = torch.tensor([1e-8], dtype=torch.bfloat16)  
assert x != 0  # No underflow!

Mixed precision is a strategy people take.
Training with fp32 works, but requires lots of memory.
Training with fp16 and even bf16 is risky, and you can get instability.
Solution is mixed precision training 
Use bf16 for parameters, activations, and gradients
Use fp32 for optimizer states
Pytorch has an automatic mixed precision (AMP) library.

It helps to think of floats as **scientific notation in binary**:

```
  −1.0110 × 2^5
  ↑   ↑        ↑
 sign mantissa exponent
```

Every float has a fixed budget of bits, split into three parts:
- **Sign** (1 bit): + or −.
- **Exponent**: how far the number can slide up or down, i.e. how big or how tiny it can be. More exponent bits gives more **dynamic range**.
- **Mantissa**: how many significant digits you keep. More mantissa bits gives more **resolution**, meaning finer steps between neighbouring numbers.

So "moving bits around" means: with the same total bits, do you spend them on reach (exponent) or on detail (mantissa)?

### Decimal analogy
Say you're only allowed 4 digits in total:
- 3 for digits and 1 for the power gives `1.23 × 10^5`. That's precise, but it can only go up to about 10^9.
- 2 for digits and 2 for the power gives `1.2 × 10^15`. That's coarse, but it can reach 10^99.

Same budget, different trade-off.

### The formats

| Format | Sign / Exp / Mantissa | Range (max) | Smallest step near 1.0 | Note |
|---|---|---|---|---|
| fp32 | 1 / 8 / 23 | ~3×10³⁸ | ~0.0000001 | the reference |
| fp16 | 1 / **5** / 10 | **65,504** | ~0.001 | good detail, tiny range |
| bf16 | 1 / **8** / 7 | ~3×10³⁸ (same as fp32) | ~0.008 | fp32's range, coarse detail |
| fp8 E4M3 | 1 / 4 / 3 | 448 | 0.125 | more detail |
| fp8 E5M2 | 1 / 5 / 2 | 57,344 | 0.25 | more range |

**bf16 is fp32 with its mantissa chopped off.** It keeps the 8 exponent bits. That's why people say it "has the same range as fp32."

### Why it matters in training
- **Not enough range causes overflow and underflow.** In fp16, anything above 65,504 becomes `inf`, and very small gradients (say 1e-8) round to **0**. That's why fp16 training needs *loss scaling*: you multiply the loss up so the gradients stay above the floor. bf16 mostly doesn't need this, which is why everyone switched to it.
- **Not enough resolution means small updates get lost.** In bf16, `1.0 + 0.001 = 1.0`, because the next number after 1.0 is about 1.008. If you add a small gradient step to a large weight, nothing happens. That's why people keep a **master copy of the weights in fp32** and do the matmuls in bf16 or fp8.
- **fp8 comes in two flavours** because both bits are precious at that size. E4M3 is often used for weights and activations (they need detail). E5M2 is used for gradients (they swing wildly in size, so they need range).

In one line: **exponent bits mean how big or how small a number can be, and mantissa bits mean how finely it can be measured.**

fp8 works on H100'S and it supports two varients

next is fp4 devleoped by NVIDIA in 2025
only 4 bit per value
Nemotron 3 super was trained in NVFP4

People generally train models in fp32,fp16 and bf16 and quantize it in 1 bit
so i guess that might be the reason why its fails on clearer tasks or deeper udnerstadning kinda tasks

Yea also by default tesnors are created on the CPU , we can move them to GPU

![alt text](image.png)


##  VRAM vs DRAM
DRAM is the **type of memory chip**. VRAM is **where the memory sits**, which is on the GPU. VRAM is actually built from DRAM too. They're not opposites.

| | System RAM | GPU VRAM |
|---|---|---|
| Where | on the motherboard, next to the CPU | on the GPU card or package |
| Kind | DDR5 (DRAM) | GDDR6/7 or **HBM** (also DRAM, much wider and faster) |
| Bandwidth | ~50–100 GB/s | ~1 TB/s (GDDR) to **~3+ TB/s (HBM, H100)** |
| Size | 16–512 GB | 24 GB (gaming) to 80–192 GB (H100/B200) |

Why it matters for LLMs:
- **The model must fit in VRAM.** Weights, gradients, optimizer state and activations all have to live there while training. That's why the precision question matters: bf16 weights take half the memory of fp32.
- **Moving data between the two is slow.** CPU RAM to GPU goes over PCIe at ~32–64 GB/s, which is far slower than VRAM's own bandwidth.
- You'll also hear about **SRAM**, the tiny but extremely fast memory on the GPU chip itself. It's tens of MB, at ~20+ TB/s. The memory hierarchy goes **SRAM (tiny, fastest) → HBM/VRAM → CPU DRAM (big, slow)**. Much of GPU optimization, like FlashAttention, is about keeping data in SRAM and making fewer trips to HBM.

