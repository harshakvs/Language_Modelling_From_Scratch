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

