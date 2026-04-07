# ChomskyBench

ChomskyBench is a benchmark designed to systematically evaluate Large Language Models (LLMs) through the lens of the **Chomsky Hierarchy**. This hierarchy provides a canonical, tiered framework for classifying formal languages based on their computational complexity, making it ideal for assessing the formal reasoning capabilities of LLMs across different levels of language complexity.

## Overview

The Chomsky Hierarchy consists of four main levels of formal languages, from least to most computationally complex:

1. **Regular Languages (Type-3)** - Recognized by finite automata
2. **Context-Free Languages (Type-2)** - Recognized by pushdown automata  
3. **Context-Sensitive Languages (Type-1)** - Recognized by linear bounded automata
4. **Recursively Enumerable Languages (Type-0)** - Recognized by Turing machines

ChomskyBench contains **115 carefully crafted problems** distributed across these categories:

| Level | Count | Description |
|-------|-------|-------------|
| re | 11 | Regular language problems (Turing machine simulations) |
| cfg | 16 | Context-free grammar and pushdown automaton problems |
| dcfg | 27 | Deterministic context-free language problems |
| csg | 38 | Context-sensitive grammar and linear bounded automaton problems |
| regular | 23 | Regular language problems (finite automata, regular expressions) |

**Total: 115 problems**

## Problem Types

Each problem in ChomskyBench includes:

- **Question**: Formal language theory problem requiring structured reasoning
- **Correct Answer**: Ground truth answer in standardized format
- **Verification Code**: Executable Python code to programmatically verify correctness
- **ID**: Unique identifier indicating problem category and type

Example problem types:
- **TM Simulation**: Multi-tape Turing machine simulation with step-by-step execution
- **LBA Computation**: Linear bounded automaton state transitions and acceptance
- **Grammar Derivations**: Context-sensitive and context-free grammar productions
- **Automaton Construction**: Building finite automata for specific languages
- **Parsing Problems**: Deterministic parsing and language membership testing

## Quick Start

### Prerequisites

```bash
pip install openai tqdm numpy pandas
```

### Configuration

1. Add your OpenRouter API key to `api_keys_openrouter.txt`:
   ```
   sk-or-v1-...
   ```

2. Configure models to evaluate in `inference.sh`:
   ```bash
   full_models=(
       deepseek/deepseek-chat-v3.1
   )
   
   short_models=(
       deepseek-v3.1-nonthinking
   )
   ```

### Running Inference

Execute the benchmark inference:

```bash
bash inference.sh
```

This will:
- Create a `results/` directory for model outputs
- Generate log files in `logs/` directory
- Produce completion files as `{short_model}_completion.jsonl`

### Evaluation

#### Standard Evaluation (Greedy Decoding)

Evaluate model accuracy:

```bash
python evaluate.py --input_dir results
```

Outputs per-category metrics:
- **Accuracy**: Percentage of completely correct answers
- **Pass Ratio**: Average correctness across all answer components
- Summary statistics for all models

#### Sampling Evaluation (Pass@k)

For sampling-based evaluation with multiple attempts:

```bash
python evaluate_sampling.py --input_dir results
```

## Understanding the Results

Evaluation results are organized by grammatical complexity order:

| Order | Category | Description |
|-------|----------|-------------|
| 0 | re | Regular language problems (re category) |
| 1 | csg | Context-sensitive grammar problems |
| 2 | cfg | Context-free grammar problems |
| 3 | dcfg | Deterministic context-free problems |
| 4 | regular | Regular language problems (regular category) |
| -1 | avg | Average across all categories |

The evaluation provides:
- **Per-category accuracy**: Correct answers / Total problems
- **Pass ratio**: Average partial correctness (e.g., if answer has 4 components, getting 2 right = 0.5)
- **Detailed logs**: Per-problem breakdowns saved to `{model}_result.jsonl`

### Custom Inference Parameters

Modify `gpt_inference_stream.py` for different decoding strategies:

```python
parser.add_argument('--mode', type=str, required=True)      # "greedy" or "sampling"
parser.add_argument('--max_tokens', type=int, default=16000)  # Maximum response length
parser.add_argument('--T', type=float, default=0)             # Temperature (0=greedy)
parser.add_argument('--top_p', type=float, default=1)         # Nucleus sampling
parser.add_argument('--N', type=int, default=1)               # Number of samples
```


## Dataset Format

Problems are stored in `ChomskyBench.jsonl` (JSON Lines format):

```json
{
  "id": "re_tm_simulation_1",
  "question": "Multi-tape TM M with 2 tapes, states Q = {...}\n\nSimulate M on input...",
  "correct_answer": "(2, 'babab_', 'aa_', 3)",
  "verification_code": "def verify_answer(answer): ..."
}
```

## License

This project is released under the MIT License.
