# DiscoX - Professional Translation Evaluation Environment

DiscoX is an OpenReward environment for evaluating translation quality between English and Chinese using expert-designed quality rubrics and LLM-based grading.

## Overview

- **Dataset**: [ByteDance-Seed/DiscoX](https://huggingface.co/datasets/ByteDance-Seed/DiscoX)
- **Tasks**: 200 professional translation tasks (bidirectional EN↔ZH)
- **Domains**: Academic papers, humanities, social sciences, and more
- **Evaluation**: Multi-rubric LLM grading using GPT-5-mini
- **Rubrics**: Expert-designed criteria covering terminology, tone, cultural nuance, and fluency

## Features

- **Bilingual Support**: Automatically detects translation direction (EN→ZH or ZH→EN)
- **Multi-Rubric Evaluation**: Each translation graded against multiple expert rubrics
- **Parallel Grading**: Efficient concurrent evaluation of all rubrics
- **Detailed Feedback**: Comprehensive criterion-by-criterion analysis
- **Normalized Scoring**: Final reward scaled to 0.0-1.0 range

## Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/EnvCommons/discox.git
cd discox

# Install dependencies
pip install -r requirements.txt

# Download dataset
python -c "from datasets import load_dataset; load_dataset('ByteDance-Seed/DiscoX', split='train').to_parquet('discox.parquet')"

# Run server
python server.py
```

### Docker

```bash
# Build image
docker build -t discox:latest .

# Run container
docker run -p 8080:8080 discox:latest
```

## Usage

### Testing with Agent

```bash
export OPENAI_API_KEY=sk-...
python test_agent.py
```

### Using OpenReward SDK

```python
from openreward import AsyncOpenReward

or_client = AsyncOpenReward()
environment = or_client.environments.get(name="EnvCommons/discox")

tasks = await environment.list_tasks(split="train")
tools = await environment.list_tools(format="openai")

# Create session with secrets
async with environment.session(
    task=tasks[0],
    secrets={"openai_api_key": "sk-..."}
) as session:
    prompt = await session.get_prompt()
    # ... agent interaction ...
```

## Environment Structure

### File Organization

```
discox/
├── discox.py          # Main environment class with multi-rubric grading
├── server.py          # Server wrapper
├── test_agent.py      # Local testing script
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container build
└── README.md          # This file
```

### Task Specification

Each task contains:
- `id`: Unique task identifier (e.g., "discox_001")
- `direction`: Translation direction ("en_to_zh" or "zh_to_en")
- `primary_domain`: Main content domain
- `secondary_domain`: Specific subdomain

### Tool: submit_translation

Submit your translation for evaluation:

```python
{
    "translation": "Your complete translation of the source text"
}
```

Returns:
- Multi-rubric feedback with criterion-by-criterion analysis
- Overall score and normalized reward (0.0-1.0)
- Detailed grader analysis for each rubric

## Evaluation Methodology

### Rubric Extraction

Expert rubrics are automatically extracted from the `reference_list` field in Chinese:
- Pattern 1: "考点N" format (most common)
- Pattern 2: Numbered lists (1. 2. etc.)
- Fallback: Holistic quality assessment

### Grading Process

1. **Parallel Evaluation**: All rubrics graded concurrently using GPT-5-mini
2. **Score Extraction**: Multi-stage parsing with fallback handling
3. **Aggregation**: Scores summed and normalized to 0.0-1.0 range
4. **Feedback Generation**: Detailed criterion-by-criterion results

### Grading Criteria

Each rubric evaluates:
- Terminology accuracy
- Semantic fidelity
- Cultural appropriateness
- Tone and register
- Overall fluency

### Score Scale

- **1.0**: Excellent (fully meets criterion)
- **0.7-0.9**: Good (mostly meets with minor issues)
- **0.4-0.6**: Fair (partially meets)
- **0.1-0.3**: Poor (significant issues)
- **0.0**: Unacceptable (fails criterion)

## Dataset Information

- **Size**: 200 tasks, ~1.98 MB (parquet format)
- **Splits**: train (all 200 tasks)
- **Languages**: English ↔ Chinese (Simplified)
- **Domains**: Academic papers, humanities, social sciences, technical documents

### Data Columns

- `prompt`: Translation instructions (Chinese)
- `ori_text`: Source text to translate
- `reference_list`: Expert rubrics (Chinese, 考点 format)
- `Primary_Domain`: Main content category
- `Secondary_Domain`: Specific subdomain
- `prompt_id`: Original task identifier

## Technical Details

### Architecture

- **Pattern**: Single-turn environment with LLM-based grading
- **Base Class**: `openreward.environments.Environment`
- **Grading Model**: GPT-5-mini (no temperature parameter)
- **Concurrency**: Async parallel rubric evaluation

### API Key Requirements

This environment requires an OpenAI API key for grading:

```python
async with environment.session(
    task=task,
    secrets={"openai_api_key": "sk-..."}  # Required
) as session:
    ...
```

### Error Handling

- Empty translations: Immediate validation error (reward=0.0)
- API failures: Conservative fallback (score=0.0 with error message)
- Score parsing failures: Multi-stage extraction with 0.0 default
- Missing rubrics: Fallback to holistic quality assessment

## Example Output

```
# Translation Evaluation Results

**Task**: discox_001
**Domain**: 学术论文 / 人文科学
**Direction**: EN→ZH

**Overall Score**: 3.25 / 4.00
**Normalized Reward**: 0.812

---

## Rubric-by-Rubric Feedback

### Rubric 1: 0.85 / 1.00
**Criterion**: 考点1："Sensibility"推荐译为"感性"

**Grader Analysis**:
Analysis: The translation correctly uses "感性" for "Sensibility",
demonstrating good terminology choice. Minor contextual refinement
could improve clarity.
Score: 0.85

---

### Rubric 2: 0.90 / 1.00
...
```

## Performance Considerations

- **Grading Time**: ~2-4 seconds per task (depends on rubric count)
- **Concurrency**: All rubrics evaluated in parallel
- **Rate Limits**: Respect OpenAI API rate limits
- **Cost**: ~$0.01-0.02 per task evaluation (GPT-5-mini pricing)

## Citation

If you use DiscoX in your research, please cite:

```bibtex
@misc{discox2024,
  title={DiscoX: A Professional Translation Benchmark with Expert Rubrics},
  author={ByteDance-Seed},
  year={2024},
  howpublished={\url{https://huggingface.co/datasets/ByteDance-Seed/DiscoX}}
}
```

## License

This environment implementation is released under the MIT License. The DiscoX dataset follows the original dataset's license terms.

## Contributing

Contributions welcome! Please submit issues or pull requests on GitHub.

## Support

For questions or issues:
- GitHub Issues: https://github.com/EnvCommons/discox/issues
- OpenReward Docs: https://docs.openreward.org/
