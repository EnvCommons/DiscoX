# DiscoX

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/DiscoX)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/ByteDance-Seed/DiscoX)

## Description

DiscoX is an OpenReward environment for evaluating professional translation quality between English and Chinese. Agents are given a source text and must produce a translation that is graded against hidden expert-designed rubrics using LLM-based evaluation. The environment covers 200 bidirectional (EN-to-ZH and ZH-to-EN) translation tasks spanning academic papers, humanities, social sciences, and technical documents.

## Capabilities

- Professional-grade translation between English and Chinese
- Domain-specific terminology usage across academic and technical fields
- Cultural and contextual adaptation in translation
- Register and tone matching for varied document types

## Compute Requirements

DiscoX is a single-turn environment with no special compute requirements. Each task consists of one prompt and one tool call. LLM-based grading is performed server-side via the OpenAI API.

## License

[ORLv1](https://openreward.ai/orlv1.md).

## Tasks

There are 200 tasks in a single `train` split. Each task presents the agent with a source text to translate, along with domain information and translation instructions. The translation direction (EN-to-ZH or ZH-to-EN) is automatically detected based on the character composition of the source text (greater than 30% Chinese characters indicates ZH-to-EN).

Tasks span the following domains:

- Academic papers
- Humanities
- Social sciences
- Technical documents

Each task specification includes:

- `id`: Unique task identifier (e.g., `discox_000`)
- `direction`: Translation direction (`en_to_zh` or `zh_to_en`)
- `primary_domain`: Main content domain
- `secondary_domain`: Specific subdomain

## Reward Structure

DiscoX uses continuous rewards in the range 0.0 to 1.0. Each translation is graded against multiple expert-designed rubrics that are hidden from the agent. Rubrics are extracted from the dataset's `reference_list` field, which contains structured evaluation criteria (typically in the Chinese "kaodian" format).

Each rubric criterion is evaluated independently by gpt-5-mini, producing a score from 0.0 to 1.0. The final reward is the sum of all rubric scores divided by the total possible points:

$$\text{reward} = \frac{\sum_{i=1}^{n} \text{score}_i}{\sum_{i=1}^{n} \text{max\_points}_i}$$

All rubrics are weighted equally at 1.0 point each. Grading covers terminology accuracy, semantic fidelity, cultural appropriateness, tone/register, and fluency.

## Data

The dataset is sourced from [ByteDance-Seed/DiscoX](https://huggingface.co/datasets/ByteDance-Seed/DiscoX) on HuggingFace and stored as `discox.parquet`. It contains 200 professional translation tasks with the following columns:

- `prompt`: Translation instructions
- `ori_text`: Source text to translate
- `reference_list`: Expert rubrics used for grading (hidden from the agent)
- `Primary_Domain`: Main content category
- `Secondary_Domain`: Specific subdomain
- `prompt_id`: Original task identifier

## Tools

DiscoX provides a single tool:

| Tool | Description |
|------|-------------|
| `submit_translation` | Submit a complete translation for evaluation. Takes a `translation` string parameter. Returns rubric-by-rubric feedback, overall score, and normalized reward (0.0-1.0). This call ends the episode. |

## Time Horizon

DiscoX is a single-turn environment. The agent receives a translation prompt and submits one translation via the `submit_translation` tool, which ends the episode. Each task requires exactly one tool call.

## Environment Difficulty

Translation quality is evaluated against expert-designed rubrics covering terminology, semantics, cultural nuance, tone, and fluency. The tasks span specialized academic and technical domains, requiring domain-specific knowledge and professional translation skill.

## Other Environment Requirements

This environment requires an OpenAI API key for LLM-based translation grading.

## Safety

DiscoX evaluates translation quality using professional source texts from academic and technical domains. The environment does not present direct safety risks. Agents interact only with provided text and a grading API. Source texts are drawn from published academic papers and professional documents.

## Citations

```bibtex
@misc{discox2024,
  title     = {DiscoX: A Professional Translation Benchmark with Expert Rubrics},
  author    = {ByteDance-Seed},
  year      = {2024},
  url       = {https://huggingface.co/datasets/ByteDance-Seed/DiscoX}
}
```
