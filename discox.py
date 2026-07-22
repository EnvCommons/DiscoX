from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, List

import openai
import pandas as pd
from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, terminal, tool
from pydantic import BaseModel, Field

# --- Data loading (AIME pattern): load once at module import time ---
# Path checking for local dev and Docker
if Path("discox.parquet").exists():
    DATA_PATH = Path("discox.parquet")
elif Path("/orwd_data/discox.parquet").exists():
    DATA_PATH = Path("/orwd_data/discox.parquet")
else:
    raise FileNotFoundError(
        "DiscoX dataset not found. Please ensure discox.parquet is available "
        "in the current directory or at /orwd_data/discox.parquet"
    )

# Load at module import
TASKS_DF = pd.read_parquet(DATA_PATH)
TASKS: List[dict[str, Any]] = TASKS_DF.to_dict(orient="records")

# Add unique IDs and detect direction
for idx, task in enumerate(TASKS):
    task["id"] = f"discox_{idx:03d}"

    # Detect EN→ZH or ZH→EN based on source text
    chinese_chars = len([c for c in task["ori_text"] if '\u4e00' <= c <= '\u9fff'])
    total_chars = len([c for c in task["ori_text"] if not c.isspace()])

    if total_chars > 0:
        task["direction"] = "zh_to_en" if chinese_chars / total_chars > 0.3 else "en_to_zh"
    else:
        task["direction"] = "unknown"


# --- Grader prompt template ---
GRADER_TEMPLATE = """You are an expert translation evaluator specializing in professional Chinese-English translation.

**Translation Direction**: {direction}
**Domain**: {domain}

**Source Text**:
{source_text}

**Translation Submitted**:
{translation}

**Evaluation Criterion**:
{criterion}

**Grading Instructions**:
1. Evaluate how well the translation meets this specific criterion
2. Consider: terminology accuracy, semantic fidelity, cultural appropriateness, tone/register, fluency
3. Score on scale 0.0 to 1.0:
   - 1.0 = Excellent (fully meets criterion)
   - 0.7-0.9 = Good (mostly meets with minor issues)
   - 0.4-0.6 = Fair (partially meets)
   - 0.1-0.3 = Poor (significant issues)
   - 0.0 = Unacceptable (fails criterion)

**Output Format**:
Analysis: [2-3 sentence explanation]
Score: [Decimal 0.0-1.0]
"""


# --- Pydantic Models ---
class TaskSpec(BaseModel):
    id: str


class SubmitTranslationInput(BaseModel):
    translation: str = Field(..., description="Your complete translation of the source text")


# --- Environment Class ---
class DiscoX(Environment):
    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split != "train":
            return []

        # Return lightweight task specs
        return [
            {
                "id": t["id"],
                "direction": t["direction"],
                "primary_domain": t["Primary_Domain"],
                "secondary_domain": t["Secondary_Domain"],
            }
            for t in TASKS
        ]

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)

        # CRITICAL: Validate OpenAI API key
        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError(
                "OpenAI API key required for LLM grading. "
                "Please provide it via the secrets parameter."
            )

        self.client = openai.AsyncClient(api_key=api_key)

        # Load task
        task_id = str(task_spec["id"])
        self.task = next((t for t in TASKS if t["id"] == task_id), None)

        if self.task is None:
            raise ValueError(f"Task {task_id} not found in dataset")

        # Parse rubrics from reference_list
        self.rubrics = self._parse_rubrics(self.task["reference_list"])

    def _parse_rubrics(self, reference_list: str) -> List[dict]:
        """Parse reference_list into structured rubrics with equal weighting (1.0 per rubric)"""

        if not reference_list or not reference_list.strip():
            # Fallback to single holistic rubric
            return [{"criterion": "Overall translation quality", "points": 1.0}]

        rubrics = []

        # Pattern 1: 考点N format (most common in DiscoX)
        kaodian_matches = re.findall(
            r"考点(\d+)[:：](.+?)(?=考点\d+[:：]|$)",
            reference_list,
            re.DOTALL
        )

        if kaodian_matches:
            for idx, content in kaodian_matches:
                rubrics.append({
                    "criterion": f"考点{idx}: {content.strip()}",
                    "points": 1.0
                })
        else:
            # Pattern 2: Numbered list (1. 2. etc.)
            numbered_matches = re.findall(
                r"(\d+)\.\s*(.+?)(?=\d+\.|$)",
                reference_list,
                re.DOTALL
            )

            if numbered_matches:
                for num, content in numbered_matches:
                    rubrics.append({
                        "criterion": content.strip(),
                        "points": 1.0
                    })
            else:
                # Fallback: treat entire text as single rubric
                rubrics.append({
                    "criterion": reference_list.strip(),
                    "points": 1.0
                })

        return rubrics

    async def get_prompt(self) -> List[TextBlock]:
        """Return translation task (NO RUBRICS - hidden from agent)"""

        direction_display = "EN→ZH" if self.task["direction"] == "en_to_zh" else "ZH→EN"

        prompt = f"""# Translation Task ({direction_display})

**Domain**: {self.task['Primary_Domain']} / {self.task['Secondary_Domain']}
**Task ID**: {self.task['id']}

**Instructions**: {self.task['prompt'].strip()}

**Source Text**:
{self.task['ori_text'].strip()}

---

Please provide your complete translation as an ordinary message (no tool call). That message is graded.
"""

        return [TextBlock(text=prompt)]

    def _parse_score(self, grading_response: str) -> float:
        """Extract score with multi-stage fallback"""

        # Priority 1: "Score: X" pattern
        match = re.search(r"Score:\s*([\d.]+)", grading_response, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Priority 2: Last decimal number in response
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", grading_response)
        if numbers:
            try:
                score = float(numbers[-1])
                # If score > 1.0, assume 0-10 scale and normalize
                if score > 1.0:
                    score = score / 10.0
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Default fallback
        return 0.0

    async def _grade_single_rubric(self, translation: str, rubric: dict) -> dict:
        """Grade against one rubric criterion"""

        direction_display = "EN→ZH" if self.task["direction"] == "en_to_zh" else "ZH→EN"
        domain = f"{self.task['Primary_Domain']} / {self.task['Secondary_Domain']}"

        grader_prompt = GRADER_TEMPLATE.format(
            direction=direction_display,
            domain=domain,
            source_text=self.task["ori_text"],
            translation=translation,
            criterion=rubric["criterion"]
        )

        try:
            response = await self.client.chat.completions.create(
                model="gpt-5-mini",  # CRITICAL: No temperature parameter
                messages=[{"role": "user", "content": grader_prompt}]
            )

            grading_text = response.choices[0].message.content or ""
            score = self._parse_score(grading_text)

            return {
                "criterion": rubric["criterion"],
                "max_points": rubric["points"],
                "score": score,
                "feedback": grading_text
            }

        except Exception as e:
            # Conservative fallback on error
            return {
                "criterion": rubric["criterion"],
                "max_points": rubric["points"],
                "score": 0.0,
                "feedback": f"Grading error: {str(e)}"
            }

    async def _grade_all_rubrics(self, translation: str) -> List[dict]:
        """Grade all rubrics in parallel using asyncio.gather"""
        tasks = [self._grade_single_rubric(translation, r) for r in self.rubrics]
        return await asyncio.gather(*tasks)

    @terminal
    @tool
    async def submit_translation(self, params: SubmitTranslationInput) -> ToolOutput:
        """Grade the assistant's translation against the rubric checklist.

        Terminal tool: hidden from the model, which replies with its complete
        translation as an ordinary message rather than calling a tool. The
        harness routes that message text here for multi-rubric LLM grading
        (gpt-5-mini). Since this is the environment's only tool, the model is
        given no tools at all.
        """

        translation = params.translation.strip()

        # Validate non-empty
        if not translation:
            return ToolOutput(
                blocks=[TextBlock(text="❌ Please provide a non-empty translation.")],
                metadata={"error": "Empty translation"},
                reward=0.0,
                finished=True
            )

        # Grade all rubrics in parallel
        rubric_results = await self._grade_all_rubrics(translation)

        # Aggregate scores
        total_score = sum(r["score"] for r in rubric_results)
        total_possible = sum(r["max_points"] for r in rubric_results)
        normalized_reward = total_score / total_possible if total_possible > 0 else 0.0

        # Format feedback
        direction_display = "EN→ZH" if self.task["direction"] == "en_to_zh" else "ZH→EN"

        feedback = f"""# Translation Evaluation Results

**Task**: {self.task["id"]}
**Domain**: {self.task["Primary_Domain"]} / {self.task["Secondary_Domain"]}
**Direction**: {direction_display}

**Overall Score**: {total_score:.2f} / {total_possible:.2f}
**Normalized Reward**: {normalized_reward:.3f}

---

## Rubric-by-Rubric Feedback

"""

        for i, result in enumerate(rubric_results, 1):
            # Truncate criterion for display
            criterion_display = result['criterion'][:150]
            if len(result['criterion']) > 150:
                criterion_display += "..."

            feedback += f"""### Rubric {i}: {result['score']:.2f} / {result['max_points']:.2f}
**Criterion**: {criterion_display}

**Grader Analysis**:
{result['feedback']}

---

"""

        return ToolOutput(
            blocks=[TextBlock(text=feedback)],
            metadata={
                "task_id": self.task["id"],
                "direction": self.task["direction"],
                "primary_domain": self.task["Primary_Domain"],
                "secondary_domain": self.task["Secondary_Domain"],
                "total_score": total_score,
                "total_possible": total_possible,
                "normalized_reward": normalized_reward,
                "num_rubrics": len(rubric_results),
                "rubric_results": rubric_results
            },
            reward=normalized_reward,
            finished=True
        )
