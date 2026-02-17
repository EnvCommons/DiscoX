import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
ENV_NAME = "local/discox"  # Use "EnvCommons/discox" for deployed version
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


async def main():
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY=sk-...")
        return

    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Local testing: base_url for local server
    environment = or_client.environments.get(
        name=ENV_NAME,
        base_url="http://localhost:8080"
    )

    tasks = await environment.list_tasks(split="train")
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks in DiscoX dataset")
    print(f"Testing first 3 tasks...\n")

    for task in tasks[:3]:
        print(f"{'='*60}")
        print(f"Task: {task.task_spec['id']} ({task.task_spec['direction']})")
        print(f"Domain: {task.task_spec['primary_domain']} / {task.task_spec['secondary_domain']}")
        print(f"{'='*60}\n")

        async with environment.session(
            task=task,
            secrets={"openai_api_key": OPENAI_API_KEY}
        ) as session:
            prompt = await session.get_prompt()
            # Handle both str and List[TextBlock] for compatibility
            prompt_text = prompt if isinstance(prompt, str) else prompt[0].text
            input_list = [{"role": "user", "content": prompt_text}]
            finished = False

            while not finished:
                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list
                )

                input_list += response.output

                for item in response.output:
                    if item.type == "function_call":
                        tool_result = await session.call_tool(
                            item.name,
                            json.loads(str(item.arguments))
                        )

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text if tool_result.blocks else ""
                        })

                        print(f"Reward: {tool_result.reward:.3f}")
                        print(f"Feedback preview: {tool_result.blocks[0].text[:200]}...\n")

                        finished = tool_result.finished

                        if finished:
                            print('✓ Task completed!\n')
                            break


if __name__ == "__main__":
    asyncio.run(main())
