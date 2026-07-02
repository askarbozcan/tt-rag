import os

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from dataclasses import dataclass

model = OpenAIChatModel(
    model_name="Qwen/Qwen3.6-35B-A3B",
    provider=OpenAIProvider(
        base_url = os.environ.get("OPENAI_BASE_URL"),
        api_key = os.environ.get("API_KEY")
    )
)

@dataclass
class IncrementDeps:
    counter: int

incrementer_agent = Agent(
    model=model,
    deps_type=IncrementDeps,
    system_prompt=[
        "You call increment tool 2 times in a row."
    ]
)


@incrementer_agent.tool
def increment(ctx: RunContext[IncrementDeps]) -> int:
    ctx.deps.counter += 1
    return ctx.deps.counter


state1 = IncrementDeps(0)
run = incrementer_agent.run_sync("increment", deps=state1)
print(run.all_messages())

state2 = IncrementDeps(5)
run = incrementer_agent.run_sync("increment", deps=state1)