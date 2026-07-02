import asyncio
from dataclasses import dataclass
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_graph import GraphBuilder, StepContext, TypeExpression

@dataclass
class WorkflowState:
    user_input: str


async def main():
    g = GraphBuilder(
        state_type=WorkflowState,
        output_type=str,
        input_type=str,
    )

    model = OpenAIChatModel(
        model_name="Qwen/Qwen3.6-35B-A3B",
        provider=OpenAIProvider(
            base_url = os.environ.get("OPENAI_BASE_URL"),
            api_key = os.environ.get("API_KEY")
        )
    )

    agent = Agent(
        model=model,
        system_prompt=[
            "You are a helpful assistant."
        ]
    )


    @g.step
    async def generate(ctx: StepContext[WorkflowState, None, str]) -> str:
        result = await agent.run(ctx.inputs)
        return result.output

    @g.step
    async def deny(ctx: StepContext[WorkflowState, None, str]) -> str:
        return "Denied"

    g.add(
        g.edge_from(g.start_node).to(
            g.decision()
            .branch(g.match(TypeExpression[str], matches=lambda inp: "pasta" in inp).to(deny))
            .branch(g.match(TypeExpression[str], matches=lambda inp: "pasta" not in inp).to(generate))
        ),
        g.edge_from(deny).to(g.end_node),
        g.edge_from(generate).to(g.end_node)
    )

    graph = g.build()

    user_input = input("> ")
    state = WorkflowState(user_input=user_input)
    result = await graph.run(inputs=user_input, state=state)

    print(result)


if __name__ == "__main__":
    asyncio.run(main())

    
    