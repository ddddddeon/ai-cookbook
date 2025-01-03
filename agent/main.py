import click
from smolagents import CodeAgent
from smolagents import LiteLLMModel
from smolagents import GradioUI

from tools import StockInfoTool


@click.command()
@click.option("--ui", is_flag=True, help="Run with Gradio UI interface")
def main(ui):
    model = LiteLLMModel("deepseek/deepseek-chat")
    agent = CodeAgent(tools=[StockInfoTool()], model=model, add_base_tools=True)

    if ui:
        GradioUI(agent).launch()
    else:
        prompt = input("> ")
        agent.run(prompt)


if __name__ == "__main__":
    main()
