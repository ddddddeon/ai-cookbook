from smolagents import CodeAgent
from smolagents import LiteLLMModel

model = LiteLLMModel("deepseek/deepseek-chat")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

prompt = input("> ")
agent.run(prompt)
