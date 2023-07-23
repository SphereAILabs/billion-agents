from .tools.tool import Tool
import typing


class ToolExecutor(Tool):
    def __init__(self, tool: Tool, variables: dict[str, typing.Any]):
        self.tool = tool
        self.name = tool.name
        self.description = tool.description
        self.input_schema = tool.input_schema
        self.output_schema = tool.output_schema
        self.variables = variables

    def __call__(self, **kwargs):
        # grab variables
        params = {}
        variable_params = self.tool.variable_params()

        for key in kwargs.keys():
            value = kwargs[key]
            if key in variable_params:
                value = self.variables[key]

            params[key] = value

        # call tool
        result = self.tool(**params)
        should_store = self.tool.should_store_output

        if should_store:
            self.variables

        return

    def variable_params(self) -> list[str]:
        return self.tool.variable_params()

    def observation(self, output) -> str:
        return self.tool.observation(output)
