from tools.tool import Tool


class ToolBox:
    """
    ToolBox contains the tools an agent has access to.
    """

    def __init__(self, tools: list[Tool]):
        # check that no tools have the same name
        tool_names = set([tool.name for tool in tools])

        if len(tool_names) != len(tools):
            raise RuntimeError("Cannot have tools with the same name in ToolBox")

        # map tool name to tool
        self.tools = {tool.name: tool for tool in tools}

    @property
    def prompt(self) -> str:
        prompt = ""
        tools = self.tools.values()

        for i, tool in enumerate(tools):
            prompt += tool.prompt

            if i != len(self.tools) - 1:
                prompt += "\n\n"

        return prompt
