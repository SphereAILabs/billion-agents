from abc import abstractmethod


class Tool:
    """
    Interface for defining a tool to be used by an agent.
    Tools have
    name: name of the tool
    description: what does the tool do? what is it useful for?
    input_schema: what do you need to pass into the tool
    output_schema: what is the observation as a result of using this tool
    """

    name = ""
    description = ""
    input_schema = ""
    output_schema = ""

    @property
    def prompt(self) -> str:
        return f"""{self.name}: {self.description}
input: {self.input_schema}
output: {self.output_schema}"""

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def rep(self) -> str:
        """
        Returns the representation of the result of calling a tool. This is
        passed into the agent as the observation.
        """
        pass
