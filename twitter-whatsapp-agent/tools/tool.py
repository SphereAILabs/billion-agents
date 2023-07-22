class Tool:
    """
    Interface for defining a tool to be used by an agent
    """

    def context(self) -> str:
        return f"""name: {self.name}
        description: {self.description}
        input:
        output:
        """

    def __call__(self):
        raise RuntimeError("Not Implemented")
