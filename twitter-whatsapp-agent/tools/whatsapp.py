import pywhatkit

from .tool import Tool


class WhatsAppTool(Tool):
    name = "WhatsApp"
    description = "Use to send a WhatsApp message to myself."
    input_schema = "(message: str)"
    output_schema = "Sent!"

    def __call__(self, message: str):
        pywhatkit.sendwhatmsg_instantly("+9179571377", message, 5)

        return "Sent!"
