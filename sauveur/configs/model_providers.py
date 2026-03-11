from enum import Enum


class Model_Providers(Enum):
    HUGGINGFACE = 'huggingface'     # its a free models provider when no model provider given
    OPENAI = 'openai'
    GOOGLE = 'google'
    AWS_BEDROCK = 'aws_bedrock'
    AZURE = 'azure'


PROVIDER_ALIASES = {
    Model_Providers.OPENAI: {"openai", "chatgpt"},
    Model_Providers.GOOGLE: {"google", "gemini"},
    Model_Providers.AWS_BEDROCK: {"aws", "aws_bedrock", "bedrock"},
    Model_Providers.AZURE: {"azure", "microsoft"},
    Model_Providers.HUGGINGFACE: {"huggingface"}
}

