from typing import Any#, Optional

from .configs.default_env_vars import Default_Env_Vars
from .configs.model_providers import Model_Providers, PROVIDER_ALIASES
from .helpers.errors_messages import Error_Messages


class _Embedder:
    def __init__(self,
        model_provider: str,
        model_name: str,
        model_dimension: int,
        model_provider_kwargs: dict[str, Any]
    ) -> None:
        """
        It is used to create the embedding object for the given model and generate embeddings for the given documents using that embedding object.

        Args:
            model_provider (str): The name of the model provider.
            model_name (str): The name of the model which generate the embeddings.
            model_dimension (int): The dimension of the model.
            model_provider_kwargs (dict[str, Any]): Additional configuration to pass to the model provider while creating the model instance.
        """
        self._model_provider = model_provider.lower()
        self._model_name = model_name.lower()
        self._model_dimension = model_dimension
        self._model_provider_kwargs = model_provider_kwargs

        if self._model_provider != Default_Env_Vars.MODEL_PROVIDER and model_provider_kwargs == Default_Env_Vars.MODEL_PROVIDER_KWARGS:
            self._model_provider_kwargs = {}


        self._embedder = self._get_embedding_object()


    def _get_embedding_object(self):
        """
        It gets the embedding object for a model provider.

        Args: None
        Returns:
            OpenAIEmbeddings
            | GoogleGenerativeAIEmbeddings
            | BedrockEmbeddings
            | AzureOpenAIEmbeddings
            | HuggingFaceEmbeddings
            : The embedding object created for the given model provider.
        Raises:
            ValueError: If the model provider is invalid or not supported.
        """
        if self._model_provider in PROVIDER_ALIASES[Model_Providers.OPENAI]:
            openai_embedder = self._create_embedder_for_openai()
            return openai_embedder

        if self._model_provider in PROVIDER_ALIASES[Model_Providers.GOOGLE]:
            google_embedder = self._create_embedder_for_google()
            return google_embedder

        if self._model_provider in PROVIDER_ALIASES[Model_Providers.AWS_BEDROCK]:
            aws_bedrock_embedder = self._create_embedder_for_aws_bedrock()
            return aws_bedrock_embedder

        if self._model_provider in PROVIDER_ALIASES[Model_Providers.AZURE]:
            azure_embedder = self._create_embedder_for_azure()
            return azure_embedder

        if self._model_provider in PROVIDER_ALIASES[Model_Providers.HUGGINGFACE]:     # default model provider
            huggingface_embedder = self._create_huggingface_embedder()
            return huggingface_embedder

        raise ValueError(Error_Messages.INVALID_MODEL_PROVIDER)


    def generate_embeddings(self, docs: list[str]|str=[]) -> list[list[float]]:
        """
        It generates embeddings for the given string or list of strings.

        Args:
            docs (list[str]|str): A string or list of strings for which embeddings will be generated.
        Returns:
            list[list[float]]: List of list of embeddings for the given strings.
        Raises:
            TypeError: If the input is not a string or list of strings.
        """
        if docs in [[],{},set(),(),'']:
            return []
        if not isinstance(docs, (list, set, tuple, str)):
            raise TypeError(Error_Messages.INVALID_DOCS)

        docs_embeddings = self._embedder.embed_documents(docs)
        return docs_embeddings


    def _create_embedder_for_openai(self):
        """
        Creates an embedding object for OpenAI.

        Args: None
        Returns:
            OpenAIEmbeddings: The embedding object created for OpenAI embeddings.
        """
        from langchain_openai import OpenAIEmbeddings

        embedder = OpenAIEmbeddings(
            model=self._model_name,
            dimensions=self._model_dimension,
            **self._model_provider_kwargs,
        )
        return embedder


    def _create_embedder_for_google(self):
        """
        Creates an embedding object for Google.

        Args: None
        Returns:
            GoogleGenerativeAIEmbeddings: The embedding object created for Google embeddings.
        """
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embedder = GoogleGenerativeAIEmbeddings(
            model=self._model_name,
            output_dimensionality=self._model_dimension,
            **self._model_provider_kwargs,
        )
        return embedder


    def _create_embedder_for_aws_bedrock(self):
        """
        Creates an embedding object for AWS Bedrock.

        Args: None
        Returns:
            BedrockEmbeddings: The embedding object created for AWS Bedrock embeddings.
        """
        from langchain_aws import BedrockEmbeddings

        embedder = BedrockEmbeddings(
            model_id=self._model_name,
            dimensions=self._model_dimension,
            **self._model_provider_kwargs,
        )
        return embedder


    def _create_embedder_for_azure(self):
        """
        Creates an embedding object for Azure.

        Args: None
        Returns:
            AzureOpenAIEmbeddings: The embedding object created for Azure OpenAI embeddings.
        """
        from langchain_openai import AzureOpenAIEmbeddings

        embedder = AzureOpenAIEmbeddings(
            model=self._model_name,
            dimensions=self._model_dimension,
            **self._model_provider_kwargs,
        )
        return embedder


    def _create_huggingface_embedder(self):
        """
        Creates an embedding object for Hugging Face.

        Args: None
        Returns:
            HuggingFaceEmbeddings: The embedding object created for Hugging Face embeddings.
        """
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings

        embedder = HuggingFaceEmbeddings(
            model=self._model_name,
            **self._model_provider_kwargs,
        )
        return embedder

