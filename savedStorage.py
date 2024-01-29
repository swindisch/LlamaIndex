import os.path
import logging
import sys
import torch

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage
)

from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores import RedisVectorStore

from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate

# Logging Level hochsetzen, damit wir alle Meldungen sehen.
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# This will wrap the default prompts that are internal to llama-index
# taken from https://huggingface.co/Writer/camel-5b-hf
query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)

# define embedding function
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# load documents
loader = SimpleDirectoryReader("./data_pg")
documents = loader.load_data()
for file in loader.input_files:
    print(file)

print(RedisVectorStore.__init__.__doc__)


vector_store = RedisVectorStore(
    index_name="pg_essays",
    index_prefix="llama",
    redis_url="redis://localhost:6379",  # Default
    overwrite=True,
)

llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.25, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="Writer/camel-5b-hf",
    model_name="Writer/camel-5b-hf",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16}
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(chunk_size=512, llm=llm, embed_model=embed_model)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context
)

# Der Index wurde geladen, nun kann er verwendet werden.
query_engine = index.as_query_engine()
response = query_engine.query("What did the author learn?")
print(response)
