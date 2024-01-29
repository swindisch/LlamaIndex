import logging
import sys

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)

from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

from llama_index import set_global_tokenizer
from transformers import AutoTokenizer
from llama_index.embeddings import HuggingFaceEmbedding

from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate


# Logging Level hochsetzen, damit wir alle Meldungen sehen.
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

set_global_tokenizer(
    AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
)

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")



# use Huggingface embeddings
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
    model_name="StabilityAI/stablelm-tuned-alpha-3b",
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"torch_dtype": torch.float16}
)

# create a service context
#service_context = ServiceContext.from_defaults(llm="local")

service_context = ServiceContext.from_defaults(
    llm=llm,
    chunk_size=1024
)

# load documents
documents = SimpleDirectoryReader("./data").load_data()

# create vector store index
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

# Der Index wurde geladen, nun kann er verwendet werden.
query_engine = index.as_query_engine()
response = query_engine.query("Was ist generative KI?")
print(response)
