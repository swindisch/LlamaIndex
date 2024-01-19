import os
from dotenv import load_dotenv, find_dotenv
import logging
import sys
from llama_index import VectorStoreIndex, SimpleDirectoryReader

load_dotenv(find_dotenv(), override=True)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("Was ist generative KI?")
print(response)
