import logging
import sys

from llama_index import VectorStoreIndex, SimpleDirectoryReader

# Logging Level hochsetzen, damit wir alle Meldungen sehen.
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Erstelle den Index aus den Dokumenten.
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, show_progress=True)

# Der Index wurde geladen, nun kann er verwendet werden.
query_engine = index.as_query_engine()
response = query_engine.query("Was ist generative KI?")
print(response)
