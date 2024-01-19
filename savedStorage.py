import os.path
import logging
import sys

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# Logging Level hochsetzen, damit wir alle Meldungen sehen.
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Falls es ein Verzeichnis gibt, Index laden sonst erstellen.
StorageDir = "./storage"
if os.path.exists(StorageDir):
    # Lade den vorhandenen Index.
    storage_context = StorageContext.from_defaults(persist_dir=StorageDir)
    index = load_index_from_storage(storage_context)
else:
    # Erstelle den Index.
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # Speichere den Index.
    index.storage_context.persist(persist_dir=StorageDir)

# Der Index wurde geladen, nun kann er verwendet werden.
query_engine = index.as_query_engine()
response = query_engine.query("Was ist generative KI?")
print(response)
