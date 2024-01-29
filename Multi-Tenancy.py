from llama_index import VectorStoreIndex
from llama_index.vector_stores.types import MetadataFilters, ExactMatchFilter
from llama_index import SimpleDirectoryReader
from llama_index.ingestion import IngestionPipeline
from llama_index.text_splitter import SentenceSplitter


reader = SimpleDirectoryReader(input_files=["data/dense_x_retrieval.pdf"])
documents_jerry = reader.load_data()

reader = SimpleDirectoryReader(input_files=["data/llm_compiler.pdf"])
documents_ravi = reader.load_data()

index = VectorStoreIndex.from_documents(documents=[])

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=20),
    ]
)

for document in documents_jerry:
    document.metadata["user"] = "Jerry"

nodes = pipeline.run(documents=documents_jerry)
# Insert nodes into the index
index.insert_nodes(nodes)

for document in documents_ravi:
    document.metadata["user"] = "Ravi"

nodes = pipeline.run(documents=documents_ravi)
# Insert nodes into the index
index.insert_nodes(nodes)

# For Jerry
jerry_query_engine = index.as_query_engine(
    filters=MetadataFilters(
        filters=[
            ExactMatchFilter(
                key="user",
                value="Jerry",
            )
        ]
    ),
    similarity_top_k=3,
)

# For Ravi
ravi_query_engine = index.as_query_engine(
    filters=MetadataFilters(
        filters=[
            ExactMatchFilter(
                key="user",
                value="Ravi",
            )
        ]
    ),
    similarity_top_k=3,
)


# Jerry has Dense X Rerieval paper and should be able to answer following question.
response = jerry_query_engine.query(
    "what are propositions mentioned in the paper?"
)
# Print response
print(response.response)
