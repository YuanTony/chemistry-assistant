# Create a vector store for chemistry knowledge

The fine-tuned model by itself is still insufficent. Especially for a small LLM (ours is only 13b compared to the 220b GPT3), it does not have the "space" to store all the "knowledge" from its training materials. LLMs often lack domain knowledge to accurately answer highly specific questions. A common technique to supplement domain knowledge to LLMs is through the RAG technique. It works like the following.

1. The domain knowledge is segmented into chunks of text.
2. Each text segment is turned into a numeric vector, aka an embedding, by the LLM. The embedding represents the LLM's "understanding" of the text.
3. The embeddings are stored in a vector database.
4. When the user asks a question, we use the LLM to turn the question into an embedding too.
5. We will search the vector database for embeddings that are similiar to the user's question.
6. The original text of the search result embeddings will be added to the prompt for the LLM to answer the question.

This approach provides the LLM with the knowledge context and even source so that it can answer the question more accurately.

> Small LLMs often have problems understanding and following the context provided by the RAG search results. That is why fine-tuning those small LLMs is so critical. With fine-tuning, the LLMs become more sensitive to the RAG materials that are in its subject area.

## Start the Qdrant vector database

I choose Qdrant as the vector database to store and manage the knowledge embeddings. With Docker, you can easily run Qdrant locally on a personal computer.

```
mkdir qdrant_storage

nohup sudo docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

Add a vector collection called `chemistry_book`.

```
curl -X PUT 'http://localhost:6333/collections/chemistry_book' \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "vectors": {
      "size": 4096,
      "distance": "Cosine",
      "on_disk": true
    }
  }'
```

Note: you can delete a collection using the following request.

```
curl -X DELETE 'http://localhost:6333/collections/chemistry_book'
```

## Install WasmEdge with GGML plugin

To generate the embeddings, I need to process the text using the fine-tuned LLM. WasmEdge is a extremely lightweight and cross-platform runtime for LLMs. I will use it here.

```
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugins wasmedge_rustls wasi_nn-ggml
source /home/azureuser/.bashrc
```

## Build the program to generate embeddings

The application to generate and store embeddings is modified from LlamaEdge examples. It reads a chemistry-related text file, and runs the LLM to generate an embedding for each paragraph of text. The embeddings and associated original text is then saved to Qdrant.

First, make sure that you have the Rust toolchain installed.

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup target add wasm32-wasi
```

Next compile the application in this directory to Wasm bytecode.

```
cargo build --target wasm32-wasi --release
```

## Generate embeddings

Now, we can run the Wasm app to generate embeddings from a text file [chemistry.txt](chemistry.txt) and save to the Qdrant `chemistry_book` collection.

```
cp target/wasm32-wasi/release/create_embeddings.wasm .
wasmedge --dir .:. --nn-preload default:GGML:AUTO:llama-2-7b-chat.Q5_K_M.gguf create_embeddings.wasm --ctx-size 4096  default chemistry_book chemistry.txt
```


