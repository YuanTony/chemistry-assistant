# Create an API server

I am going to create an OpenAI-compatible API server for the fine-tuned LLM so that it can be accessed from external web apps and messaging clients. The API server will use chemistry knowledge embeddings in the Qdrant vector store to supplement the user questions in the prompt. For demonstration purposes, I started an API server with a chatbot web UI on my Nvidia RTX 4090 server.

xxx

## Prerequisites

* Install WasmEdge with GGML plugin
* Have Qdrant database up and running
* The Qdrant collection `chemistry` is populated with embeddings from a chemistry text


## Build the API server

The Rust source code for the API server is in this directory. It is built with components from the LlamaEdge project. For every user question, I first turn it into a vector embedding, then search the Qdrant vector store, and finally insert the related text into the prompt.

To build the api server into a cross-platform Waasm app, use the following command.

```
cargo build --target wasm32-wasi --release
```


## Start the API server


Use the following command to start the API server for model file `chemistry-assistant-13b-q5_k_m.gguf` on the local computer's port `8080`. It connects to the `chemistry` collection of a local Qdrant vector store to search for related text for user questions in the prompt. The server is started as a background process.


```
nohup wasmedge --dir .:. --nn-preload default:GGML:AUTO:chemistry-assistant-13b-q5_k_m.gguf target/wasm32-wasi/release/api-server.wasm --prompt-template llama-2-chat --ctx-size 3072 --collection_name chemistry --socket-addr 127.0.0.1:8080 --log-prompts --log-stat &
```

You can now test the server by making OpenAI style web API calls.

```
```

The server responds the following.

```
```


## Test the web chatbot UI

The API server comes with a ready-to-use web UI. The HTML and JavaScript files for the UI in the `chatbot-ui` directory. You can customize the look and feel of the UI by changing the HTML files. With the API server running in the background, you can simply load the following URL in your web browser to start interacting with the chemistry assistant RAG app.

http://127.0.0.1:8080/

Using a tool like [ngrok](https://ngrok.com/), you will be able to make your local server accessible via HTTPS over the Internet.


