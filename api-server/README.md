# Create an API server

I am going to create an OpenAI-compatible API server for the fine-tuned LLM so that it can be accessed from external web apps and messaging clients. The API server will use chemistry knowledge embeddings in the Qdrant vector store to supplement the user questions in the prompt. For demonstration purposes, I started an API server with a chatbot web UI on my Nvidia RTX 4090 server.

## Prerequisites

* Install WasmEdge with GGML plugin
* Have Qdrant database up and running
* The Qdrant collection `chemistry_book` is populated with embeddings from a chemistry text


## Download the API server

We will use the LlamaEdge API server to interact with the fine-tuned LLM.
It automatically searches for embeddings for each user request, and then adds related text to the conversation prompt.
The user requests and LLM responses are sent in the OpenAI JSON format, which enables a large ecosystem of LLM tools to work with the LlamaEdge API server.
First, download the `llama-api-server.wasm` application. 
It is a cross-platform Wasm app that runs the LLM inference efficiently on all CPUs and GPUs.

```
curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
```

Next, download and unzip the `chatbot-ui` folder, which contains the HTML, CSS, and JavaScript files for the web-based chatbot.

```
curl -LO https://github.com/second-state/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
tar xzf chatbot-ui.tar.gz
rm chatbot-ui.tar.gz
```

## Start the API server

Use the following command to start the API server for model file `chemistry-assistant-13b-q5_k_m.gguf` on the local computer's port `8080`. It connects to the `chemistry_book` collection of a local Qdrant vector store to search for related text for user questions in the prompt. 
The API server uses the `all-MiniLM-L6-v2-ggml-model-f16.gguf` model for computing embeddings.
The server is started as a background process.

```
nohup wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:chemistry-assistant-13b-q5_k_m.gguf \
  --nn-preload embedding:GGML:AUTO:all-MiniLM-L6-v2-ggml-model-f16.gguf \
  llama-api-server.wasm -p llama-2-chat \
  --model-alias default,embedding \
  --model-name chemistry-assistant-13b,all-minilm-l6-v2 \
  --ctx-size 4096,256 \
  --qdrant-url http://127.0.0.1:6333 \
  --qdrant-collection-name "chemistry_book" \
  --qdrant-limit 3 \
  --qdrant-score-threshold 0.6 \
  --log-prompts &
```

You can now test the server by making OpenAI style web API calls to the `/v1/chat/completions` endpoint.
Since the API server is started with the `qdrant-` arguments, it first searches for 
embeddings that are related to the current question. The search results are inserted into the conversation context for the LLM
to generate an answer.

```
curl -s -X POST http://localhost:8080/v1/chat/completions \
    -H 'accept:application/json' \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What are the metals that liquid at room temperature?"}],"max_tokens":2048}'
```

The server responds the following.

```
{"id":"e6219b85-0453-407b-8737-f525fe15aa27","object":"chat.completion","created":1709286513,"model":"my-model","choices":[{"index":0,"message":{"role":"assistant","content":"XXX"},"finish_reason":"stop"}],"usage":{"prompt_tokens":389,"completion_tokens":78,"total_tokens":467}}
```


## Test the web chatbot UI

The API server comes with a ready-to-use web UI. The HTML and JavaScript files for the UI in the `chatbot-ui` directory. You can customize the look and feel of the UI by changing the HTML files. With the API server running in the background, you can simply load the following URL in your web browser to start interacting with the chemistry assistant RAG app.

```
http://127.0.0.1:8080/rag.html?qdrant_collection_name=chemistry_book
```

Using a tool like [ngrok](https://ngrok.com/), you will be able to make your local server accessible via HTTPS over the Internet.


