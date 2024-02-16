# Create a Discord bot

To make the chemistry assistant AI accessible to middle school students, I decided to create a Discord bot. You can join the following Discord server and start chatting with the "Chemistry Assistant" bot user.

XXX

You could also add the bot to your own Discord server! 

## Get a Discord token

The instructions are [here](https://flows.network/blog/discord-chat-bot-guide).

## Build a flow function

I created a flow function in this directory to connect Discord with the local OpenAI API server. The flow function is written in Rust and compiles to Wasm. It can be deployed as a serverless function on the flows.network service. You can use the Rust cargo toolchain to build the flow function locally.


```
cargo build --target wasm32-wasi --release
```

## Deploy the flow function

You should fork this repo under your own GitHub user. Then, follow these steps to deploy the flow function.

1. Sign up and then log into [flows.network](https://flows.network) using your GitHUb account.
2. Click on "Create a flow" and select at the bottom of the page to import a flow function from GitHub.
3. Select the repo you forked and enter `/discord` as the path for the flow function source.
4. Click on "advanced settings", and enter the variables in the table below.
5. Click on deploy.

| Variable Name | Value         |
| ------------- | ------------- |
| discord_token  | The Discord token you have generated for this app.  |
| llm_endpoint  | The base URL of your LLM API service. It should be WITHOUT the `/v1/chat_completion` component.  |
| system_prompt  | Optional: The system prompt. |

Once deployed, the flow function listens for all messages and events associated with the `discord_token` from Discord. When it detects a 1-1 message for the bot user, it will forward the request to the RAG LLM API service, and send the response back to Discord.




