use std::env;
use discord_flows::{model::Message, Bot, ProvidedBot, message_handler};
use flowsnet_platform_sdk::logger;
use llmservice_flows::{
    // chat::{ChatOptions, ChatRole, chat_history},
    chat::{ChatOptions},
    LLMServiceFlows,
};
use store_flows as store;
use serde_json::json;

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn on_deploy() {
    let token = std::env::var("discord_token").unwrap();
    let bot = ProvidedBot::new(token);
    bot.listen_to_messages().await;
}

#[message_handler]
async fn handler(msg: Message) {
    logger::init();
    let token = env::var("discord_token").unwrap();
    let placeholder_text = env::var("placeholder").unwrap_or("Typing ...".to_string());
    let system_prompt = env::var("system_prompt").unwrap_or("You are a helpful chemistry teacher. Explain in simple English and adhere strictly to facts.".to_string());
    let llm_endpoint = std::env::var("llm_endpoint").unwrap_or("".to_string());

    let bot = ProvidedBot::new(token);
    let discord = bot.get_client();
    if msg.author.bot {
        log::info!("ignored bot message");
        return;
    }
    let channel_id = msg.channel_id;
    let content = msg.content;

    if content.eq_ignore_ascii_case("/restart") {
        _ = discord.send_message(
            channel_id.into(),
            &serde_json::json!({
                "content": "Ok, I am starting a new conversation."
            }),
        ).await;
        store::set(&channel_id.to_string(), json!(true), None);
        log::info!("Restarted converstion for {}", channel_id);
        return;
    } else if content.starts_with("/") {
        _ = discord.send_message(
            channel_id.into(),
            &serde_json::json!({
                "content": "Sorry, I do not recognize this command. Do you mean to say /restart to start a new conversation?"
            }),
        ).await;
        return;
    }

    let restart = store::get(&channel_id.to_string())
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if restart {
        log::info!("Detected restart = true");
        store::set(&channel_id.to_string(), json!(false), None);
    }

    let placeholder  = discord.send_message(
        channel_id.into(),
        &serde_json::json!({
            "content": &placeholder_text
        }),
    ).await.unwrap();

    let mut llm = LLMServiceFlows::new(&llm_endpoint);
    let co = ChatOptions {
        restart: restart,
        system_prompt: Some(&system_prompt),
        token_limit: 4096,
        ..Default::default()
    };

    match llm.chat_completion(&channel_id.to_string(), &content, &co).await {
        Ok(r) => {
            _ = discord.edit_message(
                channel_id.into(), placeholder.id.into(),
                &serde_json::json!({
                    "content": r.choice
                }),
            ).await;
        }
        Err(e) => {
            _ = discord.edit_message(
                channel_id.into(), placeholder.id.into(),
                &serde_json::json!({
                    "content": "Sorry, an error has occured. Please try again later!"
                }),
            ).await;
            log::error!("OpenAI returns error: {}", e);
        }
    }

}
