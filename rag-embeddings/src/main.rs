use serde_json::{json, Value};
use time::OffsetDateTime;
use std::env;
use std::io::{BufReader, BufRead};
use std::fs::File;
use qdrant::*;
use wasi_nn::{self, GraphExecutionContext};

fn set_data_to_context(
    context: &mut GraphExecutionContext,
    data: Vec<u8>,
) -> Result<(), wasi_nn::Error> {
    context.set_input(0, wasi_nn::TensorType::U8, &[1], &data)
}

#[allow(dead_code)]
fn set_metadata_to_context(
    context: &mut GraphExecutionContext,
    data: Vec<u8>,
) -> Result<(), wasi_nn::Error> {
    context.set_input(1, wasi_nn::TensorType::U8, &[1], &data)
}

fn get_data_from_context(context: &GraphExecutionContext, index: usize) -> String {
    // Preserve for 4096 tokens with average token length 15
    const MAX_OUTPUT_BUFFER_SIZE: usize = 4096 * 15 + 128;
    let mut output_buffer = vec![0u8; MAX_OUTPUT_BUFFER_SIZE];
    let mut output_size = context.get_output(index, &mut output_buffer).unwrap();
    output_size = std::cmp::min(MAX_OUTPUT_BUFFER_SIZE, output_size);

    String::from_utf8_lossy(&output_buffer[..output_size]).to_string()
}

fn get_embd_from_context(context: &GraphExecutionContext) -> Value {
    serde_json::from_str(&get_data_from_context(context, 0)).unwrap()
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];
    let collection_name: &str = &args[2];
    let file_name: &str = &args[3];
    let mut options = json!({});
    options["embedding"] = serde_json::Value::Bool(true);
    options["ctx-size"] = serde_json::Value::from(4096);
    let ctx_size = options["ctx-size"].as_u64().unwrap();

    let graph =
        wasi_nn::GraphBuilder::new(wasi_nn::GraphEncoding::Ggml, wasi_nn::ExecutionTarget::AUTO)
            .config(options.to_string())
            .build_from_cache(model_name)
            .expect("Create GraphBuilder Failed, please check the model name or options");
    let mut context = graph
        .init_execution_context()
        .expect("Init Context Failed, please check the model");

    let client = qdrant::Qdrant::new();

    let mut id : u64 = 0;
    let mut points = Vec::<Point>::new();
    let mut current_section = String::new();
    let file = File::open(file_name)?;
    let reader = BufReader::new(file);
    for line_result in reader.lines() {
        let line = line_result?;
        if line.trim().is_empty() && (!current_section.trim().is_empty()) {
            set_data_to_context(&mut context, current_section.as_bytes().to_vec()).unwrap();
            match context.compute() {
                Ok(_) => (),
                Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::ContextFull)) => {
                    println!("\n[INFO] Context full");
                }
                Err(wasi_nn::Error::BackendError(wasi_nn::BackendError::PromptTooLong)) => {
                    println!("\n[INFO] Prompt too long");
                }
                Err(err) => {
                    println!("\n[ERROR] {}", err);
                }
            }
            let embd = get_embd_from_context(&context);

            let mut embd_vec = Vec::<f32>::new();
            for idx in 0..ctx_size as usize {
                embd_vec.push(embd["embedding"][idx].as_f64().unwrap() as f32);
            }

            println!("{} : ID={} Size={}", OffsetDateTime::now_utc(), id, embd_vec.len());
            points.push(Point{
                id: PointId::Num(id), 
                vector: embd_vec,
                payload: json!({"text": current_section}).as_object().map(|m| m.to_owned()),
            });
            id += 1;

            // Start a new section
            current_section.clear();
        } else {
            if current_section.len() < ctx_size as usize * 4 - line.len() {
                current_section.push_str(&line);
                current_section.push('\n');
            }
        }
    }

    let r = client.upsert_points(collection_name, points).await;
    println!("Upsert points result is {:?}", r);

    Ok(())
}
