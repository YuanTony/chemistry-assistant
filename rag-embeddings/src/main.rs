use serde_json::{json, Value};
use time::OffsetDateTime;
use std::env;
use std::io::{BufReader, BufRead};
use std::fs::File;
use qdrant::*;
use wasmedge_wasi_nn::{
    self, BackendError, Error, ExecutionTarget, GraphBuilder, GraphEncoding, GraphExecutionContext,
    TensorType,
};

async fn generate_upsert (context: &mut GraphExecutionContext, data: &str, client: &qdrant::Qdrant, id: u64, collection_name: &str, vector_size: usize) {
    set_data_to_context(context, data.as_bytes().to_vec()).unwrap();
    match context.compute() {
        Ok(_) => (),
        Err(Error::BackendError(BackendError::ContextFull)) => {
            println!("\n[INFO] Context full");
        }
        Err(Error::BackendError(BackendError::PromptTooLong)) => {
            println!("\n[INFO] Prompt too long");
        }
        Err(err) => {
            println!("\n[ERROR] {}", err);
        }
    }
    let embd = get_embd_from_context(&context, vector_size);

    let mut embd_vec = Vec::<f32>::new();
    for idx in 0..vector_size as usize {
        embd_vec.push(embd["embedding"][idx].as_f64().unwrap() as f32);
    }

    println!("{} : ID={} Size={}", OffsetDateTime::now_utc(), id, embd_vec.len());

    let mut points = Vec::<Point>::new();
    points.push(Point{
        id: PointId::Num(id), 
        vector: embd_vec,
        payload: json!({"source": data}).as_object().map(|m| m.to_owned()),
    });

    // Upsert each point (you can also batch points for upsert)
    let r = client.upsert_points(collection_name, points).await;
    println!("Upsert points result is {:?}", r);
}

fn set_data_to_context(
    context: &mut GraphExecutionContext,
    data: Vec<u8>,
) -> Result<(), Error> {
    context.set_input(0, TensorType::U8, &[1], &data)
}

#[allow(dead_code)]
fn set_metadata_to_context(
    context: &mut GraphExecutionContext,
    data: Vec<u8>,
) -> Result<(), Error> {
    context.set_input(1, TensorType::U8, &[1], &data)
}

fn get_data_from_context(context: &GraphExecutionContext, vector_size: usize, index: usize) -> String {
    // Preserve for tokens with average token length 15
    let max_output_buffer_size: usize = vector_size * 15 + 128;
    let mut output_buffer = vec![0u8; max_output_buffer_size];
    let mut output_size = context.get_output(index, &mut output_buffer).unwrap();
    output_size = std::cmp::min(max_output_buffer_size, output_size);

    String::from_utf8_lossy(&output_buffer[..output_size]).to_string()
}

fn get_embd_from_context(context: &GraphExecutionContext, vector_size: usize) -> Value {
    let embd = &get_data_from_context(context, vector_size, 0);
    // println!("\n[EMBED] {}", embd);
    serde_json::from_str(embd).unwrap()
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];
    let collection_name: &str = &args[2];
    let vector_size: usize = args[3].trim().parse().unwrap();
    let file_name: &str = &args[4];
    let mut options = json!({});
    options["embedding"] = serde_json::Value::Bool(true);
    // options["ctx-size"] = serde_json::Value::from(vector_size);
    // let ctx_size = options["ctx-size"].as_u64().unwrap();

    let graph =
        GraphBuilder::new(GraphEncoding::Ggml, ExecutionTarget::AUTO)
            .config(options.to_string())
            .build_from_cache(model_name)
            .expect("Create GraphBuilder Failed, please check the model name or options");
    let mut context = graph
        .init_execution_context()
        .expect("Init Context Failed, please check the model");

    let client = qdrant::Qdrant::new();

    let mut id : u64 = 0;
    let mut current_section = String::new();
    let file = File::open(file_name)?;
    let reader = BufReader::new(file);
    for line_result in reader.lines() {
        let line = line_result?;
        if line.trim().is_empty() && (!current_section.trim().is_empty()) {
            generate_upsert(&mut context, &current_section, &client, id, collection_name, vector_size).await;
            id += 1;
            // Start a new section
            current_section.clear();
        } else {
            // We do not limit the size of the chunk. If it is over the model context size, we
            // would want it to fail explicitly
            current_section.push_str(&line);
            current_section.push('\n');
        }
    }

    // The last segment
    if !current_section.trim().is_empty() {
        generate_upsert(&mut context, &current_section, &client, id, collection_name, vector_size).await;
    }
    Ok(())
}
