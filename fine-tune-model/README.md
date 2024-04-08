# Finetune an open-source LLM for the chemistry subject

This is an optional step to finetune the LLM to better answer chemistry related questions. It works by "showing" a general LLM, i.e., a Llama2 13b chat model, examples of how a chemistry teacher might answer questions. It will improve the model's capability to understand and answer these domain-specific questions. It does take significant amount of computing power to go through the finetuning process. The example I showed below took a few days of computing time on a typical desktop PC. If you do not access to computing power, it is okay to skip this tutorial, or just download the model file I have finetuned.

```
curl -LO https://huggingface.co/juntaoyuan/chemistry-assistant-13b/resolve/main/chemistry-assistant-13b-q5_k_m.gguf
```

> Small LLMs often have problems understanding and following the context provided by the RAG search results. That is why finetuning those small LLMs is important. With finetuning, the LLMs become more sensitive to the RAG materials that are in its subject area.

## Build the finetune utility from llama.cpp

The `finetune` utility in llama.cpp can work with quantitized GGUF files on CPUs, and hence dramatically reducing the hardware requirements and expen
ses for finetuning LLMs.

Checkout and download the llama.cpp source code.

```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

Build the llama.cpp binary.

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

If you have Nvidia GPU and CUDA toolkit installed, you should build llama.cpp with CUDA support.

```
mkdir build
cd build
cmake .. -DLLAMA_CUBLAS=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build . --config Release
```

## Get the base model

I used Meta's Llama2 chat 13B model as the base model. Note that I am using a Q5 quantitized GGUF model file directly to save computing resources. You can use any of the Llama2 compatible GGUF models on Hugging Face.

```
cd .. # change to the llama.cpp directory
cd models/
curl -LO https://huggingface.co/second-state/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf
```

## Create a question and answer set for fine-tuning

I came up with 1700+ pairs of QAs for the chemistry subject. It is like the following in a [CSV file](train.csv).

Question | Answer
----- | -------
What is unique about hydrogen? | It's the most abundant element in the universe, making up over 75% of all matter.
What is the main component of Jupiter? | Hydrogen is the main component of Jupiter and the other gas giant planets.
Can hydrogen be used as fuel? | Yes, hydrogen is used as rocket fuel. It can also power fuel cells to generate electricity.
What is mercury's atomic number? | The atomic number of mercury is 80
What is Mercury? | Mercury is a silver colored metal that is liquid at room temperature. It has an atomic number of 80 on the periodic table. It is toxic to humans.

> I used GPT-4 to help me come up many of these QAs.

Then, I wrote a [Python script](convert.py) to convert each row in the CSV file into a sample QA in the Llama2 chat template format. Notice that each QA pair starts with `<SFT>` as an indicator for the fine-tuning program to start a sample. The result [train.txt](train.txt) file can now be used in fine-tuning.

Put the [train.txt](train.txt) file in the `llama.cpp/models` directory with the GGUF base model.

## Finetune!

Use the following command to start the fine-tuning process on your CPUs. I am putting it in the background so that it can run continuous now.
It could several days or even a couple of weeks depending on how many CPUs you have.

```
nohup ../build/bin/finetune --model-base llama-2-13b-chat.Q5_K_M.gguf --lora-out lora.bin --train-data train.txt --sample-start '<SFT>' --adam-iter 1024 &
```

You can check the process every a few hours in the `nohup.out` file. It will report `loss` for each iteration. You can stop the process when the `loss` goes consistently under `0.1`.


## Merge

The fine-tuning process updates several layers of the LLM's neural network. Those updated layers are saved in a file called `lora.bin` and you can now merge them back to the base LLM to create the new fine-tuned LLM.

```
../build/bin/export-lora --model-base llama-2-13b-chat.Q5_K_M.gguf --lora lora.bin --model-out chemistry-assistant-13b-q5_k_m.gguf
```

The result is this file.

```
curl -LO https://huggingface.co/juntaoyuan/chemistry-assistant-13b/resolve/main/chemistry-assistant-13b-q5_k_m.gguf
```

