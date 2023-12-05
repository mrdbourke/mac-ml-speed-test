"""
Install instructions:

# !pip install pandas
# !pip install py-cpuinfo
# !pip instal llangchain
# !pip install prettytable
# !CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python

Guides followed for this script:
- https://github.com/facebookresearch/llama-recipes/blob/main/demo_apps/HelloLlamaLocal.ipynb 
- https://llama-cpp-python.readthedocs.io/en/latest/install/macos/
- https://python.langchain.com/docs/integrations/llms/llamacpp
"""

# Standard library imports
import argparse
from pathlib import Path
from timeit import default_timer as timer

# Third-party imports
import pandas as pd
from tqdm.auto import tqdm
from prettytable import PrettyTable # for nice looking results

# Local application/library specific imports
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

try:
    import cpuinfo 
    CPU_PROCESSOR = cpuinfo.get_cpu_info().get('brand_raw').replace(" ", "_")
    print(f"[INFO] Processor: {CPU_PROCESSOR}")
except Exception as e:
    print(f"Error: {e}, may have failed to get CPU_PROCESSOR name from cpuinfo, please install cpuinfo or set CPU_PROCESSOR manually") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Llama 2 on a set of questions')
    parser.add_argument('--path_to_gguf_model', default="./llama-2-7b-chat.Q4_0.gguf", type=str, help='Path to the Llama 2 model, see: https://huggingface.co/TheBloke for downloads, should be ".gguf" format')
    parser.add_argument('--num_times_per_question', default=1, type=int, help='Number of times to ask each question')
    parser.add_argument('--num_questions', default='all', type=str, help='Number of questions to ask, default "all", can be a positive integer between 1 and 20')
    parser.add_argument('--max_tokens', default=500, type=int, help='Max tokens to generate per question, default 500')
    parser.add_argument('--stream_output', default=False, type=bool, help='Stream output from Llama 2')
    args = parser.parse_args()


    # Prompt questions for the model (using "Let's think step by step..." for verbosity of output)
    # See "Let's think step by step..." paper: https://arxiv.org/abs/2205.11916 
    questions = [
        "What are the nutrition facts of an apple? Let's think step by step...",
        "What steps are involved in the water cycle? Let's think step by step...",
        "How does a computer process a command? Let's think step by step...",
        "What are the stages of a butterfly's life cycle? Let's think step by step...",
        "How does a refrigerator keep food cold? Let's think step by step...",
        "What happens when we digest food? Let's think step by step...",
        "How does an airplane stay airborne? Let's think step by step...",
        "What are the processes involved in making a cup of coffee? Let's think step by step...",
        "How do bees produce honey? Let's think step by step...",
        "What are the key steps in recycling plastic? Let's think step by step...",
        "How does a clock measure time? Let's think step by step...",
        "What is the process of photosynthesis in plants? Let's think step by step...",
        "How does a car engine work? Let's think step by step...",
        "What are the basic steps in baking bread? Let's think step by step...",
        "How do solar panels generate electricity? Let's think step by step...",
        "What are the stages of human sleep? Let's think step by step...",
        "How does a smartphone connect to the internet? Let's think step by step...",
        "What is the life cycle of a star? Let's think step by step...",
        "How does the human immune system fight viruses? Let's think step by step...",
        "What are the steps involved in creating a movie? Let's think step by step..."
    ]

    def process_questions(arg, questions=questions):
        if arg == "all":
            print(f"[INFO] num_questions arg is 'all', will ask {len(questions)} questions...")
            return questions
        else:
            arg = int(arg)

        if isinstance(arg, int) and arg > 0:
            # Make sure arg is not greater than the number of questions
            if arg > len(questions):
                print(f"[INFO] num_questions arg is '{arg}' & is greater than the number of questions '{len(questions)}', returning all questions...")
                return questions
            else:
                print(f"[INFO] num_questions arg is '{arg}', will ask {arg} questions...")
                return questions[:arg]
        else:
            raise ValueError("Argument must be 'all' or a positive integer between 1 and 20")

    questions = process_questions(arg=args.num_questions)


    ### Model setup ###

    # Set up your target model here, download from: https://huggingface.co/TheBloke, for macOS, you'll generally want "Q4_0.gguf" formatted models
    path_to_gguf_model = args.path_to_gguf_model

    # Make sure model path exists
    assert Path(path_to_gguf_model).exists(), f"Model path '{path_to_gguf_model}' does not exist, please download it from Hugging Face and save it to the local directory, see: https://huggingface.co/TheBloke for '.gguf' models to run on macOS"

    # Print model path
    print(f"[INFO] Using model at path: {path_to_gguf_model}")

    # For token-wise streaming so you'll see the answer gets generated token by token 
    # when Llama is answering your question
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # 1 GPU layer is enough for M-series chips
    n_gpu_layers=1

    llm = LlamaCpp(
        model_path=path_to_gguf_model,
        n_gpu_layers=n_gpu_layers,
        temperature=0.5,
        max_tokens=args.max_tokens,
        n_batch=512,
        top_p=1,
        f16_kv=True,
        n_ctx=2048, # context window
        callback_manager=callback_manager if args.stream_output == True else None, 
        verbose=False, # this will print output, turning off to save printing to console (may reduce speed)
    )

    # Helper function for converting a string to token count
    def character_count_to_tokens(sequence: str) -> float:
        character_to_token_ratio = 4 # 4 chars = 1 token, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them

        # Get length of char_sequence without whitespace
        character_len = len(sequence.strip())

        # Return the token length based on char length
        return character_len / character_to_token_ratio

    
    ### Ask questions ###
    NUM_TIMES = args.num_times_per_question
    TOTAL_QUESTIONS_TO_ASK = len(questions) * NUM_TIMES

    if TOTAL_QUESTIONS_TO_ASK > 200:
        print(f"[INFO] Asking {len(questions)} questions {NUM_TIMES} times each, total questions: {TOTAL_QUESTIONS_TO_ASK}")
        # print a warning
        print(f"[WARNING] Asking {TOTAL_QUESTIONS_TO_ASK} questions, this may take a while... (consider reducing the number of questions or number of times to ask each question)")
    else:
        print(f"[INFO] Asking {len(questions)} questions {NUM_TIMES} times each, total questions: {TOTAL_QUESTIONS_TO_ASK}")

    # Prompt model X times per question
    qa_results = []
    for question in tqdm(questions):
        print(f"[INFO] Asking question '{question}' {NUM_TIMES} times.")
        for i, _ in enumerate(range(NUM_TIMES)):
            start_time = timer()
            answer = llm(question)
            end_time = timer()
            total_time = end_time - start_time

            answer_char_len = len(answer.strip())
            chars_per_second = round(answer_char_len / total_time, 2)
            
            answer_token_len = character_count_to_tokens(sequence=answer)
            tokens_per_second = round(answer_token_len / total_time, 2)

            print(f"Answer char len: {answer_char_len} | Chars per second: {chars_per_second} | Answer token len: {answer_token_len} | Tokens per second: {tokens_per_second}")

            qa_results.append({"question": question,
                            "question_iter": i,
                            "total_time": total_time,
                            "answer": answer,
                            "answer_char_len": answer_char_len,
                            "chars_per_second": chars_per_second,
                            "answer_token_len": answer_token_len,
                            "tokens_per_second": tokens_per_second})

    ### Save results to CSV ###
    GPU_NAME = False
    MODEL_NAME = path_to_gguf_model.replace('./', '')
    if GPU_NAME:
        csv_filename = f"{GPU_NAME}_{MODEL_NAME}_results.csv"
    else:
        csv_filename = f"{CPU_PROCESSOR}_{MODEL_NAME}_results.csv"

    # Make the target results directory if it doesn't exist (include the parents)
    target_results_dir = "results_llama_2"
    results_path = Path("results") / target_results_dir
    results_path.mkdir(parents=True, exist_ok=True)
    csv_filepath = results_path / csv_filename

    # Turn dict into DataFrame
    import pandas as pd
    df = pd.DataFrame(qa_results)

    # Print results
    print(f"[INFO] Results on {CPU_PROCESSOR}:")
    total_questions = len(df)
    total_time_for_all_questions = round(df["total_time"].sum(), 2)
    total_tokens_generated = df["answer_token_len"].sum()
    total_chars_generated = df["answer_char_len"].sum()
    total_tokens_per_second = round(total_tokens_generated / total_time_for_all_questions, 2)
    total_chars_per_second = round(total_chars_generated / total_time_for_all_questions, 2)

    # Create a PrettyTable object
    table = PrettyTable()

    # Define the columns
    table.field_names = ["Metric", "Value"]

    # Add rows
    table.add_row(["Total questions", total_questions])
    table.add_row(["Total time for all questions (s)", total_time_for_all_questions])
    table.add_row(["Total tokens generated", total_tokens_generated])
    table.add_row(["Total chars generated", total_chars_generated])
    table.add_row(["Average tokens per second", total_tokens_per_second])
    table.add_row(["Average chars per second", total_chars_per_second])

    # Print the table
    print(table)

    # print(f"Total questions: {total_questions}")
    # print(f"Total time for all questions: {total_time_for_all_questions}")
    # print(f"Total tokens generated: {total_tokens_generated}")
    # print(f"Total chars generated: {total_chars_generated}")
    # print(f"Average tokens per second: {total_tokens_per_second}")
    # print(f"Average chars per second: {total_chars_per_second}")

    # Save to CSV
    print(f"[INFO] Saving results to: {csv_filepath}")
    df.to_csv(csv_filepath, index=False)

    
