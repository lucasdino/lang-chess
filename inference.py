import argparse
import subprocess
import utils as utils

# ======================================
# Arg parsing
# ======================================
def none_or_int(val):
    return None if val.lower() == "none" else int(val)

def parse_args():
    parser = argparse.ArgumentParser(description="Run vLLM evaluation.")

    # Model information 
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--model_version", type=str, default="llama3", help="Model version being run to ensure correct special tokens used.")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1/completions", help="Base URL for the model endpoint")
    parser.add_argument("--client_type", type=str, default="vllm", help="Type of client to use: 'vllm' or 'api'")
    parser.add_argument("--api_key", type=str, default=None, help="API key if using 'api' client type.")

    # Filenames and dirs
    parser.add_argument("--data_dir", type=str, default="data/cleaned", help="Path to the data directory")
    parser.add_argument("--data_files", nargs="+", default=None, help="List of data files to use (e.g., evals, rejsampling, train data)")
    
    # Various run details 
    parser.add_argument("--experiment_name", type=str, default='lang-chess-experiment', help="Give name for experiment for s3 bucket saving organization.")
    parser.add_argument("--run_type", type=str, default='eval', help="Specify which task you're doing (e.g., 'eval', 'rejsampling', 'hallucination', 'reasoning_strategy').")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of samples to pass into vLLM in each batch.")
    parser.add_argument("--max_samples", type=none_or_int, default=None, help="If set to None, use all your data in your --data-files; if set to int, use that as max number of samples to test on.")
    parser.add_argument("--num_generations", type=int, default=1, help="Choose number of generations for each sample.")
    
    # Logging / saving details
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Use wandb for logging")
    parser.add_argument("--print_verbose", default=False, action="store_true", help="Print all outputs.")
    parser.add_argument("--save_verbose", default=False, action="store_true", help="Save all outputs.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print all outputs.")

    # Inference hyperparams
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_p", type=float, default=0.02)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--reasoning_effort", type=str, default=None, help="Optional reasoning effort hint passed to system prompt.")

    # Other hyperparams
    parser.add_argument("--parser_max_reprompt", type=int, default=1)

    return parser.parse_args()



# ======================================
# Main run code
# ======================================
def main():
    args = parse_args()

    # Get our endpoint client setup
    if args.client_type == "vllm":
        client = utils.vLLMClient(
            model=args.model,
            base_url=args.base_url,
            generation_args={
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "min_p": args.min_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                **({"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {})
            }
        )
    elif args.client_type == "api":
        client = utils.APIClient(
            model=args.model,
            base_url=args.base_url,
            generation_args={
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "min_p": args.min_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                **({"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {})
            },
            api_key=args.api_key
        )
    else:
        raise(Exception(f"Client Type: {args.client_type} is undefined."))

    # For cases where we have data with 'correct answers' that we test against (e.g., evaluations, rejection sampling)
    if args.run_type in ["eval", "rejsampling"]:
        evaluator = utils.Evaluator(
            args = args,
            task_map = utils.TASK_MAP
        )
        print(f"Starting {args.run_type}...")
        results = evaluator.evaluate(client)
        print(f"Completed {args.run_type}.\n\nFinal Results:\n{results}")
    # For cases where we just want to generate data
    elif args.run_type in ["generate"]:
        generator = utils.Generator(
            args = args,
            task_map = utils.TASK_MAP
        )
        print(f"Starting {args.run_type}...")
        generator.generate(client)
        print(f"Completed {args.run_type}.")
    # For cases where we want the LLM to parse existing responses to extract more nuanced info (e.g., hallucinations, reasoning methods used)
    elif args.run_type in ['hallucination', 'reasoning_strategy', 'reasoning_quality']:
        llm_parser = utils.LLMParser(
            args = args,
            runtype_mapping = utils.RUNTYPE_SYSPROMPT_MAPPING
        )
        print(f"Starting {args.run_type}...")
        results = llm_parser.evaluate(client)
        print(f"Completed {args.run_type}.\n\nFinal Results:\n{results}")
    else:
        raise(Exception(f"Run Type: {args.run_type} is undefined."))

    # Optional: Save to S3 bucket (assumes aws cli is configured)
    # cmd = f"aws s3 cp {args.data_dir}/saved_data s3://llm-chess/saved_data/{args.experiment_name} --recursive"
    # print(f"S3 save command: {cmd}")
    # subprocess.run(cmd.split(), check=True)

    # Remove all files after saving
    # cmd = f"rm -f {args.data_dir}/saved_data/*"
    # print(f"Cleanup command: {cmd}")
    # subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    main()