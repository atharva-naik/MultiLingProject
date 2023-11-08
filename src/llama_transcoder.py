import json
import os
from vllm import LLM
from tqdm import tqdm
from datautils.llm_utils import RequestOutputV2
import huggingface_hub
from vllm.sampling_params import SamplingParams

access_token = os.environ.get("ACCESS_TOKEN")

if access_token:
	print("Access Token:", access_token)
else:
	print("Access Token not found in environment variables.")
	exit()

prompt_template = """Convert the following from {source_language} code to {target_language} code:

SOURCE:
{source_program}

TARGET:

"""

huggingface_hub.login(token=access_token)

def read_jsonl(filename):
	data = []
	with open(filename) as f:
		for line in f:
			data.append(json.loads(line.strip()))
	return data

def write_jsonl(filename, data):
	with open(filename, "w+") as file:
		for item in data:
			json_str = json.dumps(item)
			file.write(json_str + '\n')


if __name__ == "__main__":
	llm = LLM("codellama/CodeLlama-13b-hf")
	dataset = read_jsonl("../dataset/transcoder_test.jsonl")
	results = []
	for sample in tqdm(dataset[:1], desc="Processing"):
		source_language = sample["source"]
		target_language = sample["target"]
		source_program = sample["source_program"]
		target_program = sample["target_program"]
		SAMPLE_PROMPT = prompt_template.format(source_language=source_language, target_language=target_language, source_program=source_program, target_program=target_program)
		print(SAMPLE_PROMPT)
		res_op_list = llm.generate(SAMPLE_PROMPT, SamplingParams(max_tokens=400))
		transformed_output = [RequestOutputV2.from_v1_object(res_op).to_json() for res_op in res_op_list]
		for x in transformed_output:
			x["target"] = target_program
		results.extend(transformed_output)
	write_jsonl("../results/test_llama.jsonl", results)    
