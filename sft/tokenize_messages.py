import multiprocessing
import os
import datasets
import transformers
import trl

os.environ['HF_HOME'] = '/tmp'

def tokenize_message_sequence(tokenizer, messages, max_length=None):
    # This is the existing logic for tokenizing a single sequence
    prompts = [messages[:i+1] for i, m in enumerate(messages) if m['role'] != 'assistant']
    responses = [messages[:i+1] for i, m in enumerate(messages) if m['role'] == 'assistant']

    prompt_ids = tokenizer.apply_chat_template(prompts, add_generation_prompt=True)
    response_ids = tokenizer.apply_chat_template(responses, add_generation_prompt=False)

    input_ids = response_ids[-1] if messages and messages[-1]['role'] == 'assistant' else prompt_ids[-1]

    roles = [m['role'] for m in messages]
    prompt_lens = [len(ids) for ids in prompt_ids]
    response_lens = [len(ids) for ids in response_ids]

    p_it = iter(prompt_lens)
    r_it = iter(response_lens)
    cum_lens = [next(r_it) if r == 'assistant' else next(p_it) for r in roles]

    completion_mask = []
    prev = 0
    for r, total in zip(roles, cum_lens):
        completion_mask.extend([int(r == 'assistant')] * (total - prev))
        prev = total

    if max_length is not None:
        input_ids = input_ids[:max_length]
        completion_mask = completion_mask[:max_length]

    return {
        'input_ids': input_ids,
        'completion_mask': completion_mask,
        'seq_lengths': [len(input_ids)],
    }

def pack_dataset(dataset, seq_length, map_kwargs={}):
    dataset = dataset.select_columns(['input_ids', 'completion_mask'])
    dataset = dataset.with_format("arrow")
    dataset = dataset.map(trl.data_utils._pack_bfd, batched=True, fn_kwargs={"seq_length": seq_length}, **map_kwargs)
    dataset = dataset.with_format(None)
    return dataset

qwen_no_think_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset', type=str, required=True)
    parser.add_argument('--output_dataset', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--job_id', type=int, default=0)
    parser.add_argument('--custom_chat_template', action='store_true', default=False)
    parser.add_argument('--pack', action='store_true', default=False)
    parser.add_argument('--pack_length', type=int, default=64000)
    parser.add_argument('--drop_too_long', action='store_true', default=False)
    
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.custom_chat_template:
        tokenizer.chat_template = qwen_no_think_template

    print(f"Loading dataset from {args.input_dataset}")
    if args.input_dataset.endswith('.jsonl'):
        dataset = datasets.load_dataset('json', data_files=args.input_dataset, split='train')
    else:
        dataset = datasets.load_from_disk(args.input_dataset)
    dataset = dataset.shuffle(seed=0)

    # split the dataset into n_jobs parts
    samples_per_job = len(dataset) // args.n_jobs
    start_idx = args.job_id * samples_per_job
    if args.job_id == args.n_jobs - 1:
        end_idx = len(dataset)
    else:
        end_idx = start_idx + samples_per_job
    dataset = dataset.select(range(start_idx, end_idx))

    # filter out examples where with fewer than 2 messages
    dataset = dataset.filter(lambda x: len(x['messages']) >= 3)

    # now, tokenize the dataset...
    print(f"Tokenizing dataset of size {len(dataset)}")
    tokenize_fnc = lambda x: tokenize_message_sequence(tokenizer, x['messages'], max_length=args.pack_length)
    dataset = dataset.map(tokenize_fnc, batched=False, num_proc=min(multiprocessing.cpu_count(), 32))

    # remove all columns except for 'input_ids', 'completion_mask', 'seq_lengths'
    columns_to_keep = ['input_ids', 'completion_mask', 'seq_lengths']
    dataset = dataset.select_columns(columns_to_keep)

    if args.drop_too_long:
        dataset = dataset.filter(lambda x: x['seq_lengths'][0] <= args.pack_length)

    if args.pack:
        dataset = pack_dataset(dataset, args.pack_length)

    print(f"Saving dataset to {args.output_dataset}")
    dataset.save_to_disk(args.output_dataset)

    print(f"Done")
