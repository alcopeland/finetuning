# Code from 'Finetuning Large Language Models' by DeepLearning.AI

# # Finetuning data: compare to pretraining and basic preparation
import jsonlines
import pandas as pd
from pprint import pprint

import datasets
from datasets import load_dataset

pretrained_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
instruction_dataset_df = pd.read_json('data2.jsonl', lines=True)

examples = instruction_dataset_df.to_dict()

prompt_template_qa = """### Question:
{question}

### Answer:
{answer}"""


prompt_template_q = """### Question:
{question}

### Answer:"""

num_examples = len(examples["question"])
finetuning_dataset_text_only = []
finetuning_dataset_question_answer = []
for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]

  text_with_prompt_template_qa = prompt_template_qa.format(question=question, answer=answer)
  finetuning_dataset_text_only.append({"text": text_with_prompt_template_qa})

  text_with_prompt_template_q = prompt_template_q.format(question=question)
  finetuning_dataset_question_answer.append({"question": text_with_prompt_template_q, "answer": answer})

pprint(finetuning_dataset_text_only[0])
pprint(finetuning_dataset_question_answer[0])


# ### Common ways of storing your data

with jsonlines.open(f'lamini_docs_processed.jsonl', 'w') as writer:
    writer.write_all(finetuning_dataset_question_answer)

finetuning_dataset_name = "lamini/lamini_docs"
finetuning_dataset = load_dataset(finetuning_dataset_name)
print(finetuning_dataset)

