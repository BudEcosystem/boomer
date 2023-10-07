from dataclasses import dataclass, field
from typing import Optional
from itertools import chain

from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, HfArgumentParser, DataCollatorForSeq2Seq     
from datasets import load_dataset


IGNORE_INDEX = -100

@dataclass
class ModelArguments:
    base_model: Optional[str] = field(default="base-model")

@dataclass
class DataArguments:
    data_path: str = field(
      default=None, metadata={"help": "Path to the training data."}
    )
    max_seq_length = 4096


@dataclass
class TrainArguments(TrainingArguments):
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 1
    num_train_epochs = 3
    learning_rate = 2e-5
    fp16 = True
    logging_steps = 10
    optim = "adamw_torch"
    save_strategy = "epoch"
    output_dir = 'output'
    save_total_limit = 5
    report_to = 'wandb'
    adam_beta1 = 0.9
    adam_beta2 = 0.95

def load_data(tokenizer, dataset, max_length):
    

    def preprocess_pretrain_dataset(examples):

        text_ids = tokenizer(
            examples["text"],
            add_special_tokens=False)["input_ids"]

        concatenated_ids = list(chain(*text_ids))
        total_length = len(concatenated_ids)
        
        block_size = max_length
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of max_source_length
        result = [concatenated_ids[i: i + block_size]
                  for i in range(0, total_length, block_size)]
        
        return {
            "input_ids": result,
            "labels": result.copy()
        }
    
    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(tokenizer.decode([
            token_id if token_id != -100 else tokenizer.pad_token_id for token_id in example["labels"]
        ], skip_special_tokens=False)))
    
    column_names = dataset.column_names
    dataset = dataset.map(
        preprocess_pretrain_dataset,
        batched=True,
        remove_columns=column_names,
        num_proc=64
    )

    print_supervised_dataset_example(dataset[0])
    print(len(dataset))

    return {
        "train_dataset": dataset
    }


def train():
    
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    

    tokenizer = LlamaTokenizer.from_pretrained(model_args.base_model)
    tokenizer.pad_token_id = 0

    model = LlamaForCausalLM.from_pretrained(
        model_args.base_model,
        use_flash_attention_2=True
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total params: ", total_params)


    if data_args.data_path.endswith(".json") or data_args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_args.data_path)
    else:
        data = load_dataset(data_args.data_path)
    
    dataset =  load_data(tokenizer, data['train'], data_args.max_seq_length)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)


    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        data_collator=data_collator,
        **dataset
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=False)

    trainer.save_model()



if __name__ == "__main__":
    train()
