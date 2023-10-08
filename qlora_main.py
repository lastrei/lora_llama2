import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
from peft import get_peft_model, LoraConfig, set_peft_model_state_dict, prepare_model_for_kbit_training
import torch
from collections import defaultdict
from component.dataset import SFTDataset
import bitsandbytes as bnb
from component.loss import TargetLMLoss
from component.collator import SFTDataCollator
from component.trainer import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# os.environ["http_proxy"] = "http://127.0.0.1:10809"
# os.environ["https_proxy"] = "http://127.0.0.1:10809"
torch.cuda.empty_cache()


def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    # 统计可训练参数中，各种类型参数分布
    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


print("start")
tokenizer = AutoTokenizer.from_pretrained("YeungNLP/firefly-llama2-7b-base", trust_remote_code=True, use_fast=True)
print("tokenizer load complete")
# 部分tokenizer没有pad_token_id
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.unk_token_id
# 如果两者相同，模型训练时不会计算eos_token_id的loss
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    raise Exception('pad_token_id should not be equal to eos_token_id')

cutoff_len = 512

model = AutoModelForCausalLM.from_pretrained(
    "YeungNLP/firefly-llama2-7b-base",
    load_in_4bit=True,
    device_map='auto',
    torch_dtype=torch.float16,
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=2.0,
        llm_int8_has_fp16_weight=True,
    ),
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
print(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
target_modules = find_all_linear_names(model)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules
)

resume_from_checkpoint = "./outputs/lora_pretrain"
model = get_peft_model(model, config)
model.print_trainable_parameters()
model.config.torch_dtype = torch.float16

# 查看模型种各种类型的参数的情况
verify_model_dtype(model)
# 重新定义loss
loss_func = TargetLMLoss(ignore_index=tokenizer.pad_token_id)

if resume_from_checkpoint:
    # Check the available weights and load them
    checkpoint_name = os.path.join(
        resume_from_checkpoint, "pytorch_model.bin"
    )  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        resume_from_checkpoint = (
            False  # So the trainer won't try loading its state
        )
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")

train_dataset = SFTDataset("data/train.jsonl", tokenizer, 512)
data_collator = SFTDataCollator(tokenizer, 512)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    compute_loss=loss_func,
    tokenizer=tokenizer,
    data_collator=data_collator,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        num_train_epochs=1,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_32bit",
        save_strategy="steps",
        lr_scheduler_type="constant_with_warmup",
        save_steps=2000,
        output_dir="./outputs",
        save_total_limit=3,
        max_grad_norm=0.3,
        weight_decay=0,
        remove_unused_columns=False,
        disable_tqdm=False,
    ),
)
model.config.use_cache = False

trainer.train()
model.save_pretrained("./outputs/lora_pretrain")
