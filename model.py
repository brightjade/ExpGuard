from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from accelerate import PartialState


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        local_files_only=args.local_files_only,
    )
    tokenizer.chat_template = open(f"templates/{args.model_type.split('-')[0]}.jinja", "r").read()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(args):
    peft_config = None
    bnb_config = None
    device_string = PartialState().process_index

    if args.use_4bit_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=args.dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                load_in_8bit=args.use_8bit_quantization,
                quantization_config=bnb_config,
                torch_dtype=args.dtype if args.use_4bit_quantization or args.use_8bit_quantization or not args.do_train or args.attn_implementation == "flash_attention_2" else None,
                device_map={"": device_string},
                use_cache=False if args.do_train else True,
                attn_implementation=args.attn_implementation,
                cache_dir=args.cache_dir if args.cache_dir else None,
                local_files_only=args.local_files_only,
            )
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.05,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        if args.do_train:
            if args.use_4bit_quantization or args.use_8bit_quantization:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.use_gradient_checkpointing)

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    return model, peft_config
