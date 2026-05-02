"""
Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
from the Accelerate example zoo https://huggingface.co/docs/accelerate/usage_guides/training_zoo
"""
import os
import shutil

from dataclasses import dataclass
import torch
import datasets


@dataclass
class MyArguments:
    model: str
    train_dataset_dir: str
    subsample: float = None
    shuffle: bool = True
    myseed: int = 0
    trust_remote_code: bool = False
    attn_implementation: str = "flash_attention_2"
    ckpt_every_steps: int = 0  # if >0, call final_save() every N steps to /<output_dir>/checkpoint-<N>/
    chat_token_only_loss: bool = False  # if True, mask labels to -100 except positions where target is a chat token (65536-65539)


def get_valid_checkpoints(output_dir):
    print(f"Checking for valid checkpoints in {output_dir}...")
    if not os.path.exists(output_dir):
        return []

    # get all subdirs
    subdirs = [os.path.join(output_dir, x) for x in os.listdir(output_dir)]
    subdirs = [x for x in subdirs if os.path.isdir(x)]

    # keep only those that contain the word 'checkpoint'
    subdirs = [x for x in subdirs if 'checkpoint' in x]

    # keep only those subdirs that contain a file that ends in .pt
    subdirs = [x for x in subdirs if any(y.endswith('.pt') for y in os.listdir(x))]

    return subdirs

def is_there_valid_checkpoint(output_dir):
    subdirs = get_valid_checkpoints(output_dir)
    return len(subdirs) > 0

def clean_up_checkpoints(output_dir, remove_all=False):
    """ remove_all = True removes the entire checkpoint folder, otherwise only the optimizer states are removed """
    # ensure there is at least one '.safetensors' file in the output dir (otherwise the model might not have been saved)
    if not any(x.endswith('.safetensors') for x in os.listdir(output_dir)):
        print(f"No .safetensors file found in {output_dir}, will not remove all")
        remove_all = False
    
    for subdir in os.listdir(output_dir):
        subidr_path = os.path.join(output_dir, subdir)
        if os.path.isdir(subidr_path) and 'checkpoint' in subdir:
            if remove_all:
                shutil.rmtree(subidr_path)
            else:
                # remove only .pt and .pth files
                for file in os.listdir(subidr_path):
                    if file.endswith('.pt') or file.endswith('.pth') or file.endswith('.bin'):
                        os.remove(os.path.join(subidr_path, file))

def final_save(trainer, output_dir, source_model_dir=None):
    """Save the SFT'd model to `output_dir` on /fast (Lustre), via /tmp + dd.

    Why this is bespoke:
      - HF save_pretrained on an FSDP-wrapped model resolves the auto_map via
        the FSDP wrapper class and copies _fsdp_*.py / _fully_shard.py from
        torch internals instead of our modeling code. Bypass it entirely.
      - HF save also saves in fp32 when params are fp32-internal under FSDP.
        We force bf16.
      - Direct write to Lustre is ~20 MB/s due to page cache; dd-direct from
        local /tmp is ~940 MB/s.

    All ranks must participate in `accelerator.get_state_dict()` (FSDP gather);
    only rank 0 does the file writes.
    """
    import os, json, subprocess, shutil, glob
    import torch
    from safetensors.torch import save_file

    state_dict = trainer.accelerator.get_state_dict(trainer.model)

    if trainer.accelerator.is_main_process:
        # Cast to bf16 (fp params -> bf16, integer/buffers untouched).
        sd = {
            k: v.to(torch.bfloat16).contiguous() if torch.is_floating_point(v) else v
            for k, v in state_dict.items()
        }
        del state_dict

        # FT_TRL_KEEP_LOCAL=1 skips the dd-to-/fast step; the bf16 model stays
        # at /tmp/<basename(output_dir)> for fast eval-iteration. Otherwise we
        # use a scratch /tmp/ft_trl_final_save.
        keep_local = bool(os.environ.get("FT_TRL_KEEP_LOCAL"))
        if keep_local:
            tmp_dir = os.path.join("/tmp", os.path.basename(output_dir.rstrip("/")))
        else:
            tmp_dir = "/tmp/ft_trl_final_save"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        # Single-file safetensors. ~26 GB for 13B bf16 — well under any limit.
        # metadata={"format": "pt"} is required by some transformers versions (TGI, vLLM)
        # for from_pretrained loading; safetensors.save_file omits it by default.
        print(f"[final_save] writing {len(sd)} tensors to {tmp_dir}/model.safetensors ...")
        save_file(sd, os.path.join(tmp_dir, "model.safetensors"), metadata={"format": "pt"})

        # Copy small files from the source model dir — gives us config.json
        # (with correct auto_map), modeling_talkie.py, tokenizer, etc.
        if source_model_dir:
            for fname in (
                "config.json",
                "configuration_talkie.py",
                "modeling_talkie.py",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "generation_config.json",
                "special_tokens_map.json",
            ):
                src = os.path.join(source_model_dir, fname)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(tmp_dir, fname))

        if keep_local:
            print(f"[final_save] FT_TRL_KEEP_LOCAL=1, leaving model at {tmp_dir}")
            print(f"[final_save] done -> {tmp_dir}")
        else:
            os.makedirs(output_dir, exist_ok=True)
            print(f"[final_save] dd-ing to {output_dir} ...")
            for fname in sorted(os.listdir(tmp_dir)):
                src = os.path.join(tmp_dir, fname)
                dst = os.path.join(output_dir, fname)
                if not os.path.isfile(src):
                    continue
                size = os.path.getsize(src)
                if size > 1 << 20:
                    subprocess.run(
                        ["dd", f"if={src}", f"of={dst}", "bs=64M",
                         "oflag=direct", "status=none"],
                        check=True,
                    )
                else:
                    shutil.copy(src, dst)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"[final_save] done -> {output_dir}")

    trainer.accelerator.wait_for_everyone()

if __name__ == "__main__":
    import os
    import torch
    import datasets
    from trl import TrlParser, SFTConfig, SFTTrainer

    class ChatPreservingSFTTrainer(SFTTrainer):
        """SFTTrainer that excludes embed.weight and lm_head from weight decay.

        Under --completion_only_loss, the chat-token rows of embed/lm_head
        (<|user|>, <|assistant|>, <|system|>) never receive useful gradient
        and <|end|> rarely does. With --weight_decay 0.1 those rows collapse
        from norm ~0.86 to ~0.12 over a few hundred steps via a feedback loop:
        wd shrinks every lm_head row, lm_head_gain.w_g compensates by growing
        for trained rows, the untrained chat rows have nothing to compensate
        with, so their effective magnitude in logit space (w_g * row) drops
        and the model never learns to emit <|end|>. Excluding embed/lm_head
        (and the per-output gains) from wd breaks the loop. See
        talkie-eval/CHAT_TOKEN_COLLAPSE.md for the full diagnosis.

        The "embed"/"lm_head" name patterns also catch embed_skip and
        lm_head_gain — both are learnable gain scalars that should not get
        weight decay either (canonical: skip wd on 1-D params).
        """
        def get_decay_parameter_names(self, model):
            decay = super().get_decay_parameter_names(model)
            excluded = sorted({n for n in decay if "embed" in n or "lm_head" in n})
            decay = [n for n in decay if "embed" not in n and "lm_head" not in n]
            print(f"[wd-skip] {len(excluded)} params excluded from weight decay: {excluded}")
            return decay

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # When chat_token_only_loss is on, restrict label positions to chat
            # tokens only (65536-65539). Composes with --completion_only_loss
            # which already sets labels to -100 outside assistant turns; this
            # further intersects with "target is a chat token", so loss flows
            # only at the ~one <|end|> position per assistant turn. Gradient
            # still propagates back through the whole model.
            #
            # Pass num_items_in_batch=None so the model's built-in CE does
            # mean-over-non-ignored. HF >=4.46 normally divides by total tokens
            # (including ignored), which would make our loss ~1000x smaller than
            # the per-token magnitude (1e-3 instead of ~5) since only ~16 of
            # 65536 positions contribute — that would attenuate the gradient
            # uselessly.
            if getattr(self, "_chat_token_only_loss", False):
                labels = inputs.get("labels")
                if labels is not None:
                    chat_ids = torch.tensor([65536, 65537, 65538, 65539], device=labels.device)
                    keep = torch.isin(labels, chat_ids)
                    inputs = {**inputs, "labels": torch.where(keep, labels, torch.full_like(labels, -100))}
                num_items_in_batch = None
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

    parser = TrlParser((SFTConfig, MyArguments))
    trl_args, my_args = parser.parse_args_and_config()

    if type(my_args.train_dataset_dir) == str and my_args.train_dataset_dir.endswith('.jsonl'):
        print(f"Loading dataset from {my_args.train_dataset_dir}")
        dataset = datasets.load_dataset('json', data_files=my_args.train_dataset_dir)['train']
    # check for dataset_info.json --> dataset
    elif os.path.exists(os.path.join(my_args.train_dataset_dir, 'dataset_info.json')):
        dataset = datasets.load_from_disk(my_args.train_dataset_dir)
    else:
        # must be that each subdir is a dataset (e.g., if it was tokenized with multiple jobs)
        subdirs = [os.path.join(my_args.train_dataset_dir, x) for x in os.listdir(my_args.train_dataset_dir)]
        print(f"Loading {len(subdirs)} datasets from {my_args.train_dataset_dir}")
        dataset = [datasets.load_from_disk(x) for x in subdirs]
        dataset = datasets.concatenate_datasets(dataset)

    if my_args.subsample is not None:
        dataset = dataset.select(range(int(len(dataset)*my_args.subsample)))

    if my_args.shuffle:
        dataset = dataset.shuffle(seed=my_args.myseed)
    
    trl_args.model_init_kwargs = {
        "torch_dtype": torch.bfloat16,
    }
    if my_args.attn_implementation and my_args.attn_implementation.lower() != "none":
        trl_args.model_init_kwargs["attn_implementation"] = my_args.attn_implementation
    trl_args.lr_scheduler_kwargs = {'min_lr_rate': 0.1}

    # When the model relies on `trust_remote_code` we cannot let SFTTrainer
    # take a path string: TRL's `create_model_from_path` calls AutoConfig
    # without that flag, then `getattr(transformers, arch)` which fails for
    # custom architectures. Pre-load and pass the instance, and globally
    # auto-trust so SFTTrainer's later AutoProcessor/AutoConfig calls succeed.
    model_obj = my_args.model
    if my_args.trust_remote_code:
        def _auto_trust(trust_remote_code, *a, **kw):
            return True
        for mod_name in (
            "transformers.dynamic_module_utils",
            "transformers.models.auto.configuration_auto",
            "transformers.models.auto.tokenization_auto",
            "transformers.models.auto.processing_auto",
            "transformers.models.auto.image_processing_auto",
            "transformers.models.auto.feature_extraction_auto",
        ):
            try:
                _m = __import__(mod_name, fromlist=["resolve_trust_remote_code"])
                if hasattr(_m, "resolve_trust_remote_code"):
                    _m.resolve_trust_remote_code = _auto_trust
            except ImportError:
                pass

        from transformers import AutoModelForCausalLM as _AMC
        load_kwargs = dict(trl_args.model_init_kwargs)
        load_kwargs["trust_remote_code"] = True
        model_obj = _AMC.from_pretrained(my_args.model, **load_kwargs)

    resume_from_checkpoint = is_there_valid_checkpoint(trl_args.output_dir)
    print(f"Size of the raw dataset: {len(dataset)}")

    # if packing is False set max_length to None
    if not trl_args.packing:
        trl_args.max_length = None

    trainer = ChatPreservingSFTTrainer(
        model=model_obj,
        train_dataset=dataset,
        args=trl_args,
    )
    trainer._chat_token_only_loss = bool(my_args.chat_token_only_loss)
    if trainer._chat_token_only_loss:
        print("[chat-token-only-loss] active: labels restricted to ids {65536, 65537, 65538, 65539}")

    if my_args.ckpt_every_steps and my_args.ckpt_every_steps > 0:
        from transformers import TrainerCallback

        class CustomCheckpointCallback(TrainerCallback):
            """Triggers final_save() at every N steps. Reuses the bespoke save
            path so intermediate checkpoints are bf16 + correct auto_map +
            dd-fast (or /tmp under FT_TRL_KEEP_LOCAL=1), unlike HF Trainer's
            default save which mishandles FSDP-wrapped custom architectures."""
            def __init__(self, every_n, output_dir, source_model_dir):
                self.every_n = every_n
                self.output_dir = output_dir
                self.source_model_dir = source_model_dir
                self.trainer = None

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step <= 0 or state.global_step % self.every_n != 0:
                    return
                ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
                if self.trainer.is_world_process_zero():
                    print(f"[ckpt_cb] saving step {state.global_step} -> {ckpt_dir}")
                final_save(self.trainer, ckpt_dir, source_model_dir=self.source_model_dir)

        cb = CustomCheckpointCallback(
            my_args.ckpt_every_steps, trl_args.output_dir, my_args.model
        )
        cb.trainer = trainer
        trainer.add_callback(cb)

    print(f"Size of the dataset prior to training: {len(trainer.train_dataset)}")

    trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )

    if trainer.is_world_process_zero():
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[ft_trl] rank0 peak GPU mem: {peak:.2f} GB")

    if not os.environ.get("FT_TRL_SKIP_FINAL_SAVE"):
        final_save(trainer, trl_args.output_dir, source_model_dir=my_args.model)

    if trainer.is_world_process_zero():
        clean_up_checkpoints(trl_args.output_dir, remove_all=False)
