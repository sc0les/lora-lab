# LoRA Lab

An unfinished local experiment with LoRA adapters, Hugging Face training,
conversation formatting, and an interactive adapter-backed chat loop.
This repository contains source only, not model weights or private training data.

## Layout

- `config.py`: model, adapter, training, and output settings.
- `utils/data_prep.py`: JSON dataset loading and conversation formatting.
- `train.py`: tokenizer/model loading and the Trainer-based training loop.
- `chat.py`: interactive inference using the saved adapter.
- `check_environment.py`: device and model-access diagnostic, not a package installer.
- `examples/training_data.json`: synthetic format example only.

## Experiment setup

Use a separate Python environment and install the packages in `requirements.txt`.
Versions are not pinned because a compatible training environment has not been
revalidated for this export. Choose hardware and model access appropriate to
`config.py`; the default is an 8B model loaded in float32.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
mkdir -p data results
cp examples/training_data.json data/training_data.json
# Review the compatibility notes below before downloading or training a model.
python check_environment.py
python train.py
python chat.py
```

## Known gaps

- The hand-written `[INST]` prompt format is not aligned with the configured
  Llama 3.1-derived model's native chat template. Correct this before training.
- The sample contains too little data for useful fine-tuning.
- There is no held-out evaluation or demonstrated quality improvement.
- Automatic device mapping and the chat loop's chosen input device can disagree.
- The base model's license and access conditions are separate from this code.

No training, model download, or inference run was performed for this export.
The dependency-free unit tests validate formatting with a stub dataset module;
they do not validate Hugging Face integration:

```sh
python -m unittest discover -s tests -v
```

## History

August-October 2026 commits are retrospective imports of existing local code.
Their assigned dates organize the import; they are not original development
timestamps. Export documentation and additional checks are dated when added.
