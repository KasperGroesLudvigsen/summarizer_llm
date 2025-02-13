# summarizer_llm
This is a project for fine tuning an LLM on [ThatsGroes/synthetic-dialog-summaries-processed-clean-chatml](https://huggingface.co/datasets/ThatsGroes/synthetic-dialog-summaries-processed-clean-chatml)

The main fine tuning script is found in `train.py`. This script takes a dictionary
of configurations as input. See examples in the `training_configs` directory.

The scripts starting with `train_` are scripts that load a specific config file
and calls the `train()` function in `train.py`.

See my LLM fine tuning notebook for a walkthrough of the training code and the related concepts:
[LLM fine tuning notebook](https://colab.research.google.com/drive/18jraZF_nEv462wr7L9sMqVk6l4VACKCl#scrollTo=5e5xTWH7BgKD)

