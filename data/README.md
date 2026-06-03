# Data Directory

This directory contains the datasets used for training and fine-tuning the model.

## Available Datasets

- **TinyShakespeare (`tinyshakespeare.txt`)**: A classic character-level language modeling dataset containing works of Shakespeare. Often used for initial pre-training testing.
- **Natural Questions (NQ)**:
  - `NQ-open-train.txt`: Training set for open-domain questions.
  - `NQ-open-dev.txt`: Development/validation set.
  - `NQ-train_pairs.txt`: Training pairs for QA fine-tuning.
- **Conversation (`Conversation.csv`)**: Conversational dataset used for chatbot fine-tuning (SFT/Instruction tuning).
- **Literature (`literature-condensed.txt`)**: A condensed corpus of literature for general language pre-training.

## References
- [Tiny Stories](https://huggingface.co/datasets/roneneldan/TinyStories) (Linked for reference, but not currently in standard text format here)