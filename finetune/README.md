# GPT Fine-tuning Project

This project demonstrates fine-tuning GPT-3.5-turbo using OpenAI's API to create a specialized assistant that is extremely enthusiastic about Chris's coolness.

## Usage

```bash
pip install openai click
```

To run the assistant using the pre-trained model:
```bash
python main.py
```

To train a new model (requires train_data.jsonl in the finetune directory):
```bash
python main.py --train
```
