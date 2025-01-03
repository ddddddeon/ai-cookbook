from collections import defaultdict
import click
import time
from openai import OpenAI

openai = OpenAI()


@click.command()
@click.option("--train", is_flag=True, help="Run the training process")
def main(train):
    model = ("ft:gpt-3.5-turbo-1106:personal::AlRL2cJb",)

    if train:
        file_obj = openai.files.create(
            file=open("./train_data.jsonl", "rb"),
            purpose="fine-tune",
        )

        file_id = file_obj.id

        job = openai.fine_tuning.jobs.create(
            training_file=file_id,
            model="gpt-3.5-turbo-1106",
        )

        while True:
            status = openai.fine_tuning.jobs.retrieve(job.id).status
            if status in ["succeeded", "failed"]:
                break
            print(f"Training status: {status} - waiting 30 seconds...")
            time.sleep(30)

        if status == "succeeded":
            model = openai.fine_tuning.jobs.retrieve(job.id).fine_tuned_model
            print(f"Training complete! Model ID: {model}")
        else:
            print("Training failed!")
            return

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Is Chris cool?"},
        ],
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
