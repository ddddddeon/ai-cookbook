from collections import defaultdict

from openai import OpenAI

openai = OpenAI()

# file_obj = openai.files.create(
#     file=open("train_data.jsonl", "rb"),
#     purpose="fine-tune",
# )

# file_id = file_obj.id

# job = openai.fine_tuning.jobs.create(
#     training_file=file_id,
#     model="gpt-3.5-turbo-1106",
# )

# jobs = openai.fine_tuning.jobs.list()
# print(openai.fine_tuning.jobs.retrieve(jobs.data[0].id))

response = openai.chat.completions.create(
    model="ft:gpt-3.5-turbo-1106:personal::AlRL2cJb",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Is Chris cool?"}
    ]
)

print(response.choices[0].message.content)
