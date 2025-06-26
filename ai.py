from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import pandas as pd
import os

def get_financial_chatbot():
    class FinancialChatbot:
        def __init__(self, model_name="gpt2"):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.chat_history = []

        def generate_response(self, prompt, max_length=200, temperature=0.7):
            full_prompt = "\n".join(self.chat_history + [f"User: {prompt}", "Assistant:"])
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            self.chat_history.append(f"User: {prompt}")
            self.chat_history.append(f"Assistant: {response}")
            self.chat_history = self.chat_history[-6:]
            return response

        def clear_history(self):
            self.chat_history = []

    return FinancialChatbot

def prepare_financial_data(csv_path, output_dir="financial_data", context_length=256):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError("CSV must contain 'question' and 'answer' columns")
    formatted_data = []
    for _, row in df.iterrows():
        formatted_data.append(f"User: {row['question']}")
        formatted_data.append(f"Assistant: {row['answer']}")
    split_idx = int(0.9 * len(formatted_data))
    train_data = formatted_data[:split_idx]
    val_data = formatted_data[split_idx:]
    train_path = os.path.join(output_dir, "train.txt")
    val_path = os.path.join(output_dir, "val.txt")
    with open(train_path, 'w') as f:
        f.write("\n".join(train_data))
    with open(val_path, 'w') as f:
        f.write("\n".join(val_data))
    return train_path, val_path

def fine_tune_model(model, tokenizer, train_path, val_path, output_dir="fine_tuned_model"):
    # Use datasets.load_dataset instead of deprecated TextDataset
    train_dataset = load_dataset('text', data_files={'train': train_path})['train']
    val_dataset = load_dataset('text', data_files={'validation': val_path})['validation']
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_steps=100,
        save_steps=1000,
        warmup_steps=100,
        prediction_loss_only=True,
        learning_rate=5e-5,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer

FinancialChatbot = get_financial_chatbot()
chatbot = FinancialChatbot()
print("Basic Chatbot Demo:")
print(chatbot.generate_response("What is the current stock price of Apple?"))
print(chatbot.generate_response("What about Microsoft?"))

# To fine-tune on your own data, use:
# train_path, val_path = prepare_financial_data("financial_qa.csv")
# chatbot.model, chatbot.tokenizer = fine_tune_model(chatbot.model, chatbot.tokenizer, train_path, val_path)
# print(chatbot.generate_response("What is EBITDA?"))