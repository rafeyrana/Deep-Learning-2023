# %% [markdown]
# # CS437 PA4 Part 3 - Abstractive Text Summarization with T5 [20 marks]
# 
# Roll Number: 24100103
# 
# Name: Rafey Rana

# %% [markdown]
# ![Abstractive Summarization](./assets/abstractive.png)

# %% [markdown]
# # Introduction and HuggingFace
# 
# Summarization is the task of generating a shorter version of a longer text, while preserving its essential meaning. The goal of summarization is to make it easier for readers to understand and remember the main points of a text. There are two main approaches to summarization: extractive and abstractive.
# 
# Extractive summarization involves *selecting* the most important sentences or phrases from the original text and concatenating them to form a summary. Extractive summarization may result in summaries that are disjointed or fail to capture the overall meaning of the original text.
# 
# Abstractive summarization, on the other hand, involves *generating* a summary that may contain new phrases or sentences not present in the original text. This approach requires a deeper understanding of the text and the ability to generate coherent and grammatically correct sentences. Abstractive summarization has the potential to generate more informative and readable summaries, but it is also more challenging and less mature than extractive summarization.
# 
# In recent years, deep learning models have shown significant progress in abstractive summarization. T5, a transformer-based language model developed by Google, has shown promising results in various natural language processing tasks, including abstractive summarization. In this part, we will explore the use of T5 for abstractive summarization and evaluate its performance on a dataset of *dialogues* (the SAMSum dataset).
# 
# We will be using HuggingFace in this part. It is a framework, rather similar to PyTorch, but specialized for using and training Transformer models. It has a very rich ecosystem, provides a huge variety of models, and you will likely find something to suit your use case very easily here (note that these are BIG models, so try not to run them on an ancient computer).
# 
# As you may see in the cell below, we have imported things like `AutoModelForSeq2SeqLM` and `AutoTokenizer` from HuggingFace. These classes provide an easy interface for us to instantiate models, along with the Tokenizers that work the best **with those** models. Since this is a **framework** and not a simple library, one has to play by its rules: it expects things to be done in a very specific way.
# 
# **Important:** This part expects you to be very comfortable with reading documentation. This means being able to google how each of the classes work, what kinds of methods are available, how to go about doing specific things like setting up a Trainer etc. The documentation is very good, and there are many forum posts for all sorts of issues.
# 
# To familiarize yourself with the tool, start with the documentation [here](https://huggingface.co/docs/transformers/index) and with this introductory video [here](https://youtu.be/QEaBAZQCtwE). If you're stuck, google it.

# %%
# Installs
!pip install -q evaluate py7zr rouge_score absl-py

# Imports here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

import torch
import torch.nn as nn

import datasets
!pip install transformers
import transformers
from transformers import (
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        AutoTokenizer
)
import evaluate

# Quality of life fixes
import warnings
warnings.filterwarnings('ignore')
from pprint import pprint

import os
os.environ["WANDB_DISABLED"] = "true"

from IPython.display import clear_output

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")
print(f"Evaluate version: {evaluate.__version__}")

# Get the samsum dataset
samsum = datasets.load_dataset('samsum')
clear_output()
print("Setup done!")

# %%
# What does this dataset object look like?
samsum

# %% [markdown]
# Now that we've downloaded the dataset from HuggingFace and examined it, we see that it has already been conveniently split for us. There are over 14000 instances in the Training Set alone.
# 
# Let's look at some examples. Note how the indexing works here.

# %%
# Print out one sample from the training set
rand_idx = np.random.randint(0, len(samsum['train']))

print(f"Dialogue:\n{samsum['train'][rand_idx]['dialogue']}")
print('\n', '-'*50, '\n')
print(f"Summary:\n{samsum['train'][rand_idx]['summary']}")

# %% [markdown]
# ## Step 1: Preprocessing the Data [5 Marks]

# %% [markdown]
# The model we will be using is **T5** (the **Text-To-Text Transfer Transformer**). It is a general-case transformer that has been pretrained on nearly 800GB of text data. It's goal is to simply generate text given a prompt. The utility of this model comes in how we can prompt it to perform a large variety of tasks, by altering the input to it.
# 
# ![The T5 Model](./assets/T5.png)
# 
# For example, you could have it translate from German to English by applying the prefix in the image above. What we're interested in is having it summarize text for us. We can do this by simply prepending a `summarize:` prefix before each of the inputs, and simply have it generate what comes next.

# %%
model_ckpt = 't5-small'

# TODO: Create the Tokenizer (hint: use the AutoTokenizer pretrained checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# %% [markdown]
# As you may recall, one significant problem we face in NLP problems is packaging everything into Tensors. This is because of variable length inputs and/or targets. 
# 
# We deal with this by setting a threshold, and truncating instances longer than that, and padding instances shorter than that.
# 
# In the following cell, we will 
# 1. Concatenate the Train and Test portions of our dataset
# 2. Tokenize them
# 3. Find out the max lengths for both the inputs and the outputs (which are both sequences)
# 
# Doing this will help us be precise about what shapes we expect the data to be in. We could guess a large enough number for both measures which would save us some time, but then we'd have to explore the data's distribution to find a nice enough value like the median anyway.
# 
# **Note:** If you're uncomfortable because of Data Leakage happening here, don't be (we *are* performing this grave sin, but we don't care).

# %%
from datasets import concatenate_datasets
# Find the max lengths of the source and target samples
# The maximum total input sequence length after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([samsum["train"], samsum["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([samsum["train"], samsum["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")

# %% [markdown]
# Now that we have the max lengths for the source and targets, we can move to preprocessing our actual dataset. This is simply
# 1. Adding in the `summarize:` prefix to our dialogues
# 2. Specifying the max source length to allow for padding and/or truncation
# 3. Specifying the max target length to allow for padding and/or truncation
# 4. Indicate to the model to IGNORE the Padding tokens in the targets (otherwise it would try to learn those patterns, though we did it just for a preprocessing step)
# 5. Map this function across the whole dataset

# %%
def preprocess_function(
    sample, 
    padding="max_length", 
    max_source_length=max_source_length,
    max_target_length=max_target_length
):
    '''
    A preprocessing function that will be applied across the dataset.
    The inputs and targets will be tokenized and padded/truncated to the max lengths.

    Args:
        sample: A dictionary containing the source and target texts (keys are "dialogue" and "summary") in a list.
        padding: Whether to pad the inputs and targets to the max lengths.
        max_source_length: The maximum length of the source text.
        max_target_length: The maximum length of the target text.
    '''
    
    # TODO: Add prefix to the input for t5
    inputs = ["summarize: " + d for d in sample["dialogue"]]

    # TODO: Tokenize inputs, specifying the padding, truncation and max_length
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # TODO: Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample['summary'], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    # Format and return
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# TODO: Map this preprocessing function to our datasets using .map on the samsum variable
# inside .map, setup the following params: (batched=True, remove_columns=["dialogue", "summary", "id"])
tokenized_dataset = samsum.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# %%
# def preprocess_function(
#     sample, 
#     padding="max_length", 
#     max_source_length=max_source_length,
#     max_target_length=max_target_length
# ):
#     '''
#     A preprocessing function that will be applied across the dataset.
#     The inputs and targets will be tokenized and padded/truncated to the max lengths.

#     Args:
#         sample: A dictionary containing the source and target texts (keys are "dialogue" and "summary") in a list.
#         padding: Whether to pad the inputs and targets to the max lengths.
#         max_source_length: The maximum length of the source text.
#         max_target_length: The maximum length of the target text.
#     '''
    
#     # Add prefix to the input for t5
#     inputs = "summarize: " + sample["dialogue"]

#     # Tokenize inputs, specifying the padding, truncation and max_length
#     model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

#     # Tokenize targets with the `text_target` keyword argument
#     labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

#     # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss
#     if padding == "max_length":
#         labels["input_ids"] = [
#             [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
#         ]

#     # Format and return
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs


# # Map this preprocessing function to our datasets using .map on the samsum variable
# # inside .map, setup the following params: (batched=True, remove_columns=["dialogue", "summary", "id"])
# tokenized_dataset = samsum.map(
#     preprocess_function,
#     batched=True,
#     remove_columns=["dialogue", "summary", "id"]
# )
# print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")


# %% [markdown]
# ## Step 2: Creating a Metric for Evaluation [5 Marks]

# %% [markdown]
# So how do we actually measure how well a model performs on a task like this?
# 
# One fancy metric is the **ROUGE-score**. You can read more about it [here](https://www.freecodecamp.org/news/what-is-rouge-and-how-it-works-for-evaluation-of-summaries-e059fb8ac840/). TLDR: it is a robust way to measure how good a summary is without having to match the tokens in the summary to the tokens inside the input text (which would work better for Extractive Summarization).

# %%
# Load in the ROUGE metric
metric = evaluate.load("rouge")
clear_output()

# %% [markdown]
# Since we are working within a framework, the model will not only output it's results in a form that requires extra processing to display it nicely, but the ROUGE object we're using also has it's own peeves about how inputs to it should be structured. Specifically, it expects a newline after each sentence. On top of this we need to be careful for ignoring the PAD tokens in the model output and the labels (recall we set that token to be `-100` numerically).
# 
# When this is done, we can invoke the `compute()` method of the metric object and get a nice collection of the results.

# %%
def postprocess_text(preds, labels):
    '''
    A simple post-processing function to clean up the predictions and labels

    Args:
        preds: List[str] of predictions
        labels: List[str] of labels
    '''
    
    # TODO: strip whitespace on all sentences in preds and labels
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    
    # Fetch the predictions and labels
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Decode the predictions back to text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing for ROUGE
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # TODO: Compute ROUGE on the decoded predictions and the decoder labels
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# %% [markdown]
# ## Step 3: Creating and Training the Model [5 Marks]

# %% [markdown]
# One could try to play with the T5 model now, and see how it performs on unseen samples. You may find that the model performs surprisingly well already (because of how it has been trained before). However, we want to specialize it for our use-case: providing summaries of conversations between two people.
# 
# To do this, we have to *fine-tune* (or simply "train") the model on the SAMSum dataset that we saw above.
# 
# HuggingFace makes this very easy - all one has to do is specify the hyperparameters, which model and tokenizer to use, along with the datasets and evaluation metrics. It will handle everything from there.

# %%
# TODO: Bring in the model (look into the AutoModelForSeq2SeqLM class and use the model_ckpt variable)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

clear_output()

# %% [markdown]
# Let's go back to the processing phase for a second. Another thing we should do is to create a Data Collator: think of this as a utility that will batch together outputs and labels to perform operations like Padding, and Augmentations (like random masking of tokens).
# 
# We could still live without this, but we'd like to set things up nicely for the training phase. Take my word: it **really** helps out.

# %%
# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100

# Create the Data Collator, specifying the tokenizer, model, and label_pad_token_id
# Also set pad_to_multiple_of=8 to speed up training
data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

# %% [markdown]
# Now to actually train the model. 
# 
# We can setup the Training Arguments/Hyperparameters and create a `Trainer` object to handle everything for us.

# %%
import logging
logging.getLogger("transformers").setLevel(logging.WARNING)


# Define training hyperparameters in Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_samsum", # the output directory
    logging_strategy="epoch",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=5,
    predict_with_generate=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_steps=50,
    logging_first_step=False,
    fp16=False
)

# Hint: just index into the tokenized_dataset variable to get the training and validation data
# Hint 2: if you want to speed training up, you can use a smaller subset of the data (call .select(num_samples) on the datasets)
training_data = tokenized_dataset['train']
eval_data = tokenized_dataset['validation']

# TODO: Create the Trainer for the model
trainer = Seq2SeqTrainer(
    model=model,    # the model to be trained
    args=training_args, # training arguments we just defined
    train_dataset= training_data, # the training dataset
    eval_dataset=eval_data, # the validation dataset
    tokenizer= tokenizer, # the tokenizer we used to tokenize our data
    compute_metrics=compute_metrics, # the function we defined above to compute metrics
    data_collator= data_collator# the data collator we defined above
)

# %%
# Train the model (this will take a while!)
results = trainer.train()
clear_output()
pprint(results)

# %% [markdown]
# ## Step 4: Evaluation and Inference [5 Marks]
# 
# Now that we're done with fine-tuning the model, we can evaluate it on the Test Set based on the ROUGE score.

# %%
# TODO: use the trainer to evaluate the model since we defined our metric function there
# hint: call .evaluate() on the trainer
res = trainer.evaluate()
clear_output()

# %%
# Format the results dictionary nicely 
cols = ["eval_loss", "eval_rouge1", "eval_rouge2", "eval_rougeL", "eval_rougeLsum"]
filtered_scores = dict((x, res[x]) for x in cols)
pd.DataFrame([filtered_scores], index=[model_ckpt])

# %% [markdown]
# More importantly, let's write a function that will take in raw text that we provide, perform the necessary processing steps on it, and have the model generate the summary for us!
# 
# We can make our lives *even* simpler. A wonderful utility that HuggingFace provides is the `pipeline` ([docs](https://huggingface.co/docs/transformers/main_classes/pipelines)). All we have to do is specify
# * The Task
# * The Model
# * The Tokenizer,
# and it handles everything for us!
# 
# Now you might be thinking: "why didn't we do this at the start?"
# 
# Let's pretend you didn't think that.

# %%
from transformers import pipeline

summarizer_pipeline = pipeline("summarization",
                               model=model, 
                               tokenizer=tokenizer,
                               device=0)

# %%
## Run this cell to test the model out on a random sample from the test set (which the model HASN'T seen yet)

rand_idx = np.random.randint(low=0, high=len(samsum["test"]))
sample = samsum["test"][rand_idx]

dialog = sample["dialogue"]
true_summary = sample["summary"]

model_summary = summarizer_pipeline(dialog)
clear_output()

print(f"Dialogue: {dialog}")
print("-"*25)
print(f"True Summary: {true_summary}")
print("-"*25)
print(f"Model Summary: {model_summary[0]['summary_text']}")
print("-"*25)      

# %%
# TODO: Generate a summary for a random sample from the test set
def create_summary(input_text, model_pipeline=summarizer_pipeline):
    '''
    A function to generate a summary for a given input text.
    '''
    
    summary = summary = model_pipeline(input_text)[0]["summary_text"]
    
    return summary

text = '''
Batman: Where is he?
Joker: You have a little fight in you. I like that.
Batman: Then you're going to love me.
Joker: You're a real comedian. You know that, Batsy?
Batman: (ignores Joker's comment) Where are they?
Joker: You know, for a while there, I thought you really were a dented, angry little freak. The way you threw yourself after her.
Batman: Look at me!
Joker: You've got your hands full tonight, huh? Cops or robbers, which one are you going to stop first?
Batman: (grabs Joker) I said, where are they?
Joker: You know, you remind me of my father. I hated my father.
Batman: (slams Joker into the table) Don't talk about my parents!
Joker: (laughs) And why, Batman? Why, why, why would he do that? When he could be doing this?
(Joker slams his head into Batman's, causing Batman to stumble back)
Joker: It's okay, I'm not gonna hurt you. I'm just gonna bash your brains in. (laughs)
'''

print(f"Original Text:\n{text}")
print('\n', '-'*50, '\n')

summary = create_summary(text)

print(f"Generated Summary:\n{summary}")

# %% [markdown]
# Hopefully you had some fun with this. If you want to try to get a better model you can:
# 1. Incorporate a larger training set (if you trained with fewer samples)
# 2. Train for more epochs
# 3. Try out a different, larger model! It's as simple as changing the `model_ckpt` variable and running the remaining cells just the same!


