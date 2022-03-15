# Named Entity Recognition (NER)

The script `run_ner.py` can be used to fine-tuned a Transformer-based model for
Named Entity Recognition.

The following tables gives an overview of supported options for fine-tuning a model.

Mandatory parameters, that must be defined:

| Parameter            | Description
| -------------------- | ---------------------------------------------------------------------------------------------------------------
| `model_name_or_path` | Model name (valid model name on the Hugging Face [model hub](https://huggingface.co/)) or local path to a model
| `dataset_name`       | Valid Flair datasets name, as described [here](../../resources/docs/TUTORIAL_6_CORPUS.md)

The (optional) argument `dataset_arguments` can be used to pass special options to a Flair dataset, such as defining a special language
for a dataset, that consists of multiple corpora.

All other options incl. their default value:

| Parameter                 | Default                 | Description
| ------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------
| `layers`                  | `-1`                    | Layers to be fine-tuned
| `subtoken_pooling`        | `first`                 | Subtoken pooling strategy used for fine-tuned (could be: `first`, `last`, `first_last` or `mean`)
| `hidden_size`             | `256`                   | Hidden size for NER model
| `use_crf`                 | `False`                 | Whether to use a CRF on-top of NER model or not
| `num_epochs`              | `10`                    | The number of training epochs
| `batch_size`              | `16`                    | Batch size used for training
| `learning_rate`           | `5e-05`                 | Learning rate OneCycleLR scheduler
| `seed`                    | `42`                    | Seed used for reproducible fine-tuning results
| `device`                  | `cuda:0`                | CUDA device string
| `weight_decay`            | `0.0`                   | Weight decay for optimizer
| `embeddings_storage_mode` | `none`                  | Defines embedding storage method
| `output_dir`              | `resources/taggers/ner` | Defines output directory for final fine-tuned model

To use the recently introduced [FLERT](https://arxiv.org/abs/2011.06993) the following parameters can be set:

| Parameter                     | Default | Description
| ----------------------------- | ------- | ----------------------------------------------
| `context_size`                | `0`     | Context size (`0` means no additional context)
| `respect_document_boundaries` | `False` | Whether to respect document boundaries or not

# Example

The following example shows how to fine-tune a model for the recently released [Masakhane](https://arxiv.org/abs/2103.11811) dataset for
the Luo language. We choose XLM-RoBERTa Base for fine-tuning. In this example, the best model (choosen on performance on development set)
is used for final evaluation on the test set.

## Choosing the dataset

The Masakhane dataset was recently integrated into Flair and is accessible through the `NER_MASAKHANE` object. The `dataset_name` then is
also `NER_MASAKHANE`.

As we want to use the Luo language subset of Masakhane dataset, we need to specify it via the `languages` argument of the `NER_MASAKHANE`
class. The final value of the `dataset_arguments` then is `{"languages": "luo"}`.

## Choosing the training hyper-parameters

We use the same hyper-parameters for fine-tuning XLM-RoBERTa Base as specified in the Maskakhane paper:

| Hyper-parameter    | Value
| ------------------ | -------
| `batch_size`       | `32`
| `learning_rate`    | `5e-05`
| `num_epochs`       | `50`

## Fine-Tuning

The following command starts fine-tuning of the XLM-RoBERTa Base model on the Luo subcorpus of the Masakhane dataset:

```python
python3 run_ner.py \
    --dataset_name NER_MASAKHANE \
    --dataset_arguments '{"languages": "luo"}' \
    --model_name_or_path xlm-roberta-base \
    --batch_size 32 \
    --learning_rate 5e-05 \
    --num_epochs 50 \
    --output_dir masakhane-luo-ner-base
```

The final output (incl. evaluation results) could look like:

```bash
Results:
- F-score (micro) 0.7538
- F-score (macro) 0.7246
- Accuracy 0.6162

By class:
              precision    recall  f1-score   support

         PER     0.8626    0.7958    0.8278       142
         LOC     0.7727    0.7969    0.7846       128
        DATE     0.7458    0.6377    0.6875        69
         ORG     0.6230    0.5758    0.5984        66

   micro avg     0.7755    0.7333    0.7538       405
   macro avg     0.7510    0.7015    0.7246       405
weighted avg     0.7752    0.7333    0.7529       405
 samples avg     0.6162    0.6162    0.6162       405
```

For this example we could reach a F1-score of 75.38% with is on-par with the result (74.86%) for this language in the Masakhane paper.

## Fine-Tuning using FLERT approach

In the example, we extend our fine-tuning steps and use an additional context window of 64 tokens, as it was proposed in the
[FLERT](https://arxiv.org/abs/2011.06993) paper.

As the Masakhane dataset does not provide any document boundaries, we leave this option unused.

Then the following command can be used to start fine-tuning on the Luo subcorpus of the Masakhana dataset using the FLERT approach:

```python
python3 run_ner.py \
    --dataset_name NER_MASAKHANE \
    --dataset_arguments '{"languages": "luo"}' \
    --model_name_or_path xlm-roberta-base \
    --batch_size 32 \
    --learning_rate 5e-05 \
    --num_epochs 50 \
    --context_size 64 \
    --output_dir masakhane-luo-ner-flert
```

The final output on the test set could look like:

```bash
Results:
- F-score (micro) 0.7793
- F-score (macro) 0.7418
- Accuracy 0.657

By class:
              precision    recall  f1-score   support

         PER     0.9058    0.8803    0.8929       142
         LOC     0.7698    0.8359    0.8015       128
        DATE     0.7302    0.6667    0.6970        69
         ORG     0.5758    0.5758    0.5758        66

   micro avg     0.7783    0.7802    0.7793       405
   macro avg     0.7454    0.7397    0.7418       405
weighted avg     0.7791    0.7802    0.7789       405
 samples avg     0.6570    0.6570    0.6570       405
```

The final result on test set shows a +2,55% performance boost compared to the baseline model.

## Optional: Using configuration files

It is also possible to store all parameters in a JSON configuration file. For the example on the Masakhane NER dataset, such a
configuration file would look like:

```json
{
    "dataset_name": "NER_MASAKHANE",
    "dataset_arguments": "{\"languages\": \"luo\"}",
    "model_name_or_path": "xlm-roberta-base",
    "batch_size": 32,
    "learning_rate": 5e-05,
    "num_epochs": 50,
    "output_dir": "masakhane-luo-ner-base-2"
}
```
