# Training vs fine-tuning

There are two broad ways you train a model: The "classic" approach and the fine-tuning approach. This section
explains the differences. 


## Fine-Tuning

Fine-tuning is the current state-of-the-art approach. The main idea is that you take a pre-trained language model that 
consists of (hundreds of) millions of trained parameters. To this language model you add a simple prediction head with
randomly initialized weights. 

Since in this case, the vast majority of parameters in the model is already trained, you only need to "fine-tune" this
model. This means: Very small learning rate (LR) and just a few epochs. You are essentially just minimally modifying 
the model to adapt it to the task you want to solve.

Use this method by calling [`ModelTrainer.fine_tune()`](#flair.trainers.ModelTrainer.fine_tune).
Since most models in Flair were trained this way, this is likely the approach you'll want to use. 


## Training

On the other hand, you should use the classic training approach if the majority of the trainable parameters in your 
model is randomly initialized. This can happen for instance if you freeze the model weights of the pre-trained language 
model, leaving only the randomly initialited prediction head as trainable parameters. This training approach is also
referred to as "feature-based" or "probing" in some papers. 
 
Since the majority of parameters is randomly initialized, you need to fully train the model. This means: high learning 
rate and many epochs. 

Use this method by calling  [`ModelTrainer.train()`](#flair.trainers.ModelTrainer.train) .

```{note}
Another application of classic training is for linear probing of pre-trained language models. In this scenario, you 
"freeze" the weights of the language model (meaning that they cannot be changed) and add a prediction head that is 
trained from scratch. So, even though a language model is involved, its parameters are not trainable. This means that 
all trainable parameters in this scenario are randomly initialized, therefore necessitating the use of the classic
training approach.
```


## Paper 

If you are interested in an experimental comparison of the two above-mentioned approach, check out [our paper](https://arxiv.org/pdf/2011.06993) 
that compares fine-tuning to the feature-based approach.


## Next 

Next, learn how to load a [training dataset](how-to-load-prepared-dataset.md).