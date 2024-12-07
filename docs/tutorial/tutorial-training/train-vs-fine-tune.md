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

Most models in Flair are trained using fine-tuning. So this is likely the approach you'll want to use. 


## Training

On the other hand, you should use the classic training approach if the majority of the trainable parameters in your 
model is randomly initialized. This is essentially the "old way", before fine-tuning of transformers. 

Since the majority of parameters is randomly initialized, you need to fully train the model. This means: high learning 
rate and many epochs. 

Another application of classic training is for linear probing of pre-trained language models. In this scenario, you 
"freeze" the weights of the language model (meaning that they cannot be changed) and add a prediction head that is 
trained from scratch. So, even though a language model is involved, its parameters are not trainable. This means that 
all trainable parameters in this scenario are randomly initialized, therefore necessitating the use of the classic
training approach.

## Paper 

Our paper 