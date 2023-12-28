# Writing a good issue

You are likely reading this, because you want to create an issue.
This is great, issues are a great way to provide feedback which then can be used to enhance the library.
Here are some guidelines to make the issue as insightful as possible.

## Issue types

Before you start with the issue, you need to choose the type.
There are 3 issue types:

* **Bug Report** -> You have noticed something that doesn't work the way it is expected.
* **Feature/Enhancement request** -> You have an idea for something that would make flair better.
* **Question** -> You have a question that is unrelated to a potential bug or feature request.

### Bug Report

A bug report is one of the most common issues. It is simple: you tried something, but it didn't work as expected.
It is important to provide as much context as possible, so ensure that you ran the [collect_env.py](https://github.com/flairNLP/flair/blob/master/collect_env.py) and if required  and created a minimal reproducible example.
The minimal reproducible example has, like the name says two properties:
* it is reproducible
* it is as small as possible

**Reproducibility**

Please ensure that we can really reproduce your issue.

You might have encountered the issue while training on your custom dataset and don't want to share it. That is ok,
but maybe you can test if you can recreate the same bug by using one of the manny public datasets instead and if not,
maybe filter the problem down to a single sentence and report what property it has. 

It is also possible, that you have encountered the issue while predicting some sentences. Maybe you don't want to share
your trained model, but maybe you can recreate the issue by creating a model without training it?

Please, be sure to not add local paths, or load any data that others have no access.

**Minimal**

After ensuring reproducibility, please also take some time to make it minimal. That way, we can quicker understand
what the issue is and won't need to spend time debugging code that is unrelated to the issue.

For example, you might get an error where the stack trace shows that it occurred while saving the model. In that case,
you can verify, if the model really needs to be trained on the full dataset for 100 epochs and test if it instead would be enough
to just create a model and save it with no training involved.

### Feature/Enhancement request

For a Feature/Enhancement request, please provide not only the *what* but also the *why*, it is easier to judge how important a feature is,
when you know why it is wanted and what it could provide to the users.

### Question

Questions are the most generic types of issues, but also those whose usually lack most of the context.
Please ensure that you are not creating a Question that should actually be a bug report.

For example issues like: `[Question]: Something is wrong with ...`, `[Question]: sentence.to_dict(tag_type='ner') no longer have ...`
or `[Question]: MultiTagger cannot be loaded...` are examples for issues that clearly should be bug reports instead and
could have been resolved quicker, if enough context and a minimal reproducible example were provided.

