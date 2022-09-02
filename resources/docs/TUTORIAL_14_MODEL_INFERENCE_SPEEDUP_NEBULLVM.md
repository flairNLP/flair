# Speed up inference time of flair models with Nebullvm

**NOTICE:** The integration of nebullvm inside flair for now is only available for TransformerEmbeddings models.

This tutorial shows you how optimize a `TransformerDocumentEmbedding` or `TransformerWordEmbedding` using [Nebullvm](https://github.com/nebuly-ai/nebullvm), which allows to speedup
the inference time of the model.

## Introduction

This tutorial assumes that you have trained a model that you want to optimize, we also assume that the trained model uses transformer embeddings.
````python
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings

model = SequenceTagger.load("ner-large")
assert isinstance(model.embeddings, (TransformerWordEmbeddings, TransformerDocumentEmbeddings))
````

We need to define a few sentences that will be used by nebullvm to trace the model and optimize it.
Those sentences can be part of your dataset.
```python
from flair.data import Sentence

sentences = [
    Sentence("Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System."),
    Sentence("In the fourth century BCE, Aristotle noted that Mars disappeared behind the Moon during an occultation."),
    Sentence("Liquid water cannot exist on the surface of Mars due to low atmospheric pressure."),
    Sentence("In 2004, Opportunity detected the mineral jarosite."),
]
```

## Nebullvm optimization

If it's the first time that you run an optimization using nebullvm on this machine, we suggest that you run `import nebullvm` to install the deep learning compilers
required by the library. Be aware that this will take a few minutes, but it will be required only the first time.

```python
import nebullvm
```

You can then optimize the model with nebullvm with a simple line of code:
```python
model.embeddings = model.embeddings.optimize_nebullvm(sentences)
```
By running this cell, the embeddings are replaced by TransformerNebullvmEmbeddings, which ensure that the optimized model is used for predictions.

The `optimize_nebullvm` function can additionally take several optional parameters as input, that allow to customize the optimization:
- metric_drop_ths (float, optional): Maximum reduction in the
            selected metric accepted. No model with an higher error will be
            accepted, i.e. all optimized model having a larger error respect to
            the original one will be discarded, without even considering their
            possible speed-up. Default: None, i.e. no drop in metric accepted.
- metric (Union[Callable, str], optional): The metric to
            be used for accepting or refusing a precision-reduction
            optimization proposal. If none is given but a `metric_drop_ths` is
            received, the `nebullvm.measure.compute_relative_difference`
            metric will be used as default one. A user-defined metric can
            be passed as function accepting as inputs two tuples of tensors
            (produced by the baseline and the optimized model) and the related
            original labels.
            For more information see
            `nebullvm.measure.compute_relative_difference` and
            `nebullvm.measure.compute_accuracy_drop`. `metric`
            accepts as value also a string containing the metric name. At the
            current stage the supported metrics are `"numeric_precision"` and
            `"accuracy"`. Default: `"numeric_precision"`
- optimization_time (OptimizationTime, optional): The optimization time
            mode. It can be either 'constrained' or 'unconstrained'. For
            'constrained' mode just compilers and precision reduction
            techniques are used (no compression). 'Unconstrained' optimization
            allows the usage of more time consuming techniques as pruning and
            distillation. Note that for using many of the sophisticated
            techniques in the 'unconstrained' optimization, a small fine-tuning
            of the model will be needed. Thus we highly recommend to give as
            input_data at least 100 samples for when selecting 'unconstrained'
            optimization. Default: 'constrained'.
- dynamic_info (Dict, optional): Dictionary containing info about the
            dynamic axis. It should contain as keys both "inputs" and "outputs"
            and as values two lists of dictionaries where each dictionary
            represents the dynamic axis information for an input/output tensor.
            The inner dictionary should have as key an integer, i.e. the
            dynamic axis (considering also the batch size) and as value a
            string giving a "tag" to it, e.g. "batch_size". Default: None
- config_file (str, optional): Configuration file containing the
            parameters needed for defining the CompressionStep in the pipeline.
            Default: None.
- ignore_compilers (List, optional): List containing the compilers to be
            ignored during the OptimizerStep. Default: None.

You can find more details in the official nebullvm [documentation](https://nebuly.gitbook.io/nebuly/nebulgym/get-started) and on the [github page](https://github.com/nebuly-ai/nebullvm).

**KNOWN ISSUES:**
- Nebullvm optimization is not compatible for now with Pytorch versions 12.0 and 12.1 due to a Pytorch bug in the function that allows to export a model to onnx.
If you get an error similar to `RuntimeError: r INTERNAL ASSERT FAILED at "../aten/src/ATen/core/jit_type_base.h":545, please report a bug to PyTorch`, please try using Pytorch 1.11.

After the optimization, the usage for predictions is the same as before:
```python
sentence = Sentence('George Washington went to Washington.')

model.predict(sentence)
```

After performing the prediction, you can see the result:
```python
print(sentence)
# Expected output: Sentence: "George Washington went to Washington ." → ["George Washington"/PER, "Washington"/LOC]
```

## Results

With nebullvm the inference speed of the mode can be significantly improved, with this model we found the following results:

| Machine Type   | Baseline (s) | Nebullvm - optimized (s) | Speedup |
|----------------|--------------|--------------------------|---------|
| M1             | 0.181        | 0.0358                   | 5.1x    |
| Intel CPU      | 0.206        | 0.0953                   | 2,2x    |
| GPU (Tesla T4) | 0.0266       | 0.0129                   | 2.1x    |
