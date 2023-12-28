# Speed up prediction speed of Transformer Embeddings

This tutorial shows you, how to export a `TransformerDocumentEmbedding` or `TransformerWordEmbedding` to [ONNX](https://onnxruntime.ai/) or to use [JIT and TorchScript](https://pytorch.org/docs/stable/jit.html).
Both are ways to speed up the model prediction time and find common use in production environments.

## Assumption

This tutorial assumes, that you have trained a model that you want to put into production we also assume that the trained model uses transformer embeddings.
````python
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings

model = SequenceTagger.load("ner-large")
assert isinstance(model.embeddings, (TransformerWordEmbeddings, TransformerDocumentEmbeddings))
````

In both ways, we need to define a few sentences that will be used to trace the operations of the embeddings.
Those sentences can be part of your dataset.
```python
from flair.datasets import CONLL_03
sentences = list(CONLL_03().test)[:5]
```

## TransformerOnnxEmbeddings

To use OnnxEmbeddings, you need to install any [execution provider](https://onnxruntime.ai/docs/execution-providers/) and the onnxruntime:
which can be done via `pip install onnxruntime`

To export the OnnxEmbeddings there is only one line to run:
```python
model.embeddings = model.embeddings.export_onnx("flert-embeddings.onnx", sentences, providers=["CUDAExecutionProvider", "CPUExecutionProvider"], session_options={})
```
This creates a file `flert-embeddings.onnx` which stores the exported Onnx Model. Besides that, the embeddings are replaced by `TransformerOnnxEmbeddings` which ensure, that the created Onnx model is used for predictions.
The providers referenced are part of your production environment and are documented [here](https://onnxruntime.ai/docs/execution-providers/)
You can provide SessionOptions documented [here](https://onnxruntime.ai/docs/api/python/api_summary.html#sessionoptions), by passing each property you want to set as key in the `session_options` dictionary

The usage for predictions is the same as before:
```python
model.predict(sentences)
```
### Optimization

To use the optimization, in addition to `onnx-runtime`, also `coloredlogs` and `onnx` need to be installed, this can be done via:
`pip install onnx coloredlogs`

One advantage of ONNX is, that it can apply hardware-specific optimizations, see [here](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/README.md#optimizer-options)
We can use the following code to optimize our transformer model and update our OnxEmbeddings to use the optimized model:
```python
model.embeddings.optimize_model(
            "flert-optimized-embeddings.onnx", opt_level=2, use_gpu=True, only_onnxruntime=True, use_external_data_format=True,
        )
```
This creates not only a new file called `flert-optimized-embeddings.onnx`, but also a few more files. This is due to `use_external_data_format=True`.
Our model is so large, that the exported ONNX-Model is more than the expected limit of 2GB. For smaller models, it is recommended to use `use_external_data_format=False` and have everything packed in a single file.

### Quantization

To use the optimization, in addition to `onnx-runtime`, also `onnx` need to be installed, this can be done via:
`pip install onnx`

When you only have a CPU available, you might consider speeding up your model drastically, by using quantization:

```python
model.embeddings.quantize_model(
            "flert-quantized-embeddings.onnx", extra_options={"DisableShapeInference": True}, use_external_data_format=True
        )
```
Here again we use `use_external_data_format=True` due to the model size. Also, we use `extra_options={"DisableShapeInference": True}` as ShapeInference tries to save the model without the external data format and yields to the same problem.
**Notice:** Quantization is not supported for the CUDAExecutionProvider and yields slower predictions. Make sure, that the Provider you use also supports Quantization!


### Customized tooling

Although Optimization and Quantization have integrated support, you might want to use different tooling for model optimizations.
You can use the tools and then update the onnx-model via:
```python
model.embeddings.remove_session()
model.embeddings.onnx_model = "path-to-new-onnx-model.onnx"
model.embeddings.providers = [...] # updated providers config
model.embeddings.create_session()
```

### Saving the model

After doing all optimizations you want to do, you can use run
```python
model.save("flert_onnx_model.pt")
```
**Notice:** you now have to take care of both, the torch model and the onnx model (which might consist of several files)

Now you have an optimized onnx model that you can use for faster inference.


## TransformerJitEmbeddings

If you don't want to use ONNX, you might want to speed up your Embeddings using [TorchScript](https://pytorch.org/docs/stable/jit.html).
**Notice:** if you want to use aws neutron instead of jit, you can do so by following the same tutorial. However, you need to additionally force long sequences by setting `model.embeddings.force_max_length=True` before starting.
To do so, we need to take a look of how the `TransformerEmbeddings` work:

There are 3 parts:
* creating the input tensors will be done by `tensors = embedding.prepare_tensors(sentences)`. This returns a dictionary of tensors, depending on the inner model and its use. (for example some transformer models do not have attention_masks.)
* The `embeddings = embedding.forward(**tensors)` method calls the whole model and returns a dictionary of tensors. If the embedding is a `TokenEmbedding` it has a key `token_embeddings`. If the embedding is a `DocumentEmbedding it has a key `document_embeddings`. Notice that the embedding could be both at once and therefore return both values.
* A mapping from the embeddings to the Tokens/Sentence objects.

To use jit, we are not allowed to pass keyword arguments, and we are also not allowed to pass `None`. Also, if we want to use strict mode, we are not allowed to return a dictionary.
To deal with these limitations we need to write a wrapper torch model, but first lets inspect the tensors: 
```python
tensors = model.embeddings.prepare_tensors(sentences)
print(sorted(tensors.keys()))  # ["attention_mask", "input_ids", "overflow_to_sample_mapping", "word_ids"] 
```
This shows us the parameters that our wrapper model needs to implement. We know that the output has only the key `"token_embeddings"` as our model uses a TokenEmbedding and no DocumentEmbedding.

Hence we can create a wrapper as the following:
````python
class JitWrapper(torch.nn.Module):
    def __init__(self, embedding: TransformerWordEmbeddings):
        super().__init__()
        self.embedding = embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        sub_token_lengths: torch.LongTensor,
        token_lengths: torch.LongTensor,
        attention_mask: torch.Tensor,
        overflow_to_sample_mapping: torch.Tensor,
        word_ids: torch.Tensor,
    ):
        return self.embedding.forward(
            input_ids=input_ids,
            sub_token_lengths=sub_token_lengths,
            token_lengths=token_lengths,
            attention_mask=attention_mask,
            overflow_to_sample_mapping=overflow_to_sample_mapping,
            word_ids=word_ids,
    )["token_embeddings"]
````

Now we can use the wrapper and `torch.jit.trace` to create a script module.
**Notice: ** if you want to use [AWS Neutron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/api-compilation-python-api.html), just replace the `torch.jit.trace` call.

```python
from flair.embeddings import TransformerJitWordEmbeddings

# create a JitWrapper
wrapper = JitWrapper(model.embeddings)

# create the parameters that will be passed to the jit model in the right order.
parameter_names, parameter_list = TransformerJitWordEmbeddings.parameter_to_list(model.embeddings, wrapper, sentences)

# create the script module
script_module = torch.jit.trace(wrapper, parameter_list)

# replace the embeddings with jit embeddings.
model.embeddings = TransformerJitWordEmbeddings.create_from_embedding(script_module, model.embeddings, parameter_names)

model.save("flert_jit.pt")
```

Now you have an optimized jit model that you can use for faster inference.