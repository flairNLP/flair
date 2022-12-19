import logging
from pathlib import Path
from typing import Any, Dict, Generic, Optional, Tuple, Type, TypeVar, Union

from flair.file_utils import load_torch_state
from flair.nn import Classifier, Model

MT = TypeVar("MT", bound=Model)

log = logging.getLogger("flair")


class ModelRegisterMixin(Generic[MT]):
    MODEL_CLASSES: Dict[str, Type[MT]]

    @classmethod
    def register(cls, *args):
        name = None

        def _register(model_cls):
            nonlocal name
            if name is None:
                name = model_cls.__name__
            model_cls.model_name = name
            cls.MODEL_CLASSES[name] = model_cls
            return model_cls

        if len(args) == 1 and callable(args[0]):
            return _register(args[0])
        elif len(args) > 0:
            name = args[0]

        return _register

    @classmethod
    def _fetch_model(cls, model_name: str) -> Tuple[Optional[Type[MT]], str]:
        if Path(model_name).exists():
            return None, model_name

        for model_cls in cls.MODEL_CLASSES.values():
            try:
                new_model_name = model_cls._fetch_model(model_name)
                if new_model_name != model_name:
                    return model_cls, new_model_name
            except Exception:
                # skip any invalid loadings, e.g. not found on huggingface hub
                continue

        raise ValueError(f"Could not find any model with name '{model_name}'")

    @classmethod
    def _infer_cls(cls, state: Dict[str, Any]) -> Type[MT]:
        log.warning(
            "No information about the class is found, trying to infer the class."
            "It is advised to once load the model using the specific class (e.g. `SequenceTagger.load(...)`)"
            "and save it again, to ensure the model is loaded as the right class."
        )
        # try to guess the model by seeing if it can be initialized with the current state
        for model_cls in cls.MODEL_CLASSES.values():
            try:
                model_cls._init_model_with_state_dict(state)
            except KeyError:
                continue
            return model_cls
        raise Exception("Could not infer model type by state")

    @classmethod
    def load(cls, model_path: Union[str, Path, Dict[str, Any]]) -> MT:
        model_cls: Optional[Type[MT]] = None
        if not isinstance(model_path, dict):
            model_cls, model_path = cls._fetch_model(str(model_path))
            state = load_torch_state(model_path)
        else:
            state = model_path
        cls_name = state.pop("__cls__", None)
        # older (flair 11.3 and below) models do not contain cls information.
        if model_cls is None:
            if cls_name is None:
                cls_name = cls._infer_cls(state)
            model_cls = cls.MODEL_CLASSES[cls_name]
        return model_cls.load(state)


class AutoFlairModel(ModelRegisterMixin[Model]):
    MODEL_CLASSES: Dict[str, Type[Model]] = {}


class AutoFlairClassifier(ModelRegisterMixin[Classifier]):
    MODEL_CLASSES: Dict[str, Type[Classifier]] = {}
