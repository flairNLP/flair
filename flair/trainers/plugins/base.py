import logging
from collections import defaultdict
from inspect import isclass, signature
from itertools import count
from queue import Queue
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    NewType,
    Optional,
    Sequence,
    Set,
    Type,
    Union,
    cast,
)

log = logging.getLogger("flair")


PluginArgument = Union["BasePlugin", Type["BasePlugin"]]
HookHandleId = NewType("HookHandleId", int)

EventIdenifier = str


class TrainingInterrupt(Exception):
    """Allows plugins to interrupt the training loop."""


class Pluggable:
    """Dispatches events which attached plugins can react to."""

    valid_events: Optional[Set[EventIdenifier]] = None

    def __init__(self, *, plugins: Sequence[PluginArgument] = []) -> None:
        """Initialize a `Pluggable`.

        Args:
            plugins: Plugins which should be attached to this `Pluggable`.
        """
        self._hook_handles: Dict[EventIdenifier, Dict[HookHandleId, HookHandle]] = defaultdict(dict)

        self._hook_handle_id_counter = count()

        self._plugins: List[BasePlugin] = []

        # This flag tracks, whether an event is currently being processed (otherwise it is added to the queue)
        self._processing_events = False
        self._event_queue: Queue = Queue()

        for plugin in plugins:
            if isclass(plugin):
                # instantiate plugin
                plugin = plugin()

            plugin = cast("BasePlugin", plugin)
            plugin.attach_to(self)

    @property
    def plugins(self):
        return self._plugins

    def append_plugin(self, plugin):
        self._plugins.append(plugin)

    def validate_event(self, *events: EventIdenifier):
        for event in events:
            assert isinstance(event, EventIdenifier)

            if self.valid_events is not None and event not in self.valid_events:
                raise RuntimeError(f"Event '{event}' not recognized. Available: {', '.join(self.valid_events)}")
            return event
        return None

    def register_hook(self, func: Callable, *events: EventIdenifier):
        """Register a hook.

        Args:
            func: Function to be called when the event is emitted.
            *events: List of events to call this function on.
        """
        self.validate_event(*events)

        handle: HookHandle = HookHandle(
            HookHandleId(next(self._hook_handle_id_counter)), events=events, func=func, pluggable=self
        )

        for event in events:
            self._hook_handles[event][handle.id] = handle
        return handle

    def dispatch(self, event: EventIdenifier, *args, **kwargs) -> None:
        """Call all functions hooked to a certain event."""
        self.validate_event(event)

        self._event_queue.put((event, args, kwargs))

        if not self._processing_events:
            try:
                self._processing_events = True

                while not self._event_queue.empty():
                    event, args, kwargs = self._event_queue.get()

                    for hook in self._hook_handles[event].values():
                        hook(*args, **kwargs)
            finally:
                # Reset the flag, since an exception event might be dispatched
                self._processing_events = False

    def remove_hook(self, handle: "HookHandle"):
        """Remove a hook handle from this instance."""
        for event in handle.events:
            del self._hook_handles[event][handle.id]


class HookHandle:
    """Represents the registration information of a hook callback."""

    def __init__(
        self, _id: HookHandleId, *, events: Sequence[EventIdenifier], func: Callable, pluggable: Pluggable
    ) -> None:
        """Intitialize `HookHandle`.

        Args:
            _id: Id, the callback is stored as in the `Pluggable`.
            events: List of events, the callback is registered for.
            func: The callback function.
            pluggable: The `Pluggable` where the callback is registered.
        """
        pluggable.validate_event(*events)

        self._id = _id
        self._events = events
        self._func = func
        self._pluggable = pluggable

    @property
    def id(self) -> HookHandleId:
        """Return the id of this `HookHandle`."""
        return self._id

    @property
    def func_name(self):
        return self._func.__qualname__

    @property
    def events(self) -> Iterator[EventIdenifier]:
        """Return iterator of events whis `HookHandle` is registered for."""
        yield from self._events

    def remove(self):
        """Remove a hook from the `Pluggable` it is attached to."""
        self._pluggable.remove_hook(self)

    def __call__(self, *args, **kw):
        """Call the hook this `HookHandle` is associated with."""
        try:
            return self._func(*args, **kw)
        except TypeError as err:
            sig = signature(self._func)

            if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                # If there is no **kw argument in the callback, check if any of the passed kw args is not accepted by
                # the callback
                for name in kw:
                    if name not in sig.parameters:
                        raise TypeError(
                            f"Hook callback {self.func_name}() does not accept keyword argument '{name}'"
                        ) from err

            raise err


class BasePlugin:
    """Base class for all plugins."""

    def __init__(self) -> None:
        """Initialize the base plugin."""
        self._hook_handles: List[HookHandle] = []
        self._pluggable: Optional[Pluggable] = None

    def attach_to(self, pluggable: Pluggable):
        """Attach this plugin to a `Pluggable`."""
        assert self._pluggable is None
        assert len(self._hook_handles) == 0

        self._pluggable = pluggable

        pluggable.append_plugin(self)

        # go through all attributes
        for name in dir(self):
            try:
                func = getattr(self, name)

                # get attribute hook events (may raise an AttributeError)
                events = func._plugin_hook_events

                # register function as a hook
                handle = pluggable.register_hook(func, *events)
                self._hook_handles.append(handle)

            except AttributeError:
                continue

    def detach(self):
        """Detach a plugin from the `Pluggable` it is attached to."""
        assert self._pluggable is not None

        for handle in self._hook_handles:
            handle.remove()

        self._pluggable = None
        self._hook_handles = []

    @classmethod
    def mark_func_as_hook(cls, func: Callable, *events: EventIdenifier) -> Callable:
        """Mark method as a hook triggered by the `Pluggable`."""
        if len(events) == 0:
            events = (func.__name__,)
        func._plugin_hook_events = events  # type: ignore[attr-defined]
        return func

    @classmethod
    def hook(
        cls,
        first_arg: Optional[Union[Callable, EventIdenifier]] = None,
        *other_args: EventIdenifier,
    ) -> Callable:
        """Convience function for `BasePlugin.mark_func_as_hook`).

        Enables using the `@BasePlugin.hook` syntax.

        Can also be used as:
        `@BasePlugin.hook("some_event", "another_event")`
        """
        if first_arg is None:
            # Decorator was used with parentheses, but no args
            return cls.mark_func_as_hook

        if isinstance(first_arg, EventIdenifier):
            # Decorator was used with args (strings specifiying the events)
            def decorator_func(func: Callable):
                return cls.mark_func_as_hook(func, cast(EventIdenifier, first_arg), *other_args)

            return decorator_func

        # Decorator was used without args
        return cls.mark_func_as_hook(first_arg, *other_args)

    @property
    def pluggable(self) -> Optional[Pluggable]:
        return self._pluggable

    def __str__(self) -> str:
        return self.__class__.__name__

    def get_state(self) -> Dict[str, Any]:
        return {"__cls__": f"{self.__module__}.{self.__class__.__name__}"}


class TrainerPlugin(BasePlugin):
    @property
    def trainer(self):
        return self.pluggable

    @property
    def model(self):
        return self.trainer.model

    @property
    def corpus(self):
        return self.trainer.corpus
