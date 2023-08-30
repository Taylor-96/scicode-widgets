from typing import List, Optional, Union

from ipywidgets import VBox, Widget
from traitlets.utils.sentinel import Sentinel


class CueBox(VBox):
    """
    A box around the widget :param widget_to_cue: that adds a visual cue defined in the
    :param css_style: when the trait :param traits_to_observe: in the widget :param
    widget_to_observe: changes. If the :param widget_to_observe: is a list then for each
    each widget is observed.

    :param widget_to_observe:
        The widget to observe if the :param traits_to_observe: has changed.
    :param traits_to_observe:
        The trait from the :param widget_to_observe: to observe if changed.
        Specify `traitlets.All` to observe all traits.
    :param widget_to_cue:
        The widget to wrap the box around to give a visual cue, once :param
        traits_to_observe: has changed
        If None, then the :param widget_to_cue: is set to :param widget_to_observe:.
    :param cued:
        Specifies if it is cued on initialization
    :param css_syle:
        - **base**: the css style of the box during initialization
        - **cue**: the css style that is added when :param
          traits_to_observe: in widget :param widget_to_observe: changes.
          It is supposed to change the style of the box such that the user has a visual
          cue that :param widget_to_cue: has changed.

    Further accepts the same (keyword) arguments as :py:class:`ipywidgets.Box`.
    """

    def __init__(
        self,
        widget_to_observe: Union[List[Widget], Widget],
        traits_to_observe: Union[str, List[str], List[List[str]], Sentinel] = "value",
        widget_to_cue: Optional[Widget] = None,
        cued: bool = True,
        css_style: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        if widget_to_cue is None and isinstance(widget_to_observe, list):
            raise ValueError(
                "When widget_to_observe is a list, widget_to_cue must be"
                " given, since it is ambiguous which widget to cue."
            )
        if isinstance(widget_to_observe, list) and isinstance(traits_to_observe, list):
            if len(widget_to_observe) != len(traits_to_observe):
                raise ValueError(
                    "When widget_to_observe and traits_to_observe are "
                    "lists, they must have the same length"
                )

        self._widget_to_observe = widget_to_observe
        self._traits_to_observe = traits_to_observe

        if widget_to_cue is None and not (isinstance(self._widget_to_observe, list)):
            self._widget_to_cue = widget_to_observe
        else:
            self._widget_to_cue = widget_to_cue

        self._widget_to_observe = widget_to_observe

        if css_style is None:
            self._css_style = {
                "base": "scwidget-cue-box",
                "cue": "scwidget-cue-box--cue",
            }
        else:
            self._css_style = css_style

        super().__init__([self._widget_to_cue], *args, **kwargs)
        if isinstance(self._widget_to_observe, list):
            for i, widget in enumerate(self._widget_to_observe):
                if isinstance(self._traits_to_observe, list):
                    widget.observe(
                        self._on_traits_to_observe_changed, self._traits_to_observe[i]
                    )
                else:
                    widget.observe(
                        self._on_traits_to_observe_changed, self._traits_to_observe
                    )
        else:
            self._widget_to_observe.observe(
                self._on_traits_to_observe_changed, traits_to_observe
            )

        self.add_class(self._css_style["base"])

        self._cued = cued
        if cued:
            self.add_class(self._css_style["cue"])

    @property
    def widget_to_observe(self):
        return self._widget_to_observe

    @property
    def traits_to_observe(self):
        return self._traits_to_observe

    @property
    def widget_to_cue(self):
        return self._widget_to_cue

    @property
    def cued(self):
        return self._cued

    @cued.setter
    def cued(self, cued: bool):
        if cued:
            self.add_class(self._css_style["cue"])
        else:
            self.remove_class(self._css_style["cue"])
        self._cued = cued

    def _on_traits_to_observe_changed(self, change: dict):
        self.cued = True


class SaveCueBox(CueBox):
    """
    A box around the widget :param widget_to_cue: that adds a visual cue defined in the
    :param css_style: when the trait :param traits_to_observe: in the widget :param
    widget_to_observe: changes.

    :param widget_to_observe:
        The widget to observe if the :param traits_to_observe: has changed.
    :param traits_to_observe:
        The trait from the :param widget_to_observe: to observe if changed.
        Specify `traitlets.All` to observe all traits.
    :param cued:
        Specifies if it is cued on initialization
    :param widget_to_cue:
        The widget to wrap the box around to give a visual cue, once :param
        traits_to_observe: has changed
        If None, then the :param widget_to_cue: is set to :param widget_to_observe:.

    Further accepts the same (keyword) arguments as :py:class:`ipywidgets.Box`.
    """

    def __init__(
        self,
        widget_to_observe: Widget,
        traits_to_observe: Union[str, List[str], Sentinel] = "value",
        widget_to_cue: Optional[Widget] = None,
        cued: bool = True,
        *args,
        **kwargs,
    ):
        css_style = {
            "base": "scwidget-save-cue-box",
            "cue": "scwidget-save-cue-box--cue",
        }
        super().__init__(
            widget_to_observe, traits_to_observe, widget_to_cue, cued, css_style
        )


class CheckCueBox(CueBox):
    """
    A box around the widget :param widget_to_cue: that adds a visual cue defined in the
    :param css_style: when the trait :param traits_to_observe: in the widget :param
    widget_to_observe: changes.

    :param widget_to_observe:
        The widget to observe if the :param traits_to_observe: has changed.
    :param traits_to_observe:
        The trait from the :param widget_to_observe: to observe if changed.
        Specify `traitlets.All` to observe all traits.
    :param widget_to_cue:
        The widget to wrap the box around to give a visual cue, once :param
        traits_to_observe: has changed
        If None, then the :param widget_to_cue: is set to :param widget_to_observe:.
    :param cued:
        Specifies if it is cued on initialization

    Further accepts the same (keyword) arguments as :py:class:`ipywidgets.Box`.
    """

    def __init__(
        self,
        widget_to_observe: Widget,
        traits_to_observe: Union[str, List[str], Sentinel] = "value",
        widget_to_cue: Optional[Widget] = None,
        cued: bool = True,
        *args,
        **kwargs,
    ):
        css_style = {
            "base": "scwidget-check-cue-box",
            "cue": "scwidget-check-cue-box--cue",
        }
        super().__init__(
            widget_to_observe, traits_to_observe, widget_to_cue, cued, css_style
        )


class UpdateCueBox(CueBox):
    """
    A box around the widget :param widget_to_cue: that adds a visual cue defined in the
    :param css_style: when the trait :param traits_to_observe: in the widget :param
    widget_to_observe: changes.

    :param widget_to_observe:
        The widget to observe if the :param traits_to_observe: has changed.
    :param traits_to_observe:
        The trait from the :param widget_to_observe: to observe if changed.
        Specify `traitlets.All` to observe all traits.
    :param widget_to_cue:
        The widget to wrap the box around to give a visual cue, once :param
        traits_to_observe: has changed
        If None, then the :param widget_to_cue: is set to :param widget_to_observe:.
    :param cued:
        Specifies if it is cued on initialization

    Further accepts the same (keyword) arguments as :py:class:`ipywidgets.Box`.
    """

    def __init__(
        self,
        widget_to_observe: Widget,
        traits_to_observe: Union[str, List[str], Sentinel] = "value",
        widget_to_cue: Optional[Widget] = None,
        cued: bool = True,
        *args,
        **kwargs,
    ):
        css_style = {
            "base": "scwidget-update-cue-box",
            "cue": "scwidget-update-cue-box--cue",
        }
        super().__init__(
            widget_to_observe, traits_to_observe, widget_to_cue, cued, css_style
        )
