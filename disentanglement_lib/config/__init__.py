# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A little patch for gin configure, because google missed the __wrapped__ for decoration.
"""
import gin
from gin import config_is_locked, config_parser
from gin.config import Configurable, _REGISTRY, _decorate_fn_or_cls, _make_gin_wrapper, _validate_parameters, \
    _INTERACTIVE_MODE


def _make_configurable(fn_or_cls,
                       name=None,
                       module=None,
                       allowlist=None,
                       denylist=None,
                       subclass=False):
    if config_is_locked():
        err_str = 'Attempted to add a new configurable after the config was locked.'
        raise RuntimeError(err_str)

    name = fn_or_cls.__name__ if name is None else name
    if config_parser.IDENTIFIER_RE.match(name):
        default_module = getattr(fn_or_cls, '__module__', None)
        module = default_module if module is None else module
    elif not config_parser.MODULE_RE.match(name):
        raise ValueError("Configurable name '{}' is invalid.".format(name))

    if module is not None and not config_parser.MODULE_RE.match(module):
        raise ValueError("Module '{}' is invalid.".format(module))

    selector = module + '.' + name if module else name
    if not _INTERACTIVE_MODE and selector in _REGISTRY:
        err_str = ("A configurable matching '{}' already exists.\n\n"
                   'To allow re-registration of configurables in an interactive '
                   'environment, use:\n\n'
                   '    gin.enter_interactive_mode()')
        raise ValueError(err_str.format(selector))

    if allowlist and denylist:
        err_str = 'An allowlist or a denylist can be specified, but not both.'
        raise ValueError(err_str)

    if allowlist and not isinstance(allowlist, (list, tuple)):
        raise TypeError('allowlist should be a list or tuple.')

    if denylist and not isinstance(denylist, (list, tuple)):
        raise TypeError('denylist should be a list or tuple.')

    _validate_parameters(fn_or_cls, allowlist, 'allowlist')
    _validate_parameters(fn_or_cls, denylist, 'denylist')

    def decorator(fn):
        """Wraps `fn` so that it obtains parameters from the configuration."""
        return _make_gin_wrapper(fn, fn_or_cls, name, selector, allowlist,
                                 denylist)

    decorated_fn_or_cls = _decorate_fn_or_cls(
        decorator, fn_or_cls, subclass=subclass)
    # patch
    decorated_fn_or_cls.__wrapped__ = fn_or_cls
    _REGISTRY[selector] = Configurable(
        decorated_fn_or_cls,
        name=name,
        module=module,
        allowlist=allowlist,
        denylist=denylist,
        selector=selector)
    return decorated_fn_or_cls


gin.config._make_configurable = _make_configurable
