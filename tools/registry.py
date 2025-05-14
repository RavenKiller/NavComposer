#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
from typing import Any, Callable, DefaultDict, Optional, Type


class NonePipeline:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        if isinstance(args[0], list):
            return [None for _ in args[0]]
        else:
            return None


class IdentityPipeline:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        if isinstance(args[0], list):
            return [v.replace("\n", " ") for v in args[0]]
        else:
            return args[0].replace("\n", " ")


class Registry:
    mapping: DefaultDict[str, Any] = collections.defaultdict(dict)

    @classmethod
    def _register_impl(
        cls,
        _type: str,
        to_register: Optional[Any],
        name: Optional[str],
        assert_type: Optional[Type] = None,
    ) -> Callable:
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(to_register, assert_type)
            register_name = to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def register_action(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("action", to_register, name)

    @classmethod
    def register_object(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("object", to_register, name)

    @classmethod
    def register_scene(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("scene", to_register, name)

    @classmethod
    def register_summary(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("summary", to_register, name)

    @classmethod
    def register_speaker(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("speaker", to_register, name)

    @classmethod
    def _get_impl(cls, _type: str, name: str, default_none: str = None) -> Type:
        if name not in cls.mapping[_type]:
            print(
                "Warning: {} not exists in {} module. Use default {}".format(
                    name, _type, default_none
                )
            )
        return cls.mapping[_type].get(name, default_none)

    @classmethod
    def get_action(cls, name: str):
        return cls._get_impl("action", name, default_none=NonePipeline)

    @classmethod
    def get_object(cls, name: str):
        return cls._get_impl("object", name, default_none=NonePipeline)

    @classmethod
    def get_scene(cls, name: str):
        return cls._get_impl("scene", name, default_none=NonePipeline)

    @classmethod
    def get_summary(cls, name: str):
        return cls._get_impl("summary", name, default_none=IdentityPipeline)

    @classmethod
    def get_speaker(cls, name: str):
        return cls._get_impl("speaker", name)

    @classmethod
    def get_mapping(cls):
        return cls.mapping


registry = Registry()
