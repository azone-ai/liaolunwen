from __future__ import annotations

from enum import Enum


class MappingType(str, Enum):
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_MANY = "many_to_many"
    REORGANIZE = "reorganize"
    SHUFFLE = "shuffle"
    REDUCTION = "reduction"
    OPAQUE = "opaque"


class FusionDecision(str, Enum):
    ALWAYS = "always"
    PROFILE = "profile"
    NEVER = "never"