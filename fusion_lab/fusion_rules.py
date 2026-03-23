from __future__ import annotations

from .enums import FusionDecision, MappingType


def edge_fusion_decision(producer: MappingType, consumer: MappingType) -> FusionDecision:
    if MappingType.OPAQUE in {producer, consumer}:
        return FusionDecision.NEVER
    if producer == MappingType.MANY_TO_MANY and consumer == MappingType.MANY_TO_MANY:
        return FusionDecision.NEVER
    if producer == MappingType.REDUCTION and consumer == MappingType.REDUCTION:
        return FusionDecision.NEVER
    if producer == MappingType.ONE_TO_ONE or consumer == MappingType.ONE_TO_ONE:
        return FusionDecision.ALWAYS
    if producer == MappingType.REORGANIZE and consumer in {
        MappingType.REORGANIZE,
        MappingType.SHUFFLE,
        MappingType.ONE_TO_MANY,
        MappingType.MANY_TO_MANY,
        MappingType.REDUCTION,
    }:
        return FusionDecision.PROFILE
    if consumer == MappingType.REORGANIZE and producer in {
        MappingType.REORGANIZE,
        MappingType.SHUFFLE,
        MappingType.ONE_TO_MANY,
        MappingType.MANY_TO_MANY,
        MappingType.REDUCTION,
    }:
        return FusionDecision.PROFILE
    if producer == MappingType.SHUFFLE or consumer == MappingType.SHUFFLE:
        return FusionDecision.PROFILE
    if producer == MappingType.ONE_TO_MANY and consumer == MappingType.ONE_TO_MANY:
        return FusionDecision.PROFILE
    if producer == MappingType.ONE_TO_MANY and consumer in {
        MappingType.MANY_TO_MANY,
        MappingType.REDUCTION,
    }:
        return FusionDecision.PROFILE
    if consumer == MappingType.ONE_TO_MANY and producer in {
        MappingType.MANY_TO_MANY,
        MappingType.REDUCTION,
    }:
        return FusionDecision.PROFILE
    if producer == MappingType.MANY_TO_MANY and consumer == MappingType.REDUCTION:
        return FusionDecision.PROFILE
    if producer == MappingType.REDUCTION and consumer == MappingType.MANY_TO_MANY:
        return FusionDecision.PROFILE
    return FusionDecision.PROFILE


def dominant_mapping_type(patterns: list[MappingType]) -> MappingType:
    unique = set(patterns)
    if MappingType.OPAQUE in unique:
        return MappingType.OPAQUE
    if MappingType.MANY_TO_MANY in unique:
        return MappingType.MANY_TO_MANY
    if MappingType.REDUCTION in unique:
        return MappingType.REDUCTION
    if MappingType.ONE_TO_MANY in unique:
        return MappingType.ONE_TO_MANY
    if MappingType.SHUFFLE in unique:
        return MappingType.SHUFFLE
    if unique.issubset({MappingType.ONE_TO_ONE, MappingType.REORGANIZE}):
        return MappingType.ONE_TO_ONE
    if MappingType.REORGANIZE in unique:
        return MappingType.REORGANIZE
    return MappingType.ONE_TO_ONE