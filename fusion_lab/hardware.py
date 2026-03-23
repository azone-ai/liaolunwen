from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from .enums import MappingType


def _default_penalty_weights() -> dict[str, float]:
    return {
        "register": 0.35,
        "shared_memory": 0.25,
        "geometry": 0.20,
        "icache": 0.20,
    }


def _default_compute_efficiency() -> dict[str, float]:
    return {
        MappingType.ONE_TO_ONE.value: 0.45,
        MappingType.ONE_TO_MANY.value: 0.40,
        MappingType.MANY_TO_MANY.value: 0.70,
        MappingType.REORGANIZE.value: 0.35,
        MappingType.SHUFFLE.value: 0.35,
        MappingType.REDUCTION.value: 0.40,
        MappingType.OPAQUE.value: 0.25,
    }


def _default_memory_efficiency() -> dict[str, float]:
    return {
        MappingType.ONE_TO_ONE.value: 0.78,
        MappingType.ONE_TO_MANY.value: 0.65,
        MappingType.MANY_TO_MANY.value: 0.55,
        MappingType.REORGANIZE.value: 0.72,
        MappingType.SHUFFLE.value: 0.60,
        MappingType.REDUCTION.value: 0.52,
        MappingType.OPAQUE.value: 0.40,
    }


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    peak_flops: float
    memory_bandwidth: float
    launch_overhead_us: float
    warp_size: int = 32
    max_warps_per_sm: int = 64
    max_threads_per_block: int = 1024
    max_threads_per_sm: int = 2048
    max_blocks_per_sm: int = 16
    registers_per_thread_limit: int = 255
    registers_per_sm: int = 65536
    shared_mem_per_block: int = 49152
    shared_mem_per_sm: int = 65536
    icache_inst_limit: int = 1200
    reg_soft_ratio: float = 0.55
    smem_soft_ratio: float = 0.60
    supported_threads: tuple[int, ...] = (64, 128, 256, 512, 1024)
    penalty_weights: dict[str, float] = field(default_factory=_default_penalty_weights)
    base_compute_efficiency: dict[str, float] = field(default_factory=_default_compute_efficiency)
    base_memory_efficiency: dict[str, float] = field(default_factory=_default_memory_efficiency)

    @classmethod
    def from_dict(cls, data: dict) -> "HardwareProfile":
        merged = dict(data)
        merged.setdefault("penalty_weights", _default_penalty_weights())
        merged.setdefault("base_compute_efficiency", _default_compute_efficiency())
        merged.setdefault("base_memory_efficiency", _default_memory_efficiency())
        if "supported_threads" in merged:
            merged["supported_threads"] = tuple(int(item) for item in merged["supported_threads"])
        return cls(**merged)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "HardwareProfile":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    @property
    def launch_overhead_s(self) -> float:
        return self.launch_overhead_us * 1e-6