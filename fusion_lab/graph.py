"""Internal graph data model.

This file owns graph loading, topology helpers and group boundary analysis. It is one of the safest places to extend input graph metadata.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


DTYPE_BYTES = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "bool": 1,
}


def num_elements(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return max(total, 1)


def dtype_nbytes(dtype: str) -> int:
    return DTYPE_BYTES.get(dtype.lower(), 4)


def tensor_nbytes(shape: Iterable[int], dtype: str) -> int:
    return num_elements(shape) * dtype_nbytes(dtype)


@dataclass(frozen=True)
class ExternalTensorSpec:
    name: str
    shape: list[int]
    dtype: str = "float32"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExternalTensorSpec":
        return cls(
            name=data["name"],
            shape=[int(dim) for dim in data["shape"]],
            dtype=data.get("dtype", "float32"),
        )


@dataclass(frozen=True)
class NodeSpec:
    name: str
    op_type: str
    inputs: list[str]
    output_shape: list[int]
    dtype: str = "float32"
    attrs: dict[str, Any] = field(default_factory=dict)
    mapping_type: str | None = None
    flops: float | None = None
    weight_bytes: int | None = None
    registers_per_thread: int | None = None
    instruction_count: int | None = None
    preferred_threads: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeSpec":
        return cls(
            name=data["name"],
            op_type=data["op"],
            inputs=list(data.get("inputs", [])),
            output_shape=[int(dim) for dim in data["output_shape"]],
            dtype=data.get("dtype", "float32"),
            attrs=dict(data.get("attrs", {})),
            mapping_type=data.get("mapping_type"),
            flops=data.get("flops"),
            weight_bytes=data.get("weight_bytes"),
            registers_per_thread=data.get("registers_per_thread"),
            instruction_count=data.get("instruction_count"),
            preferred_threads=data.get("preferred_threads"),
        )

    @property
    def output_bytes(self) -> int:
        return tensor_nbytes(self.output_shape, self.dtype)


@dataclass
class GraphModel:
    name: str
    inputs: dict[str, ExternalTensorSpec]
    nodes: dict[str, NodeSpec]
    outputs: list[str]

    def __post_init__(self) -> None:
        self._preds: dict[str, list[str]] = {name: [] for name in self.nodes}
        self._succs: dict[str, list[str]] = {name: [] for name in self.nodes}
        for node in self.nodes.values():
            for ref in node.inputs:
                if ref in self.nodes:
                    self._preds[node.name].append(ref)
                    self._succs[ref].append(node.name)
                elif ref not in self.inputs:
                    raise KeyError(
                        f"Input reference '{ref}' for node '{node.name}' is not a node "
                        "or declared graph input."
                    )
        self._topo_order = self._build_topological_order()
        self._topo_index = {name: idx for idx, name in enumerate(self._topo_order)}
        self._layers = self._build_layers()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphModel":
        inputs = {
            item["name"]: ExternalTensorSpec.from_dict(item)
            for item in data.get("inputs", [])
        }
        nodes = {item["name"]: NodeSpec.from_dict(item) for item in data.get("nodes", [])}
        return cls(
            name=data["name"],
            inputs=inputs,
            nodes=nodes,
            outputs=list(data.get("outputs", [])),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "GraphModel":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    def _build_topological_order(self) -> list[str]:
        indegree = {name: len(preds) for name, preds in self._preds.items()}
        ready = deque(sorted(name for name, degree in indegree.items() if degree == 0))
        order: list[str] = []
        while ready:
            node = ready.popleft()
            order.append(node)
            for succ in sorted(self._succs[node]):
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)
        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle; fusion search requires a DAG.")
        return order

    def _build_layers(self) -> list[list[str]]:
        layer_of: dict[str, int] = {}
        for node in self._topo_order:
            preds = self.predecessors(node)
            layer_of[node] = 0 if not preds else 1 + max(layer_of[pred] for pred in preds)
        max_layer = max(layer_of.values(), default=-1)
        layers = [[] for _ in range(max_layer + 1)]
        for node in self._topo_order:
            layers[layer_of[node]].append(node)
        return layers

    def topological_order(self) -> list[str]:
        return list(self._topo_order)

    def topo_index(self, node_name: str) -> int:
        return self._topo_index[node_name]

    def predecessors(self, node_name: str) -> list[str]:
        return list(self._preds[node_name])

    def successors(self, node_name: str) -> list[str]:
        return list(self._succs[node_name])

    def layers(self) -> list[list[str]]:
        return [list(layer) for layer in self._layers]

    def layer_index(self, node_name: str) -> int:
        for layer_id, nodes in enumerate(self._layers):
            if node_name in nodes:
                return layer_id
        raise KeyError(node_name)

    def tensor_shape(self, ref: str) -> list[int]:
        if ref in self.nodes:
            return list(self.nodes[ref].output_shape)
        return list(self.inputs[ref].shape)

    def tensor_dtype(self, ref: str) -> str:
        if ref in self.nodes:
            return self.nodes[ref].dtype
        return self.inputs[ref].dtype

    def tensor_bytes(self, ref: str) -> int:
        return tensor_nbytes(self.tensor_shape(ref), self.tensor_dtype(ref))

    def weakly_connected_components(self, node_names: Iterable[str]) -> list[set[str]]:
        subset = set(node_names)
        unseen = set(subset)
        components: list[set[str]] = []
        while unseen:
            seed = min(unseen, key=self.topo_index)
            queue = deque([seed])
            component: set[str] = set()
            unseen.remove(seed)
            while queue:
                node = queue.popleft()
                component.add(node)
                neighbors = set(self.predecessors(node)) | set(self.successors(node))
                for neighbor in sorted(neighbors, key=self.topo_index):
                    if neighbor in unseen and neighbor in subset:
                        unseen.remove(neighbor)
                        queue.append(neighbor)
            components.append(component)
        components.sort(key=lambda comp: min(self.topo_index(name) for name in comp))
        return components

    def internal_edges(self, node_names: Iterable[str]) -> list[tuple[str, str]]:
        subset = set(node_names)
        edges: list[tuple[str, str]] = []
        for src in subset:
            for dst in self.successors(src):
                if dst in subset:
                    edges.append((src, dst))
        edges.sort(key=lambda item: (self.topo_index(item[0]), self.topo_index(item[1])))
        return edges

    def external_inputs_for_group(self, node_names: Iterable[str]) -> list[str]:
        group = set(node_names)
        refs: list[str] = []
        seen: set[str] = set()
        for node_name in sorted(group, key=self.topo_index):
            for ref in self.nodes[node_name].inputs:
                if ref not in group and ref not in seen:
                    seen.add(ref)
                    refs.append(ref)
        return refs

    def external_output_nodes_for_group(self, node_names: Iterable[str]) -> list[str]:
        group = set(node_names)
        external_outputs: list[str] = []
        for node_name in sorted(group, key=self.topo_index):
            succs = self.successors(node_name)
            if node_name in self.outputs or any(succ not in group for succ in succs):
                external_outputs.append(node_name)
        return external_outputs