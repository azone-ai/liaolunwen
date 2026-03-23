from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph import ExternalTensorSpec, GraphModel, NodeSpec
from .search import PlanBlock, PlanResult
from .vendor import import_onnx


def _sanitize_name(name: str) -> str:
    sanitized = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in name)
    return sanitized or 'node'


def _make_unique(base: str, used: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in used:
        candidate = f'{base}_{suffix}'
        suffix += 1
    used.add(candidate)
    return candidate


def _shape_from_value_info(value_info: Any) -> list[int]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField('shape'):
        return [1]
    shape: list[int] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField('dim_value') and dim.dim_value > 0:
            shape.append(int(dim.dim_value))
        else:
            shape.append(1)
    return shape or [1]


def _dtype_name(elem_type: int, onnx_module: Any) -> str:
    try:
        np_dtype = onnx_module.helper.tensor_dtype_to_np_dtype(elem_type)
        return np_dtype.name
    except Exception:
        fallback = {
            onnx_module.TensorProto.FLOAT16: 'float16',
            onnx_module.TensorProto.FLOAT: 'float32',
            onnx_module.TensorProto.DOUBLE: 'float64',
            onnx_module.TensorProto.INT8: 'int8',
            onnx_module.TensorProto.INT16: 'int16',
            onnx_module.TensorProto.INT32: 'int32',
            onnx_module.TensorProto.INT64: 'int64',
            onnx_module.TensorProto.BOOL: 'bool',
            onnx_module.TensorProto.UINT8: 'uint8',
        }
        return fallback.get(elem_type, 'float32')


def _attrs_to_dict(node: Any, onnx_module: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for attr in node.attribute:
        value = onnx_module.helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            attrs[attr.name] = value.decode('utf-8', errors='ignore')
        elif isinstance(value, tuple):
            attrs[attr.name] = list(value)
        else:
            attrs[attr.name] = value
    return attrs


def _enrich_attrs_from_initializers(
    op_type: str,
    attrs: dict[str, Any],
    initializer_inputs: list[str],
    tensor_shapes: dict[str, list[int]],
    onnx_inputs: list[str],
) -> None:
    op = op_type.strip().lower()
    if op in {'conv', 'convtranspose'} and initializer_inputs:
        weight_shape = tensor_shapes.get(initializer_inputs[0], [])
        data_shape = tensor_shapes.get(onnx_inputs[0], []) if onnx_inputs else []
        if len(weight_shape) >= 2:
            attrs.setdefault('out_channels', int(weight_shape[0]))
            if len(data_shape) > 1:
                attrs.setdefault('in_channels', int(data_shape[1]))
            else:
                attrs.setdefault('in_channels', int(weight_shape[1]))
        if len(weight_shape) >= 4:
            attrs.setdefault('kernel', [int(weight_shape[2]), int(weight_shape[3])])
    if op in {'gemm', 'matmul'} and initializer_inputs:
        weight_shape = tensor_shapes.get(initializer_inputs[0], [])
        data_shape = tensor_shapes.get(onnx_inputs[0], []) if onnx_inputs else []
        if len(weight_shape) >= 2:
            attrs.setdefault('k', int(weight_shape[0]))
            attrs.setdefault('n', int(weight_shape[1]))
        if len(data_shape) >= 2:
            attrs.setdefault('m', int(data_shape[0]))


def _copy_proto(proto: Any, proto_type: Any) -> Any:
    new_proto = proto_type()
    new_proto.CopyFrom(proto)
    return new_proto


@dataclass
class ImportedOnnxModel:
    source_path: Path
    model: Any
    graph_model: GraphModel
    node_name_to_proto: dict[str, Any]
    node_name_to_index: dict[str, int]
    node_name_to_output_tensor: dict[str, str]
    output_tensor_to_node_name: dict[str, str]
    tensor_consumers: dict[str, list[str]]
    graph_output_tensors: set[str]
    ordered_node_names: list[str]


def load_onnx_model(path: str | Path) -> ImportedOnnxModel:
    onnx = import_onnx()
    source_path = Path(path)
    model = onnx.load(source_path)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    graph = model.graph
    initializer_names = {initializer.name for initializer in graph.initializer}

    tensor_shapes: dict[str, list[int]] = {}
    tensor_dtypes: dict[str, str] = {}

    for initializer in graph.initializer:
        tensor_shapes[initializer.name] = [int(dim) for dim in initializer.dims] or [1]
        tensor_dtypes[initializer.name] = _dtype_name(initializer.data_type, onnx)

    for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
        if value_info.type.HasField('tensor_type') and value_info.type.tensor_type.elem_type:
            tensor_shapes[value_info.name] = _shape_from_value_info(value_info)
            tensor_dtypes[value_info.name] = _dtype_name(value_info.type.tensor_type.elem_type, onnx)

    graph_inputs = {
        value_info.name: ExternalTensorSpec(
            name=value_info.name,
            shape=tensor_shapes.get(value_info.name, [1]),
            dtype=tensor_dtypes.get(value_info.name, 'float32'),
        )
        for value_info in graph.input
        if value_info.name not in initializer_names
    }

    used_node_names: set[str] = set(graph_inputs)
    node_name_to_proto: dict[str, Any] = {}
    node_name_to_index: dict[str, int] = {}
    node_name_to_output_tensor: dict[str, str] = {}
    output_tensor_to_node_name: dict[str, str] = {}
    ordered_node_names: list[str] = []

    for index, node in enumerate(graph.node):
        outputs = [name for name in node.output if name]
        if len(outputs) != 1:
            raise NotImplementedError(
                f"Only single-output ONNX nodes are supported right now; node '{node.name or node.op_type}' "
                f"has outputs {outputs}."
            )
        internal_name = _make_unique(_sanitize_name(node.name or outputs[0] or f'{node.op_type}_{index}'), used_node_names)
        output_tensor = outputs[0]
        node_name_to_proto[internal_name] = node
        node_name_to_index[internal_name] = index
        node_name_to_output_tensor[internal_name] = output_tensor
        output_tensor_to_node_name[output_tensor] = internal_name
        ordered_node_names.append(internal_name)

    tensor_consumers: dict[str, list[str]] = {}
    nodes: dict[str, NodeSpec] = {}
    proto_to_internal = {id(proto): name for name, proto in node_name_to_proto.items()}

    for node in graph.node:
        internal_name = proto_to_internal[id(node)]
        output_tensor = node_name_to_output_tensor[internal_name]
        onnx_inputs = [tensor_name for tensor_name in node.input if tensor_name]
        inputs: list[str] = []
        initializer_inputs: list[str] = []
        for tensor_name in onnx_inputs:
            if tensor_name in initializer_names:
                initializer_inputs.append(tensor_name)
                continue
            producer = output_tensor_to_node_name.get(tensor_name)
            inputs.append(producer if producer is not None else tensor_name)
            tensor_consumers.setdefault(tensor_name, []).append(internal_name)

        attrs = _attrs_to_dict(node, onnx)
        _enrich_attrs_from_initializers(node.op_type, attrs, initializer_inputs, tensor_shapes, onnx_inputs)
        if onnx_inputs and 'input_shape' not in attrs and onnx_inputs[0] in tensor_shapes:
            attrs['input_shape'] = tensor_shapes[onnx_inputs[0]]
        attrs['_onnx_input_tensors'] = list(onnx_inputs)
        attrs['_onnx_initializer_tensors'] = list(initializer_inputs)
        attrs['_onnx_output_tensor'] = output_tensor
        attrs['_onnx_domain'] = node.domain
        attrs['_onnx_node_name'] = node.name

        nodes[internal_name] = NodeSpec(
            name=internal_name,
            op_type=node.op_type,
            inputs=inputs,
            output_shape=tensor_shapes.get(output_tensor, [1]),
            dtype=tensor_dtypes.get(output_tensor, 'float32'),
            attrs=attrs,
        )

    outputs: list[str] = []
    for value_info in graph.output:
        tensor_name = value_info.name
        outputs.append(output_tensor_to_node_name.get(tensor_name, tensor_name))

    graph_model = GraphModel(
        name=graph.name or source_path.stem,
        inputs=graph_inputs,
        nodes=nodes,
        outputs=outputs,
    )
    return ImportedOnnxModel(
        source_path=source_path,
        model=model,
        graph_model=graph_model,
        node_name_to_proto=node_name_to_proto,
        node_name_to_index=node_name_to_index,
        node_name_to_output_tensor=node_name_to_output_tensor,
        output_tensor_to_node_name=output_tensor_to_node_name,
        tensor_consumers=tensor_consumers,
        graph_output_tensors={value_info.name for value_info in graph.output},
        ordered_node_names=ordered_node_names,
    )


def export_fused_onnx(imported: ImportedOnnxModel, plan: PlanResult, output_path: str | Path) -> Path:
    onnx = import_onnx()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    multi_node_blocks = [block for block in plan.blocks if len(block.nodes) > 1]
    if not multi_node_blocks:
        model_copy = _copy_proto(imported.model, onnx.ModelProto)
        onnx.save(model_copy, output_path)
        return output_path

    block_by_node: dict[str, PlanBlock] = {}
    units: dict[str, dict[str, Any]] = {}
    order_key: dict[str, int] = {}
    new_functions: list[Any] = []

    for block in multi_node_blocks:
        unit_id = f'block_{block.block_id}'
        for node_name in block.nodes:
            block_by_node[node_name] = block
        function_proto, fused_node = _build_fused_function(onnx, imported, block)
        new_functions.append(function_proto)
        units[unit_id] = {
            'node': fused_node,
            'inputs': [name for name in fused_node.input if name],
            'outputs': [name for name in fused_node.output if name],
        }
        order_key[unit_id] = min(imported.node_name_to_index[node_name] for node_name in block.nodes)

    for node_name in imported.ordered_node_names:
        if node_name in block_by_node:
            continue
        proto = _copy_proto(imported.node_name_to_proto[node_name], onnx.NodeProto)
        unit_id = f'node_{node_name}'
        units[unit_id] = {
            'node': proto,
            'inputs': [name for name in proto.input if name],
            'outputs': [name for name in proto.output if name],
        }
        order_key[unit_id] = imported.node_name_to_index[node_name]

    producer_by_tensor: dict[str, str] = {}
    for unit_id, unit in units.items():
        for tensor_name in unit['outputs']:
            producer_by_tensor[tensor_name] = unit_id

    indegree = {unit_id: 0 for unit_id in units}
    successors = {unit_id: set() for unit_id in units}
    for unit_id, unit in units.items():
        for tensor_name in unit['inputs']:
            producer = producer_by_tensor.get(tensor_name)
            if producer is None or producer == unit_id:
                continue
            if unit_id not in successors[producer]:
                successors[producer].add(unit_id)
                indegree[unit_id] += 1

    ready = sorted([unit_id for unit_id, degree in indegree.items() if degree == 0], key=lambda item: order_key[item])
    ordered_unit_ids: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered_unit_ids.append(current)
        next_units: list[str] = []
        for succ in sorted(successors[current], key=lambda item: order_key[item]):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                next_units.append(succ)
        ready.extend(next_units)
        ready.sort(key=lambda item: order_key[item])

    if len(ordered_unit_ids) != len(units):
        raise RuntimeError('Failed to topologically sort fused ONNX units.')

    fused_graph = _copy_proto(imported.model.graph, onnx.GraphProto)
    del fused_graph.node[:]
    fused_graph.node.extend([units[unit_id]['node'] for unit_id in ordered_unit_ids])
    del fused_graph.value_info[:]

    fused_model = _copy_proto(imported.model, onnx.ModelProto)
    fused_model.graph.CopyFrom(fused_graph)
    del fused_model.functions[:]
    fused_model.functions.extend(new_functions)
    fused_model.producer_name = 'codex-fusion-lab'
    if not any(opset.domain == 'fusion_lab' for opset in fused_model.opset_import):
        fused_model.opset_import.extend([onnx.helper.make_operatorsetid('fusion_lab', 1)])

    try:
        fused_model = onnx.shape_inference.infer_shapes(fused_model)
    except Exception:
        pass

    onnx.checker.check_model(fused_model)
    onnx.save(fused_model, output_path)
    return output_path


def _build_fused_function(onnx_module: Any, imported: ImportedOnnxModel, block: PlanBlock) -> tuple[Any, Any]:
    external_inputs = _block_external_inputs(imported, block.nodes)
    external_outputs = _block_external_outputs(imported, block.nodes)

    function_name = f'FusedBlock_{block.block_id}'
    doc_string = f'Fusion block {block.block_id}: ' + ', '.join(block.nodes)
    body_nodes = [
        _copy_proto(imported.node_name_to_proto[node_name], onnx_module.NodeProto)
        for node_name in block.nodes
    ]

    function_proto = onnx_module.helper.make_function(
        domain='fusion_lab',
        fname=function_name,
        inputs=external_inputs,
        outputs=external_outputs,
        nodes=body_nodes,
        opset_imports=[_copy_proto(opset, onnx_module.OperatorSetIdProto) for opset in imported.model.opset_import],
        doc_string=doc_string,
    )

    fused_node = onnx_module.helper.make_node(
        function_name,
        inputs=external_inputs,
        outputs=external_outputs,
        name=f'fused_block_{block.block_id}',
        domain='fusion_lab',
        original_nodes=','.join(block.nodes),
        dominant_pattern=block.dominant_pattern,
    )
    return function_proto, fused_node


def _block_external_inputs(imported: ImportedOnnxModel, node_names: list[str]) -> list[str]:
    group = set(node_names)
    ordered_inputs: list[str] = []
    seen: set[str] = set()
    for node_name in node_names:
        node = imported.node_name_to_proto[node_name]
        for tensor_name in node.input:
            if not tensor_name:
                continue
            producer = imported.output_tensor_to_node_name.get(tensor_name)
            if producer not in group and tensor_name not in seen:
                seen.add(tensor_name)
                ordered_inputs.append(tensor_name)
    return ordered_inputs


def _block_external_outputs(imported: ImportedOnnxModel, node_names: list[str]) -> list[str]:
    group = set(node_names)
    ordered_outputs: list[str] = []
    for node_name in node_names:
        tensor_name = imported.node_name_to_output_tensor[node_name]
        consumers = imported.tensor_consumers.get(tensor_name, [])
        if tensor_name in imported.graph_output_tensors or any(consumer not in group for consumer in consumers):
            ordered_outputs.append(tensor_name)
    return ordered_outputs
