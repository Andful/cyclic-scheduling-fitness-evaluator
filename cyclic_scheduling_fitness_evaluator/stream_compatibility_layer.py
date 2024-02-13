from functools import reduce
from heapq import heapify, heappop, heappush
from itertools import repeat
from sdf_expander import Sdf
from enum import Enum
from math import ceil
import numpy as np
import sys
import networkx as nx
from networkx.algorithms.minors import contracted_edge
from networkx.algorithms.components import node_connected_component
from networkx import DiGraph, Graph, induced_subgraph
from networkx.algorithms.shortest_paths.weighted import negative_edge_cycle, find_negative_cycle
from zigzag.utils import pickle_deepcopy
from dataclasses import dataclass, replace
from enum import IntEnum
from stream.classes.opt.allocation.genetic_algorithm.fitness_evaluator import FitnessEvaluator
from stream.classes.cost_model.communication_manager import CommunicationLinkEvent
from typing import Optional, Any, Dict, cast, Hashable, Union, Literal
from stream.classes.workload.computation_node import ComputationNode
from stream.classes.workload.simd_node import SimdNode
from stream.classes.workload.elementwise_node import ElementwiseNode
from stream.classes.workload.flatten_node import FlattenNode
from stream.classes.workload.dummy_node import DummyNode
from stream.classes.hardware.architecture.accelerator import Accelerator
from stream.visualization.memory_usage import humanbytes
from logging import warning, info
from .mapping import WorkloadNode, WorkloadEdge, display_workload_graph, workload_graph_to_mapped_graph, display_mapped_graph, mapped_to_hsdf, earliest_first, display_mapped_hsdf, MappedEdge, MappedNode, add_minimum_buffers, Convolution1D, latest_first, OptimizationType, optimize

class IdentifierType(IntEnum):
    LoadInput = 0
    ComputationNode = 1
    DataTransfer = 2
    StoreOutput = 3

@dataclass
class StreamIdentifier:
    identifier_type: IdentifierType
    identifier: Any

@dataclass
class Mapping:
    latency: int
    energy: int
    processor: str
        
@dataclass
class ActorMetadata:
    n_execution: int
    stream_identifier: StreamIdentifier
    mapping: Optional[Mapping] = None
        
    def assign_mapping(self, mapping: Mapping) -> 'ActorMetadata':
        return replace(self, mapping=mapping)
    
def display_token_graph(token_graph):
    from IPython.display import display
    import graphviz

    dot = graphviz.Digraph()
    for (a, i) in token_graph.nodes():
        dot.node(f"{a}({i})", f"{a}({i})")

    for a, b in token_graph.edges:
        dot.edge(f"{a[0]}({a[1]})", f"{b[0]}({b[1]})", label=str(token_graph.edges[a,b]['weight']), style=token_graph.edges[a,b].get('style', None))

    display(dot)

def has_cyclic_dependence(token_graph):
    g = DiGraph()
    g.add_weighted_edges_from([(a, b, c - 1e-3) for a,b,c in token_graph.edges.data('weight')])
    return negative_edge_cycle(g)

@dataclass
class AcceleratorVirtualMachine:
    accelerator: Accelerator
    latency: int
    energy: int
    considered_dimension: str

    def __init__(self, accelerator: Accelerator, workload, considered_dimension):
        self.latency = 0
        self.energy = 0
        from stream.classes.cost_model.scheduler import initialize_offchip_tensors
        self.accelerator = accelerator
        self.workload = workload
        self.considered_dimensions = [considered_dimension, {'OX':'IX', 'OY':'IY'}[considered_dimension]]
        initialize_offchip_tensors(workload, accelerator)

    def load_weights(self, layer_core_mapping: Dict[str, int]) -> int:
        offchip_core_id = self.accelerator.offchip_core_id
        offchip_core = self.accelerator.get_core(offchip_core_id)

        time_offset = 0
        already_loaded = set()
        for n in self.workload.nodes():
            for op, tensor in n.operand_tensors.items():
                if op in n.constant_operands and all(d not in tensor.loop_dimensions for d in self.considered_dimensions):
                    #Move tensor
                    core_id = layer_core_mapping[n.name]
                    if (core_id, tensor) in already_loaded:
                        continue
                    already_loaded.add((core_id, tensor))
                    core = self.accelerator.get_core(layer_core_mapping[n.name])
                    [communication_link, *_] = self.accelerator.communication_manager.get_links_for_pair(offchip_core, core)
                    transfer_time = ceil(tensor.size/communication_link.bandwidth)
                    self.copy_tensor(offchip_core, core, tensor, time_offset, transfer_time)
                    time_offset += transfer_time
        return time_offset


    def copy_tensor(self, core1, core2, tensor, start, duration):
        info(f"Copying {tensor} at {start} from {core1} to {core2}")
        links = self.accelerator.communication_manager.get_links_for_pair(core1, core2)
        link = max(links, key= lambda link: ceil(tensor.size / link.bandwidth))
        duration2 = ceil(tensor.size / link.bandwidth)
        assert duration == duration2
        energy = link.unit_energy_cost*duration
        link.transfer(
            CommunicationLinkEvent(
                "transfer",
                start=start,
                end=start + duration,
                tensors=[tensor],
                energy=energy
            )
        )
        self.energy += energy

        self.energy += self.accelerator.get_memory_energy_cost_of_transfer(
            tensor, core1, core2, tensor.memory_operand, tensor.memory_operand
        )

        self.accelerator.memory_manager.add_tensor_to_core(
            tensor=tensor,
            core=core2,
            timestep=start,
            timestep_end=start + duration,
        )

        memory_op = tensor.memory_operand
        top_instance = self.accelerator.get_top_instance_of_core(core1, memory_op)
        if not self.accelerator.memory_manager.contains(tensor, top_instance):
            warning(f"no {tensor} in {core1} {top_instance}")

        self.latency = max(self.latency, start + duration)
        
    def free(self, core, tensor, start):
        info(f"Freeing {tensor} at {start} in {core}")
        memory_op = tensor.memory_operand
        top_instance = self.accelerator.get_top_instance_of_core(core, memory_op)
        if not self.accelerator.memory_manager.contains(tensor, top_instance):
            info(f"WARNING: no {tensor} in {core} {top_instance}")
        self.accelerator.memory_manager.remove_tensor_from_top_instance(
            top_instance,
            tensor,
            start,
        )

    def compute(self, core, cn, start, duration, energy):
        info(f"Executing {cn.name} at {start} producing {cn.operand_tensors['O']} in {core}")
        out_tensor = cn.operand_tensors['O']

        self.accelerator.memory_manager.add_tensor_to_core(
            tensor=out_tensor,
            core=core,
            timestep=start,
            timestep_end=start + duration,
        )

        cn.core_allocation = core.id
        cn.start = start
        cn.runtime = duration
        self.energy += energy

        if cn.onchip_energy is not None:
            assert cn.onchip_energy == energy


class CyclicFitnessEvaluator(FitnessEvaluator):
    metadata: Dict[str, ActorMetadata]
    accelerator: Accelerator
    sdf_relation: Union[Literal['OX'], Literal['OY']]
    optimization_type: OptimizationType


    def __init__(self, workload, original_workload, node_hw_performances, layer_groups_flexible, accelerator, sdf_relation, optimization_type: OptimizationType):
        self.weights = [-1, -1]
        self.metrics = ["energy estimate", "cycle time"]
        self.original_workload = original_workload
        self.workload = workload
        self.layer_groups_flexible = layer_groups_flexible
        self.accelerator = accelerator
        # custom
        self.sdf_relation = sdf_relation
        self.optimization_type = optimization_type
        
        original_workload = self.original_workload
        
        self.mappings = {
            layer.name: {
                core.id: Mapping(p.latency_total1, p.energy_total, processor=f"Core {core.id}") 
                for (core, p) in cme.items()
            } for (layer, cme) in node_hw_performances.items()
        }

        import sympy as sp
        
        ox = sp.Symbol("ox")
        fx = sp.Symbol("fx")
        oy = sp.Symbol("oy")
        fy = sp.Symbol("fy")
        
        self.finer_nodes: dict[str, ComputationNode] = {n.name: n for n in self.workload.nodes() if n.id[1] == 0}

        workload = pickle_deepcopy(original_workload)
        
        for src, trg in list(workload.edges()):
            if isinstance(trg, DummyNode):
                assert(workload.out_degree(src) == 1)
                contracted_edge(workload, (src, trg), False, False) 
        
        mrsdf = DiGraph()
        nodes: dict[str, WorkloadNode] = {}
        for node in workload.nodes():
            finer_node = self.finer_nodes[node.name]
            is_source = workload.in_degree(node) == 0
            is_sink = workload.out_degree(node) == 0
            n_execution = node.loop_dim_size[self.sdf_relation]//finer_node.loop_dim_size[self.sdf_relation]
        
            if isinstance(node, DummyNode):
                raise ValueError("There should not be any DummyNode")
                
            if isinstance(node, FlattenNode):
                raise ValueError(f"Unsupported Node {type(node)}")
                
            if not isinstance(node, ComputationNode):
                continue
            
            if isinstance(node, SimdNode) or isinstance(node, ElementwiseNode):
                x_stride = 1
                x_dilation = 1
                y_stride = 1
                y_dilation = 1
                x_padding = 0
                y_padding = 0
                x_kernel = 1
                y_kernel = 1
            else:
                relations = node.layer_attrs['dimension_relations']

                if len(relations) != 2:
                    sys.exit("Matrix Multiplication Unsupported")

                x_wise, y_wise = relations
                x_poly = sp.Poly(x_wise.split("=")[1])
                y_poly = sp.Poly(y_wise.split("=")[1])

                x_stride = int(x_poly.coeff_monomial(ox))
                x_dilation = int(x_poly.coeff_monomial(fx))
                y_stride = int(y_poly.coeff_monomial(oy))
                y_dilation = int(y_poly.coeff_monomial(fy))
                x_padding, _ = node.padding['IX']
                y_padding, _ = node.padding['IY']
                x_kernel = node.loop_dim_size['FX']
                y_kernel = node.loop_dim_size['FY']
            
            if self.sdf_relation == 'OX':
                kernel = x_kernel
                padding = x_padding
                stride = x_stride
            elif self.sdf_relation == 'OY':
                kernel = y_kernel
                padding = y_padding
                stride = y_stride
            else:
                raise ValueError(f"Unknown relation {self.sdf_relation}")
            
            finer_trg = self.finer_nodes[node.name]
            production_rate = finer_trg.loop_dim_size[self.sdf_relation]
            assert (finer_trg.operand_tensors['O'].size % production_rate) == 0
            
            get_size_or_assume_0 = lambda x: x.size if x is not None else 0

            cn = WorkloadNode(
                id=node.name,
                conv1d=Convolution1D(
                    kernel_size=kernel,
                    padding=padding,
                    stride=stride,
                    production_rate=production_rate,
                    output_token_size=finer_trg.operand_tensors['O'].size // production_rate,
                    input_token_size=get_size_or_assume_0(finer_trg.operand_tensors.get('I')), # Assume this is not first layer if it has no I
                ),
                n_execution=n_execution,
            )
            nodes[node.name] = cn
            mrsdf.add_node(cn)

        for (_s, _t) in workload.edges:
            s = nodes[_s.name]
            t = nodes[_t.name]
            mrsdf.add_edge(s, t, weight=WorkloadEdge(
                initial_tokens=t.in_initial_tokens(),
                production_rate=s.production_rate,
                consumption_rate=t.consumption_rate(),
            ))
        
        self.fixed_core_allocations = {
            n: next(iter(layer_possible_mappings.keys()))
            for n, layer_possible_mappings in self.mappings.items()
            if len(layer_possible_mappings.keys()) == 1
        }
        
        self.mrsdf = mrsdf
    
    def get_fitness(self, core_allocations, return_scme=False):
        from dataclasses import replace
        layer_core_mapping: Dict[str, int] = self.fixed_core_allocations | { self.finer_nodes[layer_name].name: core_id for layer_name, core_id in zip(self.layer_groups_flexible, core_allocations) }
        
        offchip = self.accelerator.get_core(self.accelerator.offchip_core_id)

        def get_channel_from_cores(a, b):
            link = self.accelerator.communication_manager.get_links_for_pair(a, b)
            return link[0]
        
        get_channel_link = lambda a, b: get_channel_from_cores(self.get_core(a, layer_core_mapping), self.get_core(b, layer_core_mapping))
        get_load_channel_link = lambda a: get_channel_from_cores(offchip, self.get_core(a, layer_core_mapping))
        get_store_channel_link = lambda a: get_channel_from_cores(self.get_core(a, layer_core_mapping), offchip)

        def get_channel(a, b):
            c = get_channel_link(a, b)
            return (c.sender, c.receiver)
        def get_load_channel(a):
            c = get_load_channel_link(a)
            return (c.sender, c.receiver)
        def get_store_channel(a):
            c = get_store_channel_link(a)
            return (c.sender, c.receiver)

        get_transfer_time = lambda a,b: ceil(self.finer_nodes[a].operand_tensors['O'].size/get_channel_link(a, b).bandwidth)
        get_load_time = lambda a: ceil(self.finer_nodes[a].operand_tensors['I'].size/get_channel_from_cores(offchip, self.get_core(a, layer_core_mapping)).bandwidth)
        get_store_time = lambda a: ceil(self.finer_nodes[a].operand_tensors['O'].size/get_channel_from_cores(self.get_core(a, layer_core_mapping), offchip).bandwidth)

        #display_workload_graph(self.mrsdf)

        mapped = workload_graph_to_mapped_graph(
            workload=self.mrsdf,
            core_mapping=lambda s: self.get_core(s, layer_core_mapping),
            get_channel=get_channel,
            get_load_channel=get_load_channel,
            get_load_time=get_load_time,
            get_store_channel=get_store_channel,
            get_store_time=get_store_time,
            get_execution_time=lambda a: self.mappings[a][layer_core_mapping[a]].latency,
            get_transfer_time=get_transfer_time
        )

        #display_mapped_graph(mapped)

        (hsdf, _) = mapped_to_hsdf(mapped)

        #display_mapped_hsdf(hsdf)
        add_minimum_buffers(hsdf)
        cycle_time, t = optimize(hsdf, self.optimization_type)

        total_buffer_size = sum(e.t.buffer_token_size*e.initial_tokens for e in map(lambda e: cast(MappedEdge, e[2]), hsdf.edges.data('weight')) if e.t.t == 'TensorBuffer')
            

        #earliest_first(hsdf, self.mrsdf)
        latest_first(hsdf, self.mrsdf)

        def get_load_energy(i):
            c = get_load_channel_link(i)
            return ceil(self.finer_nodes[i].operand_tensors['I'].size/c.bandwidth)*c.unit_energy_cost
        
        def get_store_energy(i):
            c = get_store_channel_link(i)
            return ceil(self.finer_nodes[i].operand_tensors['O'].size/c.bandwidth)*c.unit_energy_cost
        
        def get_transfer_energy(i1, i2):
            c = get_channel_link(i1, i2)
            return ceil(self.finer_nodes[i1].operand_tensors['O'].size/c.bandwidth)*c.unit_energy_cost

        energy = sum(self.mappings[n.id][layer_core_mapping[n.id]].energy * n.n_execution for n in self.mrsdf.nodes) + \
                sum(get_load_energy(n.id) * n.n_execution for n in self.mrsdf.nodes if self.mrsdf.in_degree(n) == 0) + \
                sum(get_store_energy(n.id) * n.n_execution for n in self.mrsdf.nodes if self.mrsdf.out_degree(n) == 0) + \
                sum(get_transfer_energy(n1.id, n2.id) * n1.n_execution for n1, n2 in self.mrsdf.edges if self.get_core(n1.id, layer_core_mapping) != self.get_core(n2.id, layer_core_mapping))
        if not return_scme:
            return energy, cycle_time
        else:

            def count_repetition(repetition: dict[Hashable, int], a):
                (actor, _) = a
                repetition.setdefault(actor, 0)
                repetition[actor] += 1
                return repetition

            actor_repetitions = reduce(count_repetition, hsdf.nodes, dict())

            workload = pickle_deepcopy(self.workload)
            accelerator = pickle_deepcopy(self.accelerator)
            vm = AcceleratorVirtualMachine(accelerator, workload, self.sdf_relation)
            time_offset = vm.load_weights(layer_core_mapping)

            actor_priority_queue = list(zip(t + time_offset, repeat(0), hsdf.nodes()))
            heapify(actor_priority_queue)

            # Handle events in cronological order

            computational_nodes = { (node.name, node.id[1]): node for node in workload.nodes() }
            while len(actor_priority_queue) != 0:
                t, instance, (actor, i) = heappop(actor_priority_queue)
                repetition = actor_repetitions[actor]
                actor = cast(MappedNode, actor)
                if actor.t.t == 'Computation':
                    cn = computational_nodes.get((actor.id, instance*repetition + i))
                    if cn is None:
                        continue
                    core = self.get_core(actor.id, layer_core_mapping)
                    duration = actor.execution_time

                    
                    vm.compute(core, cn, t, duration, self.mappings[actor.id][layer_core_mapping[actor.id]].energy)
                elif actor.t.t == 'Transfer':
                    (_, (n2, _)) = next(filter(lambda x: x[1][0].t.t == 'Computation', hsdf.out_edges((actor, i))))
                    index = instance*repetition + i
                    
                    cn1 = computational_nodes.get((actor.id, index))
                    if cn1 is None:
                        continue
                    cn2 = computational_nodes.get((n2.id, 0)) # No need to keep track of index
                    if cn2 is None: #This shouls not be none
                        continue
                    core1 = self.get_core(cn1.name, layer_core_mapping)
                    core2 = self.get_core(cn2.name, layer_core_mapping)
                    vm.copy_tensor(core1, core2, cn1.operand_tensors['O'], t, actor.execution_time)
                elif actor.t.t == 'Load':
                    cn = computational_nodes.get((actor.id, instance*repetition + i))
                    if cn is None:
                        continue
                    core = self.get_core(cn.name, layer_core_mapping)
                    vm.copy_tensor(offchip, core, cn.operand_tensors['I'], t, actor.execution_time)
                elif actor.t.t == 'Store':
                    cn = computational_nodes.get((actor.id, instance*repetition + i))
                    if cn is None:
                        continue
                    core = self.get_core(cn.name, layer_core_mapping)
                    vm.copy_tensor(core, offchip, cn.operand_tensors['O'], t, actor.execution_time)
                elif actor.t.t == 'MemoryFree':
                    cn = computational_nodes.get((actor.id, instance*repetition + i))
                    if cn is None:
                        continue
                    vm.free(actor.t.processor, cn.operand_tensors[actor.t.operator], t)
                else:
                    continue
                    
                heappush(actor_priority_queue, (t + cycle_time, instance + 1, (actor, i)))

            from stream.classes.cost_model.cost_model import StreamCostModelEvaluation

            scme = StreamCostModelEvaluation(
                workload,
                accelerator,
                scheduling_order=[],
                operands_to_prefetch=[],
            )
            scme.latency = vm.latency

            self.workload
            scme.energy = vm.energy
            return energy, cycle_time, scme

    def get_core(self, actor_name: str, layer_core_mapping: Dict[str, int]):
        return self.accelerator.get_core(layer_core_mapping[actor_name])

@dataclass
class MinimumMemory:
    t = "MinimumMemory"

@dataclass
class MinimumLatency:
    t = "MinimumLatency"

@dataclass
class MinimumLatency:
    t ="MinimumLatency"
        
class CyclicFitnessEvaluatorBuilder:
    sdf_relation: Union[Literal['OX'], Literal['OY']]

    def __init__(self, sdf_relation, optimization_type: OptimizationType = MinimumMemory()):
        self.sdf_relation = sdf_relation
        self.optimization_type = optimization_type
        
    def __call__(
        self,
        workload,
        accelerator,
        node_hw_performances,
        layer_groups_flexible,
        scheduler_candidate_selection,
        operands_to_prefetch,
        original_workload,
    ):
        m = { n.id[0]: n.name for n in workload.nodes }
        return CyclicFitnessEvaluator(
            sdf_relation=self.sdf_relation,
            optimization_type=self.optimization_type,
            workload=workload,
            original_workload=original_workload,
            node_hw_performances=node_hw_performances,
            layer_groups_flexible=list(map(lambda a: m[a[0]], layer_groups_flexible)),
            accelerator=accelerator
        )
