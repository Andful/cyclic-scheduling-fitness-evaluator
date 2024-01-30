from functools import reduce
from heapq import heapify, heappop, heappush
from itertools import repeat
from sdf_expander import Sdf
from enum import Enum
from math import ceil
import sys
from networkx.algorithms.minors import contracted_edge
from networkx.algorithms.components import node_connected_component
from networkx import DiGraph, Graph, induced_subgraph
from networkx.algorithms.shortest_paths.weighted import negative_edge_cycle, find_negative_cycle
from zigzag.utils import pickle_deepcopy
from dataclasses import dataclass, replace
from enum import IntEnum
from stream.classes.opt.allocation.genetic_algorithm.fitness_evaluator import FitnessEvaluator
from stream.classes.cost_model.communication_manager import CommunicationLinkEvent
from typing import Optional, Any, Dict, cast, Hashable
from stream.classes.workload.computation_node import ComputationNode
from stream.classes.workload.simd_node import SimdNode
from stream.classes.workload.elementwise_node import ElementwiseNode
from stream.classes.workload.flatten_node import FlattenNode
from stream.classes.workload.dummy_node import DummyNode
from stream.classes.hardware.architecture.accelerator import Accelerator
from .mapping import WorkloadNode, WorkloadEdge, display_workload_graph, workload_graph_to_mapped_graph, display_mapped_graph, mapped_to_hsdf, minimum_memory_schedule, display_mapped_hsdf, MappedEdge, MappedNode

from cyclic_scheduling import CyclicSchedulingProblem

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

    def __init__(self, accelerator: Accelerator, workload):
        self.latency = 0
        from stream.classes.cost_model.scheduler import initialize_offchip_tensors
        self.accelerator = accelerator
        self.workload = workload
        initialize_offchip_tensors(workload, accelerator)

    def copy_tensor(self, core1, core2, tensor, start, duration, energy):
        #print(f"Copyimg {tensor} at {start} from {core1} to {core2}")
        [communication_link, *_] = self.accelerator.communication_manager.get_links_for_pair(core1, core2)
        communication_link.transfer(
            CommunicationLinkEvent(
                "transfer",
                start=start,
                end=start + duration,
                tensors=[tensor],
                energy=energy
            )
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
            print(f"WARNING: no {tensor} in {core1} {top_instance}")

        self.latency = max(self.latency, start + duration)
        
    def free(self, core, tensor, start):
        #print(f"Freeimg {tensor} at {start} in {core}")
        memory_op = tensor.memory_operand
        top_instance = self.accelerator.get_top_instance_of_core(core, memory_op)
        if not self.accelerator.memory_manager.contains(tensor, top_instance):
            print(f"WARNING: no {tensor} in {core} {top_instance}")
        self.accelerator.memory_manager.remove_tensor_from_top_instance(
            top_instance,
            tensor,
            start,
        )

    def compute(self, core, cn, start, duration):
        #print(f"Executing {cn.name} at {start} producing {cn.operand_tensors['O']} in {core}")
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


class CyclicFitnessEvaluator(FitnessEvaluator):
    metadata: Dict[str, ActorMetadata]
    accelerator: Accelerator

    def __init__(self, workload, original_workload, node_hw_performances, layer_groups_flexible, accelerator, start_layer, layer_cuts, sdf_relation):
        self.weights = [-1, -1]
        self.metrics = ["energy", "latency"]
        self.original_workload = original_workload
        self.workload = workload
        self.layer_groups_flexible = layer_groups_flexible
        self.accelerator = accelerator
        # custom
        self.start_layer = start_layer
        self.layer_cuts = layer_cuts
        self.sdf_relation = sdf_relation
        
        
        original_workload = self.original_workload
        start_layer = self.start_layer
        layer_cuts = self.layer_cuts
        
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
        
        self.finer_nodes = {n.name: n for n in self.workload.nodes() if n.id[1] == 0}

        workload = pickle_deepcopy(original_workload)
        
        start_nodes = list(n for n in workload.nodes() if isinstance(n, ComputationNode) and n.name == start_layer)
        
        assert len(start_nodes) > 0
        
        [start_node, *_] = start_nodes
        
        for src, trg in list(workload.edges()):
            if hasattr(src, 'name') and hasattr(trg, 'name') and (src.name, trg.name) in layer_cuts:
                workload.remove_edge(src, trg)
                
        workload_nodes = node_connected_component(Graph(workload), start_node)
        workload = induced_subgraph(workload, workload_nodes).copy()
        
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
            
            cn = WorkloadNode(
                id=node.name,
                kernel_size=kernel,
                padding=padding,
                stride=stride,
                production_rate=production_rate,
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
        core_allocations = [0, 0, 1, 0, 0]
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

        minimum_memory_schedule(hsdf, self.mrsdf)

        cyclic_scheduling = CyclicSchedulingProblem()

        for core in filter(lambda x: x is not None, set(cast(MappedNode, a).processor for a in mapped.nodes)):
            cyclic_scheduling.add_processor(core)
        cyclic_scheduling.add_processor("other")

        node_to_index = {n: i for i, n in enumerate(hsdf.nodes)}

        layers = sorted(((actor.name, actor.id[0]) for actor in self.finer_nodes.values()), key=lambda x: x[1])
        from matplotlib.colors import to_hex
        from matplotlib import colormaps
        import numpy as np
        rainbow = colormaps["rainbow"]
        colors = dict(zip(map(lambda x: x[0], layers), map(to_hex, rainbow(np.linspace(0, 1, len(layers))))))
        nodes = list(hsdf.nodes)
        for a in nodes:
            cyclic_scheduling.add_actor(f"{node_to_index[a]})", a[0].execution_time, "other" if a[0].processor is None else a[0].processor, color=colors[a[0].id])

        for a, b, c in hsdf.edges.data('weight'):
            cyclic_scheduling.add_channel(f"{node_to_index[a]})", f"{node_to_index[b]})", c.initial_tokens)

        solution = cyclic_scheduling.solution(relaxed=True)

        if not return_scme:
            return 0, solution.cycle_time
        else:

            cycle_time = solution.cycle_time

            def count_repetition(repetition: dict[Hashable, int], a):
                (actor, _) = a
                repetition.setdefault(actor, 0)
                repetition[actor] += 1
                return repetition

            actor_repetitions = reduce(count_repetition, nodes, dict())
            actor_priority_queue = list(zip(solution.t, repeat(0), hsdf.nodes()))
            heapify(actor_priority_queue)

            workload = pickle_deepcopy(self.workload)
            accelerator = pickle_deepcopy(self.accelerator)
            vm = AcceleratorVirtualMachine(accelerator, workload)

            energy = 0 #TODO fix
            latency = 0

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
                    
                    vm.compute(core, cn, t, duration)
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
                    vm.copy_tensor(core1, core2, cn1.operand_tensors['O'], t, actor.execution_time, 0) # TODO: ifx
                elif actor.t.t == 'Load':
                    cn = computational_nodes.get((actor.id, instance*repetition + i))
                    if cn is None:
                        continue
                    core = self.get_core(cn.name, layer_core_mapping)
                    vm.copy_tensor(offchip, core, cn.operand_tensors['I'], t, actor.execution_time, 0) # TODO fix
                elif actor.t.t == 'Store':
                    cn = computational_nodes.get((actor.id, instance*repetition + i))
                    if cn is None:
                        continue
                    core = self.get_core(cn.name, layer_core_mapping)
                    vm.copy_tensor(core, offchip, cn.operand_tensors['O'], t, actor.execution_time, 0) # TODO fix
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
            #print(f"LATENCY {vm.latency}")
            store = cast(MappedNode, next(filter(lambda n: n.t.t == "Store", mapped.nodes)))
            #print(f"ESTIMATED LATENCY {}")
            scme.energy = energy
            scme.cyclic_scheduling = cyclic_scheduling
            return 1, solution.cycle_time, scme

    def get_core(self, actor_name: str, layer_core_mapping: Dict[str, int]):
        return self.accelerator.get_core(layer_core_mapping[actor_name])
   
        
        
class CyclicFitnessEvaluatorBuilder:
    def __init__(self, start_layer, layer_cuts, sdf_relation):
        self.start_layer = start_layer
        self.layer_cuts = layer_cuts
        self.sdf_relation = sdf_relation
        
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
            start_layer=self.start_layer,
            layer_cuts=self.layer_cuts,
            sdf_relation=self.sdf_relation,
            workload=workload,
            original_workload=original_workload,
            node_hw_performances=node_hw_performances,
            layer_groups_flexible=list(map(lambda a: m[a[0]], layer_groups_flexible)),
            accelerator=accelerator
        )