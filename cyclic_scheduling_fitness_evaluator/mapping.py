from dataclasses import dataclass
from enum import Enum
from typing import Literal, Union, Optional, Generic, TypeVar, Hashable, Callable, cast
from networkx import DiGraph
import networkx as nx
from math import ceil
from sdf_expander import Sdf
from itertools import chain
import numpy as np


ID = TypeVar('ID', bound=Hashable)
PROCESSOR = TypeVar('PROCESSOR', bound=Hashable)
CORE = TypeVar('CORE', bound=Hashable)
CHANNEL = TypeVar('CHANNEL', bound=Hashable)
# Initial sdf

@dataclass
class Convolution1D:
    kernel_size: int
    padding: int
    stride: int
    production_rate: int
    output_token_size: int
    input_token_size: int

@dataclass
class WorkloadNode(Generic[ID]):

    id: ID
    conv1d: Convolution1D
    n_execution: int

    def __hash__(self) -> int:
        return hash(self.id)

    def production_rate(self):
        return self.conv1d.production_rate
    def consumption_rate(self):
        return self.production_rate() * self.conv1d.stride

    def in_initial_tokens(self):
        # After how many input rows can it fire
        return self.consumption_rate() - self.conv1d.kernel_size + self.conv1d.padding

    def freeing_initial_tokens(self):
        # After how many firing can it release memory
        return -self.conv1d.padding  # TODO


@dataclass
class WorkloadEdge:
    initial_tokens: int
    production_rate: int
    consumption_rate: int


def display_workload_graph(graph: DiGraph):
    from IPython.display import display
    import graphviz

    dot = graphviz.Digraph()
    for n in graph.nodes():
        name = str(n.id)
        dot.node(name, name)

    for s, t, c in graph.edges.data('weight'):
        dot.edge(str(s.id), str(t.id), label=str(c.initial_tokens), taillabel=str(
            c.production_rate), headlabel=str(c.consumption_rate))

    display(dot)


def display_mapped_graph(graph: DiGraph):
    from IPython.display import display
    import graphviz

    def name(n):
        return f"{n.t.t}({n.id})"
    
    nodes = list(graph.nodes())

    dot = graphviz.Digraph()
    for n in graph.nodes():
        dot.node(f"A({nodes.index(n)})", str(name(n)))

    for s, t, c in graph.edges.data('weight'):
        dot.edge(f"A({nodes.index(s)})", f"A({nodes.index(t)})", label=f"{c.t.t}({c.initial_tokens})", taillabel=str(
            c.production_rate), headlabel=str(c.consumption_rate))

    display(dot)

def display_mapped_hsdf(graph: DiGraph):
    from IPython.display import display
    import graphviz

    def name(n):
        return f"{n[0].t.t}({n[0].id},{n[1]})"

    dot = graphviz.Digraph()
    for n in graph.nodes():
        dot.node(name(n), name(n))

    for s, t, c in graph.edges.data('weight'):
        dot.edge(name(s), name(t), label=f"{c.t.t}({c.initial_tokens})", taillabel=str(
            c.production_rate), headlabel=str(c.consumption_rate), style='dashed' if cast(MappedEdge, c).t.t == 'Scheduling' else None)

    display(dot)


@dataclass
class TransferNodeType(Generic[CORE]):
    t: Literal['Transfer']
    core1: CORE
    core2: CORE
    conv1d: Convolution1D
    n_execution: int
    def __init__(self, core1: CORE, core2: CORE, conv1d: Convolution1D, n_execution: int):
        self.t = 'Transfer'
        self.core1 = core1
        self.core2 = core2
        self.conv1d = conv1d
        self.n_execution = n_execution


@dataclass
class ComputationNodeType:
    t: Literal['Computation']
    conv1d: Convolution1D
    n_execution: int

    def __init__(self, conv1d: Convolution1D, n_execution: int):
        self.t = 'Computation'
        self.conv1d = conv1d
        self.n_execution = n_execution


@dataclass
class MemoryFreeNodeType:
    t: Literal['MemoryFree']
    processor: Hashable
    operator: str

    def __init__(self, processor: Hashable, operator: str):
        self.t = 'MemoryFree'
        self.processor = processor
        self.operator = operator


@dataclass
class LoadType(Generic[CORE]):
    t: Literal['Load']
    core2: CORE
    output_token_size: int
    def __init__(self, core2: CORE, output_token_size):
        self.t = 'Load'
        self.core2 = core2
        self.output_token_size = output_token_size


@dataclass
class StoreType:
    t: Literal['Store']

    def __init__(self):
        self.t = 'Store'


MappedNodeType = Union[TransferNodeType,
                       ComputationNodeType, MemoryFreeNodeType, LoadType, StoreType]


@dataclass
class MappedNode(Generic[ID, PROCESSOR]):
    t: MappedNodeType
    id: ID
    execution_time: int
    processor: PROCESSOR

    def __hash__(self) -> int:
        return hash((self.t.t, self.id))
    
    def __lt__(self, o) -> int:
        return hash(self) < hash(o)


@dataclass
class DataEdgeType:
    t: Literal['Data']

    def __init__(self):
        self.t = 'Data'


@dataclass
class IntoTransferEdgeType:
    t: Literal['IntoTransfer']

    def __init__(self):
        self.t = 'IntoTransfer'


@dataclass
class OutFromTransferEdgeType:
    t: Literal['OutFromTransfer']

    def __init__(self):
        self.t = 'OutFromTransfer'


@dataclass
class TensorUsedEdgeType:
    t: Literal['TensorUsed']

    def __init__(self):
        self.t = 'TensorUsed'


@dataclass
class SameProcessorEdgeType:
    t: Literal['SameProcessor']

    def __init__(self):
        self.t = 'SameProcessor'


@dataclass
class TensorBufferEdgeType:
    t: Literal['TensorBuffer']
    buffer_token_size: int

    def __init__(self, buffer_token_size: int):
        self.t = 'TensorBuffer'
        self.buffer_token_size = buffer_token_size


@dataclass
class RemoveAutoConcurrencyEdgeType:
    t: Literal['RemoveAutoConcurrency']

    def __init__(self):
        self.t = 'RemoveAutoConcurrency'


@dataclass
class SchedulingEdgeType:
    t: Literal['Scheduling']

    def __init__(self):
        self.t = 'Scheduling'


MappedEdgeType = Union[DataEdgeType, IntoTransferEdgeType, OutFromTransferEdgeType,
                       TensorUsedEdgeType, TensorBufferEdgeType, RemoveAutoConcurrencyEdgeType, SchedulingEdgeType]


@dataclass
class MappedEdge:
    t: MappedEdgeType
    initial_tokens: int
    production_rate: int
    consumption_rate: int


UNMAPPED_ACTOR_ID = TypeVar('UNMAPPED_ACTOR_ID', bound=Hashable)
MAPPED_ACTOR_ID = TypeVar('MAPPED_ACTOR_ID', bound=Hashable)

@dataclass
class MinimumMemory:
    t: Literal['MinimumMemory']

    def __init__(self):
        self.t = 'MinimumMemory'

@dataclass
class MinimumLatency:
    t: Literal['MinimumLatency']

    def __init__(self):
        self.t = 'MinimumLatency'

@dataclass
class BoundLatencyMinimumMemory:
    t: Literal['BoundLatencyMinimumMemory']

    def __init__(self):
        self.t = 'BoundLatencyMinimumMemory'

@dataclass
class BoundMemoryMinimumLatency:
    t: Literal['BoundMemoryMinimumLatency']

    def __init__(self):
        self.t = 'BoundMemoryMinimumLatency'

OptimizationType = Union[MinimumMemory, MinimumLatency, BoundLatencyMinimumMemory, BoundMemoryMinimumLatency]



def workload_graph_to_mapped_graph(
        workload: DiGraph,
        core_mapping: Callable[[UNMAPPED_ACTOR_ID], CORE],
        get_channel: Callable[[UNMAPPED_ACTOR_ID, UNMAPPED_ACTOR_ID], CHANNEL],
        get_load_channel: Callable[[CORE], CHANNEL],
        get_load_time: Callable[[CORE], int],
        get_store_channel: Callable[[CORE], CHANNEL],
        get_store_time: Callable[[CORE], int],
        get_execution_time: Callable[[UNMAPPED_ACTOR_ID], int],
        get_transfer_time: Callable[[UNMAPPED_ACTOR_ID, UNMAPPED_ACTOR_ID], int],
) -> DiGraph:
    mapped = DiGraph()
    corresponding_mapped: dict[WorkloadNode[UNMAPPED_ACTOR_ID], MappedNode] = {
        n: MappedNode(
            t=ComputationNodeType(n.conv1d, n.n_execution),
            id=n.id,
            execution_time=get_execution_time(n.id),
            processor=core_mapping(n.id)
        ) for n in map(lambda n: cast(WorkloadNode, n), workload.nodes())
    }

    corresponding_workload = {b: a for a, b in corresponding_mapped.items()}

    for n in corresponding_mapped.values():
        mapped.add_node(n)

    #Convert nodes and add actors representing transfers
    for wn1, n in corresponding_mapped.items():
        core1 = core_mapping(wn1.id)

        child_per_core: dict[CORE, list[WorkloadNode]] = dict()

        for (_, wn2) in workload.out_edges(wn1):
            wn2 = cast(WorkloadNode, wn2)
            e = child_per_core.setdefault(core_mapping(wn2.id), [])
            e.append(wn2)

        for core2, childs_wn in child_per_core.items():
            if core1 == core2:
                for wn2 in childs_wn:
                    mapped.add_edge(
                        corresponding_mapped[wn1],
                        corresponding_mapped[wn2],
                        weight=MappedEdge(
                            t=DataEdgeType(),
                            initial_tokens=wn2.in_initial_tokens(),
                            production_rate=wn1.production_rate(),
                            consumption_rate=wn2.consumption_rate(),
                        )
                    )
            else:
                for wn2 in childs_wn:
                    transfer = MappedNode(
                        id=wn1.id,
                        t=TransferNodeType(core1, core2, wn1.conv1d, wn1.n_execution),
                        execution_time=get_transfer_time(wn1.id, wn2.id),
                        processor=get_channel(wn1.id, wn2.id)
                    )
                    mapped.add_node(transfer)
                    mapped.add_edge(
                        corresponding_mapped[wn1],
                        transfer,
                        weight=MappedEdge(
                            t=IntoTransferEdgeType(),
                            initial_tokens=0,
                            production_rate=wn1.production_rate(),
                            consumption_rate=wn1.production_rate(),
                        )
                    )
                    mapped.add_edge(
                        transfer,
                        corresponding_mapped[wn2],
                        weight=MappedEdge(
                            t=OutFromTransferEdgeType(),
                            initial_tokens=wn2.in_initial_tokens(),
                            production_rate=wn1.production_rate(),
                            consumption_rate=wn2.consumption_rate(),
                        )
                    )

    #Add Store Node
    for n1 in list(mapped.nodes):
        n1 = cast(MappedNode, n1)
        if mapped.out_degree(n1) == 0:
            store_node = MappedNode(
                t=StoreType(),
                id=n1.id,
                execution_time=get_store_time(n1.id),
                processor=get_store_channel(n1.id)
            )
            assert n1.t.t == 'Computation'
            mapped.add_node(store_node)
            production_rate = corresponding_workload[n1].production_rate()
            mapped.add_edge(n1, store_node, weight=MappedEdge(
                t=IntoTransferEdgeType(),
                initial_tokens=0,
                production_rate=production_rate,
                consumption_rate=production_rate
            ))

    #Add Load Node
    for n1 in list(mapped.nodes):
        if mapped.in_degree(n1) == 0:
            consumption_rate = corresponding_workload[n1].consumption_rate()
            assert (n1.t.conv1d.input_token_size % consumption_rate) == 0
            load_node = MappedNode(
                t=LoadType(n1.processor, n1.t.conv1d.input_token_size//consumption_rate), # because of how Stream works
                id=n1.id,
                execution_time=get_load_time(n1.id),
                processor=get_load_channel(n1.id)
            )
            mapped.add_node(load_node)
            mapped.add_edge(load_node, n1, weight=MappedEdge(
                t=OutFromTransferEdgeType(),
                initial_tokens=0,
                production_rate=consumption_rate, # because of how Stream works
                consumption_rate=consumption_rate # because of how Stream works
            ))

    #Add edge to remove autoconcurrency
    for n in mapped.nodes:
        mapped.add_edge(n, n, weight=MappedEdge(
            t=RemoveAutoConcurrencyEdgeType(),
            production_rate=1,
            consumption_rate=1,
            initial_tokens=1,
        ))

    return mapped

def add_minimum_buffers(hsdf: DiGraph):
    NodeType = tuple[MappedNode, int]
    #Assume Load is done as long as Store is possible
    load_nodes = list(filter(lambda n: n[0].t.t == 'Load', map(lambda n: cast(NodeType, n), hsdf.nodes)))
    store_nodes = list(filter(lambda n: n[0].t.t == 'Store', map(lambda n: cast(NodeType, n), hsdf.nodes)))
    #Prioritize latest instances of a store
    store_nodes = sorted(store_nodes, key=lambda n: n[1], reverse=True)

    def weight(_1: NodeType, _2: NodeType, d: MappedEdge):
        return d['weight'].initial_tokens

    for sn in store_nodes:
        distances = nx.single_source_bellman_ford_path_length(hsdf.reverse(copy=False), sn, weight)
        for ln in load_nodes:
            hsdf.add_edge(sn, ln, weight=MappedEdge(
                t=SchedulingEdgeType(),
                initial_tokens=1 - distances[ln],
                production_rate=1,
                consumption_rate=1,
            ))

    for n1 in list(map(lambda n: cast(NodeType, n), hsdf.nodes())):
        #Only the memory of these action must be freed
        if n1[0].t.t not in ['Computation', 'Transfer', 'Load']:
                continue
        children = list(map(lambda d: cast(MappedEdge, d[2]), hsdf.out_edges(data='weight')))
        if len(children) == 0:
            continue
        free_node = (MappedNode(
            t=MemoryFreeNodeType( n1[0].t.core2 if n1[0].t.t == "Transfer" or n1[0].t.t == "Load" else n1[0].processor, "I" if n1[0].t.t == "Load" else "O"),
            id=n1[0].id,
            execution_time=0,
            processor=None
        ), n1[1])

        hsdf.add_node(free_node)
        
        for n2 in list(map(lambda d: cast(NodeType, d[1]), hsdf.out_edges(n1, data='weight'))):
            if n2[0].t.t not in ['Computation', 'Transfer', 'Store']:
                continue
            if n2[0].t.t == 'Computation':
                conv1d = cast(Convolution1D, n2[0].t.conv1d)
                instance = n1[1] % conv1d.kernel_size # Which part of the stride does it occupy?
                hsdf.add_edge(n2, free_node, weight=MappedEdge(
                    t=TensorUsedEdgeType(),
                    initial_tokens=-((conv1d.padding - instance)//conv1d.stride),
                    production_rate=1,
                    consumption_rate=1,
                ))
            else:
                # Can free immediatly after transfer
                hsdf.add_edge(n2, free_node, weight=MappedEdge(
                    t=TensorUsedEdgeType(),
                    initial_tokens=0,
                    production_rate=1,
                    consumption_rate=1,
                )) 
        
        tokens_in_path = nx.bellman_ford_path_length(hsdf.reverse(copy=False), free_node, n1, weight)
        hsdf.add_edge(free_node, n1, weight=MappedEdge(
            t=TensorBufferEdgeType(n1[0].t.output_token_size if n1[0].t.t == 'Load' else n1[0].t.conv1d.output_token_size),
            initial_tokens=1 - tokens_in_path,
            production_rate=1,
            consumption_rate=1,
        ))



def optimize(hsdf, optimization_type: OptimizationType) -> (int, np.ndarray):
    if optimization_type.t == "MinimumMemory":
        priority = ["throughput", "buffer_size"]
    elif optimization_type.t == "MinimumLatency":
        priority = ["buffer_size", "throughput"]
    else:
        raise ValueError('Not implemented')
    
    from gurobipy import Model, GRB, Var
    m = Model()
    processors = set(cast(MappedNode, n).processor for (n, _) in hsdf.nodes)
    processors.remove(None)
    buffers: list[tuple[MappedEdge, Var]] = []

    t = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="t")
    if optimization_type.t == "MinimumMemory": # TODO: numerically better solution
        m.addConstr(t*1e+9 >= 1, name='test') # No deadlock
    buffer_size = 0
    m.setObjectiveN(-t, 1, priority.index("throughput"), name="throughput")

    start_times = { n: m.addVar(vtype=GRB.CONTINUOUS, lb=0) for n in hsdf.nodes }
    

    for a, b, e in hsdf.edges.data('weight'):
        a = cast(tuple[MappedNode, int], a)
        b = cast(tuple[MappedNode, int], b)
        e = cast(MappedEdge, e)

        if e.t.t == 'TensorBuffer':
            pass
            
            bs = m.addVar(vtype=GRB.INTEGER, lb=e.initial_tokens)
            token_size = e.t.buffer_token_size
            buffer_size += token_size * bs
            m.addConstr(start_times[b] >= start_times[a] + t*a[0].execution_time - bs)
            buffers.append((e, bs))
            #assert e.initial_tokens > 0
        elif e.t.t == 'Scheduling':
            pass
        else:
            m.addConstr(start_times[b] >= start_times[a] + t*a[0].execution_time - e.initial_tokens)

    for a in hsdf.nodes:
        if a[0].processor == None:
            continue
        for b in hsdf.nodes:
            if a > b and a[0].processor == b[0].processor:
                initial_tokens = m.addVar(vtype=GRB.INTEGER, lb=-float('inf'), ub=float('inf'))
                m.addConstr(start_times[b] >= start_times[a] + t*a[0].execution_time - initial_tokens)
                m.addConstr(start_times[a] >= start_times[b] + t*b[0].execution_time - (1 - initial_tokens))

    processors = set(a[0].processor for a in hsdf.nodes)

    for processor in processors:
        cycle_bound = 0
        for a in hsdf.nodes:
            if a[0].processor == processor:
                cycle_bound += a[0].execution_time

        m.addConstr(cycle_bound*t <= 1)
    
    m.setObjectiveN(buffer_size, 0, priority.index("buffer_size"), name="buffer_size")
    m.write('model.lp')
    m.setParam(GRB.Param.OutputFlag, 0)
    m.optimize()

    for e, v in buffers:
        e.initial_tokens = round(v.X)

    cycle_time = ceil(1/t.X)
    st = np.array([round(start_times[n].X * cycle_time) for n in hsdf.nodes])
    return cycle_time, st

       

def mapped_to_hsdf(mapped: DiGraph):
    sdf = Sdf()

    for a in mapped.nodes:
        sdf.add_actor(a)

    for a, b, c in mapped.edges.data('weight'):
        sdf.add_channel(a, c.production_rate, b,
                        c.consumption_rate, c.initial_tokens)

    hsdf = sdf.to_hsdf()

    result = DiGraph()

    for a in hsdf.actors():
        result.add_node(a)

    for a, b, initial_tokens in hsdf.channels():
        if result.has_edge(a, b):
            result.edges[a, b]['weight'].initial_tokens = min(result.edges[a, b]['weight'].initial_tokens, initial_tokens)
        else:
            d = cast(MappedEdge, mapped[a[0]][b[0]]['weight'])
            result.add_edge(a, b, weight=MappedEdge(
                t=d.t,
                initial_tokens=int(initial_tokens),
                production_rate=1,
                consumption_rate=1,
            ))
    return result, hsdf.repetitions_vector

def earliest_first(hsdf: DiGraph, workload: DiGraph):
    processors = set(cast(MappedNode, n).processor for (n, _) in hsdf.nodes)
    processors.remove(None)

    def weight(_1, _2, d):
        return d['weight'].initial_tokens

    order = [n.id for n in nx.algorithms.topological_sort(workload)]

    ordered_nodes = sorted(hsdf.nodes, key=lambda x: (order.index(x[0].id), x[1]))

    for o in ordered_nodes:
        if o[0].processor == None:
            continue
        tokens = nx.algorithms.single_source_bellman_ford_path_length(hsdf.reverse(False), o, weight)
        for target, t in tokens.items():
            if target[0].processor == o[0].processor:
                hsdf.add_edge(o, target, weight=MappedEdge(
                    t=SchedulingEdgeType(),
                    initial_tokens=1 - t,
                    production_rate=1,
                    consumption_rate=1,
                ))

def latest_first(hsdf: DiGraph, workload: DiGraph):
    processors = set(cast(MappedNode, n).processor for (n, _) in hsdf.nodes)
    processors.remove(None)

    def weight(_1, _2, d):
        return d['weight'].initial_tokens

    order = [n.id for n in nx.algorithms.topological_sort(workload)]

    ordered_nodes = sorted(hsdf.nodes, key=lambda x: (order.index(x[0].id), x[1]), reverse=True)

    for o in ordered_nodes:
        if o[0].processor == None:
            continue
        tokens = nx.algorithms.single_source_bellman_ford_path_length(hsdf.reverse(False), o, weight)
        for target, t in tokens.items():
            if target[0].processor == o[0].processor:
                hsdf.add_edge(o, target, weight=MappedEdge(
                    t=SchedulingEdgeType(),
                    initial_tokens=1 - t,
                    production_rate=1,
                    consumption_rate=1,
                ))

