from dataclasses import dataclass
from enum import Enum
from typing import Literal, Union, Optional, Generic, TypeVar, Hashable, Callable, cast
from networkx import DiGraph
import networkx as nx
from math import ceil
from sdf_expander import Sdf
from itertools import chain


ID = TypeVar('ID', bound=Hashable)
PROCESSOR = TypeVar('PROCESSOR', bound=Hashable)
CORE = TypeVar('CORE', bound=Hashable)
CHANNEL = TypeVar('CHANNEL', bound=Hashable)
# Initial sdf


@dataclass
class WorkloadNode(Generic[ID]):

    id: ID
    kernel_size: int
    padding: int
    stride: int
    production_rate: int
    n_execution: int

    def __hash__(self) -> int:
        return hash(self.id)

    def consumption_rate(self):
        return self.production_rate * self.stride

    def in_initial_tokens(self):
        # After how many input rows can it fire
        return self.consumption_rate() - self.kernel_size + self.padding

    def freeing_initial_tokens(self):
        # After how many firing can it release memory
        return -self.padding  # TODO


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
class TransferNodeType:
    t: Literal['Transfer']
    core1: CORE
    core2: CORE
    def __init__(self, core1: CORE, core2: CORE):
        self.t = 'Transfer'
        self.core1 = core1
        self.core2 = core2


@dataclass
class ComputationNodeType:
    t: Literal['Computation']

    def __init__(self):
        self.t = 'Computation'


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
class LoadType:
    t: Literal['Load']
    core2: CORE
    def __init__(self, core2: CORE):
        self.t = 'Load'
        self.core2 = core2


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

    def __init__(self):
        self.t = 'TensorBuffer'


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


def workload_graph_to_mapped_graph(
        workload: DiGraph,
        core_mapping: Callable[[UNMAPPED_ACTOR_ID], CORE],
        get_channel: Callable[[CORE, CORE], CHANNEL],
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
            t=ComputationNodeType(),
            id=n.id,
            execution_time=get_execution_time(n.id),
            processor=core_mapping(n.id)
        ) for n in workload.nodes()
    }

    corresponding_workload = {b: a for a, b in corresponding_mapped.items()}

    for n in corresponding_mapped.values():
        mapped.add_node(n)

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
                            production_rate=wn1.production_rate,
                            consumption_rate=wn2.consumption_rate(),
                        )
                    )
            else:
                for wn2 in childs_wn:
                    transfer = MappedNode(
                        id=wn1.id,
                        t=TransferNodeType(core1, core2),
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
                            production_rate=wn1.production_rate,
                            consumption_rate=wn1.production_rate,
                        )
                    )
                    mapped.add_edge(
                        transfer,
                        corresponding_mapped[wn2],
                        weight=MappedEdge(
                            t=OutFromTransferEdgeType(),
                            initial_tokens=wn2.in_initial_tokens(),
                            production_rate=wn1.production_rate,
                            consumption_rate=wn2.consumption_rate(),
                        )
                    )

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
            production_rate = corresponding_workload[n1].production_rate
            mapped.add_edge(n1, store_node, weight=MappedEdge(
                t=IntoTransferEdgeType(),
                initial_tokens=0,
                production_rate=production_rate,
                consumption_rate=production_rate
            ))

    for n1 in list(mapped.nodes):
        if mapped.in_degree(n1) == 0:
            load_node = MappedNode(
                t=LoadType(n1.processor),
                id=n1.id,
                execution_time=get_load_time(n1.id),
                processor=get_load_channel(n1.id)
            )
            mapped.add_node(load_node)
            consumption_rate = corresponding_workload[n1].consumption_rate()
            mapped.add_edge(load_node, n1, weight=MappedEdge(
                t=OutFromTransferEdgeType(),
                initial_tokens=0,
                production_rate=consumption_rate,
                consumption_rate=consumption_rate
            ))

    nodes_and_children = {cast(MappedNode, n): [(cast(MappedNode, m), cast(
        MappedEdge, d)) for (_, m, d) in mapped.out_edges(n, 'weight')] for n in mapped.nodes}

    for n1, children in nodes_and_children.items():
        if len(children) == 0:
            continue
        free_node = MappedNode(
            t=MemoryFreeNodeType( n1.t.core2 if n1.t.t == "Transfer" or n1.t.t == "Load" else n1.processor, "I" if n1.t.t == "Load" else "O"),
            id=n1.id,
            execution_time=0,
            processor=None
        )

        mapped.add_node(free_node)
        for n2, d in children:
            if n2.t.t == 'Computation':
                wn = corresponding_workload[n2]
                initial_tokens = wn.freeing_initial_tokens()
            elif n2.t.t == 'Transfer' or n2.t.t == 'Store':
                #TODO understand the buffer requiriments
                initial_tokens = min(chain([1], (corresponding_workload[n3].freeing_initial_tokens() for (n3, _) in nodes_and_children[n2])))  # Can Free as soon as transfered
            else:
                raise ValueError("unreachable")

            d = cast(MappedEdge, mapped.edges[n1, n2]['weight'])

            mapped.add_edge(n2, free_node, weight=MappedEdge(
                t=TensorUsedEdgeType(),
                production_rate=d.consumption_rate,
                consumption_rate=d.production_rate,
                initial_tokens=initial_tokens
            ))

            buffer_size = ceil(d.consumption_rate/d.production_rate -
                               d.initial_tokens/d.production_rate - initial_tokens/d.production_rate)
            if mapped.has_edge(free_node, n1):
                buffer = mapped.edges[free_node, n1]['weight']
                buffer.initial_tokens = max(buffer.initial_tokens, buffer_size)
            else:
                mapped.add_edge(free_node, n1, weight=MappedEdge(
                    t=TensorBufferEdgeType(),
                    production_rate=1,
                    consumption_rate=1,
                    initial_tokens=buffer_size  # Minimum Buffer for now size
                ))

    for n in mapped.nodes:
        mapped.add_edge(n, n, weight=MappedEdge(
            t=RemoveAutoConcurrencyEdgeType(),
            production_rate=1,
            consumption_rate=1,
            initial_tokens=1,
        ))

    return mapped


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

def minimum_memory_schedule(hsdf: DiGraph, workload: DiGraph):
    processors = set(cast(MappedNode, n).processor for (n, _) in hsdf.nodes)
    processors.remove(None)
    for n,m,d in hsdf.edges.data('weight'):
        if cast(MappedEdge, d).t.t == 'TensorBuffer':
            d.initial_tokens = float('inf')
    #Avoid infinie throuput by instantiating maximum buffer size
    for n in filter(hsdf.edges.data('weight')):
        [buffer] = [d for _,_, d in hsdf.in_edges(n).data() if cast(MappedEdge, d).t.t == 'TensorBuffer']
        
        buffer.






    def weight(_1, _2, d):
        return d['weight'].initial_tokens
    #assert not nx.algorithms.find_negative_cycle(graph, list(graph.nodes)[0], weight)
    for p in processors:
        actors = [(n, i) for (n, i) in hsdf.nodes if cast(
            MappedNode, n).processor == p]
        

    order = [n.id for n in nx.algorithms.topological_sort(workload)]

    ordered_nodes = sorted(hsdf.nodes, key=lambda x: (order.index(x[0].id), x[1]))

    cycles = nx.algorithms.find_negative_cycle(hsdf, next(iter(hsdf.nodes)), lambda _1, _2, d: cast(MappedEdge, d['weight']).initial_tokens - 0.0001)

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


