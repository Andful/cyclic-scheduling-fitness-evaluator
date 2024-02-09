from stream.inputs.examples.hardware.cores.Eyeriss_like import get_core as get_eyeriss_core
from stream.inputs.examples.hardware.cores.TPU_like import get_core as get_tpu_core
from stream.inputs.examples.hardware.cores.offchip import get_offchip_core
from stream.inputs.examples.hardware.nocs.mesh_2d import get_2d_mesh
from stream.classes.hardware.architecture.accelerator import Accelerator

cores = [get_tpu_core(0), get_eyeriss_core(1)]
offchip_core_id = 2
offchip_core = get_offchip_core(id=offchip_core_id)

cores_graph = get_2d_mesh(cores, 2, 1, 64, 32, None, None, offchip_core)

accelerator = Accelerator(
    "Eyeriss-like-dual-core", cores_graph, offchip_core_id=offchip_core_id
)