import argparse
import json
import re
import urllib.request
import gzip
import collections
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("Please install torch and torch-geometric: pip install torch torchvision torchaudio torch-geometric")

# Path to the locally cloned MacroPlacement dataset
DATASETS_ROOT = Path(__file__).parent / "datasets" / "MacroPlacement" / "Flows"

# Mapping user-friendly PDK names to folder names in the repo
PDK_FOLDER_MAP = {
    "Nangate45": "NanGate45",
    "NanGate45": "NanGate45",
    "ASAP7": "ASAP7",
    "SKY130HD": "SKY130HD",
}

GITHUB_BASE = "https://raw.githubusercontent.com/TILOS-AI-Institute/MacroPlacement/main"

# Node type integer codes stored in graph.node_type
NODE_TYPE_MACRO = 0       # Hard macro (SRAM etc.) — placement targets
NODE_TYPE_SOFT_MACRO = 1  # Soft macro (Grp_* std-cell cluster) — fixed context
NODE_TYPE_PORT = 2        # Chip I/O port — fixed context


def discover_local_designs(datasets_root: Optional[Path] = None) -> List[Dict]:
    """Return all design-PDK pairs that have a local output_CT_Grouping netlist."""
    root = datasets_root or DATASETS_ROOT
    found = []
    if not root.exists():
        return found
    for pdk_dir in sorted(root.iterdir()):
        if not pdk_dir.is_dir():
            continue
        for design_dir in sorted(pdk_dir.iterdir()):
            if not design_dir.is_dir():
                continue
            ct_dir = design_dir / "netlist" / "output_CT_Grouping"
            pb_file = ct_dir / "netlist.pb.txt"
            if not pb_file.exists():
                continue
            plc_file = None
            for name in ("legalized.plc", "initial.plc"):
                candidate = ct_dir / name
                if candidate.exists():
                    plc_file = candidate
                    break
            found.append({
                "design": design_dir.name,
                "pdk": pdk_dir.name,
                "pb_path": pb_file,
                "plc_path": plc_file,
            })
    return found


def load_local_netlist(design: str, pdk: str, datasets_root: Optional[Path] = None) -> Tuple[Optional[str], Optional[str]]:
    """Read netlist.pb.txt and .plc from the local MacroPlacement clone."""
    root = datasets_root or DATASETS_ROOT
    pdk_folder = PDK_FOLDER_MAP.get(pdk, pdk)
    ct_dir = root / pdk_folder / design / "netlist" / "output_CT_Grouping"

    netlist_content = None
    pb_file = ct_dir / "netlist.pb.txt"
    if pb_file.exists():
        netlist_content = pb_file.read_text(encoding="utf-8")

    plc_content = None
    for plc_name in ("legalized.plc", "initial.plc"):
        plc_file = ct_dir / plc_name
        if plc_file.exists():
            plc_content = plc_file.read_text(encoding="utf-8")
            break

    return netlist_content, plc_content


def fetch_url(url: str, is_gzip: bool = False) -> Optional[str]:
    print(f"Fetching: {url}")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = response.read()
            if is_gzip or url.endswith(".gz"):
                return gzip.decompress(data).decode("utf-8")
            return data.decode("utf-8")
    except Exception as e:
        print(f"  -> Failed to fetch {url}: {e}")
        return None


def fetch_netlist_and_plc(design: str, pdk: str) -> Tuple[Optional[str], Optional[str]]:
    """Remote fallback: try known GitHub paths for netlist.pb.txt and .plc."""
    pdk_folder = PDK_FOLDER_MAP.get(pdk, pdk)
    paths_to_try = [
        f"Flows/{pdk_folder}/{design}/netlist/output_CT_Grouping",
        f"Flows/{pdk_folder}/{design}/netlist",
        f"Flows/{pdk_folder}/{design}",
    ]

    netlist_content = None
    for path in paths_to_try:
        url = f"{GITHUB_BASE}/{path}/netlist.pb.txt.gz"
        netlist_content = fetch_url(url, is_gzip=True)
        if not netlist_content:
            url = f"{GITHUB_BASE}/{path}/netlist.pb.txt"
            netlist_content = fetch_url(url)
        if netlist_content:
            break

    plc_content = None
    for path in paths_to_try:
        for plc_name in ("legalized.plc", "initial.plc"):
            url = f"{GITHUB_BASE}/{path}/{plc_name}"
            plc_content = fetch_url(url)
            if plc_content:
                break
        if plc_content:
            break

    return netlist_content, plc_content


def parse_pb_txt(content: str) -> Dict:
    """
    Parse a Circuit Training pb.txt netlist.

    The format has six node types:
      MACRO      — hard macro (SRAM etc.), the RL placement targets
      macro      — soft macro (Grp_* std-cell cluster)
      PORT       — chip I/O port (fixed position on boundary)
      MACRO_PIN  — pin belonging to a hard macro
      macro_pin  — pin belonging to a soft macro
      (metadata) — ignored __metadata__ node

    Connectivity is encoded on PIN nodes via the `input:` field:
    "input: B" on pin A means A drives signal to B (A → B).

    Returns a dict with:
      macros       : {name: {width, height, x, y, idx}}
      soft_macros  : {name: {width, height, x, y, idx}}
      ports        : {name: {x, y, side, idx}}
      edge_weights : Counter((parent_a, parent_b) → # connecting pins)
    """
    macros: Dict = {}
    soft_macros: Dict = {}
    ports: Dict = {}
    # Maps every pin name to the name of the macro/group/port it belongs to
    pin_to_parent: Dict[str, str] = {}
    # All pin connections; processed after pin_to_parent is fully built
    pin_connections: List[Tuple[str, str]] = []  # (src_pin, dst_pin)

    blocks = content.split("node {\n")

    for node_idx, block in enumerate(blocks[1:]):
        if not block.strip():
            continue

        name_m = re.search(r'name:\s+"(.*?)"', block)
        if not name_m:
            continue
        name = name_m.group(1)
        if name == "__metadata__":
            continue

        # Parse all attr blocks in one sweep
        float_attrs: Dict[str, float] = {}
        str_attrs: Dict[str, str] = {}
        for attr_block in block.split("attr {")[1:]:
            key_m = re.search(r'key:\s+"(.*?)"', attr_block)
            if not key_m:
                continue
            key = key_m.group(1)
            if "f:" in attr_block:
                val_m = re.search(r"f:\s+([0-9.E+\-]+)", attr_block)
                if val_m:
                    float_attrs[key] = float(val_m.group(1))
            if "placeholder:" in attr_block:
                val_m = re.search(r'placeholder:\s+"(.*?)"', attr_block)
                if val_m:
                    str_attrs[key] = val_m.group(1)

        node_type = str_attrs.get("type")
        if not node_type:
            continue

        inputs = re.findall(r'input:\s+"(.*?)"', block)

        if node_type == "MACRO":
            macros[name] = {
                "width": float_attrs.get("width", 50.0),
                "height": float_attrs.get("height", 50.0),
                "x": float_attrs.get("x", 0.0),
                "y": float_attrs.get("y", 0.0),
                "idx": node_idx,
            }

        elif node_type == "macro":
            soft_macros[name] = {
                "width": float_attrs.get("width", 10.0),
                "height": float_attrs.get("height", 10.0),
                "x": float_attrs.get("x", 0.0),
                "y": float_attrs.get("y", 0.0),
                "idx": node_idx,
            }

        elif node_type == "PORT":
            ports[name] = {
                "x": float_attrs.get("x", 0.0),
                "y": float_attrs.get("y", 0.0),
                "side": str_attrs.get("side"),
                "idx": node_idx,
            }
            pin_to_parent[name] = name
            # PORT nodes directly carry input: connections
            for dst_pin in inputs:
                pin_connections.append((name, dst_pin))

        elif node_type in ("MACRO_PIN", "macro_pin"):
            macro_name = str_attrs.get("macro_name")
            if macro_name:
                pin_to_parent[name] = macro_name
                for dst_pin in inputs:
                    pin_connections.append((name, dst_pin))

    # Build edge weights between parent macros/groups/ports
    edge_weights: collections.Counter = collections.Counter()
    for src_pin, dst_pin in pin_connections:
        src_parent = pin_to_parent.get(src_pin)
        dst_parent = pin_to_parent.get(dst_pin)
        if src_parent and dst_parent and src_parent != dst_parent:
            # Canonical key so (A,B) and (B,A) merge into one undirected edge
            key = (src_parent, dst_parent) if src_parent < dst_parent else (dst_parent, src_parent)
            edge_weights[key] += 1

    return {
        "macros": macros,
        "soft_macros": soft_macros,
        "ports": ports,
        "edge_weights": edge_weights,
    }


def parse_plc(content: str) -> Tuple[float, float, Dict]:
    """Parse a Circuit Training .plc placement file for canvas size."""
    placements = {}
    canvas_w = 0.0
    canvas_h = 0.0

    for line in content.splitlines():
        if line.startswith("# Width :") or "Width :" in line:
            m = re.search(r"Width\s*:\s*([0-9.]+)\s*Height\s*:\s*([0-9.]+)", line)
            if m:
                canvas_w = float(m.group(1))
                canvas_h = float(m.group(2))
        elif not line.startswith("#") and line.strip():
            parts = line.strip().split()
            if len(parts) >= 4:
                node_idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                orient = parts[3]
                placements[node_idx] = {"x": x, "y": y, "orient": orient}

    return canvas_w, canvas_h, placements


def build_pyg_graph(parsed: Dict, placements: Dict, canvas_w: float, canvas_h: float) -> Tuple[Data, Dict]:
    """
    Build a PyG Data graph from the parsed netlist.

    Node ordering: MACRO nodes first (indices 0..n_macro-1), then soft_macro,
    then PORT.  This keeps MACRO indices stable and backward-compatible.

    Graph attributes:
      x              : [N, 5] float features [w, h, area, norm_w, norm_h]
      pos            : [N, 2] normalised (x/W, y/H) initial positions
      edge_index     : [2, E] undirected edges derived from real pin connectivity
      edge_attr      : [E, 1] connection count per undirected macro pair
      node_type      : [N] long  (0=MACRO, 1=soft_macro, 2=PORT)
      placeable_mask : [N] bool  True for MACRO nodes (RL placement targets)
      macro_indices  : [n_macro] long — indices of hard MACRO nodes
      canvas_size    : [2] float [canvas_w, canvas_h]
      num_macros     : int   (Python-level attribute)
    """
    macros = parsed["macros"]
    soft_macros = parsed["soft_macros"]
    ports = parsed["ports"]
    edge_weights = parsed["edge_weights"]

    # Node ordering: MACRO → soft_macro → PORT
    macro_names = sorted(macros.keys())
    soft_macro_names = sorted(soft_macros.keys())
    port_names = sorted(ports.keys())

    all_names = macro_names + soft_macro_names + port_names
    n_macro = len(macro_names)
    n_soft = len(soft_macro_names)
    n_port = len(port_names)
    n_total = n_macro + n_soft + n_port

    print(f"  Hard MACROs: {n_macro}  Soft macros: {n_soft}  Ports: {n_port}  Total nodes: {n_total}")

    if n_total == 0:
        empty = Data(
            x=torch.empty((0, 5), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            pos=torch.empty((0, 2), dtype=torch.float),
            num_nodes=0,
        )
        empty.num_macros = 0
        return empty, {}

    name_to_idx = {name: i for i, name in enumerate(all_names)}

    # Apply .plc overrides to MACRO positions (keyed by original pb.txt node index)
    idx_to_macro_name: Dict[int, str] = {d["idx"]: n for n, d in macros.items()}
    for pb_idx, placement in placements.items():
        macro_name = idx_to_macro_name.get(pb_idx)
        if macro_name and macro_name in macros:
            macros[macro_name]["x"] = placement["x"]
            macros[macro_name]["y"] = placement["y"]

    def safe_div(a: float, b: float) -> float:
        return a / b if b > 0 else 0.0

    # Build node features [w, h, area, norm_w, norm_h]
    features = []
    pos_list = []

    for name in macro_names:
        d = macros[name]
        w, h = max(d["width"], 1e-6), max(d["height"], 1e-6)
        features.append([w, h, w * h, safe_div(w, canvas_w), safe_div(h, canvas_h)])
        pos_list.append([safe_div(d["x"], canvas_w), safe_div(d["y"], canvas_h)])

    for name in soft_macro_names:
        d = soft_macros[name]
        w, h = max(d["width"], 1e-6), max(d["height"], 1e-6)
        features.append([w, h, w * h, safe_div(w, canvas_w), safe_div(h, canvas_h)])
        pos_list.append([safe_div(d["x"], canvas_w), safe_div(d["y"], canvas_h)])

    for name in port_names:
        d = ports[name]
        features.append([1.0, 1.0, 1.0, safe_div(1.0, canvas_w), safe_div(1.0, canvas_h)])
        pos_list.append([safe_div(d["x"], canvas_w), safe_div(d["y"], canvas_h)])

    node_types_list = (
        [NODE_TYPE_MACRO] * n_macro
        + [NODE_TYPE_SOFT_MACRO] * n_soft
        + [NODE_TYPE_PORT] * n_port
    )

    # Build edges from real pin connectivity
    edges_fwd: List[List[int]] = []
    edge_w: List[float] = []

    for (a, b), weight in edge_weights.items():
        idx_a = name_to_idx.get(a)
        idx_b = name_to_idx.get(b)
        if idx_a is None or idx_b is None:
            continue
        edges_fwd.append([idx_a, idx_b])
        edges_fwd.append([idx_b, idx_a])
        edge_w.extend([float(weight), float(weight)])

    x_tensor = torch.tensor(features, dtype=torch.float)
    pos_tensor = torch.tensor(pos_list, dtype=torch.float)
    node_type_tensor = torch.tensor(node_types_list, dtype=torch.long)

    if edges_fwd:
        edge_index = torch.tensor(edges_fwd, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_w, dtype=torch.float).unsqueeze(1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    placeable_mask = node_type_tensor == NODE_TYPE_MACRO
    macro_indices = torch.where(placeable_mask)[0]

    graph = Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos_tensor,
        node_type=node_type_tensor,
        placeable_mask=placeable_mask,
        macro_indices=macro_indices,
        canvas_size=torch.tensor([canvas_w, canvas_h], dtype=torch.float),
        num_nodes=n_total,
    )
    # Convenience scalar (Python-level, not saved as tensor)
    graph.num_macros = n_macro

    lef_macros = {
        n: {"width": macros[n]["width"], "height": macros[n]["height"]}
        for n in macro_names
    }
    return graph, lef_macros


def process_design(design: str, pdk: str, output_dir: Path, datasets_root: Optional[Path] = None) -> bool:
    """Parse one design-PDK pair and save the graph + metadata. Returns True on success."""
    print(f"\nProcessing design: {design}  PDK: {pdk}")

    # Try local data first, fall back to remote
    netlist_content, plc_content = load_local_netlist(design, pdk, datasets_root)
    if not netlist_content:
        print("  Local data not found, trying remote GitHub …")
        netlist_content, plc_content = fetch_netlist_and_plc(design, pdk)
    if not netlist_content:
        print(f"  ERROR: Could not find netlist for {design}/{pdk}")
        return False

    print("  Parsing protobuf netlist …")
    parsed = parse_pb_txt(netlist_content)
    n_macros = len(parsed["macros"])
    n_soft = len(parsed["soft_macros"])
    n_ports = len(parsed["ports"])
    n_edges_raw = len(parsed["edge_weights"])
    print(f"  Nodes — MACRO: {n_macros}  soft_macro: {n_soft}  PORT: {n_ports}")
    print(f"  Unique macro-pair connections: {n_edges_raw}")

    canvas_w, canvas_h = 400.0, 400.0
    placements: Dict = {}
    if plc_content:
        canvas_w, canvas_h, placements = parse_plc(plc_content)
        print(f"  Canvas: {canvas_w} x {canvas_h} µm")

    graph, lef_macros = build_pyg_graph(parsed, placements, canvas_w, canvas_h)
    if graph.num_nodes == 0:
        print(f"  ERROR: Graph construction produced no nodes for {design}/{pdk}")
        return False

    edge_index = graph.edge_index
    n_edges_undirected = (edge_index.shape[1] // 2) if (edge_index is not None and edge_index.numel() > 0) else 0
    print(f"  Graph — nodes: {graph.num_nodes}  edges (undirected): {n_edges_undirected}")
    print(f"  Hard MACROs in graph: {graph.num_macros}")

    summary = {
        "design": design,
        "pdk": pdk,
        "node_counts": {
            "macro": n_macros,
            "soft_macro": n_soft,
            "port": n_ports,
            "total": graph.num_nodes,
        },
        "edge_count_undirected": n_edges_undirected,
        "canvas_size_um": [canvas_w, canvas_h],
        "feature_dim": int(graph.x.shape[1]),
        "lef_macros": lef_macros,
        "source": "TILOS-AI-Institute/MacroPlacement output_CT_Grouping",
    }

    out_dir = output_dir / design / pdk
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / f"metadata-{design}-{pdk}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    graph_path = out_dir / f"{design}_{pdk}_graph.pt"
    torch.save({"graph": graph, "metadata": summary}, graph_path)

    print(f"  Saved: {graph_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="TILOS MacroPlacement real-connectivity graph builder")
    parser.add_argument("--design", type=str, default=None,
                        help="Design name (e.g. ariane133). Omit to process all local designs.")
    parser.add_argument("--pdk", type=str, default=None,
                        choices=list(PDK_FOLDER_MAP.keys()),
                        help="PDK name. Required when --design is specified.")
    parser.add_argument("--output-dir", type=str, default="preprocessed_dataset",
                        help="Root output directory.")
    parser.add_argument("--datasets-root", type=str, default=None,
                        help="Override path to MacroPlacement/Flows directory.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    datasets_root = Path(args.datasets_root) if args.datasets_root else None

    if args.design:
        if not args.pdk:
            parser.error("--pdk is required when --design is specified")
        process_design(args.design, args.pdk, output_dir, datasets_root)
    else:
        # Batch mode: process every locally available design-PDK pair
        designs = discover_local_designs(datasets_root)
        if not designs:
            print("No local designs found. Specify --design and --pdk for remote fetch.")
            return
        print(f"Found {len(designs)} local design-PDK pairs.")
        successes = 0
        for entry in designs:
            ok = process_design(entry["design"], entry["pdk"], output_dir, datasets_root)
            if ok:
                successes += 1
        print(f"\nDone: {successes}/{len(designs)} designs processed successfully.")


if __name__ == "__main__":
    main()
