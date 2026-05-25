import os
import argparse
import json
import re
import urllib.request
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("Please install torch and torch-geometric: pip install torch torchvision torchaudio torch-geometric")

# Mapping user-friendly names to specific repo structure variations
PDK_FOLDER_MAP = {
    "Nangate45": "NanGate45",
    "ASAP7": "ASAP7",
    "SKY130HD": "SKY130HD"
}

GITHUB_BASE = "https://raw.githubusercontent.com/TILOS-AI-Institute/MacroPlacement/main"

def fetch_url(url: str, is_gzip: bool = False) -> Optional[str]:
    print(f"Fetching: {url}")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = response.read()
            if is_gzip or url.endswith(".gz"):
                return gzip.decompress(data).decode('utf-8')
            return data.decode('utf-8')
    except Exception as e:
        print(f"  -> Failed to fetch {url}: {e}")
        return None

def fetch_netlist_and_plc(design: str, pdk: str) -> Tuple[Optional[str], Optional[str]]:
    """Attempts to fetch netlist.pb.txt.gz and initial.plc from various known paths"""
    pdk_folder = PDK_FOLDER_MAP.get(pdk, pdk)
    
    # Check known output grouping paths
    paths_to_try = [
        f"Flows/{pdk_folder}/{design}/netlist/output_CT_Grouping",
        f"Flows/{pdk_folder}/{design}/netlist",
        f"Flows/{pdk_folder}/{design}"
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
        # Prefer legalized if available, otherwise initial
        url = f"{GITHUB_BASE}/{path}/legalized.plc"
        plc_content = fetch_url(url)
        if not plc_content:
            url = f"{GITHUB_BASE}/{path}/initial.plc"
            plc_content = fetch_url(url)
        if plc_content:
            break
            
    return netlist_content, plc_content

def parse_pb_txt(content: str):
    """Robust regex-based streaming parser for Circuit Training pb.txt"""
    nodes = []
    
    # Split by node definitions
    blocks = content.split('node {\n')
    for block in blocks:
        if not block.strip(): continue
        
        name_match = re.search(r'name:\s+"(.*?)"', block)
        if not name_match: continue
        
        node = {
            'name': name_match.group(1),
            'inputs': [],
            'type': None,
            'macro_name': None,
            'width': 0.0,
            'height': 0.0,
            'x': 0.0,
            'y': 0.0
        }
        
        for inp in re.finditer(r'input:\s+"(.*?)"', block):
            node['inputs'].append(inp.group(1))
            
        attr_blocks = block.split('attr {')[1:]
        for attr_block in attr_blocks:
            key_match = re.search(r'key:\s+"(.*?)"', attr_block)
            if not key_match: continue
            key = key_match.group(1)
            
            val = None
            if 'placeholder:' in attr_block:
                m = re.search(r'placeholder:\s+"(.*?)"', attr_block)
                if m: val = m.group(1)
            elif 's:' in attr_block:
                m = re.search(r's:\s+"(.*?)"', attr_block)
                if m: val = m.group(1)
            elif 'f:' in attr_block:
                m = re.search(r'f:\s+([0-9.E+-]+)', attr_block)
                if m: val = float(m.group(1))
                
            node[key] = val
            
        nodes.append(node)
        
    return nodes

def parse_plc(content: str):
    placements = {}
    canvas_w = 0.0
    canvas_h = 0.0
    
    for line in content.splitlines():
        if line.startswith('# Width :'):
            m = re.search(r'Width\s*:\s*([0-9.]+)\s*Height\s*:\s*([0-9.]+)', line)
            if m:
                canvas_w = float(m.group(1))
                canvas_h = float(m.group(2))
        elif not line.startswith('#') and line.strip():
            parts = line.strip().split()
            if len(parts) >= 4:
                node_idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                orient = parts[3]
                placements[node_idx] = {'x': x, 'y': y, 'orient': orient}
                
    return canvas_w, canvas_h, placements

def build_pyg_graph(nodes, placements, canvas_w, canvas_h):
    # Filter to only MACRO blocks (skip ports, soft macros etc for this specific graph type)
    macro_nodes = []
    # Map original node name to new macro index
    name_to_macro_idx = {}
    
    for idx, node in enumerate(nodes):
        if node.get('type') == 'MACRO':
            macro_idx = len(macro_nodes)
            name_to_macro_idx[node['name']] = macro_idx
            
            # Incorporate placement if available
            if idx in placements:
                node['x'] = placements[idx]['x']
                node['y'] = placements[idx]['y']
                
            macro_nodes.append(node)

    num_macros = len(macro_nodes)
    print(f"Extracted {num_macros} MACRO nodes.")
    
    if num_macros == 0:
        return None, {}

    # Node features: [width, height, area, normalized_w, normalized_h]
    features = []
    pos_list = []
    parsed_lef_macros = {}
    
    for macro in macro_nodes:
        w = macro.get('width', 50.0)
        h = macro.get('height', 50.0)
        area = w * h
        norm_w = w / canvas_w if canvas_w > 0 else 0.0
        norm_h = h / canvas_h if canvas_h > 0 else 0.0
        
        features.append([w, h, area, norm_w, norm_h])
        
        px = macro.get('x', 0.0)
        py = macro.get('y', 0.0)
        norm_x = px / canvas_w if canvas_w > 0 else 0.0
        norm_y = py / canvas_h if canvas_h > 0 else 0.0
        pos_list.append([norm_x, norm_y])
        
        macro_name = macro.get('macro_name')
        if macro_name:
            parsed_lef_macros[macro_name] = {'width': w, 'height': h}

    x_tensor = torch.tensor(features, dtype=torch.float)
    pos_tensor = torch.tensor(pos_list, dtype=torch.float)
    
    # Edges
    edges = []
    for macro in macro_nodes:
        target_idx = name_to_macro_idx[macro['name']]
        for inp in macro['inputs']:
            # Find the source node index
            # inputs could be another macro, or a soft macro (Grp_*)
            # We only build edges between macros here
            source_idx = name_to_macro_idx.get(inp)
            if source_idx is not None:
                edges.append([source_idx, target_idx])
                edges.append([target_idx, source_idx]) # Undirected for simple GNN

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    graph = Data(
        x=x_tensor,
        edge_index=edge_index,
        pos=pos_tensor,
        num_nodes=num_macros
    )
    
    return graph, parsed_lef_macros

def main():
    parser = argparse.ArgumentParser(description="TILOS MacroPlacement Network Parser")
    parser.add_argument("--design", type=str, required=True, help="e.g., ariane133, MemPool_tile")
    parser.add_argument("--pdk", type=str, default="Nangate45", choices=["Nangate45", "ASAP7", "SKY130HD"])
    parser.add_argument("--output-dir", type=str, default="preprocessed_dataset")
    args = parser.parse_args()

    print(f"\nProcessing design: {args.design} with PDK: {args.pdk}")

    netlist_content, plc_content = fetch_netlist_and_plc(args.design, args.pdk)
    
    if not netlist_content:
        print("ERROR: Could not find netlist.pb.txt for this design on GitHub.")
        return
        
    print("Parsing protobuf netlist...")
    nodes = parse_pb_txt(netlist_content)
    print(f"Extracted {len(nodes)} total nodes from netlist.")

    canvas_w, canvas_h = 400.0, 400.0
    placements = {}
    if plc_content:
        print("Parsing placement coordinates...")
        canvas_w, canvas_h, placements = parse_plc(plc_content)
        print(f"Canvas: {canvas_w} x {canvas_h}")

    print("Building PyG Graph...")
    graph, parsed_lef_macros = build_pyg_graph(nodes, placements, canvas_w, canvas_h)
    
    if not graph:
        print("ERROR: Graph construction failed.")
        return

    # Create detailed JSON matching the standard format expected
    summary = {
        "design": args.design,
        "pdk": args.pdk,
        "known_info": {
            "macro_count": graph.num_nodes,
            "macro_type": "Parsed from pb.txt",
            "approx_macro_size_um": [float(graph.x[0][0]), float(graph.x[0][1])] if graph.num_nodes > 0 else [0.0, 0.0],
            "canvas_size_um": [canvas_w, canvas_h],
            "stdcells_flops": "Omitted in graph representation"
        },
        "parsed_lef_macros": parsed_lef_macros,
        "source_files": {
            "primary_rtl": None,
            "rtl_paths": [],
            "sv2v_paths": [],
            "note": "Netlist fetched directly from TILOS-AI-Institute/MacroPlacement GitHub output_CT_Grouping"
        },
        "netlist_path": None,
        "files": {},
        "note": "Graph node dimensions, coordinates, and types parsed directly from pb.txt and plc."
    }

    out_dir = Path(args.output_dir) / args.design / args.pdk
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = out_dir / f"metadata-{args.design}-{args.pdk}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetadata saved to: {summary_path}")

    graph_path = out_dir / f"{args.design}_{args.pdk}_graph.pt"
    save_data = {
        "graph": graph,
        "metadata": summary
    }
    torch.save(save_data, graph_path)
    
    print(f"GNN-ready graph saved to: {graph_path}")
    print(f"   Nodes (macros): {graph.num_nodes}")
    print(f"   Edges: {graph.num_edges // 2 if graph.edge_index.numel() > 0 else 0} (undirected)")
    print(f"   Feature dim: {graph.x.shape[1] if graph.num_nodes > 0 else 0}")

if __name__ == "__main__":
    main()