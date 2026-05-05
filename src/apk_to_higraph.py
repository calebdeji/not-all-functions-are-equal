import os
import json
import hashlib
import logging
import argparse
import multiprocessing as mp
from collections import defaultdict
from tqdm import tqdm

try:
    from androguard.misc import AnalyzeAPK
    from androguard.core.analysis.analysis import ExternalMethod, MethodAnalysis
    from androguard.core.dex import EncodedMethod
    from androguard.util import set_log
except ImportError:
    print("Androguard (or a component) not found. Please install it: pip install androguard")
    exit(1)

logger = logging.getLogger("apk_graph_extractor")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Pre-compile KEY_API_PATTERNS for faster matching
KEY_API_PATTERNS = [
    "Landroid/telephony/SmsManager;",
    "Ljava/net/HttpURLConnection;",
    "Ljava/net/Socket;",
    "Ldalvik/system/DexClassLoader;",
    "Ljavax/crypto/",
    "Ljava/io/File;",
    "Landroid/location/LocationManager;",
    "Landroid/content/ContentResolver;",
    "Landroid/provider/ContactsContract;",
    "Landroid/app/admin/DevicePolicyManager;",
    "Ljava/lang/Runtime;->exec",
    "Ljava/lang/reflect/Method;->invoke",
    "Landroid/util/Base64;",
    "Landroid/accounts/AccountManager;->getPassword",
    "Landroid/accounts/AccountManager;->getUserData",
    "Landroid/app/ActivityManager;->killBackgroundProcesses",
    "Landroid/hardware/Camera;->open",
    "Landroid/media/AudioRecord;->startRecording",
    "Landroid/os/RecoverySystem;->installPackage",
    "Landroid/os/RecoverySystem;->rebootWipeUserData",
    "Landroid/telephony/TelephonyManager;->getDeviceId",
    "Landroid/telephony/TelephonyManager;->getLine1Number",
    "Landroid/webkit/WebView;->addJavascriptInterface",
    "Landroid/webkit/WebView;->loadUrl",
]

def _get_canonical_name(node_obj):
    """Helper function to get a canonical string name for a method node object."""
    if isinstance(node_obj, MethodAnalysis):
        method_proto = node_obj.get_method()
        if isinstance(method_proto, EncodedMethod):
            return f"{method_proto.get_class_name()}->{method_proto.get_name()}{method_proto.get_descriptor()}"
        return None
    elif isinstance(node_obj, ExternalMethod):
        return f"{node_obj.class_name}->{node_obj.name}{node_obj.descriptor}"
    elif isinstance(node_obj, EncodedMethod):
        return f"{node_obj.get_class_name()}->{node_obj.get_name()}{node_obj.get_descriptor()}"
    return None

def generate_fcg(dx):
    """Generate Function Call Graph from Androguard Analysis object."""
    cg = dx.get_call_graph()
    
    # Get all nodes from the call graph
    raw_nodes_from_cg = list(cg.nodes())
    
    # Create mapping from node objects to FCG IDs
    node_object_to_fcg_id = {node_obj: i for i, node_obj in enumerate(raw_nodes_from_cg)}
    
    # Identify key API node IDs
    key_api_node_ids = set()
    for fcg_id, node_obj in enumerate(raw_nodes_from_cg):
        canonical_name = _get_canonical_name(node_obj)
        if canonical_name and any(pattern in canonical_name for pattern in KEY_API_PATTERNS):
            key_api_node_ids.add(fcg_id)

    # Create FCG edges, filtering based on key APIs
    temp_filtered_fcg_source_ids = []
    temp_filtered_fcg_target_ids = []
    
    for u_obj, v_obj in cg.edges():
        u_id = node_object_to_fcg_id.get(u_obj)
        v_id = node_object_to_fcg_id.get(v_obj)
        
        if u_id is not None and v_id is not None:
            if u_id in key_api_node_ids or v_id in key_api_node_ids:
                temp_filtered_fcg_source_ids.append(u_id)
                temp_filtered_fcg_target_ids.append(v_id)
            
    nodes_to_keep_original_ids = key_api_node_ids.copy()
    nodes_to_keep_original_ids.update(temp_filtered_fcg_source_ids)
    nodes_to_keep_original_ids.update(temp_filtered_fcg_target_ids)

    # Remap original IDs of kept nodes to new, dense IDs (0 to N-1)
    sorted_kept_original_ids = sorted(list(nodes_to_keep_original_ids))
    original_id_to_new_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_kept_original_ids)}
    
    # Create a list of function names, ordered by the new IDs
    function_names = [None] * len(sorted_kept_original_ids)
    for original_id in sorted_kept_original_ids:
        new_id = original_id_to_new_id_map[original_id]
        if 0 <= original_id < len(raw_nodes_from_cg):
            node_obj = raw_nodes_from_cg[original_id]
            function_names[new_id] = _get_canonical_name(node_obj)
        else:
            function_names[new_id] = None

    # Translate filtered edges to use new IDs
    final_fcg_source_ids = [original_id_to_new_id_map[old_id] for old_id in temp_filtered_fcg_source_ids if old_id in original_id_to_new_id_map]
    final_fcg_target_ids = [original_id_to_new_id_map[old_id] for old_id in temp_filtered_fcg_target_ids if old_id in original_id_to_new_id_map]
    
    return raw_nodes_from_cg, [final_fcg_source_ids, final_fcg_target_ids], node_object_to_fcg_id, nodes_to_keep_original_ids, original_id_to_new_id_map, function_names

def generate_cfg_for_method(method_obj, fcg_id, dx):
    """Generate Control Flow Graph for a given method using its MethodAnalysis object."""
    ma_of_method = None
    encoded_method_for_meta = None

    if isinstance(method_obj, MethodAnalysis):
        ma_of_method = method_obj
        encoded_method_for_meta = ma_of_method.get_method()
    elif isinstance(method_obj, EncodedMethod):
        ma_of_method = dx.get_method_analysis(method_obj)
        if ma_of_method:
            encoded_method_for_meta = ma_of_method.get_method()
        else:
            encoded_method_for_meta = method_obj
    
    if not ma_of_method:
        return None

    # Get the encoded method
    encoded_method = ma_of_method.get_method() 
    if not encoded_method:
        return None

    # Check method has basic blocks and code
    if not hasattr(ma_of_method, 'get_basic_blocks') or not (hasattr(encoded_method, 'get_code') and encoded_method.get_code()):
        return None
        
    # Get Basic Blocks
    basic_blocks_container = ma_of_method.get_basic_blocks()
    if not basic_blocks_container:
        return None
        
    all_basic_blocks_list = []
    try:
        for bb in basic_blocks_container:
            all_basic_blocks_list.append(bb)
    except Exception:
        return None

    if not all_basic_blocks_list:
        return None
    
    # Create CFG data structure
    cfg_data = {
        "block_number": len(all_basic_blocks_list),
        "global_function_id": fcg_id,
        "method_name": encoded_method.get_name(),
        "class_name": encoded_method.get_class_name(),
        "descriptor": encoded_method.get_descriptor()
    }
    
    bb_obj_to_local_id = {bb: i for i, bb in enumerate(all_basic_blocks_list)}
    
    block_attributes = []
    block_features = []
    
    # Map of all basic blocks to their outgoing edges count
    bb_offspring_count = defaultdict(int)
    
    # First pass to count offspring for each basic block
    for source_bb_obj in all_basic_blocks_list:
        if hasattr(source_bb_obj, "childs"):
            try:
                for child_info_tuple in source_bb_obj.childs:
                    if len(child_info_tuple) >= 3:
                        target_bb_obj = child_info_tuple[2]
                        if target_bb_obj in bb_obj_to_local_id:
                            bb_offspring_count[source_bb_obj] += 1
            except Exception:
                pass
    
    # Process each basic block
    for bb_obj in all_basic_blocks_list:
        try:
            # Get instructions
            instructions = list(bb_obj.get_instructions()) 
            instruction_count = len(instructions)
            
            # Count different types of instructions
            call_count = 0        # invoke*
            transfer_count = 0    # goto, if-*, switch
            arithmetic_count = 0  # add-*, sub-*, mul-*, div-*, rem-*, neg-*
            logic_count = 0       # and-*, or-*, xor-*, not-*, shl-*, shr-*
            compare_count = 0     # cmp*, if-cmp*
            move_count = 0        # move*, *-to-*
            termination_count = 0 # return*
            declaration_count = 0 # const*, new-*
            constants_count = 0   # const-string, const-*
            
            for inst in instructions:
                opcode = inst.get_name()
                
                # Categorize instruction by opcode - optimized checks
                if "invoke" in opcode:
                    call_count += 1
                elif "goto" in opcode or "if-" in opcode or "switch" in opcode:
                    transfer_count += 1
                elif any(x in opcode for x in ["add", "sub", "mul", "div", "rem", "neg"]):
                    arithmetic_count += 1
                elif any(x in opcode for x in ["and", "or", "xor", "not", "shl", "shr"]):
                    logic_count += 1
                elif "cmp" in opcode:
                    compare_count += 1
                elif "move" in opcode or "-to-" in opcode:
                    move_count += 1
                elif "return" in opcode:
                    termination_count += 1
                elif "const" in opcode or "new-" in opcode:
                    declaration_count += 1
                    if "const-string" in opcode or ("const-" in opcode and "class" not in opcode):
                        constants_count += 1
            
            attrs = {
                "start_addr": bb_obj.start,
                "end_addr": bb_obj.end,
                "instruction_count": instruction_count
            }
            block_attributes.append(attrs)
            
            # Create feature vector
            feature_vector = [
                call_count,
                transfer_count,
                arithmetic_count,
                logic_count,
                compare_count,
                move_count,
                termination_count,
                declaration_count,
                instruction_count,
                constants_count,
                bb_offspring_count[bb_obj]
            ]
            block_features.append(feature_vector)
            
        except Exception:
            # Create a default feature vector if an error occurs
            block_attributes.append({"start_addr": bb_obj.start, "end_addr": bb_obj.end, "instruction_count": 0})
            block_features.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    cfg_data["block_attributes"] = block_attributes
    cfg_data["block_features"] = block_features
    
    # Build CFG edges
    cfg_source_indices = []
    cfg_target_indices = []
    
    for source_bb_obj in all_basic_blocks_list:
        source_local_id = bb_obj_to_local_id[source_bb_obj]
        
        if hasattr(source_bb_obj, "childs"):
            try:
                for child_info_tuple in source_bb_obj.childs:
                    if len(child_info_tuple) >= 3:
                        target_bb_obj_candidate = child_info_tuple[2] 
                        if target_bb_obj_candidate in bb_obj_to_local_id:
                            target_local_id = bb_obj_to_local_id[target_bb_obj_candidate]
                            cfg_source_indices.append(source_local_id)
                            cfg_target_indices.append(target_local_id)
            except Exception:
                pass
    
    cfg_data["block_edges"] = [cfg_source_indices, cfg_target_indices]
    return cfg_data

def process_apk(apk_path, apk_hash, verbose=False):
    """Process a single APK file and extract its hierarchical graph structure."""
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        a, d_list, dx = AnalyzeAPK(apk_path)
    except Exception as e:
        logger.error(f"Failed to analyze APK {apk_path}: {str(e)}")
        return None
    
    output = {}
    
    # Use the provided APK hash
    output["hash"] = apk_hash
    
    # Extract APK metadata
    output["package_name"] = a.get_package()
    output["version_name"] = a.get_androidversion_name()
    output["version_code"] = a.get_androidversion_code()
    
    # Generate Function Call Graph
    fcg_nodes, fcg_edges, node_object_to_fcg_id, nodes_to_keep_original_ids, original_id_to_new_id_map, function_names = generate_fcg(dx)
    output["function_edges"] = fcg_edges
    output["function_count"] = len(nodes_to_keep_original_ids)
    output["function_names"] = function_names
    
    # Generate Control Flow Graphs for internal methods
    control_flow_edges_map = {}
    processed_canonical_names = set()
    
    # Process nodes in the original order (which defines fcg_id)
    for fcg_id, node_obj in enumerate(fcg_nodes):
        # Skip if the node is not in the filtered graph
        if fcg_id not in nodes_to_keep_original_ids:
            continue
            
        # Only generate CFGs for internal methods (not ExternalMethod)
        if isinstance(node_obj, ExternalMethod):
            continue
            
        canonical_name = _get_canonical_name(node_obj)
        if not canonical_name or canonical_name in processed_canonical_names:
            continue
                
        processed_canonical_names.add(canonical_name)
            
        try:
            cfg_data = generate_cfg_for_method(node_obj, fcg_id, dx)
            if cfg_data:
                block_edges = cfg_data.get("block_edges")
                if block_edges and len(block_edges) == 2 and block_edges[0]:
                    # Use the NEW, remapped ID as the key for control_flow_edges_map
                    new_fcg_id = original_id_to_new_id_map.get(fcg_id)
                    if new_fcg_id is not None:
                        control_flow_edges_map[str(new_fcg_id)] = {
                            "block_number": cfg_data.get("block_number", 0),
                            "block_edges": block_edges,
                            "block_features": cfg_data.get("block_features", [])
                        }
        except Exception:
            pass
    
    output["control_flow_edges"] = control_flow_edges_map
    return output

def process_apk_wrapper(args):
    """Wrapper function for multiprocessing"""
    apk_path, output_dir, verbose, family, year = args

    apk_filename = os.path.basename(apk_path)
    apk_hash = os.path.splitext(apk_filename)[0].lower()

    output_json_path = os.path.join(output_dir, f"{apk_hash}.json")

    if os.path.exists(output_json_path):
        logger.info(f"Output file {output_json_path} already exists. Skipping {apk_path}.")
        return (True, apk_path)

    try:
        apk_data = process_apk(apk_path, apk_hash, verbose)
        if apk_data:
            apk_data["family"] = family
            if year is not None:
                apk_data["year"] = year
            with open(output_json_path, 'w') as f:
                json.dump(apk_data, f)
            return (True, apk_path)
        return (False, apk_path)
    except Exception as e:
        logger.error(f"Error processing {apk_path} in worker: {str(e)}")
        return (False, f"{apk_path}: {str(e)}")

def main():
    import csv as _csv
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import Utils, _FLAT_DATASET_NAMES

    parser = argparse.ArgumentParser(
        description="Generate hierarchical graph data (FCG & CFGs) from APKs using Androguard."
    )
    parser.add_argument("--dataset", default="higraph",
                        choices=["higraph", "maldroid", "kronodroid"],
                        help="Dataset to process (default: higraph).")
    parser.add_argument("--years", nargs="+", default=None,
                        help="HiGraph only — one or more years to process (default: all available).")
    parser.add_argument("--apk_root", default=None,
                        help="Root directory of raw APKs. Defaults to raw_format/{dataset}.")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for JSON files. Defaults to dataset/{dataset}/graph/raw_data/higraph_base_data (higraph) or dataset/{dataset}/graph/raw (flat).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--limit", "-l", type=int, help="Limit the number of APKs to process")
    parser.add_argument("--processes", "-p", type=int, default=mp.cpu_count(),
                        help=f"Number of processes to use (default: {mp.cpu_count()})")
    parser.add_argument("--no-extension", action="store_true",
                        help="Process all files regardless of extension")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        set_log(level="DEBUG")
    else:
        logger.setLevel(logging.INFO)
        set_log(level="ERROR")

    flat = args.dataset in _FLAT_DATASET_NAMES
    apk_root = args.apk_root or Utils.get_raw_apk_root(args.dataset)
    truth_dir = Utils.get_refined_truth_root(args.dataset)
    graph_root = Utils.get_graph_root(args.dataset)

    if args.out_dir:
        base_out = args.out_dir
    elif flat:
        base_out = os.path.join(graph_root, "raw")
    else:
        base_out = os.path.join(graph_root, "raw_data", "higraph_base_data")

    all_tasks = []

    if flat:
        # ── Flat dataset (maldroid, kronodroid) ──────────────────────────────
        csv_path = os.path.join(truth_dir, f"labelled_samples_with_family_{args.dataset}_apks.csv")
        if not os.path.exists(csv_path):
            logger.error(f"CSV not found: {csv_path}")
            return

        hash_to_family = {}
        with open(csv_path, newline="") as f:
            for row in _csv.DictReader(f):
                hash_to_family[row["hash"].strip().lower()] = row["family"].strip()

        os.makedirs(base_out, exist_ok=True)

        apk_files = [
            f for f in os.listdir(apk_root)
            if args.no_extension or f.endswith(".apk")
        ]
        if args.limit:
            apk_files = apk_files[:args.limit]

        for filename in apk_files:
            sha256 = os.path.splitext(filename)[0].lower()
            family = hash_to_family.get(sha256)
            if family is None:
                logger.warning(f"Hash not in CSV, skipping: {sha256}")
                continue
            all_tasks.append((os.path.join(apk_root, filename), base_out, args.verbose, family, None))

        logger.info(f"[{args.dataset}] {len(all_tasks)} APKs to process → {base_out}")

    else:
        # ── HiGraph (yearly) ─────────────────────────────────────────────────
        import glob as _glob
        if args.years:
            years = args.years
        else:
            csv_pattern = os.path.join(truth_dir, "labelled_samples_with_family_*.csv")
            years = sorted(
                os.path.basename(p).replace("labelled_samples_with_family_", "").replace(".csv", "")
                for p in _glob.glob(csv_pattern)
            )

        total = 0
        for year in years:
            csv_path = os.path.join(truth_dir, f"labelled_samples_with_family_{year}.csv")
            if not os.path.exists(csv_path):
                logger.warning(f"CSV not found for year {year}, skipping.")
                continue

            hash_to_family = {}
            with open(csv_path, newline="") as f:
                for row in _csv.DictReader(f):
                    hash_to_family[row["hash"].strip().lower()] = row["family"].strip()

            apk_dir = os.path.join(apk_root, year)
            out_dir  = os.path.join(base_out, year)

            if not os.path.isdir(apk_dir):
                logger.warning(f"APK directory not found: {apk_dir}, skipping year {year}.")
                continue

            os.makedirs(out_dir, exist_ok=True)

            apk_files = [
                f for f in os.listdir(apk_dir)
                if args.no_extension or f.endswith(".apk")
            ]
            if args.limit:
                apk_files = apk_files[:args.limit]

            for filename in apk_files:
                sha256 = os.path.splitext(filename)[0].lower()
                family = hash_to_family.get(sha256)
                if family is None:
                    logger.warning(f"Hash not in CSV, skipping: {sha256}")
                    continue
                all_tasks.append((os.path.join(apk_dir, filename), out_dir, args.verbose, family, year))
                total += 1

            logger.info(f"  [{year}] {len(apk_files)} APKs queued → {out_dir}")

        logger.info(f"Total APKs to process: {total}")

    if not all_tasks:
        logger.error("No APKs found to process. Exiting.")
        return

    successful_count = 0
    failed_apks = []

    with mp.Pool(processes=args.processes) as pool:
        with tqdm(total=len(all_tasks), desc="Processing APKs") as pbar:
            for success, result in pool.imap_unordered(process_apk_wrapper, all_tasks):
                if success:
                    successful_count += 1
                else:
                    logger.error(f"Failed: {result}")
                    failed_apks.append(result)
                pbar.update(1)

    logger.info(f"Done — {successful_count}/{len(all_tasks)} succeeded.")
    if failed_apks:
        logger.warning(f"Failed APKs: {failed_apks}")

if __name__ == '__main__':
    main()