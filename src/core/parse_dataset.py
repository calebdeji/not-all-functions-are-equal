# Local Degree Profile (LDP)
"""
# ( d(v) ): The degree of node ( v ) (the number of its direct neighbors).
# ( \min(N(v)) ): The minimum degree among all neighbors of ( v ).
# ( \max(N(v)) ): The maximum degree among all neighbors of ( v ).
# ( \mu(N(v)) ): The average (mean) degree of the neighbors of ( v ).
# ( \sigma(N(v)) ): The standard deviation of the degrees of the neighbors of ( v ).
"""

import os
import shutil
import json
import traceback

import torch
import pandas as pd
from torch_geometric.data import Dataset, Data


from src.core.utils import Utils


def get_dataset_root():
    return Utils.get_graph_root('higraph')




class HiGraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'cfg_node_batch':
            return self.num_functions

        elif key == 'fcg_edge_index':
            # This tells the DataLoader to increment fcg_edge_index
            # by the number of nodes in 'function_features', not 'x'.
            return self.function_features.shape[0]

        else:
            # Default behavior for all other keys (like 'edge_index')
            if isinstance(value, torch.Tensor) and value.numel() == 0:
                return 0
            return super().__inc__(key, value, *args, **kwargs)

class HiGraphDataset(Dataset):
    def __init__(self, root, year: str, node_dim = 11, transform = None, pre_transform = None, pre_filter = None):
        self.year = year
        r = os.path.join(root, self.year)
        self.csv_data = self._get_csv_data_frame()
        self.node_dim = node_dim
        self.family_mapping = HiGraphDataset.get_family_mapping()
        super().__init__(r, transform, pre_transform, pre_filter)

    def _get_csv_data_frame(self):
        csv_file_name = f"labelled_samples_with_family_{self.year}.csv"
        print(f"{csv_file_name} and year {self.year}")
        csv_file_path = os.path.join(Utils.get_project_root(), 'dataset_refined_truth', 'higraph', csv_file_name)

        csv_file_df = pd.read_csv(csv_file_path)

        return csv_file_df

    def _get_local_data_path(self):
        return os.path.join(get_dataset_root(), 'raw_data', 'higraph_base_data', self.year)

    @property
    def raw_file_names(self):
        csv_file_df = self.csv_data
        row_tuples = csv_file_df.itertuples(index=True, name='Row')
        file_names = []
        for row in row_tuples:
            file_hash = f'{row.hash}.json'
            file_names.append(file_hash)
        return file_names

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.raw_file_names))]




    def download(self):
        local_data_path = self._get_local_data_path()
        raw_dir = self.raw_dir
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
        else:
            print(f"Raw directory {raw_dir} already exists ")
            shutil.rmtree(raw_dir)
            os.makedirs(raw_dir)

        for file_name in self.raw_file_names:
            file_path = os.path.join(raw_dir, file_name)

            src_path = os.path.join(local_data_path, file_name)

            if os.path.exists(src_path):
                shutil.copy(src_path, file_path)
                print(f"Downloaded {file_name} to {file_path}")
            else:
                raise FileNotFoundError(f"File {file_name} not found in {local_data_path}")

    @staticmethod
    def get_family_mapping():
        with open(os.path.join(Utils.get_project_root(), 'dataset_refined_truth', 'higraph', 'global_families_mapping.json')) as f:
            family_to_index = json.load(f)

        return family_to_index

    @staticmethod
    def compute_degrees(edge_list, num_nodes):

        if len(edge_list) == 0:
            raise Exception('No edges found')


        edges = torch.tensor(edge_list, dtype=torch.long)

        sources = edges[0]
        destinations = edges[1]


        out_degrees = torch.bincount(sources, minlength=num_nodes)
        in_degrees = torch.bincount(destinations, minlength=num_nodes)

        return in_degrees, out_degrees


    @staticmethod
    def compute_cyclomatic_complexity(num_nodes, edge_list):
        """
        Calculates Cyclomatic Complexity: M = E - N + 2P
        Assuming P=1 (single function).
        """
        if num_nodes == 0:
            return torch.tensor([0.0])

        E = len(edge_list[0]) if len(edge_list) > 0 else 0
        N = num_nodes

        # M = E - N + 2
        complexity = E - N + 2

        # Clamp at 1.0 (Complexity cannot be less than 1)
        complexity = max(1.0, complexity)

        # Log-scale because complexity follows a power law (prevents huge values dominating)
        return torch.log1p(torch.tensor([complexity], dtype=torch.float))

    @staticmethod
    def compute_fcg_ldp(edges_list, num_nodes, in_degrees, out_degrees):
        if num_nodes == 0 or len(edges_list) != 2 or len(edges_list[0]) != len(edges_list[1]):
            raise Exception(f'Corrupt Data! Numbe of nodes {num_nodes} and edges {len(edges_list)} -> edges {edges_list}')

        edges = torch.tensor(edges_list, dtype=torch.long)
        sources, destinations = edges[0], edges[1]




        total_degrees = in_degrees + out_degrees

        ldp_features = torch.zeros((num_nodes, 5), dtype=torch.float)

        ldp_features[:, 0] = total_degrees.float()

        for v in range(num_nodes):

            out_neighbors = destinations[sources == v]

            in_neighbors = sources[destinations == v]

            all_neighbors = torch.cat((in_neighbors, out_neighbors))
            unique_neighbors = torch.unique(all_neighbors)

            if len(unique_neighbors) > 0:
                neighbor_degs = total_degrees[unique_neighbors].float()

                ldp_features[v, 1] = neighbor_degs.min()
                ldp_features[v, 2] = neighbor_degs.max()
                ldp_features[v, 3] = neighbor_degs.mean()

                if len(unique_neighbors) > 1:
                    ldp_features[v, 4] = neighbor_degs.std(unbiased=False)
                else:
                    ldp_features[v, 4] = 0.0

        # print(f"LDP Features for Node 1: {ldp_matrix[1].tolist()}")
        return ldp_features



    def get_labels(self):
        return torch.load(os.path.join(self.processed_dir, 'labels.pt'), weights_only=True)

    def process(self):
        idx = 0
        all_labels = []
        csv_file_df = self.csv_data

        for raw_path in self.raw_paths:
            with open(raw_path, 'r') as f:
                json_data = json.load(f)

            json_data_hash = json_data['hash']
            matching_row = csv_file_df.loc[csv_file_df['hash'] == json_data_hash]

            if len(matching_row) != 1:
                raise Exception("Unknown family value")


            fcg_edges = json_data['function_edges']
            function_count = json_data['function_count']
            control_flow_edges = json_data['control_flow_edges']
            control_flow_edges_map = (
                control_flow_edges if isinstance(control_flow_edges, dict)
                else json.loads(control_flow_edges)
            )

            # function_names = json_data['function_names']

            in_degrees, out_degrees = HiGraphDataset.compute_degrees(fcg_edges, function_count)
            ldp_features = HiGraphDataset.compute_fcg_ldp(fcg_edges, function_count, in_degrees, out_degrees)

            new_function_features = []

            for i in range(function_count):

                func_id = str(i)


                if func_id in control_flow_edges_map:
                    type_feature = torch.tensor([1.0], dtype=torch.float)
                else:
                    type_feature = torch.tensor([0.0], dtype=torch.float)

                if func_id in control_flow_edges_map:
                    cfg_info = control_flow_edges_map[func_id]

                    block_features_list = cfg_info['block_features']
                    block_edges_list = cfg_info['block_edges']

                    num_cfg_nodes = len(block_features_list)

                    complexity_feat = HiGraphDataset.compute_cyclomatic_complexity(num_cfg_nodes, block_edges_list)

                else:
                    complexity_feat = torch.tensor([0.0], dtype=torch.float)

                ldp_feat = ldp_features[i].tolist()
                ldp_feat = torch.log1p(torch.tensor(ldp_feat, dtype=torch.float))
                ldp_feat = ldp_feat[1:]
                in_deg = torch.log1p(torch.tensor([in_degrees[i]], dtype=torch.float))
                out_deg = torch.log1p(torch.tensor([out_degrees[i]], dtype=torch.float))
                final_features = torch.cat([type_feature, complexity_feat, ldp_feat, in_deg, out_deg], dim=0)

                new_function_features.append(final_features)


            x_list, ei_list, batch_list = [], [], []
            node_offset = 0
            for i in range(function_count):
                func_id = str(i)

                if func_id in control_flow_edges_map:
                    cfg_info = control_flow_edges_map[func_id]

                    block_features_list = cfg_info['block_features']
                    x_raw_full = torch.tensor(block_features_list, dtype=torch.float)

                    x_content = x_raw_full[:, :self.node_dim - 1]
                    block_edges_list = cfg_info['block_edges']
                    internal_edge_index = torch.tensor(block_edges_list, dtype=torch.long)

                    x = torch.log1p(x_content)

                    if internal_edge_index.numel() > 0:
                        ei_list.append(internal_edge_index + node_offset)

                else:
                    x = torch.zeros(1, self.node_dim - 1, dtype=torch.float)

                n = x.shape[0]
                x_list.append(x)
                batch_list.append(torch.full((n,), i, dtype=torch.long))
                node_offset += n

            x_cat = torch.cat(x_list, dim=0)
            ei_cat = torch.cat(ei_list, dim=1) if ei_list else torch.empty((2, 0), dtype=torch.long)
            batch_cat = torch.cat(batch_list, dim=0)
            fcg_edge_index = torch.tensor(fcg_edges, dtype=torch.long)

            data = HiGraphData()
            data.x = x_cat
            data.edge_index = ei_cat
            data.cfg_node_batch = batch_cat
            data.function_features = torch.stack(new_function_features)
            data.fcg_edge_index = fcg_edge_index
            data.family = json_data['family'].lower()
            data.hash = json_data_hash
            data_year = int(json_data['year'])
            data_year_parsed_value = data_year - 2010
            data.year_value = torch.tensor([data_year_parsed_value], dtype=torch.long)
            data.num_functions = torch.tensor([function_count], dtype=torch.long)
            data.y = self.family_mapping[json_data['family'].lower()]

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            print(f"[*] Processed {idx} for year {data_year}")
            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            all_labels.append(data.y)
            idx += 1

        torch.save(torch.tensor(all_labels, dtype=torch.long), os.path.join(self.processed_dir, 'labels.pt'))
        print(f"[*] Saved labels.pt with {len(all_labels)} entries")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False,
                          map_location=torch.device('cpu'))
        return data



class _FlatGraphDataset(Dataset):
    """
    Base class for year-free graph datasets (FlatGraph).

    Expects root/<dataset>/raw/*.json with fields:
        hash, family, function_count, function_edges, control_flow_edges
    control_flow_edges may be a dict or a JSON-encoded string — both handled.
    Family-to-index mapping is built from the raw files and saved alongside
    the processed tensors so it is stable across runs.
    """

    def __init__(self, root, node_dim=11, transform=None, pre_transform=None, pre_filter=None):
        self.node_dim = node_dim
        self._raw_file_names_cache = None
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        if self._raw_file_names_cache is None:
            self._raw_file_names_cache = sorted(
                f for f in os.listdir(self.raw_dir) if f.endswith('.json')
            )
        return self._raw_file_names_cache

    @property
    def processed_file_names(self):
        count_path = os.path.join(self.processed_dir, 'count.pt')
        if os.path.exists(count_path):
            n = torch.load(count_path, weights_only=True).item()
        else:
            n = len(self.raw_file_names)  # first run: triggers processing
        return [f'data_{i}.pt' for i in range(n)]

    def download(self):
        pass  # raw files already in place

    def _build_family_mapping(self):
        families = set()
        for fname in self.raw_file_names:
            with open(os.path.join(self.raw_dir, fname)) as f:
                d = json.load(f)
            families.add(d['family'].lower())
        return {fam: idx for idx, fam in enumerate(sorted(families))}

    def process(self):
        family_mapping = self._build_family_mapping()
        torch.save(family_mapping, os.path.join(self.processed_dir, 'family_mapping.pt'))

        idx = 0
        all_labels = []

        for raw_path in self.raw_paths:
            with open(raw_path) as f:
                json_data = json.load(f)

            fcg_edges = json_data['function_edges']
            function_count = json_data['function_count']

            if function_count == 0 or len(fcg_edges[0]) == 0:
                print(f'[!] Skipping {json_data["hash"]} — empty graph')
                continue

            cfe_raw = json_data['control_flow_edges']
            control_flow_edges_map = cfe_raw if isinstance(cfe_raw, dict) else json.loads(cfe_raw)

            in_degrees, out_degrees = HiGraphDataset.compute_degrees(fcg_edges, function_count)
            ldp_features = HiGraphDataset.compute_fcg_ldp(fcg_edges, function_count, in_degrees, out_degrees)

            new_function_features = []
            for i in range(function_count):
                func_id = str(i)

                if func_id in control_flow_edges_map:
                    type_feature = torch.tensor([1.0], dtype=torch.float)
                    cfg_info = control_flow_edges_map[func_id]
                    num_cfg_nodes = len(cfg_info['block_features'])
                    complexity_feat = HiGraphDataset.compute_cyclomatic_complexity(num_cfg_nodes, cfg_info['block_edges'])
                else:
                    type_feature = torch.tensor([0.0], dtype=torch.float)
                    complexity_feat = torch.tensor([0.0], dtype=torch.float)

                ldp_feat = torch.log1p(torch.tensor(ldp_features[i].tolist(), dtype=torch.float))[1:]
                in_deg = torch.log1p(torch.tensor([in_degrees[i]], dtype=torch.float))
                out_deg = torch.log1p(torch.tensor([out_degrees[i]], dtype=torch.float))
                new_function_features.append(torch.cat([type_feature, complexity_feat, ldp_feat, in_deg, out_deg]))

            x_list, ei_list, batch_list = [], [], []
            node_offset = 0
            for i in range(function_count):
                func_id = str(i)
                if func_id in control_flow_edges_map:
                    cfg_info = control_flow_edges_map[func_id]
                    x_raw = torch.tensor(cfg_info['block_features'], dtype=torch.float)
                    x = torch.log1p(x_raw[:, :self.node_dim - 1])
                    internal_ei = torch.tensor(cfg_info['block_edges'], dtype=torch.long)
                    if internal_ei.numel() > 0:
                        ei_list.append(internal_ei + node_offset)
                else:
                    x = torch.zeros(1, self.node_dim - 1, dtype=torch.float)

                n = x.shape[0]
                x_list.append(x)
                batch_list.append(torch.full((n,), i, dtype=torch.long))
                node_offset += n

            data = HiGraphData()
            data.x = torch.cat(x_list, dim=0)
            data.edge_index = torch.cat(ei_list, dim=1) if ei_list else torch.empty((2, 0), dtype=torch.long)
            data.cfg_node_batch = torch.cat(batch_list, dim=0)
            data.function_features = torch.stack(new_function_features)
            data.fcg_edge_index = torch.tensor(fcg_edges, dtype=torch.long)
            data.family = json_data['family'].lower()
            data.hash = json_data['hash']
            data.num_functions = torch.tensor([function_count], dtype=torch.long)
            data.y = family_mapping[json_data['family'].lower()]

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            all_labels.append(data.y)
            idx += 1
            print(f'[*] Processed {idx}/{len(self.raw_file_names)}')

        torch.save(torch.tensor(all_labels, dtype=torch.long), os.path.join(self.processed_dir, 'labels.pt'))
        torch.save(torch.tensor(idx), os.path.join(self.processed_dir, 'count.pt'))
        print(f'[*] Saved labels.pt with {len(all_labels)} entries')

    def get_labels(self):
        return torch.load(os.path.join(self.processed_dir, 'labels.pt'), weights_only=True)

    def get_family_mapping(self):
        return torch.load(os.path.join(self.processed_dir, 'family_mapping.pt'), weights_only=True)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False,
                          map_location=torch.device('cpu'))


class FlatGraph(_FlatGraphDataset):
    """HiGraph-format dataset for year-free flat datasets (e.g. maldroid, drebin)."""
    pass


if __name__ == '__main__':
    try:
        maldroid_ds = FlatGraph(root=Utils.get_graph_root('maldroid'))
    except Exception as e:
        print(f"[!] An error occured while loading the dataset: {e}")
        traceback.print_exc()

