from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch

try:
    from rdkit import Chem
    RDKit_AVAILABLE = True
except ImportError:
    RDKit_AVAILABLE = False
    warnings.warn("RDKit not available. Install with: conda install -c conda-forge rdkit")

ATOM_TYPES = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']
ATOM_TYPE_MAP = {atype: i for i, atype in enumerate(ATOM_TYPES)}


def resolve_sair_pocket_path(structures_dir: str, entry_id: str, verify: bool = True) -> Optional[str]:
    if not structures_dir or not entry_id:
        return None
    candidates = (
        Path(structures_dir) / entry_id / "0" / "pocket_noH.pdb",
        Path(structures_dir) / entry_id / "pocket_noH.pdb",
        Path(structures_dir) / entry_id / entry_id / "pocket_noH.pdb",
    )
    if not verify:
        return str(candidates[0])
    return next((str(path) for path in candidates if path.exists()), None)


def resolve_plinder_pocket_path(plinder_root: str, system_id: str, verify: bool = True) -> Optional[str]:
    if not plinder_root or not system_id:
        return None
    path = Path(plinder_root) / "systems_processed" / system_id / "pocket_noH.pdb"
    if not verify:
        return str(path)
    return str(path) if path.exists() else None


class PocketProcessor:
    def __init__(self,
                 pocket_radius: float = 25.0,
                 max_atoms: int = 425,
                 include_hydrogens: bool = False,
                 centroid: Optional[Tuple[float, float, float]] = None,
                 node_scalar_dim: Optional[int] = None):
        self.pocket_radius = pocket_radius
        self.max_atoms = max_atoms
        self.include_hydrogens = include_hydrogens
        self.centroid = np.array(centroid) if centroid else None
        self.node_scalar_dim = node_scalar_dim or (len(ATOM_TYPES) + 1)
        if self.node_scalar_dim < 1:
            raise ValueError("node_scalar_dim must be positive")
        self.atom_feature_dim = min(len(ATOM_TYPES), self.node_scalar_dim - 1)
        
        if not RDKit_AVAILABLE:
            raise ImportError("RDKit is required for pocket processing. Install with: conda install -c conda-forge rdkit")
    
    def _parse_pdb(self, pdb_path: str) -> Tuple[np.ndarray, List[Dict]]:
        coords = []
        atoms = []
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    element = line[76:78].strip()
                    if not element:
                        element = line[12:14].strip()[0]
                    
                    if not self.include_hydrogens and element == 'H':
                        continue
                    
                    coords.append([x, y, z])
                    atoms.append({
                        'element': element,
                        'residue': line[17:20].strip(),
                        'chain': line[21],
                        'res_num': int(line[22:26]),
                        'is_het': line.startswith('HETATM')
                    })
        
        return np.array(coords), atoms
    
    def _get_pocket_center(self, all_coords: np.ndarray) -> np.ndarray:
        if self.centroid is not None:
            return self.centroid
        
        if len(all_coords) == 0:
            raise ValueError("No atoms found to determine pocket center")
        
        return np.mean(all_coords, axis=0)
    
    def _extract_pocket(self, 
                        all_coords: np.ndarray, 
                        atoms: List[Dict],
                        center: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        if len(all_coords) == 0:
            raise ValueError("No atoms found in PDB file")
        
        distances = np.linalg.norm(all_coords - center, axis=1)
        pocket_mask = distances <= self.pocket_radius
        
        pocket_coords = all_coords[pocket_mask]
        pocket_atoms = [atoms[i] for i in np.where(pocket_mask)[0]]
        
        return pocket_coords, pocket_atoms
    
    def _encode_atom_features(self, atoms: List[Dict]) -> torch.Tensor:
        features = []
        
        for atom in atoms:
            element = atom['element']
            feature = np.zeros(self.node_scalar_dim, dtype=np.float32)
            if element in ATOM_TYPE_MAP:
                idx = ATOM_TYPE_MAP[element]
                if idx < self.atom_feature_dim:
                    feature[idx] = 1.0
            
            aromatic = 0.0
            feature[-1] = aromatic
            features.append(feature)
        
        return torch.tensor(np.array(features), dtype=torch.float32)
    
    def _compute_edge_features(self, 
                               coords: np.ndarray,
                               edge_index: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        n_edges = len(src)
        
        rel_pos = coords[dst] - coords[src]
        distances = np.linalg.norm(rel_pos, axis=1)
        unit_rel_pos = rel_pos / np.maximum(distances[:, None], 1e-6)
        edge_vector = torch.tensor(unit_rel_pos, dtype=torch.float32).unsqueeze(1)

        centers = np.linspace(0, 8, 16)
        width = 0.5
        
        gaussian_features = np.zeros((n_edges, 16))
        for i, d in enumerate(distances):
            gaussian_features[i] = np.exp(-0.5 * ((d - centers) / width) ** 2)
        
        type_features = np.zeros((n_edges, 16))
        type_features[:, 0] = 1
        
        edge_scalar = torch.tensor(
            np.concatenate([gaussian_features, type_features], axis=1),
            dtype=torch.float32
        )
        
        return edge_scalar, edge_vector
    
    def _build_knn_graph(self, coords: np.ndarray, k: int = 8) -> np.ndarray:
        from scipy.spatial.distance import cdist
        
        n_atoms = len(coords)
        if n_atoms == 0:
            return np.zeros((2, 0), dtype=np.int64)
        
        dists = cdist(coords, coords)
        nearest = np.argsort(dists, axis=1)[:, :k+1]
        
        edges = []
        for i in range(n_atoms):
            for j in nearest[i]:
                if i != j:
                    edges.append([i, j])
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        
        return np.array(edges, dtype=np.int64).T
    
    def process_pdb(self, pdb_path: str) -> Dict[str, torch.Tensor]:
        all_coords, atoms = self._parse_pdb(pdb_path)
        center = self._get_pocket_center(all_coords)
        pocket_coords, pocket_atoms = self._extract_pocket(all_coords, atoms, center)
        
        if len(pocket_coords) == 0:
            raise ValueError(f"No pocket atoms found within {self.pocket_radius}Å of center")
        
        if len(pocket_coords) > self.max_atoms:
            pocket_coords = pocket_coords[:self.max_atoms]
            pocket_atoms = pocket_atoms[:self.max_atoms]
        centered_coords = pocket_coords - center
        
        edge_index = self._build_knn_graph(centered_coords, k=8)
        node_scalar = self._encode_atom_features(pocket_atoms)
        
        node_vector = torch.zeros(len(pocket_coords), 3, 3, dtype=torch.float32)
        centered_tensor = torch.tensor(centered_coords, dtype=torch.float32)
        node_vector[:, 0, :] = centered_tensor
        norms = torch.linalg.norm(centered_tensor, dim=1, keepdim=True).clamp_min(1e-6)
        node_vector[:, 1, :] = centered_tensor / norms
        
        edge_scalar, edge_vector = self._compute_edge_features(centered_coords, edge_index)
        
        return {
            'node_scalar': node_scalar,
            'node_vector': node_vector,
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'edge_scalar': edge_scalar,
            'edge_vector': edge_vector,
            'n_atoms': len(pocket_coords),
            'center': center,
        }
    
    def process_plinder_system(self, system_dir: str, pocket_file: str = "pocket_noH.pdb") -> Dict[str, torch.Tensor]:
        pdb_path = Path(system_dir) / pocket_file
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        
        return self.process_pdb(str(pdb_path))


class PLINDERProteinPocketConditionDataset:
    def __init__(self, 
                 csv_path: str,
                 data_root: str,
                 processor: Optional[PocketProcessor] = None,
                 pocket_file: str = "pocket_noH.pdb"):
        import pandas as pd
        
        self.data_root = Path(data_root)
        self.pocket_file = pocket_file
        self.processor = processor or PocketProcessor()
        
        self.df = pd.read_csv(csv_path)
        
        required_cols = ['system_id', 'fragment_selfies']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.system_ids = self.df['system_id'].tolist()
        self.fragment_selfies_strings = self.df['fragment_selfies'].tolist()
        self.gvp_embeddings = None
    
    def __len__(self):
        return len(self.system_ids)
    
    def __getitem__(self, idx):
        system_id = self.system_ids[idx]
        fragment_selfies = self.fragment_selfies_strings[idx]
        
        pocket_path = self.data_root / "systems_processed" / system_id / self.pocket_file
        
        if not pocket_path.exists():
            raise FileNotFoundError(f"Pocket file not found: {pocket_path}")
        
        pocket_data = self.processor.process_pdb(str(pocket_path))
        
        return {
            'system_id': system_id,
            'fragment_selfies': fragment_selfies,
            'pocket_data': pocket_data,
        }
    
    def precompute_gvp_embeddings(self, model, device='cpu'):
        self.gvp_embeddings = {}
        
        for idx in range(len(self)):
            system_id = self.system_ids[idx]
            fragment_selfies = self.fragment_selfies_strings[idx]
            
            sample = self[idx]
            pocket_data = sample['pocket_data']
            
            with torch.no_grad():
                pocket_embedding = model.encode_pocket_geometry(
                    node_features=(pocket_data['node_scalar'], pocket_data['node_vector']),
                    edge_index=pocket_data['edge_index'],
                    edge_features=(pocket_data['edge_scalar'], pocket_data['edge_vector'])
                )
            
            self.gvp_embeddings[system_id] = {
                'embedding': pocket_embedding.cpu(),
                'fragment_selfies': fragment_selfies
            }
        
        print(f"Pre-computed GVP embeddings for {len(self.gvp_embeddings)} systems")
    
    def get_condition_batch(self, system_ids: List[str], model, device='cpu'):
        embeddings = []
        fragment_selfies_list = []
        
        for system_id in system_ids:
            if self.gvp_embeddings is not None and system_id in self.gvp_embeddings:
                emb = self.gvp_embeddings[system_id]['embedding']
                fragment_selfies = self.gvp_embeddings[system_id]['fragment_selfies']
            else:
                idx = self.system_ids.index(system_id)
                sample = self[idx]
                pocket_data = sample['pocket_data']
                
                with torch.no_grad():
                    emb = model.encode_pocket_geometry(
                        node_features=(pocket_data['node_scalar'], pocket_data['node_vector']),
                        edge_index=pocket_data['edge_index'],
                        edge_features=(pocket_data['edge_scalar'], pocket_data['edge_vector'])
                    )
                fragment_selfies = sample['fragment_selfies']
            
            embeddings.append(emb)
            fragment_selfies_list.append(fragment_selfies)
        
        prot_poc_gvp_emb = torch.stack(embeddings, dim=0).to(device)
        
        return {
            'prot_poc_gvp_emb': prot_poc_gvp_emb,
            'fragment_selfies': fragment_selfies_list,
        }
