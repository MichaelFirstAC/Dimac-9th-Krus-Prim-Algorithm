import sys
import gzip
import time
import heapq
from collections import defaultdict

# Increase recursion depth for deep graphs (just in case)
sys.setrecursionlimit(10**6)

class UnionFind:
    """Helper structure for Kruskal's Algorithm"""
    def __init__(self, n):
        self.parent = list(range(n + 1))
        self.rank = [0] * (n + 1)

    def find(self, i):
        if self.parent[i] != i:
            # Path compression
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i != root_j:
            # Union by rank
            if self.rank[root_i] < self.rank[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            if self.rank[root_i] == self.rank[root_j]:
                self.rank[root_i] += 1
            return True
        return False

def read_dimacs_graph(filename):
    """Reads a .gr.gz file and returns edges and adjacency list."""
    edges = []
    adj = defaultdict(list)
    num_nodes = 0
    
    print(f"Loading {filename}...")
    
    # Open gzip file directly
    with gzip.open(filename, 'rt') as f:
        for line in f:
            if line.startswith('p'):
                parts = line.split()
                num_nodes = int(parts[2])
            elif line.startswith('a'):
                # Format: a u v w
                parts = line.split()
                u, v, w = int(parts[1]), int(parts[2]), int(parts[3])
                
                # Store for Kruskal (Edge List)
                edges.append((u, v, w))
                
                # Store for Prim (Adjacency List)
                # DIMACS graphs are directed in file, but road networks 
                # for MST are usually treated as undirected. 
                # We add both directions just to be safe for MST logic.
                adj[u].append((w, v))
                adj[v].append((w, u))
                
    return num_nodes, edges, adj

def run_kruskal(num_nodes, edges):
    start_time = time.time()
    
    uf = UnionFind(num_nodes)
    mst_weight = 0
    edges_count = 0
    
    # Kruskal's Step 1: Sort edges by weight
    # This is usually the bottleneck (O(E log E))
    sorted_edges = sorted(edges, key=lambda x: x[2])
    
    for u, v, w in sorted_edges:
        if uf.union(u, v):
            mst_weight += w
            edges_count += 1
            
    end_time = time.time()
    return mst_weight, end_time - start_time

def run_prim(num_nodes, adj):
    start_time = time.time()
    
    # Priority Queue stores (weight, node)
    pq = [(0, 1)] # Start at node 1 (DIMACS is 1-indexed)
    visited = set()
    mst_weight = 0
    
    while pq and len(visited) < num_nodes:
        w, u = heapq.heappop(pq)
        
        if u in visited:
            continue
            
        visited.add(u)
        mst_weight += w
        
        for weight, neighbor in adj[u]:
            if neighbor not in visited:
                heapq.heappush(pq, (weight, neighbor))
                
    end_time = time.time()
    return mst_weight, end_time - start_time

def benchmark(datasets):
    print(f"{'Dataset':<10} | {'Algo':<8} | {'Run 1':<6} | {'Run 2':<6} | {'Run 3':<6} | {'Avg':<6} | {'Cost':<10}")
    print("-" * 75)

    for name, filename in datasets:
        try:
            num_nodes, edges, adj = read_dimacs_graph(filename)
        except FileNotFoundError:
            print(f"Error: {filename} not found. skipping.")
            continue

        # --- Test Kruskal ---
        k_times = []
        k_cost = 0
        for i in range(3):
            cost, duration = run_kruskal(num_nodes, edges)
            k_times.append(duration)
            k_cost = cost # Should be same every time
        
        k_avg = sum(k_times) / 3
        print(f"{name:<10} | Kruskal  | {k_times[0]:.2f}   | {k_times[1]:.2f}   | {k_times[2]:.2f}   | {k_avg:.2f}   | {k_cost}")

        # --- Test Prim ---
        p_times = []
        p_cost = 0
        for i in range(3):
            cost, duration = run_prim(num_nodes, adj)
            p_times.append(duration)
            p_cost = cost
            
        p_avg = sum(p_times) / 3
        print(f"{name:<10} | Prim     | {p_times[0]:.2f}   | {p_times[1]:.2f}   | {p_times[2]:.2f}   | {p_avg:.2f}   | {p_cost}")
        print("-" * 75)

if __name__ == "__main__":
    # Map Friendly Name -> Filename
    # Ensure these files are in the same directory as the script
    datasets = [
        ("NY", "USA-road-d.NY.gr.gz"),
        ("BAY", "USA-road-d.BAY.gr.gz"),
        ("COL", "USA-road-d.COL.gr.gz"),
        ("FLA", "USA-road-d.FLA.gr.gz")
    ]
    
    print("Starting Benchmark... (This may take a few minutes)")
    benchmark(datasets)