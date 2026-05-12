#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <set>
#include <unordered_set>
#include <torch/extension.h>
#include <unordered_map>

namespace py = pybind11;

// Helper for mapping (i,j) -> index
inline int pair_to_index(int a, int b, int M) {
    return a * M - (a * (a + 1)) / 2 + b - a - 1;
}

std::pair<std::vector<std::pair<int,int>>, std::vector<std::pair<int,int>>>
slg2_cpp(const std::vector<std::pair<int,int>>& input_edges) {

    // ---- Step 1: unique undirected edges ----
    std::vector<std::pair<int,int>> edges;
    for (auto &e : input_edges) {
        int u = e.first;
        int v = e.second;
        if (u < v)
            edges.emplace_back(u, v);
    }

    int M = edges.size();

    // ---- Step 2: node -> edges ----
    std::unordered_map<int, std::set<int>> node_to_edges;

    for (int eid = 0; eid < M; ++eid) {
        auto [u, v] = edges[eid];
        node_to_edges[u].insert(eid);
        node_to_edges[v].insert(eid);
    }

    // ---- Step 3: Incidence ----
    std::vector<std::set<int>> Inc(M);

    for (int eid = 0; eid < M; ++eid) {
        auto [u, v] = edges[eid];

        std::set<int> tmp = node_to_edges[u];
        tmp.insert(node_to_edges[v].begin(), node_to_edges[v].end());
        tmp.erase(eid);

        Inc[eid] = std::move(tmp);
    }

    // ---- Step 4: V_L2 ----
    std::vector<std::pair<int,int>> V_L2;
    for (int i = 0; i < M; ++i)
        for (int j = i + 1; j < M; ++j)
            V_L2.emplace_back(i, j);

    // ---- Step 5: E_L2 ----
    std::vector<std::pair<int,int>> E_L2;

    for (int i = 0; i < (int)V_L2.size(); ++i) {
        auto [e_u, e_v] = V_L2[i];

        std::set<int> Inc_A = Inc[e_u];
        Inc_A.insert(Inc[e_v].begin(), Inc[e_v].end());

        std::unordered_set<int> seen;

        for (int e_x : Inc_A) {

            // e_y < e_x
            for (int e_y = 0; e_y < e_x; ++e_y) {
                int j = pair_to_index(e_y, e_x, M);

                if (j > i && seen.find(j) == seen.end()) {
                    E_L2.emplace_back(i, j);
                    seen.insert(j);
                }
            }

            // e_y > e_x
            for (int e_y = e_x + 1; e_y < M; ++e_y) {
                int j = pair_to_index(e_x, e_y, M);

                if (j > i && seen.find(j) == seen.end()) {
                    E_L2.emplace_back(i, j);
                    seen.insert(j);
                }
            }
        }
    }

    return {V_L2, E_L2};
}



std::tuple<torch::Tensor, torch::Tensor> slg2_pg(torch::Tensor edge_index) {
    // Ensure tensor is on CPU, contiguous, and int64
    edge_index = edge_index.cpu().contiguous().toType(torch::kInt64);
    
    int num_edges_in = edge_index.size(1);
    auto edge_ptr = edge_index.accessor<int64_t, 2>();

    // ---- Step 1: unique undirected edges ----
    std::vector<std::pair<int,int>> edges;
    edges.reserve(num_edges_in);
    for (int i = 0; i < num_edges_in; ++i) {
        int u = edge_ptr[0][i];
        int v = edge_ptr[1][i];
        if (u < v) {
            edges.emplace_back(u, v);
        }
    }

    int M = edges.size();

    // ---- Step 2 & 3: Node to edges & Incidence ----
    std::unordered_map<int, std::set<int>> node_to_edges;
    for (int eid = 0; eid < M; ++eid) {
        auto [u, v] = edges[eid];
        node_to_edges[u].insert(eid);
        node_to_edges[v].insert(eid);
    }

    std::vector<std::set<int>> Inc(M);
    for (int eid = 0; eid < M; ++eid) {
        auto [u, v] = edges[eid];
        std::set<int> tmp = node_to_edges[u];
        tmp.insert(node_to_edges[v].begin(), node_to_edges[v].end());
        tmp.erase(eid);
        Inc[eid] = std::move(tmp);
    }

    // ---- Step 4: V_L2 (Using flattened vector for fast Tensor conversion) ----
    std::vector<int64_t> v_l2_flat;
    v_l2_flat.reserve(M * (M - 1)); 
    for (int i = 0; i < M; ++i) {
        for (int j = i + 1; j < M; ++j) {
            v_l2_flat.push_back(i);
            v_l2_flat.push_back(j);
        }
    }
    int num_v_l2 = v_l2_flat.size() / 2;

    // ---- Step 5: E_L2 (Using flattened vector) ----
    std::vector<int64_t> e_l2_flat;
    for (int i = 0; i < num_v_l2; ++i) {
        int e_u = v_l2_flat[2*i];
        int e_v = v_l2_flat[2*i + 1];

        std::set<int> Inc_A = Inc[e_u];
        Inc_A.insert(Inc[e_v].begin(), Inc[e_v].end());
        std::unordered_set<int> seen;

        for (int e_x : Inc_A) {
            for (int e_y = 0; e_y < e_x; ++e_y) {
                int j = pair_to_index(e_y, e_x, M);
                if (j > i && seen.find(j) == seen.end()) {
                    e_l2_flat.push_back(i); e_l2_flat.push_back(j); seen.insert(j);
                }
            }
            for (int e_y = e_x + 1; e_y < M; ++e_y) {
                int j = pair_to_index(e_x, e_y, M);
                if (j > i && seen.find(j) == seen.end()) {
                    e_l2_flat.push_back(i); e_l2_flat.push_back(j); seen.insert(j);
                }
            }
        }
    }

    // ---- Step 6: Direct Tensor Creation (Zero Python Overhead) ----
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    
    torch::Tensor V_L2_tensor;
    if (num_v_l2 > 0) {
        // from_blob creates a tensor pointing to C++ memory. clone() takes ownership.
        // t().contiguous() instantly gives us the [2, N] shape PyG expects!
        V_L2_tensor = torch::from_blob(v_l2_flat.data(), {num_v_l2, 2}, opts).clone().t().contiguous();
    } else {
        V_L2_tensor = torch::empty({2, 0}, opts);
    }

    torch::Tensor E_L2_tensor;
    int num_e_l2 = e_l2_flat.size() / 2;
    if (num_e_l2 > 0) {
        E_L2_tensor = torch::from_blob(e_l2_flat.data(), {num_e_l2, 2}, opts).clone().t().contiguous();
    } else {
        E_L2_tensor = torch::empty({2, 0}, opts);
    }

    return std::make_tuple(V_L2_tensor, E_L2_tensor);
}

// Module Definition
PYBIND11_MODULE(slg2lib, m) {
    m.def("slg2", &slg2_cpp, "Compute SLG2 graph (Legacy Python List version)");
    m.def("slg2_pg", &slg2_pg, "Compute SLG2 graph natively from PyTorch Tensors");
}