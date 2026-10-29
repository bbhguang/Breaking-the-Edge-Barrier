#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;
using Node = uint32_t;
using Count = uint64_t;
constexpr double KIB = 1024.0;
constexpr Count ELEMENT_BYTES = 32;
constexpr Count REPORT_BYTES = 16;

struct Graph {
    Node n = 0;
    std::vector<Count> offsets;
    std::vector<Node> neighbors;
    Count degree(Node u) const { return offsets[u + 1] - offsets[u]; }
};

struct Plan {
    Count job_id, projection_seed, sampling_seed, noise_seed, shuffle_seed;
    Node tau;
    double p, noise_scale;
};

struct TrialResult {
    double estimate;
    Count sampled_instances, projected_arcs, sampled_wedge_sum;
    double user_outbound_kib, shuffler_inbound_kib, expected_user_outbound_kib;
};

static std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

static Count integer(const std::string& token) {
    const auto s = trim(token);
    Count value = 0;
    const auto parsed = std::from_chars(s.data(), s.data() + s.size(), value);
    if (s.empty() || parsed.ec != std::errc() || parsed.ptr != s.data() + s.size())
        throw std::runtime_error("Expected a nonnegative integer: " + s);
    return value;
}

static double real(const std::string& token) {
    const auto s = trim(token);
    size_t used = 0;
    const double value = std::stod(s, &used);
    if (used != s.size() || !std::isfinite(value))
        throw std::runtime_error("Expected a finite number: " + s);
    return value;
}

static void set_threads(const std::string& arg) {
    const Count value = integer(arg);
    if (value < 1 || value > 4096) throw std::runtime_error("THREADS must be in [1,4096]");
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(static_cast<int>(value));
#else
    if (value != 1) std::cerr << "OpenMP is unavailable; using one thread.\n";
#endif
}

static Graph read_graph(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open input graph");
    std::string line;
    if (!std::getline(in, line) || trim(line) != "#nodes")
        throw std::runtime_error("Expected '#nodes' on the first line");
    if (!std::getline(in, line)) throw std::runtime_error("Missing node count");
    const Count n = integer(line);
    if (n < 2 || n > std::numeric_limits<Node>::max())
        throw std::runtime_error("Node count must be in [2, 2^32-1]");
    if (!std::getline(in, line) || trim(line) != "node,node")
        throw std::runtime_error("Expected 'node,node' on the third line");
    Graph g;
    g.n = static_cast<Node>(n);
    std::vector<std::pair<Node, Node>> edges;
    edges.reserve(static_cast<size_t>(fs::file_size(path) / 12));
    std::vector<Count> degrees(n, 0);
    Count self_loops = 0, line_number = 3;
    while (std::getline(in, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const auto comma = line.find(',');
        if (comma == std::string::npos || line.find(',', comma + 1) != std::string::npos)
            throw std::runtime_error("Invalid edge at line " + std::to_string(line_number));
        const Count u = integer(line.substr(0, comma)), v = integer(line.substr(comma + 1));
        if (u >= n || v >= n) throw std::runtime_error("Node ID outside [0,n) at line " + std::to_string(line_number));
        if (u == v) { ++self_loops; continue; }
        edges.emplace_back(static_cast<Node>(u), static_cast<Node>(v));
        ++degrees[u]; ++degrees[v];
    }
    g.offsets.assign(n + 1, 0);
    for (Count u = 0; u < n; ++u) g.offsets[u + 1] = g.offsets[u] + degrees[u];
    g.neighbors.resize(g.offsets.back());
    auto cursor = g.offsets;
    for (const auto& edge : edges) {
        g.neighbors[cursor[edge.first]++] = edge.second;
        g.neighbors[cursor[edge.second]++] = edge.first;
    }
    std::vector<std::pair<Node, Node>>().swap(edges);
    Count duplicates = 0;
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:duplicates)
    for (Count u = 0; u < n; ++u) {
        auto first = g.neighbors.begin() + g.offsets[u];
        auto last = g.neighbors.begin() + g.offsets[u + 1];
        std::sort(first, last);
        const Count unique = static_cast<Count>(std::unique(first, last) - first);
        duplicates += degrees[u] - unique;
        degrees[u] = unique;
    }
    if (duplicates) {
        Graph clean;
        clean.n = g.n;
        clean.offsets.assign(n + 1, 0);
        for (Count u = 0; u < n; ++u) clean.offsets[u + 1] = clean.offsets[u] + degrees[u];
        clean.neighbors.resize(clean.offsets.back());
        #pragma omp parallel for schedule(dynamic, 64)
        for (Count u = 0; u < n; ++u)
            std::copy_n(g.neighbors.begin() + g.offsets[u], degrees[u], clean.neighbors.begin() + clean.offsets[u]);
        g = std::move(clean);
    }
    std::cout << "graph n=" << g.n << " edges=" << g.neighbors.size()/2
              << " removed_self_loops=" << self_loops
              << " removed_duplicate_edges=" << duplicates/2 << '\n';
    return g;
}

static Count count_triangles(const Graph& g) {
    std::vector<Count> offsets(static_cast<Count>(g.n) + 1, 0);
    auto precedes = [&](Node u, Node v) {
        return g.degree(u) < g.degree(v) || (g.degree(u) == g.degree(v) && u < v);
    };
    #pragma omp parallel for schedule(dynamic, 64)
    for (Count u = 0; u < g.n; ++u)
        for (Count i = g.offsets[u]; i < g.offsets[u + 1]; ++i)
            offsets[u + 1] += precedes(static_cast<Node>(u), g.neighbors[i]);
    for (Count u = 0; u < g.n; ++u) offsets[u + 1] += offsets[u];
    if (offsets.back()*2 != g.neighbors.size()) throw std::runtime_error("Input graph is not undirected");
    std::vector<Node> forward(offsets.back());
    #pragma omp parallel for schedule(dynamic, 64)
    for (Count u = 0; u < g.n; ++u) {
        Count at = offsets[u];
        for (Count i = g.offsets[u]; i < g.offsets[u + 1]; ++i)
            if (precedes(static_cast<Node>(u), g.neighbors[i])) forward[at++] = g.neighbors[i];
    }
    Count triangles = 0;
    #pragma omp parallel reduction(+:triangles)
    {
        std::vector<uint8_t> marked(g.n, 0);
        #pragma omp for schedule(dynamic, 32)
        for (Count u = 0; u < g.n; ++u) {
            for (Count i = offsets[u]; i < offsets[u + 1]; ++i) marked[forward[i]] = 1;
            for (Count i = offsets[u]; i < offsets[u + 1]; ++i) {
                const Node v = forward[i];
                for (Count j = offsets[v]; j < offsets[v + 1]; ++j) triangles += marked[forward[j]];
            }
            for (Count i = offsets[u]; i < offsets[u + 1]; ++i) marked[forward[i]] = 0;
        }
    }
    return triangles;
}

static bool little_endian() { const uint16_t v = 1; return *reinterpret_cast<const uint8_t*>(&v) == 1; }

template<class T> static void write_values(std::ofstream& out, const T* data, size_t count) {
    if (little_endian()) out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(sizeof(T)*count));
    else for (size_t i = 0; i < count; ++i)
        for (size_t j = 0; j < sizeof(T); ++j) out.put(static_cast<char>((data[i] >> (8*j)) & 255));
    if (!out) throw std::runtime_error("Failed to write graph cache");
}

template<class T> static void read_values(std::ifstream& in, T* data, size_t count) {
    if (little_endian()) in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(sizeof(T)*count));
    else for (size_t i = 0; i < count; ++i) {
        data[i] = 0;
        for (size_t j = 0; j < sizeof(T); ++j) {
            const int byte = in.get();
            if (byte < 0) throw std::runtime_error("Truncated graph cache");
            data[i] |= static_cast<T>(static_cast<unsigned char>(byte)) << (8*j);
        }
    }
    if (!in) throw std::runtime_error("Truncated graph cache");
}

static void write_cache(const Graph& g, Count triangles, const fs::path& dir) {
    fs::create_directories(dir);
    const std::string magic = "STPGRP01";
    {
        std::ofstream out(dir / "graph.bin.tmp", std::ios::binary | std::ios::trunc);
        out.write(magic.data(), 8);
        const Count n = g.n, size = g.neighbors.size();
        write_values(out, &n, 1); write_values(out, &size, 1);
        write_values(out, g.offsets.data(), g.offsets.size());
        write_values(out, g.neighbors.data(), g.neighbors.size());
    }
    std::vector<Node> degrees(g.n);
    Node maximum = 0;
    for (Count u = 0; u < g.n; ++u) maximum = std::max(maximum, degrees[u] = static_cast<Node>(g.degree(static_cast<Node>(u))));
    {
        std::ofstream out(dir / "degrees.bin.tmp", std::ios::binary | std::ios::trunc);
        write_values(out, degrees.data(), degrees.size());
    }
    {
        std::ofstream out(dir / "metadata.json.tmp", std::ios::trunc);
        out << "{\"format_version\":1,\"n\":" << g.n << ",\"edges\":" << g.neighbors.size()/2
            << ",\"triangles\":" << triangles << ",\"max_degree\":" << maximum << "}\n";
        if (!out) throw std::runtime_error("Failed to write cache metadata");
    }
    for (const auto* file : {"graph.bin", "degrees.bin", "metadata.json"})
        fs::rename(dir / (std::string(file) + ".tmp"), dir / file);
}

static Graph read_cache(const fs::path& dir) {
    std::ifstream in(dir / "graph.bin", std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open graph cache");
    char magic[8]; in.read(magic, 8);
    if (!in || std::memcmp(magic, "STPGRP01", 8)) throw std::runtime_error("Unsupported graph cache format");
    Count n, size;
    read_values(in, &n, 1); read_values(in, &size, 1);
    if (n < 2 || n > std::numeric_limits<Node>::max() || size > n*(n-1))
        throw std::runtime_error("Invalid graph cache dimensions");
    const Count actual = fs::file_size(dir / "graph.bin");
    if (actual < 24 || size > (actual - 24)/4 || n + 1 > (actual - 24)/8 ||
        actual != 24 + 8*(n+1) + 4*size)
        throw std::runtime_error("Graph cache size mismatch");
    if (fs::file_size(dir / "degrees.bin") != n*sizeof(Node) || !fs::exists(dir / "metadata.json"))
        throw std::runtime_error("Missing or invalid degree cache");
    Graph g;
    g.n = static_cast<Node>(n); g.offsets.resize(n+1); g.neighbors.resize(size);
    read_values(in, g.offsets.data(), g.offsets.size());
    read_values(in, g.neighbors.data(), g.neighbors.size());
    if (g.offsets.front() != 0 || g.offsets.back() != size)
        throw std::runtime_error("Invalid cached offsets");
    for (Count u = 0; u < n; ++u) {
        if (g.offsets[u] > g.offsets[u+1] || g.offsets[u+1] > size)
            throw std::runtime_error("Invalid cached offsets");
        for (Count i = g.offsets[u]; i < g.offsets[u+1]; ++i)
            if (g.neighbors[i] >= n || g.neighbors[i] == u ||
                (i > g.offsets[u] && g.neighbors[i] <= g.neighbors[i-1]))
                throw std::runtime_error("Invalid cached adjacency row");
    }
    std::vector<Node> degrees(n);
    std::ifstream degree_file(dir / "degrees.bin", std::ios::binary);
    read_values(degree_file, degrees.data(), degrees.size());
    for (Count u = 0; u < n; ++u)
        if (degrees[u] != g.degree(static_cast<Node>(u))) throw std::runtime_error("Cached degree mismatch");
    return g;
}

static Count splitmix64(Count x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30))*0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27))*0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static std::mt19937_64 node_rng(Count seed, Node node) { return std::mt19937_64(splitmix64(seed ^ splitmix64(node))); }

static Count bounded(std::mt19937_64& rng, Count bound) {
    const Count threshold = -bound % bound;
    Count value;
    do { value = rng(); } while (value < threshold);
    return value % bound;
}

static Graph project(const Graph& g, Node tau, Count seed) {
    Graph projected;
    projected.n = g.n;
    projected.offsets.assign(static_cast<Count>(g.n) + 1, 0);
    for (Count u = 0; u < g.n; ++u)
        projected.offsets[u+1] = projected.offsets[u] + std::min<Count>(g.degree(static_cast<Node>(u)), tau);
    projected.neighbors.resize(projected.offsets.back());
    #pragma omp parallel for schedule(dynamic, 32)
    for (Count u = 0; u < g.n; ++u) {
        const Count degree = g.degree(static_cast<Node>(u));
        const Count k = projected.degree(static_cast<Node>(u));
        auto dest = projected.neighbors.begin() + projected.offsets[u];
        if (k == degree) { std::copy_n(g.neighbors.begin() + g.offsets[u], k, dest); continue; }
        auto rng = node_rng(seed, static_cast<Node>(u));
        // Floyd's algorithm samples a uniform subset without replacing neighbors.
        std::unordered_set<Count> chosen;
        chosen.reserve(static_cast<size_t>(2*k + 1));
        for (Count j = degree-k; j < degree; ++j) {
            const Count candidate = bounded(rng, j+1);
            if (!chosen.insert(candidate).second) chosen.insert(j);
        }
        for (Count position : chosen) *dest++ = g.neighbors[g.offsets[u] + position];
        std::sort(projected.neighbors.begin() + projected.offsets[u], dest);
    }
    return projected;
}

static Count intersection(const Graph& g, Node u, Node v) {
    Count i = g.offsets[u], j = g.offsets[v], total = 0;
    const Count end_i = g.offsets[u+1], end_j = g.offsets[v+1];
    while (i < end_i && j < end_j) {
        const Node a = g.neighbors[i], b = g.neighbors[j];
        if (a == b) { ++total; ++i; ++j; }
        else if (a < b) ++i;
        else ++j;
    }
    return total;
}

static double laplace(std::mt19937_64& rng, double scale) {
    // A midpoint grid keeps the uniform variate strictly inside (0,1).
    const double u = (static_cast<double>(rng() >> 12) + 0.5)*0x1.0p-52;
    return u < 0.5 ? scale*std::log(2*u) : -scale*std::log(2*(1-u));
}

static TrialResult run_projected(const Graph& projected, const Plan& plan) {
    Count sampled = 0, wedge_sum = 0;
    std::vector<double> reports(projected.n);
    #pragma omp parallel for schedule(dynamic, 32) reduction(+:sampled,wedge_sum)
    for (Count u = 0; u < projected.n; ++u) {
        auto sampling_rng = node_rng(plan.sampling_seed, static_cast<Node>(u));
        auto noise_rng = node_rng(plan.noise_seed, static_cast<Node>(u));
        Count local_wedges = 0;
        for (Count i = projected.offsets[u]; i < projected.offsets[u+1]; ++i) {
            const double uniform = static_cast<double>(sampling_rng() >> 11)*0x1.0p-53;
            if (plan.p == 1.0 || uniform < plan.p) {
                ++sampled;
                local_wedges += intersection(projected, static_cast<Node>(u), projected.neighbors[i]);
            }
        }
        wedge_sum += local_wedges;
        reports[u] = static_cast<double>(local_wedges) + laplace(noise_rng, plan.noise_scale);
    }
    std::mt19937_64 shuffle_rng(plan.shuffle_seed);
    // Fisher-Yates explicitly permutes the released wedge reports before aggregation.
    for (Count i = reports.size(); i > 1; --i) std::swap(reports[i-1], reports[bounded(shuffle_rng, i)]);
    const long double sum = std::accumulate(reports.begin(), reports.end(), static_cast<long double>(0));
    const double estimate = static_cast<double>(sum/(6.0L*plan.p));
    if (!std::isfinite(estimate)) throw std::runtime_error("Non-finite triangle estimate; check noise calibration");
    const long double instance_bytes = 3.0L*ELEMENT_BYTES*plan.tau;
    const double outgoing = static_cast<double>((instance_bytes*sampled/projected.n + REPORT_BYTES)/KIB);
    const double expected = static_cast<double>((instance_bytes*projected.neighbors.size()*plan.p/projected.n + REPORT_BYTES)/KIB);
    return {estimate, sampled, projected.neighbors.size(), wedge_sum,
            outgoing, REPORT_BYTES/KIB, expected};
}

static std::vector<std::string> fields(const std::string& line) {
    std::vector<std::string> result;
    std::stringstream input(line);
    std::string token;
    while (std::getline(input, token, ',')) result.push_back(trim(token));
    return result;
}

static std::vector<Plan> read_plans(const fs::path& path, const Graph& graph) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open plans CSV");
    const std::vector<std::string> header = {"job_id","tau","p","noise_scale","projection_seed","sampling_seed","noise_seed","shuffle_seed"};
    std::string line;
    if (!std::getline(in, line) || fields(line) != header) throw std::runtime_error("Unexpected plans CSV header");
    std::vector<Plan> plans;
    std::unordered_set<Count> jobs;
    while (std::getline(in, line)) {
        if (trim(line).empty()) continue;
        const auto f = fields(line);
        if (f.size() != 8) throw std::runtime_error("Expected eight plan fields");
        const Count tau = integer(f[1]);
        if (tau < 1 || tau >= graph.n) throw std::runtime_error("tau must be in [1,n-1]");
        const Plan p = {integer(f[0]), integer(f[4]), integer(f[5]), integer(f[6]), integer(f[7]),
                        static_cast<Node>(tau), real(f[2]), real(f[3])};
        if (p.p <= 0 || p.p > 1 || p.noise_scale < 0) throw std::runtime_error("Require 0 < p <= 1 and noise_scale >= 0");
        if (static_cast<long double>(graph.n)*tau*tau > std::numeric_limits<Count>::max())
            throw std::runtime_error("Wedge count upper bound exceeds uint64 range");
        if (!jobs.insert(p.job_id).second) throw std::runtime_error("Duplicate job_id");
        plans.push_back(p);
    }
    if (plans.empty()) throw std::runtime_error("Plans CSV has no trials");
    return plans;
}

static void execute(const fs::path& cache, const fs::path& plans_file, const fs::path& results_file) {
    const Graph graph = read_cache(cache);
    const auto plans = read_plans(plans_file, graph);
    if (!results_file.parent_path().empty()) fs::create_directories(results_file.parent_path());
    std::ofstream out(results_file, std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot open results CSV");
    out << "job_id,estimate,sampled_instances,projected_arcs,sampled_wedge_sum,user_outbound_kib,shuffler_inbound_kib,expected_user_outbound_kib\n";
    out << std::setprecision(17);
    for (const auto& plan : plans) {
        const auto projected = project(graph, plan.tau, plan.projection_seed);
        const auto r = run_projected(projected, plan);
        out << plan.job_id << ',' << r.estimate << ',' << r.sampled_instances << ',' << r.projected_arcs
            << ',' << r.sampled_wedge_sum << ',' << r.user_outbound_kib << ',' << r.shuffler_inbound_kib
            << ',' << r.expected_user_outbound_kib << '\n';
        out.flush();
        if (!out) throw std::runtime_error("Failed to write trial result");
        std::cout << "completed job_id=" << plan.job_id << " sampled_instances=" << r.sampled_instances << std::endl;
    }
}

int main(int argc, char** argv) {
    try {
        if (argc == 5 && std::string(argv[1]) == "prepare") {
            set_threads(argv[4]);
            const auto graph = read_graph(argv[2]);
            const Count triangles = count_triangles(graph);
            write_cache(graph, triangles, argv[3]);
            std::cout << "prepared triangles=" << triangles << '\n';
        } else if (argc == 6 && std::string(argv[1]) == "run") {
            set_threads(argv[5]);
            execute(argv[2], argv[3], argv[4]);
        } else {
            throw std::runtime_error("Usage: shuftri_engine prepare INPUT CACHE_DIR THREADS | run CACHE_DIR PLANS_CSV RESULTS_CSV THREADS");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
