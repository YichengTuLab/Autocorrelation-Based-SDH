// acs.c -- Autocorrelation-Based Spatial Distance Histogram (ACS).
//
// Reference implementation for the paper's frequency-space SDH algorithms.
// One binary computes a d-dimensional spatial distance histogram by three
// interchangeable methods, selectable at runtime:
//
//   dualtree : Dual-Tree             (spatial-index baseline)
//   autocorr : Basic ACS             (one leaf-level FFT)
//   ring     : Ring Enumeration (RE) (row-wise shell traversal; main method)
//
// Dimensionality d = 1..4 via --dim D. Each method runs in exact (approx=0) or
// approximate (approx=1) mode. Input points are either generated internally
// (uniform i.i.d., fixed seed) or read from a whitespace-separated coordinate
// file given in the RDF_INPUT environment variable.
//
// Build:
//   gcc -O3 -fopenmp acs.c -o acs -lfftw3 -lfftw3_omp -lm
//   (or simply `make` from the repository root)
//
// Build with -DNO_INNER_STATS to disable the inner per-pair counter.
//
// See ../README.md for the full command-line interface and reproduction steps.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>
#include <omp.h>
#include <complex.h>
#include <fftw3.h>

// ============================================================================
// COMPILE-TIME LIMITS
// ============================================================================

#define MAX_DIM 4

// Runtime dimensionality; set once in main from --dim D (default 2).
static int g_dim = 2;

// ============================================================================
// TIMING UTILITIES
// ============================================================================

double get_wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void print_time(const char *label, double seconds) {
    if (seconds < 1.0) {
        printf("  %-35s %.3f ms\n", label, seconds * 1000.0);
    } else if (seconds < 60.0) {
        printf("  %-35s %.3f s\n", label, seconds);
    } else {
        int minutes = (int)(seconds / 60);
        double rem  = seconds - minutes * 60;
        printf("  %-35s %d min %.2f s\n", label, minutes, rem);
    }
}

// ============================================================================
// POINT (fixed-size POD, D <= MAX_DIM)
// ============================================================================

typedef struct {
    double x[MAX_DIM];
} Point;

// ============================================================================
// N-DIMENSIONAL TREE NODE
// ============================================================================
//
// Generalization of QuadNode.  Each internal node has 2^D children, indexed
// by a D-bit number whose bit k selects the high half along axis k.
//
// IMPORTANT: at D=2 we preserve the existing 2D child ordering exactly so
// histograms match rdf_MP-29.c byte-for-byte.  The 2D code uses:
//   bit 0 = (x >= mid)   (1 = right half)
//   bit 1 = (y <  mid)   (1 = bottom half, i.e. y FLIPPED)
// At D>=3 we use the natural ordering: bit k = (x[k] >= mid[k]).
// We funnel everything through nD_child_index_for_point() so the convention
// is in one place.

#define MAX_CHILDREN (1 << MAX_DIM)   // = 16 at MAX_DIM=4

typedef struct TreeNode {
    double lo[MAX_DIM], hi[MAX_DIM];
    int count;
    int level;
    Point *points;                 // valid only at leaves; into compacted buffer
    struct TreeNode *children[MAX_CHILDREN];
    int is_leaf;
} TreeNode;

// Compute the child index for point p inside this node, using the convention
// that depends on g_dim.
static inline int nD_child_index_for_point(const TreeNode *node, Point p) {
    int idx = 0;
    if (g_dim == 2) {
        // Preserve the 2D quirk: bit 0 = (x>=mid_x), bit 1 = (y<mid_y).
        double mid_x = 0.5 * (node->lo[0] + node->hi[0]);
        double mid_y = 0.5 * (node->lo[1] + node->hi[1]);
        if (p.x[0] >= mid_x) idx |= 1;
        if (p.x[1] <  mid_y) idx |= 2;
        return idx;
    }
    // Natural ordering: bit k set iff x[k] >= mid[k].
    for (int k = 0; k < g_dim; k++) {
        double mid_k = 0.5 * (node->lo[k] + node->hi[k]);
        if (p.x[k] >= mid_k) idx |= (1 << k);
    }
    return idx;
}

// Set the lo/hi box of child `idx` of `parent`, using the same convention.
static inline void nD_set_child_box(const TreeNode *parent, int idx,
                                    double *child_lo, double *child_hi) {
    if (g_dim == 2) {
        double mid_x = 0.5 * (parent->lo[0] + parent->hi[0]);
        double mid_y = 0.5 * (parent->lo[1] + parent->hi[1]);
        // bit 0: 0 -> [lo_x, mid_x], 1 -> [mid_x, hi_x]
        // bit 1: 0 -> [mid_y, hi_y], 1 -> [lo_y, mid_y]   (FLIPPED)
        if ((idx & 1) == 0) { child_lo[0] = parent->lo[0]; child_hi[0] = mid_x; }
        else                { child_lo[0] = mid_x;          child_hi[0] = parent->hi[0]; }
        if ((idx & 2) == 0) { child_lo[1] = mid_y;          child_hi[1] = parent->hi[1]; }
        else                { child_lo[1] = parent->lo[1];  child_hi[1] = mid_y; }
        return;
    }
    for (int k = 0; k < g_dim; k++) {
        double mid_k = 0.5 * (parent->lo[k] + parent->hi[k]);
        if (((idx >> k) & 1) == 0) {
            child_lo[k] = parent->lo[k];
            child_hi[k] = mid_k;
        } else {
            child_lo[k] = mid_k;
            child_hi[k] = parent->hi[k];
        }
    }
}

// ============================================================================
// STATISTICS (unchanged from 2D)
// ============================================================================

typedef struct {
    long long exact_distance_computations;
    long long pruned_pairs;
    long long leaf_comparisons;
    long long prune_operations;
    long long displacement_distance_calls;
    long long coarse_resolved_pairs;
} Statistics;

#ifdef NO_INNER_STATS
  #define STAT_INNER_INC(field) ((void)0)
#else
  #define STAT_INNER_INC(field) ((field)++)
#endif

// ============================================================================
// METHOD ENUM
// ============================================================================

typedef enum {
    METHOD_DUALTREE      = 0,
    METHOD_AUTOCORR      = 1,
    METHOD_AUTOCORR_RING = 4
} ComputeMethod;

// File-scope flag for approximate-mode heuristic distribution.
int g_approx_mode = 0;

// Whether leaf point arrays were compacted into one buffer (so free_tree must
// NOT free per-leaf arrays — main frees the single buffer).  When compaction
// is skipped (approx mode), each leaf owns its own malloc'd array and
// free_tree must free it.
int g_leaves_compacted = 0;


// Use the k-d (alternating single-axis cut) coarse pyramid instead of the
// octree pyramid.  Set via --kd or RDF_KD=1.  Octree remains the default.
int g_use_kd = 0;

// Global counter for displacement-distance-range calls (informational).
static long long g_displacement_distance_calls = 0;

// ============================================================================
// Forward declarations of utilities used early.
// ============================================================================

int  compute_tree_depth(int n_points, double avg_points_per_leaf);


// ============================================================================
// TREE DEPTH FROM AVERAGE LEAF SIZE
// ============================================================================
//
// In nD, each subdivision splits ONE cell into 2^D, so the number of leaves
// at depth L is (2^D)^L = 2^(L*D).  Choose L so 2^(L*D) ~= n_points / avg.

int compute_tree_depth(int n_points, double avg_points_per_leaf) {
    if (avg_points_per_leaf <= 0.0 || n_points <= 0) return 0;
    double ratio = (double)n_points / avg_points_per_leaf;
    // depth such that (2^D)^depth >= ratio, i.e. depth >= log(ratio)/(D*log2).
    double depth_d = log(ratio) / (g_dim * log(2.0));
    int depth = (int)ceil(depth_d);
    if (depth < 1) depth = 1;
    if (depth > 30) depth = 30;
    return depth;
}

// ============================================================================
// TREE CONSTRUCTION
// ============================================================================

TreeNode* create_node(const double *lo, const double *hi, int level) {
    TreeNode *node = (TreeNode*)malloc(sizeof(TreeNode));
    for (int k = 0; k < g_dim; k++) {
        node->lo[k] = lo[k];
        node->hi[k] = hi[k];
    }
    // Pad unused axes to zero so we never read uninitialized memory.
    for (int k = g_dim; k < MAX_DIM; k++) {
        node->lo[k] = 0.0;
        node->hi[k] = 0.0;
    }
    node->count = 0;
    node->level = level;
    node->points = NULL;
    node->is_leaf = 1;
    for (int i = 0; i < MAX_CHILDREN; i++) node->children[i] = NULL;
    return node;
}

void subdivide(TreeNode *node) {
    int nchild = 1 << g_dim;
    int child_level = node->level + 1;
    for (int idx = 0; idx < nchild; idx++) {
        double child_lo[MAX_DIM], child_hi[MAX_DIM];
        for (int k = 0; k < MAX_DIM; k++) { child_lo[k] = 0.0; child_hi[k] = 0.0; }
        nD_set_child_box(node, idx, child_lo, child_hi);
        node->children[idx] = create_node(child_lo, child_hi, child_level);
    }
    node->is_leaf = 0;
}

void build_tree_to_depth(TreeNode *node, int target_depth) {
    if (node->level >= target_depth) return;
    subdivide(node);
    int nchild = 1 << g_dim;
    for (int i = 0; i < nchild; i++) {
        if (node->children[i]) build_tree_to_depth(node->children[i], target_depth);
    }
}

void insert_point(TreeNode *node, Point p) {
    node->count++;
    if (node->is_leaf) {
        node->points = (Point*)realloc(node->points, node->count * sizeof(Point));
        node->points[node->count - 1] = p;
    } else {
        int idx = nD_child_index_for_point(node, p);
        insert_point(node->children[idx], p);
    }
}

// ============================================================================
// TREE ANALYSIS
// ============================================================================

int get_max_depth(TreeNode *node) {
    if (!node || node->is_leaf) return node ? node->level : 0;
    int max_child = 0;
    int nchild = 1 << g_dim;
    for (int i = 0; i < nchild; i++) {
        if (node->children[i]) {
            int d = get_max_depth(node->children[i]);
            if (d > max_child) max_child = d;
        }
    }
    return max_child;
}

void get_tree_stats(TreeNode *node, int *total_nodes, int *leaf_nodes,
                    int *max_leaf_size, int *min_leaf_size) {
    if (!node) return;
    (*total_nodes)++;
    if (node->is_leaf) {
        (*leaf_nodes)++;
        if (node->count > *max_leaf_size) *max_leaf_size = node->count;
        if (*min_leaf_size < 0 || node->count < *min_leaf_size) *min_leaf_size = node->count;
    } else {
        int nchild = 1 << g_dim;
        for (int i = 0; i < nchild; i++)
            get_tree_stats(node->children[i], total_nodes, leaf_nodes,
                           max_leaf_size, min_leaf_size);
    }
}

// ============================================================================
// COLLECT LEAF NODES
// ============================================================================

void collect_leaf_nodes(TreeNode *node, TreeNode ***leaves, int *count, int *capacity) {
    if (!node) return;
    if (node->is_leaf) {
        if (*count >= *capacity) {
            *capacity *= 2;
            *leaves = (TreeNode**)realloc(*leaves, (*capacity) * sizeof(TreeNode*));
        }
        (*leaves)[*count] = node;
        (*count)++;
    } else {
        int nchild = 1 << g_dim;
        for (int i = 0; i < nchild; i++) {
            if (node->children[i]) collect_leaf_nodes(node->children[i], leaves, count, capacity);
        }
    }
}

// ============================================================================
// COMPACT LEAF POINT ARRAYS INTO ONE CONTIGUOUS BUFFER
// ============================================================================
//
// Same idea as the 2D file: sort leaves by their D-dimensional cell
// coordinate (lex order over axes 0..D-1, with axis 0 most-significant
// at D=2 to match the 2D file's row-major-by-i ordering).

typedef struct {
    TreeNode *leaf;
    int cell[MAX_DIM];
} LeafOrdered;

static int cmp_leaf_ordered(const void *a, const void *b) {
    const LeafOrdered *la = (const LeafOrdered *)a;
    const LeafOrdered *lb = (const LeafOrdered *)b;
    for (int k = 0; k < g_dim; k++) {
        if (la->cell[k] != lb->cell[k]) return (la->cell[k] < lb->cell[k]) ? -1 : 1;
    }
    return 0;
}

Point* compact_leaf_points(TreeNode *root, int n_points, int tree_depth) {
    int capacity = 1024;
    int leaf_count = 0;
    TreeNode **leaves = (TreeNode**)malloc(capacity * sizeof(TreeNode*));
    collect_leaf_nodes(root, &leaves, &leaf_count, &capacity);

    int m = 1 << tree_depth;
    double cell_size = 1.0 / m;
    LeafOrdered *ordered = (LeafOrdered*)malloc(leaf_count * sizeof(LeafOrdered));
    for (int kk = 0; kk < leaf_count; kk++) {
        ordered[kk].leaf = leaves[kk];
        for (int k = 0; k < g_dim; k++)
            ordered[kk].cell[k] = (int)(leaves[kk]->lo[k] / cell_size);
        for (int k = g_dim; k < MAX_DIM; k++) ordered[kk].cell[k] = 0;
    }
    qsort(ordered, leaf_count, sizeof(LeafOrdered), cmp_leaf_ordered);
    free(leaves);

    Point *big_buffer = (Point*)malloc((size_t)n_points * sizeof(Point));
    if (!big_buffer) {
        fprintf(stderr, "Error: failed to allocate compacted point buffer (%lld bytes)\n",
                (long long)((size_t)n_points * sizeof(Point)));
        free(ordered);
        exit(1);
    }
    size_t cursor = 0;
    for (int kk = 0; kk < leaf_count; kk++) {
        TreeNode *leaf = ordered[kk].leaf;
        if (leaf->count > 0 && leaf->points) {
            memcpy(&big_buffer[cursor], leaf->points,
                   (size_t)leaf->count * sizeof(Point));
            free(leaf->points);
            leaf->points = &big_buffer[cursor];
            cursor += leaf->count;
        } else {
            if (leaf->points) free(leaf->points);
            leaf->points = NULL;
        }
    }
    free(ordered);
    g_leaves_compacted = 1;
    return big_buffer;
}


// ============================================================================
// LEAF GRID
// ============================================================================
//
// In nD this is a flat m^D array of counts plus a flat m^D array of node*
// pointers, indexed by:
//     idx = i_0 * m^(D-1) + i_1 * m^(D-2) + ... + i_{D-1}
// We also keep a list of NON-EMPTY cells (with the same compact bbox/count
// info as the 2D CellCoord) and row-start pointers indexed by i_0 for the
// row-merge fast path used at displacements that straddle bin boundaries.
//
// "Row" here means: cells sharing the same i_0 value (the first axis).
// At D=2 i_0=i, so this matches the 2D file's row notion exactly.
// At D>=3 it's a (D-1)-dimensional slab of cells.

typedef struct {
    int cell[MAX_DIM];           // cell coordinates along each axis
    float blo[MAX_DIM];          // bounding box low corner (float OK; only feeds binning)
    float bhi[MAX_DIM];          // bounding box high corner
    int   count;
} CellCoord;

typedef struct {
    int *grid_flat;              // flat m^D array of counts
    TreeNode **node_map_flat;    // flat m^D array of TreeNode*
    CellCoord *non_empty_cells;  // list of non-empty cells (sorted lex by cell[])
    int num_non_empty;
    int *row_start;              // row_start[i0]..row_start[i0+1] = slice of
                                 //   non_empty_cells with cell[0] == i0. Length m+1.
    int m;
    long long stride;            // m^D, total cells
    double cell_size;
} LeafGrid;

// Convert a D-dim cell coord to a flat index.  cell[0] is most significant.
static inline long long grid_index_from_cell(const int *cell, int m, int D) {
    long long idx = 0;
    for (int k = 0; k < D; k++) idx = idx * m + cell[k];
    return idx;
}

LeafGrid* build_leaf_grid(TreeNode *root, int tree_depth) {
    int m = 1 << tree_depth;
    long long stride = 1;
    for (int k = 0; k < g_dim; k++) stride *= m;

    LeafGrid *lg = (LeafGrid*)malloc(sizeof(LeafGrid));
    lg->m = m;
    lg->stride = stride;
    lg->cell_size = 1.0 / m;
    lg->num_non_empty = 0;

    lg->grid_flat     = (int*)     calloc((size_t)stride, sizeof(int));
    lg->node_map_flat = (TreeNode**)calloc((size_t)stride, sizeof(TreeNode*));
    if (!lg->grid_flat || !lg->node_map_flat) {
        fprintf(stderr, "Error: failed to allocate flat leaf grid (m=%d, D=%d, stride=%lld)\n",
                m, g_dim, stride);
        exit(1);
    }

    // Collect leaves.
    int capacity = 1024;
    int leaf_count = 0;
    TreeNode **leaves = (TreeNode**)malloc(capacity * sizeof(TreeNode*));
    collect_leaf_nodes(root, &leaves, &leaf_count, &capacity);

    int *row_count = (int*)calloc(m, sizeof(int));

    for (int kk = 0; kk < leaf_count; kk++) {
        TreeNode *leaf = leaves[kk];
        int cell[MAX_DIM] = {0};
        int in_bounds = 1;
        for (int k = 0; k < g_dim; k++) {
            int ci = (int)(leaf->lo[k] / lg->cell_size);
            if (ci < 0 || ci >= m) { in_bounds = 0; break; }
            cell[k] = ci;
        }
        if (!in_bounds) continue;
        long long idx = grid_index_from_cell(cell, m, g_dim);
        lg->grid_flat[idx]     = leaf->count;
        lg->node_map_flat[idx] = leaf;
        if (leaf->count > 0) {
            row_count[cell[0]]++;
            lg->num_non_empty++;
        }
    }
    free(leaves);

    lg->non_empty_cells = (CellCoord*)malloc(lg->num_non_empty * sizeof(CellCoord));
    lg->row_start = (int*)malloc((m + 1) * sizeof(int));
    {
        int cum = 0;
        for (int i = 0; i < m; i++) { lg->row_start[i] = cum; cum += row_count[i]; }
        lg->row_start[m] = cum;
    }

    // Counting-sort: walk all m^D cells in lex order (cell[0] major, cell[D-1] minor)
    // and place each non-empty cell into its row bucket.  Within a row the
    // walk visits cells in lex order over axes 1..D-1, which becomes the
    // tie-break in non_empty_cells.
    {
        int *cursor = (int*)malloc(m * sizeof(int));
        memcpy(cursor, lg->row_start, m * sizeof(int));
        int cell[MAX_DIM] = {0};
        // Iterate over all m^D cells in lex order.
        for (long long flat = 0; flat < stride; flat++) {
            // Decode cell from flat index (lex order: cell[0] major).
            long long t = flat;
            for (int k = g_dim - 1; k >= 0; k--) {
                cell[k] = (int)(t % m);
                t /= m;
            }
            int cnt = lg->grid_flat[flat];
            if (cnt > 0) {
                int idx = cursor[cell[0]]++;
                CellCoord *c = &lg->non_empty_cells[idx];
                TreeNode *leaf = lg->node_map_flat[flat];
                for (int k = 0; k < g_dim; k++) {
                    c->cell[k] = cell[k];
                    c->blo[k] = (float)leaf->lo[k];
                    c->bhi[k] = (float)leaf->hi[k];
                }
                for (int k = g_dim; k < MAX_DIM; k++) {
                    c->cell[k] = 0;
                    c->blo[k] = 0.0f;
                    c->bhi[k] = 0.0f;
                }
                c->count = cnt;
            }
        }
        free(cursor);
    }
    free(row_count);
    return lg;
}

void free_leaf_grid(LeafGrid *lg) {
    free(lg->grid_flat);
    free(lg->node_map_flat);
    free(lg->non_empty_cells);
    free(lg->row_start);
    free(lg);
}


// ============================================================================
// DISTANCE UTILITIES
// ============================================================================
//
// Displacement-level distance range: given a D-dim integer cell displacement
// disp[] = (dx_0, dx_1, ..., dx_{D-1}) between two cells of side `cell_size`,
// returns [d_min, d_max], the extremes of |p1 - p2| where p1, p2 live in the
// two cells (axis-aligned boxes).  Min along axis k is (|disp[k]| - 1) when
// nonzero (cells touch), else 0; max along axis k is |disp[k]| + 1.

static inline void get_displacement_distance_range_nD(const int *disp,
                                                      double cell_size,
                                                      double *d_min,
                                                      double *d_max) {
    g_displacement_distance_calls++;
    double sum_min = 0.0, sum_max = 0.0;
    for (int k = 0; k < g_dim; k++) {
        int a = disp[k]; if (a < 0) a = -a;
        int lo = (a > 0) ? (a - 1) : 0;
        int hi = a + 1;
        sum_min += (double)lo * lo;
        sum_max += (double)hi * hi;
    }
    *d_min = cell_size * sqrt(sum_min);
    *d_max = cell_size * sqrt(sum_max);
}

// Bounding-box distance range between two arbitrary nD axis-aligned boxes.
// Min along axis k is the gap (0 if boxes overlap); max is the far-corner span.
static inline void get_bbox_distance_range_nD(const double *lo1, const double *hi1,
                                              const double *lo2, const double *hi2,
                                              double *d_min, double *d_max) {
    double sum_min = 0.0, sum_max = 0.0;
    for (int k = 0; k < g_dim; k++) {
        double gap = lo1[k] - hi2[k];
        double gap2 = lo2[k] - hi1[k];
        if (gap2 > gap) gap = gap2;
        if (gap < 0.0) gap = 0.0;
        double span = hi1[k] - lo2[k];
        double span2 = hi2[k] - lo1[k];
        if (span2 > span) span = span2;
        sum_min += gap * gap;
        sum_max += span * span;
    }
    *d_min = sqrt(sum_min);
    *d_max = sqrt(sum_max);
}

// Same but for the float bbox stored in CellCoord (the approximate-mode path).
static inline void get_cellcoord_distance_range(const CellCoord *c1, const CellCoord *c2,
                                                double *d_min, double *d_max) {
    double sum_min = 0.0, sum_max = 0.0;
    for (int k = 0; k < g_dim; k++) {
        double gap1 = (double)c1->blo[k] - (double)c2->bhi[k];
        double gap2 = (double)c2->blo[k] - (double)c1->bhi[k];
        double gap = (gap1 > gap2) ? gap1 : gap2;
        if (gap < 0.0) gap = 0.0;
        double span1 = (double)c1->bhi[k] - (double)c2->blo[k];
        double span2 = (double)c2->bhi[k] - (double)c1->blo[k];
        double span = (span1 > span2) ? span1 : span2;
        sum_min += gap * gap;
        sum_max += span * span;
    }
    *d_min = sqrt(sum_min);
    *d_max = sqrt(sum_max);
}

// Exact Euclidean distance between two points in g_dim dimensions.
static inline double point_distance(const Point *a, const Point *b) {
    double s = 0.0;
    for (int k = 0; k < g_dim; k++) {
        double d = a->x[k] - b->x[k];
        s += d * d;
    }
    return sqrt(s);
}

// ============================================================================
// BIN SPEC (verbatim from rdf_MP-29.c; pure-1D, no dimension dependence)
// ============================================================================

typedef struct {
    const double *edges;
    int num_bins;
    int is_uniform;
    double uniform_width;
    double width_min;
    double width_max;
} BinSpec;

static void binspec_init(BinSpec *bs, const double *edges, int num_bins) {
    bs->edges = edges;
    bs->num_bins = num_bins;
    bs->width_min = edges[1] - edges[0];
    bs->width_max = bs->width_min;
    int uniform = 1;
    for (int i = 0; i < num_bins; i++) {
        double w = edges[i + 1] - edges[i];
        if (w < bs->width_min) bs->width_min = w;
        if (w > bs->width_max) bs->width_max = w;
        if (i > 0 && fabs(w - (edges[1] - edges[0])) > 1e-12 * w) uniform = 0;
    }
    bs->is_uniform = uniform;
    bs->uniform_width = uniform ? (edges[1] - edges[0]) : 0.0;
}

static inline int binspec_bin(const BinSpec *bs, double d) {
    int b;
    if (bs->is_uniform) {
        b = (int)(d / bs->uniform_width);
        if (b < 0) b = 0;
        if (b >= bs->num_bins) b = bs->num_bins - 1;
    } else {
        if (d <= bs->edges[0]) return 0;
        if (d >= bs->edges[bs->num_bins]) return bs->num_bins - 1;
        int lo = 0, hi = bs->num_bins;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (bs->edges[mid] <= d) lo = mid; else hi = mid;
        }
        b = lo;
    }
    return b;
}

// Starting tree level for the dual-tree pyramid.  Same logic as the 2D file,
// generalized to nD: at level L the cell size is 1/2^L and the cell diagonal
// is sqrt(D)/2^L.  We want diagonal <= largest bin width so a (0,0,...) cell
// pair fits in one bin.  Solving: L >= log2(sqrt(D)/width_max).
int compute_starting_level(double bin_width, double domain_size) {
    double diag_factor = sqrt((double)g_dim);
    double ratio = diag_factor * domain_size / bin_width;
    int k = (int)ceil(log2(ratio));
    if (k < 0) k = 0;
    return k;
}

static int compute_starting_level_bs(const BinSpec *bs, double domain_size) {
    return compute_starting_level(bs->width_max, domain_size);
}


// ============================================================================
// APPROXIMATE-MODE HEURISTIC DISTRIBUTION
// ============================================================================

void distribute_pairs_heuristic_disp(long long n_pairs,
                                     double d_min, double d_max,
                                     int bmin, int bmax,
                                     long long *histogram, const BinSpec *bs,
                                     Statistics *stats) {
    if (n_pairs == 0) return;
    if (bmin == bmax) {
        histogram[bmin] += n_pairs;
        stats->pruned_pairs += n_pairs;
        stats->coarse_resolved_pairs += n_pairs;
        return;
    }
    double span = d_max - d_min;
    long long assigned = 0;
    int best_bin = bmin;
    double best_overlap = -1.0;
    for (int b = bmin; b <= bmax; b++) {
        double bin_lo = bs->edges[b];
        double bin_hi = bs->edges[b + 1];
        double ov_lo = fmax(bin_lo, d_min);
        double ov_hi = fmin(bin_hi, d_max);
        double overlap = ov_hi - ov_lo;
        if (overlap < 0.0) overlap = 0.0;
        double frac = (span > 0.0) ? (overlap / span) : 0.0;
        long long share = (long long)(frac * (double)n_pairs);
        histogram[b] += share;
        assigned += share;
        if (overlap > best_overlap) { best_overlap = overlap; best_bin = b; }
    }
    long long remainder = n_pairs - assigned;
    if (remainder != 0) histogram[best_bin] += remainder;
    stats->pruned_pairs += n_pairs;
    stats->coarse_resolved_pairs += n_pairs;
}

void distribute_pairs_heuristic_cells(const CellCoord *c1, const CellCoord *c2,
                                      long long *histogram, const BinSpec *bs,
                                      int same_cell, Statistics *stats) {
    if (!c1 || !c2 || c1->count == 0 || c2->count == 0) return;
    stats->leaf_comparisons++;
    long long n_pairs = same_cell
        ? (long long)c1->count * (c1->count - 1) / 2
        : (long long)c1->count * (long long)c2->count;
    if (n_pairs == 0) return;
    double d_min, d_max;
    get_cellcoord_distance_range(c1, c2, &d_min, &d_max);
    int bmin = binspec_bin(bs, d_min);
    int bmax = binspec_bin(bs, d_max);
    if (bmin == bmax) {
        histogram[bmin] += n_pairs;
        stats->pruned_pairs += n_pairs;
        return;
    }
    double span = d_max - d_min;
    long long assigned = 0;
    int best_bin = bmin;
    double best_overlap = -1.0;
    for (int b = bmin; b <= bmax; b++) {
        double bin_lo = bs->edges[b];
        double bin_hi = bs->edges[b + 1];
        double ov_lo = fmax(bin_lo, d_min);
        double ov_hi = fmin(bin_hi, d_max);
        double overlap = ov_hi - ov_lo;
        if (overlap < 0.0) overlap = 0.0;
        double frac = (span > 0.0) ? (overlap / span) : 0.0;
        long long share = (long long)(frac * (double)n_pairs);
        histogram[b] += share;
        assigned += share;
        if (overlap > best_overlap) { best_overlap = overlap; best_bin = b; }
    }
    long long remainder = n_pairs - assigned;
    if (remainder != 0) histogram[best_bin] += remainder;
    stats->pruned_pairs += n_pairs;
}

// ============================================================================
// EXACT LEAF-PAIR DISTANCE COMPUTATION
// ============================================================================

void compute_exact_distances_between_cells(TreeNode *cell1, TreeNode *cell2,
                                           long long *histogram, const BinSpec *bs,
                                           int same_cell, Statistics *stats) {
    if (!cell1 || !cell2 || cell1->count == 0 || cell2->count == 0) return;
    stats->leaf_comparisons++;
    for (int i = 0; i < cell1->count; i++) {
        int j_start = same_cell ? i + 1 : 0;
        for (int j = j_start; j < cell2->count; j++) {
            double dist = point_distance(&cell1->points[i], &cell2->points[j]);
            int bin = binspec_bin(bs, dist);
            histogram[bin]++;
            STAT_INNER_INC(stats->exact_distance_computations);
        }
    }
}

static inline void process_cell_pair(TreeNode *q1, TreeNode *q2,
                                     const CellCoord *cc1, const CellCoord *cc2,
                                     long long *histogram, const BinSpec *bs,
                                     int same_cell, Statistics *stats) {
    if (g_approx_mode) {
        distribute_pairs_heuristic_cells(cc1, cc2, histogram, bs, same_cell, stats);
    } else {
        compute_exact_distances_between_cells(q1, q2, histogram, bs, same_cell, stats);
    }
}


// ============================================================================
// DUAL-TREE METHOD
// ============================================================================

void collect_nodes_at_level(TreeNode *node, int target_level, TreeNode ***nodes,
                            int *count, int *capacity) {
    if (!node || node->count == 0) return;
    if (node->level == target_level) {
        if (*count >= *capacity) {
            *capacity *= 2;
            *nodes = (TreeNode**)realloc(*nodes, (*capacity) * sizeof(TreeNode*));
        }
        (*nodes)[*count] = node;
        (*count)++;
    } else if (node->level < target_level && !node->is_leaf) {
        int nchild = 1 << g_dim;
        for (int i = 0; i < nchild; i++) {
            if (node->children[i])
                collect_nodes_at_level(node->children[i], target_level, nodes, count, capacity);
        }
    }
}

void dual_tree_histogram(TreeNode *n1, TreeNode *n2, long long *histogram,
                         const BinSpec *bs, int same_node, Statistics *stats) {
    if (n1->count == 0 || n2->count == 0) return;

    if (n1->level != n2->level) {
        fprintf(stderr, "Error: dual_tree_histogram called with unequal levels "
                "(%d vs %d). Uniform-depth assumption violated.\n",
                n1->level, n2->level);
        exit(1);
    }

    // Compute integer cell displacement between cell centers along each axis.
    double cell_size = n1->hi[0] - n1->lo[0];
    int disp[MAX_DIM] = {0};
    for (int k = 0; k < g_dim; k++) {
        double c1 = 0.5 * (n1->lo[k] + n1->hi[k]);
        double c2 = 0.5 * (n2->lo[k] + n2->hi[k]);
        disp[k] = (int)floor((c2 - c1) / cell_size + 0.5);
    }

    double d_min, d_max;
    get_displacement_distance_range_nD(disp, cell_size, &d_min, &d_max);
    int bin_min = binspec_bin(bs, d_min);
    int bin_max = binspec_bin(bs, d_max);

    if (bin_min == bin_max) {
        long long pairs = same_node
            ? ((long long)n1->count * (long long)(n1->count - 1)) / 2
            : (long long)n1->count * (long long)n2->count;
        histogram[bin_min] += pairs;
        stats->pruned_pairs += pairs;
        stats->coarse_resolved_pairs += pairs;
        stats->prune_operations++;
        return;
    }

    if (n1->is_leaf && n2->is_leaf) {
        if (g_approx_mode) {
            long long n_pairs = same_node
                ? (long long)n1->count * (n1->count - 1) / 2
                : (long long)n1->count * (long long)n2->count;
            distribute_pairs_heuristic_disp(n_pairs, d_min, d_max,
                                            bin_min, bin_max,
                                            histogram, bs, stats);
            return;
        }
        stats->leaf_comparisons++;
        for (int i = 0; i < n1->count; i++) {
            int j_start = same_node ? i + 1 : 0;
            for (int j = j_start; j < n2->count; j++) {
                double dist = point_distance(&n1->points[i], &n2->points[j]);
                int bin = binspec_bin(bs, dist);
                histogram[bin]++;
                STAT_INNER_INC(stats->exact_distance_computations);
            }
        }
        return;
    }

    int nchild = 1 << g_dim;
    if (same_node) {
        if (!n1->is_leaf) {
            for (int i = 0; i < nchild; i++) {
                if (n1->children[i]) {
                    dual_tree_histogram(n1->children[i], n1->children[i],
                                        histogram, bs, 1, stats);
                    for (int j = i + 1; j < nchild; j++) {
                        if (n1->children[j]) {
                            dual_tree_histogram(n1->children[i], n1->children[j],
                                                histogram, bs, 0, stats);
                        }
                    }
                }
            }
        }
    } else {
        if (!n1->is_leaf && !n2->is_leaf) {
            for (int i = 0; i < nchild; i++) {
                for (int j = 0; j < nchild; j++) {
                    if (n1->children[i] && n2->children[j]) {
                        dual_tree_histogram(n1->children[i], n2->children[j],
                                            histogram, bs, 0, stats);
                    }
                }
            }
        } else if (!n1->is_leaf || !n2->is_leaf) {
            fprintf(stderr, "Error: dual_tree_histogram reached asymmetric "
                    "recursion. Tree is not uniform-depth?\n");
            exit(1);
        }
    }
}

// ===========================================================================
// TASK-BASED PARALLEL DUAL-TREE
// Same computation as dual_tree_histogram, but (a) accumulates into a
// per-thread histogram (indexed by omp_get_thread_num(), so no races and no
// atomics on the hot path) and (b) spawns OpenMP tasks for child node-pairs
// when a node is large enough, so a single dominant recursion is split across
// threads. Merged into the caller's histogram afterwards.
// ===========================================================================
static long long   **g_thread_hist  = NULL;
static Statistics  **g_thread_stats = NULL;   // array of cache-line-aligned ptrs
static const BinSpec *g_bs_task     = NULL;
static long long     g_dt_cutoff    = 512;   // spawn tasks only when count > this

void dual_tree_task(TreeNode *n1, TreeNode *n2, int same_node) {
    if (n1->count == 0 || n2->count == 0) return;

    double cell_size = n1->hi[0] - n1->lo[0];
    int disp[MAX_DIM] = {0};
    for (int k = 0; k < g_dim; k++) {
        double c1 = 0.5 * (n1->lo[k] + n1->hi[k]);
        double c2 = 0.5 * (n2->lo[k] + n2->hi[k]);
        disp[k] = (int)floor((c2 - c1) / cell_size + 0.5);
    }
    double d_min, d_max;
    get_displacement_distance_range_nD(disp, cell_size, &d_min, &d_max);
    int bin_min = binspec_bin(g_bs_task, d_min);
    int bin_max = binspec_bin(g_bs_task, d_max);

    int tid = omp_get_thread_num();
    long long *hist  = g_thread_hist[tid];
    Statistics *stats = g_thread_stats[tid];

    if (bin_min == bin_max) {
        long long pairs = same_node
            ? ((long long)n1->count * (long long)(n1->count - 1)) / 2
            : (long long)n1->count * (long long)n2->count;
        hist[bin_min] += pairs;
        stats->pruned_pairs += pairs;
        stats->coarse_resolved_pairs += pairs;
        stats->prune_operations++;
        return;
    }
    if (n1->is_leaf && n2->is_leaf) {
        if (g_approx_mode) {
            long long n_pairs = same_node
                ? (long long)n1->count * (n1->count - 1) / 2
                : (long long)n1->count * (long long)n2->count;
            distribute_pairs_heuristic_disp(n_pairs, d_min, d_max, bin_min, bin_max,
                                            hist, g_bs_task, stats);
            return;
        }
        stats->leaf_comparisons++;
        for (int i = 0; i < n1->count; i++) {
            int j_start = same_node ? i + 1 : 0;
            for (int j = j_start; j < n2->count; j++) {
                double dist = point_distance(&n1->points[i], &n2->points[j]);
                hist[binspec_bin(g_bs_task, dist)]++;
                STAT_INNER_INC(stats->exact_distance_computations);
            }
        }
        return;
    }

    int nchild = 1 << g_dim;
    int spawn = (n1->count > g_dt_cutoff);   // only big subtrees become tasks
    if (same_node) {
        if (!n1->is_leaf) {
            for (int i = 0; i < nchild; i++) {
                if (!n1->children[i]) continue;
                TreeNode *ci = n1->children[i];
                if (spawn) {
                    #pragma omp task firstprivate(ci)
                    dual_tree_task(ci, ci, 1);
                } else dual_tree_task(ci, ci, 1);
                for (int j = i + 1; j < nchild; j++) {
                    if (!n1->children[j]) continue;
                    TreeNode *cj = n1->children[j];
                    if (spawn) {
                        #pragma omp task firstprivate(ci, cj)
                        dual_tree_task(ci, cj, 0);
                    } else dual_tree_task(ci, cj, 0);
                }
            }
        }
    } else {
        if (!n1->is_leaf && !n2->is_leaf) {
            for (int i = 0; i < nchild; i++) {
                for (int j = 0; j < nchild; j++) {
                    if (!n1->children[i] || !n2->children[j]) continue;
                    TreeNode *ci = n1->children[i], *cj = n2->children[j];
                    if (spawn) {
                        #pragma omp task firstprivate(ci, cj)
                        dual_tree_task(ci, cj, 0);
                    } else dual_tree_task(ci, cj, 0);
                }
            }
        }
    }
}

typedef struct {
    int i, j;
    int is_self;
} DTTask;

void compute_histogram_dualtree(TreeNode *root, const BinSpec *bs,
                                long long *histogram, Statistics *stats) {
    int num_bins = bs->num_bins;
    int starting_level = compute_starting_level_bs(bs, 1.0);
    int max_depth = get_max_depth(root);
    if (starting_level > max_depth) starting_level = max_depth;
    printf("   Starting level: %d\n", starting_level);

    // 2^(D*L) nodes at level L.
    long long approx_nodes_at_level = 1;
    for (int i = 0; i < starting_level * g_dim; i++) {
        approx_nodes_at_level *= 2;
        if (approx_nodes_at_level > 1000000000LL) break;
    }
    if (approx_nodes_at_level > 100000) {
        printf("   Warning: starting level has ~%lld potential nodes.\n",
               approx_nodes_at_level);
    }

    int capacity = 256;
    int node_count = 0;
    TreeNode **start_nodes = (TreeNode**)malloc(capacity * sizeof(TreeNode*));
    collect_nodes_at_level(root, starting_level, &start_nodes, &node_count, &capacity);
    printf("   Actual nodes at starting level: %d\n", node_count);

    /* Stream (i, j) pairs directly during the parallel loop — no tasks[] array.
       Frees the 16 B/pair the array used (up to 1.6 TB at 100 B pairs), so the
       hard memory bottleneck is now the tree itself (tens of GB), not pair
       bookkeeping. Cap remains as a wall-time sanity check. */
    long long max_tasks = ((long long)node_count * (node_count + 1)) / 2;
    printf("   Total tasks: %lld (streamed)\n", max_tasks);
    if (max_tasks > 10000000000000LL) { /* 10 T wall-time cap (~115 days @ 1 M/s) */
        fprintf(stderr,
                "Error: too many tasks (%lld) — would take >100 days. Use the 'autocorr' method.\n",
                max_tasks);
        free(start_nodes);
        exit(1);
    }

    /* TASK-BASED parallel dual-tree. Each top-level (i<=j) start-node pair is a
       task; large node-pairs recursively spawn child tasks (cutoff g_dt_cutoff),
       so one dominant recursion is split across threads. Per-thread histograms
       avoid races; merged afterwards. Same computation/result as the serial
       dual_tree_histogram. */
    int nthreads = omp_get_max_threads();
    g_bs_task = bs;
    g_thread_hist  = (long long**)malloc(nthreads * sizeof(long long*));
    g_thread_stats = (Statistics**)malloc(nthreads * sizeof(Statistics*));
    // cache-line-aligned + padded per-thread buffers -> no false sharing across
    // sockets (each thread's hist and stats own their cache line(s)).
    size_t hist_bytes = (((size_t)num_bins * sizeof(long long) + 63) / 64) * 64;
    for (int t = 0; t < nthreads; t++) {
        g_thread_hist[t]  = (long long*)aligned_alloc(64, hist_bytes);
        memset(g_thread_hist[t], 0, hist_bytes);
        g_thread_stats[t] = (Statistics*)aligned_alloc(64, 64);  // 64 >= sizeof(Statistics)
        memset(g_thread_stats[t], 0, 64);
    }

    const char *cutoff_env = getenv("DT_CUTOFF");
    if (cutoff_env) g_dt_cutoff = atoll(cutoff_env);   // tunable for big machines

    long long total_pairs = ((long long)node_count * (node_count + 1)) / 2;
    double disc = (double)(2 * (long long)node_count + 1);
    /* Top-level (i<=j) start-node pairs are distributed by a worksharing for
       (NOT one task per pair -- that serialized the spawn for narrow-w where
       node_count is large). Each pair calls dual_tree_task, which spawns child
       tasks INTERNALLY for big sub-pairs, so dominant recursions are split too.
       Explicit tasks complete at the parallel region's implicit barrier. */
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 8) nowait
        for (long long p = 0; p < total_pairs; p++) {
            long long i = (long long)((disc - sqrt(disc * disc - 8.0 * (double)p)) / 2.0);
            long long Ti = i * (long long)node_count - i * (i - 1) / 2;
            while (Ti > p) { i--; Ti = i * (long long)node_count - i * (i - 1) / 2; }
            for (;;) {
                long long Ti1 = (i + 1) * (long long)node_count - (i + 1) * i / 2;
                if (i + 1 < node_count && Ti1 <= p) { i++; Ti = Ti1; } else break;
            }
            long long j = i + (p - Ti);
            dual_tree_task(start_nodes[i], start_nodes[j], (i == j));
        }
    }

    for (int t = 0; t < nthreads; t++) {
        for (int b = 0; b < num_bins; b++) histogram[b] += g_thread_hist[t][b];
        stats->exact_distance_computations += g_thread_stats[t]->exact_distance_computations;
        stats->pruned_pairs                += g_thread_stats[t]->pruned_pairs;
        stats->leaf_comparisons            += g_thread_stats[t]->leaf_comparisons;
        stats->prune_operations            += g_thread_stats[t]->prune_operations;
        stats->displacement_distance_calls += g_thread_stats[t]->displacement_distance_calls;
        stats->coarse_resolved_pairs       += g_thread_stats[t]->coarse_resolved_pairs;
        free(g_thread_hist[t]); free(g_thread_stats[t]);
    }
    free(g_thread_hist); free(g_thread_stats);
    g_thread_hist = NULL; g_thread_stats = NULL;

    free(start_nodes);
}


// ============================================================================
// FREE TREE
// ============================================================================

void free_tree(TreeNode *node) {
    if (!node) return;
    int nchild = 1 << g_dim;
    if (!node->is_leaf) {
        for (int i = 0; i < nchild; i++) free_tree(node->children[i]);
    }
    // If leaves were compacted, leaf->points points into the single compacted
    // buffer (freed by main) and must NOT be freed here.  If compaction was
    // skipped (approx mode), each leaf owns its own malloc'd array and we free
    // it here.
    if (!g_leaves_compacted && node->is_leaf && node->points) {
        free(node->points);
        node->points = NULL;
    }
    free(node);
}

// ============================================================================
// BIN EDGE HELPERS (verbatim from 2D)
// ============================================================================

static double* make_uniform_edges(double width, int num_bins) {
    double *edges = (double*)malloc((num_bins + 1) * sizeof(double));
    for (int i = 0; i <= num_bins; i++) edges[i] = i * width;
    return edges;
}

static double* make_log_edges(double d_min, double d_max, int num_bins) {
    if (d_min <= 0 || d_max <= d_min || num_bins < 1) {
        fprintf(stderr, "Error: log bins require d_min>0, d_max>d_min, num_bins>=1\n");
        exit(1);
    }
    double *edges = (double*)malloc((num_bins + 1) * sizeof(double));
    double ratio = d_max / d_min;
    for (int i = 0; i <= num_bins; i++) {
        double t = (double)i / num_bins;
        edges[i] = d_min * pow(ratio, t);
    }
    edges[0] = 0.0;
    return edges;
}

static double* load_edges_file(const char *path, int *num_bins_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Error: cannot open bin-edges file: %s\n", path); exit(1); }
    int cap = 64, n = 0;
    double *edges = (double*)malloc(cap * sizeof(double));
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n' || *p == '#') continue;
        double v = atof(p);
        if (n == cap) { cap *= 2; edges = (double*)realloc(edges, cap * sizeof(double)); }
        edges[n++] = v;
    }
    fclose(f);
    if (n < 2) { fprintf(stderr, "Error: bin-edges file needs >= 2 edges (got %d)\n", n); exit(1); }
    for (int i = 1; i < n; i++) {
        if (edges[i] <= edges[i - 1]) {
            fprintf(stderr, "Error: bin edges must be strictly ascending\n"); exit(1);
        }
    }
    *num_bins_out = n - 1;
    return edges;
}

static void print_edges_summary(const BinSpec *bs) {
    int n = bs->num_bins;
    printf("Bins:                    %d (%s, width_min=%.6f, width_max=%.6f)\n",
           n, bs->is_uniform ? "uniform" : "variable",
           bs->width_min, bs->width_max);
    if (n <= 10) {
        printf("Edges:                   ");
        for (int i = 0; i <= n; i++) printf("%.4f%s", bs->edges[i], i == n ? "" : ", ");
        printf("\n");
    } else {
        printf("Edges (first 5):         %.4f, %.4f, %.4f, %.4f, %.4f, ...\n",
               bs->edges[0], bs->edges[1], bs->edges[2], bs->edges[3], bs->edges[4]);
        printf("Edges (last 3):          ..., %.4f, %.4f, %.4f\n",
               bs->edges[n - 2], bs->edges[n - 1], bs->edges[n]);
    }
}


// Forward declarations of methods 1 and 4 (defined later in this file).
void compute_histogram_autocorrelation_nD(TreeNode *root, int tree_depth,
                                          long long *histogram, const BinSpec *bs,
                                          int num_threads, Statistics *stats);
void compute_histogram_autocorrelation_ring_nD(TreeNode *root, int tree_depth,
                                               long long *histogram, const BinSpec *bs,
                                               int num_threads, int n_points,
                                               Statistics *stats);

// ============================================================================
// USAGE / MAIN
// ============================================================================

void print_usage(const char *program_name) {
    printf("Usage: %s <num_points> <bin_width> [num_threads] [avg_points_per_leaf] [method] [approx] [options]\n",
           program_name);
    printf("\nPositional arguments:\n");
    printf("  num_points           : number of random points (integer > 0)\n");
    printf("  bin_width            : histogram bin width (float > 0)\n");
    printf("  num_threads          : OpenMP threads (default: max)\n");
    printf("  avg_points_per_leaf  : target average points per leaf (default: 10)\n");
    printf("  method               : dualtree | autocorr | ring  (default: autocorr)\n");
    printf("  approx               : 0=exact, 1=approximate heuristic (default: 0)\n");
    printf("\nOptions:\n");
    printf("  --dim D              : dimensionality, 1..%d (default: 2)\n", MAX_DIM);
    printf("  --bin-scheme log:N:DMIN:DMAX\n");
    printf("  --bin-edges-file PATH\n");
    printf("  --force-binsearch\n");
    printf("  --out PATH          : write final histogram (bin<TAB>count per line) to PATH\n");
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Error: invalid number of arguments.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    int n_points = atoi(argv[1]);
    double bin_width = atof(argv[2]);
    int num_threads = (argc >= 4 && argv[3][0] != '-') ? atoi(argv[3]) : omp_get_max_threads();
    double avg_points_per_leaf = (argc >= 5 && argv[4][0] != '-') ? atof(argv[4]) : 10.0;
    int method = METHOD_AUTOCORR;
    if (argc >= 6 && argv[5][0] != '-') {
        const char *ms = argv[5];
        if      (!strcmp(ms,"dualtree") || !strcmp(ms,"dual-tree"))           method = METHOD_DUALTREE;
        else if (!strcmp(ms,"autocorr") || !strcmp(ms,"autocorrelation"))     method = METHOD_AUTOCORR;
        else if (!strcmp(ms,"ring")     || !strcmp(ms,"ring-enumeration"))    method = METHOD_AUTOCORR_RING;
        else method = -1;  // invalid; rejected by validation below
    }
    int approx = (argc >= 7 && argv[6][0] != '-') ? atoi(argv[6]) : 0;

    int flag_start = argc;
    for (int k = 3; k < argc; k++) {
        if (argv[k][0] == '-') { flag_start = k; break; }
    }

    const char *bin_scheme = NULL;
    const char *bin_edges_path = NULL;
    const char *out_path = NULL;
    int force_binsearch = 0;
    int dim = 2;
    for (int k = flag_start; k < argc; k++) {
        if (strcmp(argv[k], "--bin-scheme") == 0 && k + 1 < argc) {
            bin_scheme = argv[++k];
        } else if (strcmp(argv[k], "--bin-edges-file") == 0 && k + 1 < argc) {
            bin_edges_path = argv[++k];
        } else if (strcmp(argv[k], "--out") == 0 && k + 1 < argc) {
            out_path = argv[++k];
        } else if (strcmp(argv[k], "--force-binsearch") == 0) {
            force_binsearch = 1;
        } else if (strcmp(argv[k], "--dim") == 0 && k + 1 < argc) {
            dim = atoi(argv[++k]);
        } else if (strcmp(argv[k], "--kd") == 0) {
            g_use_kd = 1;
        } else {
            fprintf(stderr, "Error: unknown option: %s\n", argv[k]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (n_points <= 0 || bin_width <= 0.0 || num_threads <= 0 || avg_points_per_leaf <= 0.0) {
        fprintf(stderr, "Error: invalid parameters.\n"); return 1;
    }
    if (approx != 0 && approx != 1) {
        fprintf(stderr, "Error: approx flag must be 0 or 1.\n"); return 1;
    }
    if (dim < 1 || dim > MAX_DIM) {
        fprintf(stderr, "Error: --dim must be in [1, %d]\n", MAX_DIM); return 1;
    }
    g_dim = dim;
    g_approx_mode = approx;

    if (method != METHOD_DUALTREE && method != METHOD_AUTOCORR &&
        method != METHOD_AUTOCORR_RING) {
        fprintf(stderr, "Error: method must be one of: dualtree, autocorr, ring.\n"); return 1;
    }

    omp_set_num_threads(num_threads);
    int tree_depth = compute_tree_depth(n_points, avg_points_per_leaf);

    // Build BinSpec.
    double *edges = NULL;
    int num_bins = 0;
    BinSpec bs;
    if (bin_edges_path) {
        edges = load_edges_file(bin_edges_path, &num_bins);
        printf("Loaded %d bin edges from %s.\n", num_bins + 1, bin_edges_path);
    } else if (bin_scheme) {
        if (strncmp(bin_scheme, "log:", 4) != 0) {
            fprintf(stderr, "Error: --bin-scheme: only 'log:N:DMIN:DMAX' supported\n"); return 1;
        }
        int nb; double dmin, dmax;
        if (sscanf(bin_scheme, "log:%d:%lf:%lf", &nb, &dmin, &dmax) != 3) {
            fprintf(stderr, "Error: --bin-scheme: bad format\n"); return 1;
        }
        edges = make_log_edges(dmin, dmax, nb);
        num_bins = nb;
    } else {
        // Max diagonal of unit hypercube is sqrt(D).
        double max_dist = sqrt((double)g_dim);
        num_bins = (int)ceil(max_dist / bin_width) + 1;
        edges = make_uniform_edges(bin_width, num_bins);
    }
    binspec_init(&bs, edges, num_bins);
    if (force_binsearch) { bs.is_uniform = 0; printf("Note: --force-binsearch enabled\n"); }

    const char *method_name = (method == METHOD_DUALTREE) ? "Dual-Tree" :
                              (method == METHOD_AUTOCORR) ? "Autocorrelation" :
                                  "Ring Enumeration (Leaf-AM Direct, Anchors)";

    printf("========================================\n");
    printf("Distance Histogram - %s Method (nD)\n", method_name);
    printf("========================================\n");
    printf("Dimensionality:          %d\n", g_dim);
    printf("Number of points:        %d\n", n_points);
    print_edges_summary(&bs);
    printf("Number of threads:       %d\n", num_threads);
    printf("Target avg per leaf:     %.2f\n", avg_points_per_leaf);
    printf("Tree depth:              %d\n", tree_depth);
    printf("Method:                  %s\n", method_name);
    printf("Mode:                    %s\n", approx ? "APPROXIMATE" : "Exact");
    printf("========================================\n\n");

    double start_total = get_wall_time();

    // Generate points (or load from file via RDF_INPUT).
    double start_phase = get_wall_time();
    Point *points = (Point*)malloc(n_points * sizeof(Point));
    if (!points) { fprintf(stderr, "Error: alloc failed\n"); return 1; }
    double time_generation;
    const char *infile = getenv("RDF_INPUT");
    if (infile) {
        FILE *fp = fopen(infile, "r");
        if (!fp) { fprintf(stderr, "Error: cannot open RDF_INPUT '%s'\n", infile); return 1; }
        int cnt = 0;
        double mn[MAX_DIM], mx[MAX_DIM];
        for (int k = 0; k < MAX_DIM; k++) { mn[k] = 1e300; mx[k] = -1e300; }
        char line[1024];
        while (cnt < n_points && fgets(line, sizeof(line), fp)) {
            double v[MAX_DIM] = {0};
            char *p = line; int ok = 1;
            for (int k = 0; k < g_dim; k++) {
                char *end; v[k] = strtod(p, &end);
                if (end == p) { ok = 0; break; }
                p = end;
            }
            if (!ok) continue;
            for (int k = 0; k < MAX_DIM; k++) points[cnt].x[k] = 0.0;
            for (int k = 0; k < g_dim; k++) {
                points[cnt].x[k] = v[k];
                if (v[k] < mn[k]) mn[k] = v[k];
                if (v[k] > mx[k]) mx[k] = v[k];
            }
            cnt++;
        }
        fclose(fp);
        if (cnt == 0) { fprintf(stderr, "Error: no points read from '%s'\n", infile); return 1; }
        double span = 0.0;
        for (int k = 0; k < g_dim; k++) { double s = mx[k] - mn[k]; if (s > span) span = s; }
        if (span <= 0) span = 1.0;
        double sc = 0.999999 / span;
        for (int i = 0; i < cnt; i++)
            for (int k = 0; k < g_dim; k++) points[i].x[k] = (points[i].x[k] - mn[k]) * sc;
        n_points = cnt;
        time_generation = get_wall_time() - start_phase;
        printf("Loaded %d points from %s (uniform-scaled)\n", cnt, infile);
        print_time("Time:", time_generation);
        printf("\n");
    } else {
        srand(42);
        for (int i = 0; i < n_points; i++) {
            for (int k = 0; k < MAX_DIM; k++) points[i].x[k] = 0.0;
            for (int k = 0; k < g_dim; k++) points[i].x[k] = (double)rand() / RAND_MAX;
        }
        time_generation = get_wall_time() - start_phase;
        printf("Generating points...\n");
        print_time("Time:", time_generation);
        printf("\n");
    }

    // Build tree.
    start_phase = get_wall_time();
    printf("Building tree...\n");
    double root_lo[MAX_DIM] = {0}, root_hi[MAX_DIM] = {0};
    for (int k = 0; k < g_dim; k++) { root_lo[k] = 0.0; root_hi[k] = 1.0; }
    TreeNode *root = create_node(root_lo, root_hi, 0);
    build_tree_to_depth(root, tree_depth);
    for (int i = 0; i < n_points; i++) insert_point(root, points[i]);
    double time_tree = get_wall_time() - start_phase;
    int total_nodes = 0, leaf_nodes = 0, max_leaf = 0, min_leaf = -1;
    get_tree_stats(root, &total_nodes, &leaf_nodes, &max_leaf, &min_leaf);
    printf("Tree built.\n");
    print_time("Time:", time_tree);
    printf("   Leaf nodes: %d\n", leaf_nodes);
    printf("   Avg points/leaf: %.1f\n\n", (double)n_points / leaf_nodes);

    // Compact.  This reorders all leaf point arrays into one contiguous
    // buffer for cache locality during EXACT per-point distance computation.
    // In approximate mode no method ever dereferences leaf->points (they work
    // purely from cell counts / autocorrelation and per-cell bounding boxes),
    // so compaction is pure overhead there and we skip it.  When skipped,
    // each leaf keeps its own malloc'd points array, freed by free_tree.
    Point *compacted_points = NULL;
    if (!g_approx_mode) {
        start_phase = get_wall_time();
        printf("Compacting leaf point arrays...\n");
        compacted_points = compact_leaf_points(root, n_points, tree_depth);
        double time_compact = get_wall_time() - start_phase;
        print_time("Time:", time_compact);
        printf("\n");
    } else {
        printf("Skipping leaf compaction (approximate mode; points not read).\n\n");
    }

    // Compute histogram.
    start_phase = get_wall_time();
    printf("Computing histogram using %s method...\n", method_name);
    long long *histogram = (long long*)calloc(num_bins, sizeof(long long));
    Statistics stats = {0};
    if (method == METHOD_DUALTREE) {
        compute_histogram_dualtree(root, &bs, histogram, &stats);
    } else if (method == METHOD_AUTOCORR) {
        compute_histogram_autocorrelation_nD(root, tree_depth, histogram, &bs,
                                             num_threads, &stats);
    } else {
        compute_histogram_autocorrelation_ring_nD(root, tree_depth, histogram,
                                                  &bs, num_threads, n_points, &stats);
    }
    double time_histogram = get_wall_time() - start_phase;
    printf("Histogram computed.\n");
    print_time("Total algorithm time:", time_histogram);
    printf("\n");

    // Display.
    printf("========================================\n");
    printf("Distance Histogram Results\n");
    printf("========================================\n");
    printf("%-5s %-25s %-15s\n", "Bin", "Range", "Count");
    printf("----------------------------------------------------------------\n");
    long long total_pairs = 0;
    int non_empty_bins = 0;
    for (int i = 0; i < num_bins; i++) {
        double lower = bs.edges[i];
        double upper = bs.edges[i + 1];
        if (histogram[i] > 0) non_empty_bins++;
        printf("%-5d [%.4f, %.4f)%*s%lld\n", i, lower, upper,
               (int)(25 - 10), "", histogram[i]);
        total_pairs += histogram[i];
    }
    printf("================================================================\n");
    printf("Non-empty bins: %d / %d\n", non_empty_bins, num_bins);
    printf("Total pairs:    %lld\n", total_pairs);
    printf("================================================================\n");

    long long expected = ((long long)n_points * (n_points - 1)) / 2;
    printf("\nVerification:\n");
    printf("  Expected pairs: %lld\n", expected);
    printf("  Counted pairs:  %lld\n", total_pairs);
    printf("  Difference:     %lld\n", total_pairs - expected);
    printf("  Status:         %s\n\n", total_pairs == expected ? "PASS" : "FAIL");

    // Stats.
    printf("========================================\n");
    printf("Algorithm Statistics\n");
    printf("========================================\n");
    printf("Exact distance computations:     %lld\n", stats.exact_distance_computations);
    printf("Pruned pairs:                    %lld\n", stats.pruned_pairs);
    printf("Leaf comparisons:                %lld\n", stats.leaf_comparisons);
    printf("Prune operations:                %lld\n", stats.prune_operations);
    printf("Displacement distance calls:     %lld\n", g_displacement_distance_calls);
    long long total_counted = stats.exact_distance_computations + stats.pruned_pairs;
    if (total_counted > 0) {
        printf("Pruning efficiency:              %.2f%%\n",
               100.0 * stats.coarse_resolved_pairs / total_counted);
    }
    printf("========================================\n");

    double time_total = get_wall_time() - start_total;
    printf("\nPerformance Summary\n");
    printf("========================================\n");
    print_time("Point generation:", time_generation);
    print_time("Tree construction:", time_tree);
    print_time("Histogram computation:", time_histogram);
    print_time("Total time:", time_total);
    if (time_histogram > 0)
        printf("Throughput: %.2f M pairs/sec\n",
               (double)total_pairs / time_histogram / 1e6);
    printf("========================================\n");

    if (out_path != NULL) {
        FILE *of = fopen(out_path, "w");
        if (of == NULL) {
            fprintf(stderr, "Error: could not open --out file '%s' for writing\n",
                    out_path);
        } else {
            for (int i = 0; i < num_bins; i++) {
                fprintf(of, "%d\t%lld\n", i, histogram[i]);
            }
            fclose(of);
            printf("Wrote histogram (%d bins) to %s\n", num_bins, out_path);
        }
    }

    free(points);
    free(histogram);
    free_tree(root);
    free(compacted_points);
    free(edges);
    printf("\nCompleted successfully.\n");
    return 0;
}

// ============================================================================
// Stubs for methods 1 and 3 (filled in below)
// ============================================================================

// ----------------------------------------------------------------------------
// N-D autocorrelation via FFTW r2c/c2r.
//
// Returns a newly-allocated double array of size (2m-1)^D, indexed by:
//   ac[ (d_0 + m-1) * (2m-1)^(D-1) + (d_1 + m-1) * (2m-1)^(D-2) + ... ]
// where d_k ranges over -(m-1) .. +(m-1).
// ----------------------------------------------------------------------------

static double* compute_autocorrelation_flat_nD(const int *grid_flat,
                                               int m, int num_threads,
                                               double *time_autocorr) {
    double t0 = get_wall_time();
    int padded = 2 * m;
    int D = g_dim;

    int dims[MAX_DIM];
    for (int k = 0; k < D; k++) dims[k] = padded;

    long long real_size = 1;
    for (int k = 0; k < D; k++) real_size *= padded;
    long long n_complex_last = padded / 2 + 1;
    long long complex_size = 1;
    for (int k = 0; k < D - 1; k++) complex_size *= padded;
    complex_size *= n_complex_last;

    fftw_init_threads();
    fftw_plan_with_nthreads(num_threads);

    {
        char wisdom_path[256];
        snprintf(wisdom_path, sizeof(wisdom_path),
                 "/tmp/fftw_wisdom_nD_%d_%d_%d.dat", D, padded, num_threads);
        FILE *wf = fopen(wisdom_path, "r");
        if (wf) { fftw_import_wisdom_from_file(wf); fclose(wf); }
    }

    double       *in_r = (double*)      fftw_malloc(sizeof(double)       * real_size);
    fftw_complex *spec = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * complex_size);
    if (!in_r || !spec) {
        fprintf(stderr, "Error: FFTW allocation failed\n");
        exit(1);
    }

    fftw_plan plan_fwd = fftw_plan_dft_r2c(D, dims, in_r, spec, FFTW_MEASURE);
    fftw_plan plan_inv = fftw_plan_dft_c2r(D, dims, spec, in_r,
                                           FFTW_MEASURE | FFTW_DESTROY_INPUT);

    {
        char wisdom_path[256];
        snprintf(wisdom_path, sizeof(wisdom_path),
                 "/tmp/fftw_wisdom_nD_%d_%d_%d.dat", D, padded, num_threads);
        FILE *wf = fopen(wisdom_path, "w");
        if (wf) { fftw_export_wisdom_to_file(wf); fclose(wf); }
    }

    // Zero-pad and copy grid into in_r.  Row-major over D axes (axis 0 major).
    memset(in_r, 0, sizeof(double) * real_size);
    long long grid_stride = 1; for (int k = 0; k < D; k++) grid_stride *= m;
    #pragma omp parallel for
    for (long long flat = 0; flat < grid_stride; flat++) {
        int cell[MAX_DIM]; long long t = flat;
        for (int k = D - 1; k >= 0; k--) { cell[k] = (int)(t % m); t /= m; }
        long long pad_idx = 0;
        for (int k = 0; k < D; k++) pad_idx = pad_idx * padded + cell[k];
        in_r[pad_idx] = (double)grid_flat[flat];
    }

    fftw_execute(plan_fwd);

    #pragma omp parallel for
    for (long long i = 0; i < complex_size; i++) {
        double re = creal(spec[i]);
        double im = cimag(spec[i]);
        spec[i] = (re * re + im * im) + 0.0 * I;
    }

    fftw_execute(plan_inv);

    long long ac_per_axis = 2 * m - 1;
    long long ac_size = 1; for (int k = 0; k < D; k++) ac_size *= ac_per_axis;
    double *ac = (double*)calloc((size_t)ac_size, sizeof(double));
    if (!ac) { fprintf(stderr, "Error: ac alloc failed\n"); exit(1); }
    double norm = 1.0 / (double)real_size;

    #pragma omp parallel for
    for (long long out_idx = 0; out_idx < ac_size; out_idx++) {
        int dvec[MAX_DIM]; long long t = out_idx;
        for (int k = D - 1; k >= 0; k--) {
            int u = (int)(t % ac_per_axis);
            dvec[k] = u - (m - 1);
            t /= ac_per_axis;
        }
        long long in_idx = 0;
        for (int k = 0; k < D; k++) {
            int u = (dvec[k] + padded) % padded;
            in_idx = in_idx * padded + u;
        }
        ac[out_idx] = in_r[in_idx] * norm;
    }

    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_inv);
    fftw_free(in_r);
    fftw_free(spec);

    *time_autocorr = get_wall_time() - t0;
    return ac;
}

// ----------------------------------------------------------------------------
// Method-4-only variant: returns the FFTW result buffer in_r directly,
// skipping the extract-into-(2m-1)^D-array step entirely.  Layout is the
// padded one (2m)^D with FFTW's natural axis wrap.  Indexing for signed
// displacement d_k in [-(m-1), m-1]:
//   wrap(d_k) = d_k < 0 ? d_k + padded : d_k
//   in_idx = wrap(d_0) * padded^(D-1) + wrap(d_1) * padded^(D-2) + ... + wrap(d_{D-1})
//   value = in_r[in_idx] * (*norm_out)
//
// The caller is responsible for fftw_free(buf).  *norm_out is set to
// 1 / real_size (= 1 / (2m)^D) so the caller multiplies by it at read time.
//
// Profile showed the original function spends ~33-43% of its wall time on the
// extract loop alone; the ring method only reads the canonical half of the result,
// and reads it via known displacement vectors, so it doesn't need the
// remapping into the (2m-1)^D layout.
// ----------------------------------------------------------------------------

static double* compute_autocorrelation_padded_nD(const int *grid_flat,
                                                 int m, int num_threads,
                                                 double *time_autocorr,
                                                 double *norm_out) {
    double t0 = get_wall_time();
    int padded = 2 * m;
    int D = g_dim;

    int dims[MAX_DIM];
    for (int k = 0; k < D; k++) dims[k] = padded;

    long long real_size = 1;
    for (int k = 0; k < D; k++) real_size *= padded;
    long long n_complex_last = padded / 2 + 1;
    long long complex_size = 1;
    for (int k = 0; k < D - 1; k++) complex_size *= padded;
    complex_size *= n_complex_last;

    fftw_init_threads();
    fftw_plan_with_nthreads(num_threads);

    {
        char wisdom_path[256];
        snprintf(wisdom_path, sizeof(wisdom_path),
                 "/tmp/fftw_wisdom_nD_%d_%d_%d.dat", D, padded, num_threads);
        FILE *wf = fopen(wisdom_path, "r");
        if (wf) { fftw_import_wisdom_from_file(wf); fclose(wf); }
    }

    double       *in_r = (double*)      fftw_malloc(sizeof(double)       * real_size);
    fftw_complex *spec = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * complex_size);
    if (!in_r || !spec) {
        fprintf(stderr, "Error: FFTW allocation failed\n");
        exit(1);
    }

    fftw_plan plan_fwd = fftw_plan_dft_r2c(D, dims, in_r, spec, FFTW_MEASURE);
    fftw_plan plan_inv = fftw_plan_dft_c2r(D, dims, spec, in_r,
                                           FFTW_MEASURE | FFTW_DESTROY_INPUT);

    {
        char wisdom_path[256];
        snprintf(wisdom_path, sizeof(wisdom_path),
                 "/tmp/fftw_wisdom_nD_%d_%d_%d.dat", D, padded, num_threads);
        FILE *wf = fopen(wisdom_path, "w");
        if (wf) { fftw_export_wisdom_to_file(wf); fclose(wf); }
    }

    memset(in_r, 0, sizeof(double) * real_size);
    long long grid_stride = 1; for (int k = 0; k < D; k++) grid_stride *= m;
    #pragma omp parallel for
    for (long long flat = 0; flat < grid_stride; flat++) {
        int cell[MAX_DIM]; long long t = flat;
        for (int k = D - 1; k >= 0; k--) { cell[k] = (int)(t % m); t /= m; }
        long long pad_idx = 0;
        for (int k = 0; k < D; k++) pad_idx = pad_idx * padded + cell[k];
        in_r[pad_idx] = (double)grid_flat[flat];
    }

    fftw_execute(plan_fwd);

    #pragma omp parallel for
    for (long long i = 0; i < complex_size; i++) {
        double re = creal(spec[i]);
        double im = cimag(spec[i]);
        spec[i] = (re * re + im * im) + 0.0 * I;
    }

    fftw_execute(plan_inv);

    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_inv);
    fftw_free(spec);

    *norm_out = 1.0 / (double)real_size;
    *time_autocorr = get_wall_time() - t0;
    return in_r;   // caller must fftw_free(in_r)
}

typedef struct {
    int  disp[MAX_DIM];
    long long pair_count;
    int  bin_min, bin_max;
    double d_min, d_max;
    char is_single_bin;
    char is_intra;
} DispTaskND;

static int cmp_tasks_desc_nD(const void *a, const void *b) {
    const DispTaskND *ta = (const DispTaskND *)a;
    const DispTaskND *tb = (const DispTaskND *)b;
    long long ca = ta->is_single_bin ? 0 : ta->pair_count;
    long long cb = tb->is_single_bin ? 0 : tb->pair_count;
    if (ca != cb) return (ca > cb) ? -1 : 1;
    return 0;
}

// Is `disp` a canonical (upper-half) representative?  Scan k=0..D-1; the first
// non-zero coordinate must be positive.  All-zero disp is canonical.
// At D=2 this becomes: dx>0, OR (dx==0 AND dy>=0) -- matches the 2D file's
// convention.
static inline int disp_is_canonical(const int *disp, int D) {
    for (int k = 0; k < D; k++) {
        if (disp[k] > 0) return 1;
        if (disp[k] < 0) return 0;
    }
    return 1;
}

// Row-merge fallback for an arbitrary D-dim displacement.  Walk the
// non-empty cells; for each, look up partner cell + disp in the flat grid.
static void row_merge_displacement(const LeafGrid *lg, const int *disp,
                                   long long *histogram, const BinSpec *bs,
                                   Statistics *stats) {
    int D = g_dim;
    int m = lg->m;
    for (int k_idx = 0; k_idx < lg->num_non_empty; k_idx++) {
        const CellCoord *cc1 = &lg->non_empty_cells[k_idx];
        int partner_cell[MAX_DIM];
        int oor = 0;
        for (int k = 0; k < D; k++) {
            partner_cell[k] = cc1->cell[k] + disp[k];
            if (partner_cell[k] < 0 || partner_cell[k] >= m) { oor = 1; break; }
        }
        if (oor) continue;
        long long pflat = grid_index_from_cell(partner_cell, m, D);
        if (lg->grid_flat[pflat] == 0) continue;
        TreeNode *cell1 = lg->node_map_flat[grid_index_from_cell(cc1->cell, m, D)];
        TreeNode *cell2 = lg->node_map_flat[pflat];
        CellCoord cc2;
        for (int k = 0; k < D; k++) {
            cc2.cell[k] = partner_cell[k];
            cc2.blo[k] = (float)cell2->lo[k];
            cc2.bhi[k] = (float)cell2->hi[k];
        }
        for (int k = D; k < MAX_DIM; k++) { cc2.cell[k]=0; cc2.blo[k]=0; cc2.bhi[k]=0; }
        cc2.count = cell2->count;
        process_cell_pair(cell1, cell2, cc1, &cc2, histogram, bs, 0, stats);
    }
}

void compute_histogram_autocorrelation_nD(TreeNode *root, int tree_depth,
                                          long long *histogram, const BinSpec *bs,
                                          int num_threads, Statistics *stats) {
    int num_bins = bs->num_bins;
    int D = g_dim;

    printf("   Building leaf grid (flat m^D)...\n");
    LeafGrid *lg = build_leaf_grid(root, tree_depth);
    int m = lg->m;
    long long grid_stride = lg->stride;
    long long ac_per_axis = 2 * m - 1;
    long long ac_size = 1; for (int k = 0; k < D; k++) ac_size *= ac_per_axis;

    printf("   Grid size: m=%d, D=%d, total cells: %lld\n", m, D, grid_stride);
    printf("   Non-empty cells: %d (%.2f%% occupancy)\n",
           lg->num_non_empty, 100.0 * lg->num_non_empty / (double)grid_stride);
    {
        long long pad_doubles = 1;
        for (int k = 0; k < D; k++) pad_doubles *= (2 * m);
        printf("   Computing %dD autocorrelation (padded side %d, %lld doubles)...\n",
               D, 2 * m, pad_doubles);
    }

    double time_autocorr;
    double *ac = compute_autocorrelation_flat_nD(lg->grid_flat, m,
                                                 num_threads, &time_autocorr);
    printf("   Autocorrelation time: %.3f s\n", time_autocorr);

    if (g_approx_mode) {
        // Approximate mode: Algorithm 2, lines 4-8 in one pass.  For every occupied canonical class, compute the
        // distance bounds and buckets, then add the count to its bucket or split it with Distribute.  No task
        // records are stored.  Exact mode keeps the task list below, which it sorts by pair count to balance the
        // point-to-point work across threads.
        double t_fused_start = get_wall_time();
        long long intra_pair_count = 0;
        for (int k = 0; k < lg->num_non_empty; k++) {
            long long c = lg->non_empty_cells[k].count;
            intra_pair_count += c * (c - 1) / 2;
        }
        #pragma omp parallel
        {
            long long *local_histogram = (long long*)calloc(num_bins, sizeof(long long));
            Statistics local_stats = {0};
            #pragma omp for schedule(static) nowait
            for (long long out_idx = 0; out_idx < ac_size; out_idx++) {
                int dvec[MAX_DIM]; long long tt = out_idx;
                for (int k = D - 1; k >= 0; k--) {
                    int u = (int)(tt % ac_per_axis);
                    dvec[k] = u - (m - 1);
                    tt /= ac_per_axis;
                }
                if (!disp_is_canonical(dvec, D)) continue;
                long long pair_count = (long long)round(ac[out_idx]);
                if (pair_count == 0) continue;
                int all_zero = 1;
                for (int k = 0; k < D; k++) if (dvec[k] != 0) { all_zero = 0; break; }
                if (all_zero) pair_count = intra_pair_count;
                double d_min, d_max;
                get_displacement_distance_range_nD(dvec, lg->cell_size, &d_min, &d_max);
                int bmin = binspec_bin(bs, d_min);
                int bmax = binspec_bin(bs, d_max);
                if (bmin == bmax) {
                    local_histogram[bmin] += pair_count;
                    local_stats.pruned_pairs += pair_count;
                    local_stats.coarse_resolved_pairs += pair_count;
                    local_stats.prune_operations++;
                } else {
                    distribute_pairs_heuristic_disp(pair_count, d_min, d_max, bmin, bmax,
                                                    local_histogram, bs, &local_stats);
                }
            }
            #pragma omp critical
            {
                for (int b = 0; b < num_bins; b++) histogram[b] += local_histogram[b];
                stats->exact_distance_computations += local_stats.exact_distance_computations;
                stats->pruned_pairs                += local_stats.pruned_pairs;
                stats->leaf_comparisons            += local_stats.leaf_comparisons;
                stats->prune_operations            += local_stats.prune_operations;
                stats->displacement_distance_calls += local_stats.displacement_distance_calls;
                stats->coarse_resolved_pairs       += local_stats.coarse_resolved_pairs;
            }
            free(local_histogram);
        }
        printf("   Fused classification and bucket assignment (%.3f s)\n", get_wall_time() - t_fused_start);
        free(ac);
        free_leaf_grid(lg);
        return;
    }

    // Build displacement tasks (exact mode).
    printf("   Building displacement tasks (ac_size=%lld)...\n", ac_size);
    double t_build_start = get_wall_time();
    DispTaskND **t_bufs = (DispTaskND**)calloc(num_threads, sizeof(DispTaskND*));
    long long *t_counts = (long long*)calloc(num_threads, sizeof(long long));
    long long *t_caps   = (long long*)calloc(num_threads, sizeof(long long));
    for (int t = 0; t < num_threads; t++) {
        long long cap = (ac_size / num_threads) + 64;
        if (cap > 1<<20) cap = 1<<20;  // start modest; realloc as needed
        t_bufs[t] = (DispTaskND*)malloc((size_t)cap * sizeof(DispTaskND));
        t_caps[t] = cap;
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        DispTaskND *buf = t_bufs[tid];
        long long cnt = 0;
        long long cap = t_caps[tid];

        #pragma omp for schedule(static)
        for (long long out_idx = 0; out_idx < ac_size; out_idx++) {
            int dvec[MAX_DIM]; long long tt = out_idx;
            for (int k = D - 1; k >= 0; k--) {
                int u = (int)(tt % ac_per_axis);
                dvec[k] = u - (m - 1);
                tt /= ac_per_axis;
            }
            if (!disp_is_canonical(dvec, D)) continue;
            long long pair_count = (long long)round(ac[out_idx]);
            if (pair_count == 0) continue;

            double d_min, d_max;
            get_displacement_distance_range_nD(dvec, lg->cell_size, &d_min, &d_max);
            int bmin = binspec_bin(bs, d_min);
            int bmax = binspec_bin(bs, d_max);

            if (cnt >= cap) {
                cap *= 2;
                buf = (DispTaskND*)realloc(buf, (size_t)cap * sizeof(DispTaskND));
            }
            DispTaskND *tp = &buf[cnt++];
            for (int k = 0; k < MAX_DIM; k++) tp->disp[k] = (k < D) ? dvec[k] : 0;
            tp->pair_count = pair_count;
            tp->bin_min = bmin;
            tp->bin_max = bmax;
            tp->d_min = d_min;
            tp->d_max = d_max;
            tp->is_single_bin = (bmin == bmax) ? 1 : 0;
            int all_zero = 1;
            for (int k = 0; k < D; k++) if (dvec[k] != 0) { all_zero = 0; break; }
            tp->is_intra = all_zero ? 1 : 0;
        }
        t_bufs[tid] = buf;
        t_counts[tid] = cnt;
        t_caps[tid] = cap;
    }

    long long num_tasks = 0;
    for (int t = 0; t < num_threads; t++) num_tasks += t_counts[t];
    DispTaskND *tasks = (DispTaskND*)malloc((size_t)num_tasks * sizeof(DispTaskND));
    long long pos = 0;
    for (int t = 0; t < num_threads; t++) {
        memcpy(tasks + pos, t_bufs[t],
               (size_t)t_counts[t] * sizeof(DispTaskND));
        pos += t_counts[t];
        free(t_bufs[t]);
    }
    free(t_bufs); free(t_counts); free(t_caps);

    if (!g_approx_mode) qsort(tasks, num_tasks, sizeof(DispTaskND), cmp_tasks_desc_nD);
    double t_build = get_wall_time() - t_build_start;
    printf("   Built %lld displacement tasks (%.3f s)\n", num_tasks, t_build);

    // Intra-cell pair count (NOT from ac, which counts ordered self-pairs).
    long long intra_pair_count = 0;
    for (int k = 0; k < lg->num_non_empty; k++) {
        long long c = lg->non_empty_cells[k].count;
        intra_pair_count += c * (c - 1) / 2;
    }

    #pragma omp parallel
    {
        long long *local_histogram = (long long*)calloc(num_bins, sizeof(long long));
        Statistics local_stats = {0};

        #pragma omp for schedule(dynamic, 1) nowait
        for (long long ti = 0; ti < num_tasks; ti++) {
            DispTaskND *tp = &tasks[ti];
            long long pair_count = tp->is_intra ? intra_pair_count : tp->pair_count;


            if (tp->is_single_bin) {
                local_histogram[tp->bin_min] += pair_count;
                local_stats.pruned_pairs += pair_count;
                local_stats.coarse_resolved_pairs += pair_count;
                local_stats.prune_operations++;
            } else {
                if (g_approx_mode) {
                    distribute_pairs_heuristic_disp(
                        pair_count, tp->d_min, tp->d_max,
                        tp->bin_min, tp->bin_max,
                        local_histogram, bs, &local_stats);
                } else if (tp->is_intra) {
                    for (int k = 0; k < lg->num_non_empty; k++) {
                        long long flat = grid_index_from_cell(
                            lg->non_empty_cells[k].cell, m, D);
                        TreeNode *cell = lg->node_map_flat[flat];
                        if (cell && cell->count > 1) {
                            process_cell_pair(cell, cell,
                                              &lg->non_empty_cells[k],
                                              &lg->non_empty_cells[k],
                                              local_histogram, bs, 1,
                                              &local_stats);
                        }
                    }
                } else {
                    row_merge_displacement(lg, tp->disp,
                                           local_histogram, bs, &local_stats);
                }
            }
        }

        #pragma omp critical
        {
            for (int b = 0; b < num_bins; b++) histogram[b] += local_histogram[b];
            stats->exact_distance_computations += local_stats.exact_distance_computations;
            stats->pruned_pairs                += local_stats.pruned_pairs;
            stats->leaf_comparisons            += local_stats.leaf_comparisons;
            stats->prune_operations            += local_stats.prune_operations;
            stats->displacement_distance_calls += local_stats.displacement_distance_calls;
            stats->coarse_resolved_pairs       += local_stats.coarse_resolved_pairs;
        }
        free(local_histogram);
    }

    free(tasks);
    free(ac);
    free_leaf_grid(lg);
}

// ============================================================================
// RING ENUMERATION (RE) OVER LEAF AUTOCORRELATION MATRIX
// ============================================================================
//
// Idea. The leaf autocorrelation M_leaf[dx_0, ..., dx_{D-1}] caches the pair
// count at every cell displacement.  The map disp -> bin (via d_min, d_max and
// binspec_bin) is geometric -- independent of the points.  For fixed values
// of the first D-1 axes (a "prefix"), the map (bmin(|q|), bmax(|q|)) is
// monotone non-decreasing in |q| = |dx_{D-1}|.  So along each prefix's
// last-axis "row" the canonical-|q| range decomposes into intervals where
// (bmin, bmax) is constant: resolved-in-bin-k intervals (bmin == bmax == k)
// alternating with straddler runs (bmin < bmax).
//
// The two transitions in (bmin, bmax) are captured by two anchor arrays per
// prefix:
//   Bmax[k] = smallest |q| with d_max(|q|) >= edges[k]    (smallest |q| with
//             bmax >= k)
//   Bmin[k] = smallest |q| with d_min(|q|) >= edges[k]    (smallest |q| with
//             bmin >= k)
// Both are derived in closed form from the bound geometry (one inverse-sqrt
// per bin), so the whole row's bin-membership table is 2 * num_bins integers.
// Bmin[k] >= Bmax[k] always (because d_max >= d_min), so:
//   resolved-in-bin-k  : |q| in [Bmin[k], Bmax[k+1] - 1]
//   straddler-bmin=k   : |q| in [Bmax[k+1], Bmin[k+1] - 1]
// Resolved intervals contribute as a tight contiguous sum over M_leaf; that
// sum has no per-cell bin computation, no branch.  Straddler cells get the
// usual per-displacement straddler treatment (heuristic in approx mode, row-
// merge cell-pair walk in exact mode).
//
// In n-D, "row" means: hold all but the last axis fixed.  The prefix is the
// (D-1)-vector (dx_0, ..., dx_{D-2}) -- iterated in canonical order to avoid
// double counting (first nonzero axis must be positive; an all-zero prefix
// further restricts the last axis to |q| >= 0).  For prefixes with at least
// one nonzero coordinate, the canonical convention allows BOTH signs on the
// last axis, so we sum M_leaf over the resolved positive-q interval plus its
// mirror negative-q interval.
//
// The data structure is literally just M_leaf plus the implicit anchor map;
// nothing else is built.  No pyramid, no propagation, no resolve mask.

// ----------------------------------------------------------------------------
// Anchor inversion.  For the given (D-1)-vector prefix:
//   S_a = sum_{k<D-1} max(|prefix_k| - 1, 0)^2
//   S_b = sum_{k<D-1} (|prefix_k| + 1)^2
// Last-axis distance bounds:
//   d_max(|q|)^2 / cell^2 = S_b + (|q| + 1)^2
//   d_min(|q|)^2 / cell^2 = S_a + max(|q| - 1, 0)^2
//
// Bmax[k] = smallest integer |q| >= 0 with d_max(|q|) >= edges[k].
//           (|q|+1)^2 >= R_k^2 - S_b  with R_k = edges[k] / cell.
//           If R_k^2 <= S_b: Bmax[k] = 0.
//           Else: |q| >= ceil(sqrt(R_k^2 - S_b)) - 1, clamped to >=0.
// Bmin[k] = smallest integer |q| >= 0 with d_min(|q|) >= edges[k].
//           If R_k^2 <= S_a: Bmin[k] = 0 (any |q| works, even |q|=0 gives
//                                          d_min^2/cell^2 = S_a >= R_k^2).
//           Else: need (|q|-1)^2 >= R_k^2 - S_a with |q|>=1, so
//                 |q| >= ceil(sqrt(R_k^2 - S_a)) + 1.
//
// Endpoints are nudged with binspec_bin against the actual d_min, d_max to
// absorb any sqrt-rounding mismatch with how the rest of the code classifies
// cells (so bit-identical histograms with the autocorr method when straddler
// handling matches).
//
// Returns 0..num_bins anchors (one per bin edge plus the upper edge), so
// each output array has num_bins + 1 slots.  Anchors are non-decreasing.
//
// Rk2[] is a per-row-INVARIANT precomputation: Rk2[k] = (edges[k] / cell)^2.
// Computing it once globally (rather than per row) lifts num_bins+1
// divisions and multiplications out of every prefix iteration.

static inline void compute_row_anchors(double S_a, double S_b,
                                       int num_bins, const double *Rk2,
                                       int m,
                                       int *Bmax, int *Bmin) {
    Bmax[0] = 0;
    Bmin[0] = 0;
    for (int k = 1; k <= num_bins; k++) {
        double Rk2k = Rk2[k];

        // ----- Bmax[k]: smallest |q| with d_max(|q|) >= edges[k].
        int hi_anchor;
        if (Rk2k <= S_b) {
            hi_anchor = 0;
        } else {
            double need = sqrt(Rk2k - S_b);
            hi_anchor = (int)ceil(need) - 1;
            if (hi_anchor < 0) hi_anchor = 0;
        }
        if (hi_anchor > m) hi_anchor = m;
        Bmax[k] = hi_anchor;

        // ----- Bmin[k]: smallest |q| with d_min(|q|) >= edges[k].
        int lo_anchor;
        if (Rk2k <= S_a) {
            lo_anchor = 0;
        } else {
            double need = sqrt(Rk2k - S_a);
            lo_anchor = (int)ceil(need) + 1;
            if (lo_anchor < 1) lo_anchor = 1;
        }
        if (lo_anchor > m) lo_anchor = m;
        Bmin[k] = lo_anchor;
    }
}


// Worker that processes the rows for one fixed "outermost" axis value dx0,
// recursing over the rest of the prefix coordinates in canonical order.
//
// `prefix` is a length-D buffer; we fill prefix[0..D-2] during recursion and
// the function calls itself recursively, processing the last axis at the leaf
// of the recursion.

typedef struct {
    int D;
    int m;
    long long ac_per_axis;
    const double *ac;          // M_leaf, double precision (no integer conversion) -- legacy
    const double *in_r;        // padded FFTW output (raw, un-normalized) -- preferred
    double norm;               // multiply in_r reads by this to get pair counts
    const LeafGrid *lg;
    const BinSpec *bs;
    double cell;
    const double *Rk2;         // precomputed (edges[k]/cell)^2, length num_bins+1
    long long *local_hist;
    Statistics *local_stats;
    int *Bmax;   // scratch arrays of size num_bins + 1
    int *Bmin;
    int *signed_disp;          // scratch buffer of length MAX_DIM
    long long *r_count;        // [diag] resolved intervals
    long long *r_pairs;        // [diag] resolved pair count
    long long *s_count;        // [diag] straddler cells processed
} RingWorker;

// Convert a canonical signed displacement to its flat index in ac[].
// Layout: ac[ (d_0 + m-1) * (2m-1)^(D-1) + (d_1 + m-1) * (2m-1)^(D-2) + ... ].
static inline long long ac_index_from_disp(const int *disp, int D, int m,
                                           long long ac_per_axis) {
    long long idx = 0;
    for (int k = 0; k < D; k++) idx = idx * ac_per_axis + (disp[k] + (m - 1));
    return idx;
}

// Process a fully-specified canonical prefix's last-axis row.  The prefix
// occupies disp[0..D-2]; this function sums over |q| = dx_{D-1} according to
// the row's anchor decomposition.  `prefix_has_nonzero` tells us whether the
// prefix has any nonzero coordinate (so the last axis is signed-free) or is
// all zero (last axis restricted to >= 0).
//
// Approx-mode straddlers go through distribute_pairs_heuristic_disp.
// Exact-mode straddlers go through row_merge_displacement.

static void process_prefix_row(int *disp, int prefix_has_nonzero,
                               double S_a, double S_b, RingWorker *w) {
    int D = w->D;
    int m = w->m;
    const BinSpec *bs = w->bs;
    int num_bins = bs->num_bins;

    // 1) Compute anchors for this prefix.
    compute_row_anchors(S_a, S_b, num_bins, w->Rk2, m, w->Bmax, w->Bmin);

    // 2) Walk bins.
    //
    // For each bin k in [0, num_bins):
    //    resolved-in-k:  |q| in [Bmin[k], Bmax[k+1] - 1]
    //    straddler band: |q| in [Bmax[k+1], Bmin[k+1] - 1]
    //
    // We clip |q| to [0, m-1].  If Bmax[k+1] > m, the row has no bin-k+ cells
    // (all higher-bin transitions occur beyond the grid).
    //
    // For each resolved interval, do the bulk sum of M_leaf over the
    // appropriate signed last-axis positions:
    //   - positive side: dx_{D-1} = +|q|, summed over |q| in [lo, hi].
    //   - negative side (only when prefix_has_nonzero AND lo >= 1): summed
    //                  over |q| in [max(lo, 1), hi] mapped to dx_{D-1} = -|q|.
    //   - for the all-zero prefix, the negative side is dropped entirely.

    // Build the in_r index for the prefix part (axes 0..D-2 of disp).
    // Layout: in_r has dims (padded)^D in row-major (axis 0 most significant).
    // wrap(v) = v < 0 ? v + padded : v.  padded = 2m, so for prefix coords in
    // [-(m-1), m-1] the wrap is just a single branch.
    //
    // Positive-side last-axis access (q >= 0): in_r[prefix_base + q].
    // Negative-side last-axis access (q < 0):  in_r[prefix_base + q + padded].
    //
    // Both are stride-1 contiguous slabs along the last axis -- identical
    // access pattern to the original ac-based code (since ac was just a
    // remapping of in_r). The savings come from skipping the extract loop
    // that built ac in the first place.
    int *sdisp = w->signed_disp;
    for (int k = 0; k < D; k++) sdisp[k] = disp[k];

    const int padded = 2 * m;
    long long prefix_base = 0;
    for (int k = 0; k < D - 1; k++) {
        int v = disp[k];
        int wv = (v < 0) ? (v + padded) : v;
        prefix_base = prefix_base * (long long)padded + (long long)wv;
    }
    prefix_base *= (long long)padded;   // multiply once more for the last-axis stride

    const double norm = w->norm;
    // Positive-side row pointer: in_r[prefix_base + q] for q in [0, m-1].
    const double *row_pos = w->in_r + prefix_base;
    // Negative-side row pointer: in_r[prefix_base + q + padded] for q < 0.
    // Equivalent to indexing in_r[prefix_base + padded + q].  Negative q's of
    // magnitude up to m-1 land in [padded - (m-1), padded - 1] = [m+1, 2m-1].
    const double *row_neg = w->in_r + prefix_base + padded;

    for (int k = 0; k < num_bins; k++) {
        int lo = w->Bmin[k];
        int hi_excl = w->Bmax[k + 1];   // resolved upper bound is hi_excl - 1
        int hi = hi_excl - 1;

        if (lo > m - 1) {
            // No more resolved cells in this row for any bin >= k.
            break;
        }
        if (hi > m - 1) hi = m - 1;

        if (lo <= hi) {
            // --- Resolved interval --------------------------------------
            double s_pos = 0.0;
            for (int q = lo; q <= hi; q++) {
                s_pos += (row_pos[q] * norm);   // dx_{D-1} = +q
            }
            long long add_pos = (long long)llround(s_pos);
            if (add_pos != 0) {
                w->local_hist[k] += add_pos;
                w->local_stats->pruned_pairs += add_pos;
                w->local_stats->coarse_resolved_pairs += add_pos;
                (*w->r_pairs) += add_pos;
            }
            if (prefix_has_nonzero) {
                int mlo = (lo == 0) ? 1 : lo;
                if (mlo <= hi) {
                    double s_neg = 0.0;
                    for (int q = mlo; q <= hi; q++) {
                        s_neg += (row_neg[-q] * norm);  // dx_{D-1} = -q
                    }
                    long long add_neg = (long long)llround(s_neg);
                    if (add_neg != 0) {
                        w->local_hist[k] += add_neg;
                        w->local_stats->pruned_pairs += add_neg;
                        w->local_stats->coarse_resolved_pairs += add_neg;
                        (*w->r_pairs) += add_neg;
                    }
                }
            }
            w->local_stats->prune_operations++;
            (*w->r_count)++;
        }

        // --- Straddler band for bmin = k ------------------------------------
        //
        // Cells with bmin == k AND bmax > k.  Among cells with bmin == k
        // (|q| in [Bmin[k], Bmin[k+1] - 1]), the resolved ones (bmax == k)
        // are at |q| in [Bmax[k], Bmax[k+1] - 1].  Since Bmin[k] >= Bmax[k],
        // the straddler band is [max(Bmin[k], Bmax[k+1]), Bmin[k+1] - 1].
        //
        // Important: do NOT use Bmax[k+1] alone as the lower bound -- when
        // bmax skips bin k entirely (no |q| has bmax == k), Bmax[k+1] can be
        // smaller than Bmin[k], and the bmin=k row segment doesn't reach
        // that far.  Taking the max ensures the band is the bmin=k cells
        // that are straddlers, with no overlap with adjacent bins' bands.
        int strad_lo = hi_excl;
        if (strad_lo < lo) strad_lo = lo;
        int strad_hi = w->Bmin[k + 1] - 1;
        if (strad_hi > m - 1) strad_hi = m - 1;
        if (strad_lo < 0) strad_lo = 0;
        if (strad_lo <= strad_hi) {
            // Optimization A: avoid get_displacement_distance_range_nD and
            // binspec_bin per cell.  d_min, d_max derive from S_a, S_b (the
            // row's prefix-axis sums) plus a single |q|-dependent term.
            // bmin = k by construction (cells in this band have bmin == k).
            // bmax(q) is monotone non-decreasing; we track it via a cursor
            // that advances as q grows, using the anchor array Bmax[].
            const int bmin_fixed = (k < num_bins) ? k : (num_bins - 1);
            int bmax_cur = k + 1;
            // Advance bmax_cur so that q == strad_lo lands in the right slot.
            while (bmax_cur < num_bins && w->Bmax[bmax_cur + 1] <= strad_lo)
                bmax_cur++;
            for (int q = strad_lo; q <= strad_hi; q++) {
                // Advance bmax cursor: Bmax[j] is monotone, q is monotone.
                while (bmax_cur < num_bins && w->Bmax[bmax_cur + 1] <= q)
                    bmax_cur++;
                int bmax_eff = (bmax_cur < num_bins) ? bmax_cur : (num_bins - 1);
                // |q|-derived distance interval (same for both signs of q).
                double aq = (q > 0) ? (double)(q - 1) : 0.0;
                double bq = (double)(q + 1);
                double d_min = w->cell * sqrt(S_a + aq * aq);
                double d_max = w->cell * sqrt(S_b + bq * bq);

                // Positive side cell: dx_{D-1} = +q.
                long long cnt_pos = (long long)llround((row_pos[q] * norm));
                if (cnt_pos > 0) {
                    sdisp[D - 1] = q;
                    if (g_approx_mode) {
                        distribute_pairs_heuristic_disp(cnt_pos, d_min, d_max,
                                                        bmin_fixed, bmax_eff,
                                                        w->local_hist, bs,
                                                        w->local_stats);
                    } else {
                        int all_zero = 1;
                        for (int kk = 0; kk < D; kk++)
                            if (sdisp[kk] != 0) { all_zero = 0; break; }
                        if (all_zero) {
                            for (int kc = 0; kc < w->lg->num_non_empty; kc++) {
                                long long flat = grid_index_from_cell(
                                    w->lg->non_empty_cells[kc].cell, m, D);
                                TreeNode *cellq = w->lg->node_map_flat[flat];
                                if (cellq && cellq->count > 1) {
                                    process_cell_pair(cellq, cellq,
                                                      &w->lg->non_empty_cells[kc],
                                                      &w->lg->non_empty_cells[kc],
                                                      w->local_hist, bs, 1,
                                                      w->local_stats);
                                }
                            }
                        } else {
                            row_merge_displacement(w->lg, sdisp,
                                                   w->local_hist, bs,
                                                   w->local_stats);
                        }
                    }
                    (*w->s_count)++;
                }
                if (prefix_has_nonzero && q > 0) {
                    long long cnt_neg = (long long)llround((row_neg[-q] * norm));
                    if (cnt_neg > 0) {
                        sdisp[D - 1] = -q;
                        if (g_approx_mode) {
                            distribute_pairs_heuristic_disp(cnt_neg, d_min, d_max,
                                                            bmin_fixed, bmax_eff,
                                                            w->local_hist, bs,
                                                            w->local_stats);
                        } else {
                            row_merge_displacement(w->lg, sdisp,
                                                   w->local_hist, bs,
                                                   w->local_stats);
                        }
                        (*w->s_count)++;
                    }
                }
            }
        }
    }

    // Tail beyond the last bin's upper edge: any |q| with d_min already past
    // the last bin's upper edge.  These cells have bmin >= num_bins, which
    // binspec_bin clamps to num_bins - 1.  Same optimization as above:
    // d_min, d_max derive from S_a, S_b + |q|; (bmin, bmax) = (num_bins-1,
    // num_bins-1) by clamping.
    int tail_lo = w->Bmin[num_bins];
    if (tail_lo < 0) tail_lo = 0;
    if (tail_lo < m) {
        const int bclamp = num_bins - 1;
        for (int q = tail_lo; q < m; q++) {
            double aq = (q > 0) ? (double)(q - 1) : 0.0;
            double bq = (double)(q + 1);
            double d_min = w->cell * sqrt(S_a + aq * aq);
            double d_max = w->cell * sqrt(S_b + bq * bq);

            long long cnt_pos = (long long)llround((row_pos[q] * norm));
            if (cnt_pos > 0) {
                sdisp[D - 1] = q;
                if (g_approx_mode) {
                    distribute_pairs_heuristic_disp(cnt_pos, d_min, d_max,
                                                    bclamp, bclamp,
                                                    w->local_hist, bs,
                                                    w->local_stats);
                } else {
                    row_merge_displacement(w->lg, sdisp,
                                           w->local_hist, bs, w->local_stats);
                }
                (*w->s_count)++;
            }
            if (prefix_has_nonzero && q > 0) {
                long long cnt_neg = (long long)llround((row_neg[-q] * norm));
                if (cnt_neg > 0) {
                    sdisp[D - 1] = -q;
                    if (g_approx_mode) {
                        distribute_pairs_heuristic_disp(cnt_neg, d_min, d_max,
                                                        bclamp, bclamp,
                                                        w->local_hist, bs,
                                                        w->local_stats);
                    } else {
                        row_merge_displacement(w->lg, sdisp,
                                               w->local_hist, bs,
                                               w->local_stats);
                    }
                    (*w->s_count)++;
                }
            }
        }
    }
}

// Recursive prefix enumeration over axes [axis .. D-2] in canonical order.
// `disp[0..axis-1]` is already filled.  `prefix_any_nonzero` tracks whether
// any of those filled coordinates is nonzero.  `prefix_strictly_pos_first` is
// 1 if the canonical positivity constraint has already been satisfied (i.e.,
// the first nonzero among the filled coordinates is positive) -- this lets us
// freely enumerate signs from the next axis onward.  S_a, S_b are the
// running absolute-value sums for the prefix.

static void ring_recurse_prefix(int axis, int *disp,
                                int prefix_any_nonzero,
                                int prefix_canonical_settled,
                                double S_a, double S_b,
                                RingWorker *w) {
    int D = w->D;
    int m = w->m;

    if (axis == D - 1) {
        // Prefix complete (length D-1).  Process the last-axis row.
        process_prefix_row(disp, prefix_any_nonzero, S_a, S_b, w);
        return;
    }

    // Determine the signed range for this axis under canonical order:
    //   - if prefix is all-zero so far (prefix_any_nonzero == 0):
    //       this axis can be 0 or positive (since the first nonzero must be
    //       positive); explore disp[axis] = 0..m-1.
    //   - else (prefix_canonical_settled == 1 already):
    //       this axis is free; explore disp[axis] = -(m-1)..m-1.
    int lo, hi;
    if (!prefix_any_nonzero) { lo = 0; hi = m - 1; }
    else                     { lo = -(m - 1); hi = m - 1; }

    for (int v = lo; v <= hi; v++) {
        disp[axis] = v;
        int av = (v < 0) ? -v : v;
        double a = (av > 0) ? (double)(av - 1) : 0.0;
        double b = (double)(av + 1);
        double new_Sa = S_a + a * a;
        double new_Sb = S_b + b * b;
        int new_any_nz = prefix_any_nonzero || (v != 0);
        int new_settled = prefix_canonical_settled ||
                          (v != 0);  // first nonzero is positive (since v>=0
                                     // when !prefix_any_nonzero, and v can be
                                     // any sign otherwise -- so this flag's
                                     // value is just "we've passed the rule").
        ring_recurse_prefix(axis + 1, disp, new_any_nz, new_settled,
                            new_Sa, new_Sb, w);
    }
}

// Top-level Ring Enumeration (RE) entry point.
void compute_histogram_autocorrelation_ring_nD(TreeNode *root, int tree_depth,
                                               long long *histogram,
                                               const BinSpec *bs,
                                               int num_threads, int n_points,
                                               Statistics *stats) {
    (void)n_points;  /* reserved for entry-point symmetry */
    int D = g_dim;
    int num_bins = bs->num_bins;

    printf("   Building leaf grid (flat m^D)...\n");
    LeafGrid *lg = build_leaf_grid(root, tree_depth);
    int m = lg->m;
    double cell = lg->cell_size;

    long long grid_stride = lg->stride;
    long long ac_per_axis = 2 * m - 1;
    long long ac_size = 1; for (int k = 0; k < D; k++) ac_size *= ac_per_axis;

    printf("   Grid size: m=%d, D=%d, total cells: %lld\n", m, D, grid_stride);
    printf("   Non-empty cells: %d (%.2f%% occupancy)\n",
           lg->num_non_empty, 100.0 * lg->num_non_empty / (double)grid_stride);

    double t_start = get_wall_time();


    // ---- Leaf FFT.  Use the padded-output variant: skip the extract-into-
    // (2m-1)^D-array step entirely and index into the raw padded buffer
    // (in_r) directly from the row scan.  Profiling showed the extract loop
    // costs ~33-43% of the FFT time in D=2/D=3 -- it's pure index remapping
    // that Ring Enumeration doesn't need.
    double time_autocorr;
    double norm;
    double *in_r = compute_autocorrelation_padded_nD(lg->grid_flat, m,
                                                     num_threads,
                                                     &time_autocorr, &norm);
    printf("   Autocorrelation time: %.3f s\n", time_autocorr);

    // ---- Intra-cell pair count routing.
    //
    // In the autocorr method, the origin displacement (0,...,0) is enumerated as a
    // canonical task with pair_count overridden to intra_pair_count.  Its
    // [d_min, d_max] from get_displacement_distance_range_nD is
    // [0, cell*sqrt(D)] (because the bound formula uses |disp|+1 along each
    // axis for d_max).  In approximate mode this routes intra pairs through
    // the heuristic distribution across bins 0 .. bmax.  In exact mode the
    // intra-cell route iterates non-empty cells and does an exact in-cell
    // pair distance computation.  We must mirror this behaviour exactly to
    // get bit-identical histograms with the autocorr method.
    long long intra_pair_count = 0;
    for (int k = 0; k < lg->num_non_empty; k++) {
        long long c = lg->non_empty_cells[k].count;
        intra_pair_count += c * (c - 1) / 2;
    }
    if (intra_pair_count > 0) {
        // Compute the origin displacement's d_min, d_max via the standard
        // bound formula (matches the autocorr method exactly).
        int origin_disp[MAX_DIM] = {0};
        double d_min, d_max;
        get_displacement_distance_range_nD(origin_disp, cell, &d_min, &d_max);
        int bmin = binspec_bin(bs, d_min);
        int bmax = binspec_bin(bs, d_max);

        if (g_approx_mode) {
            distribute_pairs_heuristic_disp(intra_pair_count, d_min, d_max,
                                            bmin, bmax,
                                            histogram, bs, stats);
        } else {
            // Exact mode: same intra-cell walk the autocorr method does.
            for (int k = 0; k < lg->num_non_empty; k++) {
                long long flat = grid_index_from_cell(
                    lg->non_empty_cells[k].cell, m, D);
                TreeNode *cellq = lg->node_map_flat[flat];
                if (cellq && cellq->count > 1) {
                    process_cell_pair(cellq, cellq,
                                      &lg->non_empty_cells[k],
                                      &lg->non_empty_cells[k],
                                      histogram, bs, 1, stats);
                }
            }
        }
    }

    // ---- Zero out the origin entry of in_r so the row scan doesn't double-
    // count the intra-cell pair.  (The row scan would otherwise treat the
    // origin as a resolved bin-0 cell.)  In the padded layout, the origin
    // displacement (all zeros) maps to in_r[0].
    in_r[0] = 0.0;

    // ---- Diagnostic counters.
    long long total_resolved_intervals = 0;
    long long total_resolved_pairs = 0;
    long long total_straddler_cells = 0;

    // ---- Row-invariant precompute: Rk2[k] = (edges[k] / cell)^2.
    // Shared across all rows (Optimization A: hoist out of compute_row_anchors).
    double *Rk2 = (double*)malloc((num_bins + 1) * sizeof(double));
    for (int k = 0; k <= num_bins; k++) {
        double r = bs->edges[k] / cell;
        Rk2[k] = r * r;
    }

    // ---- Parallelize over the outermost axis dx_0 = 0..m-1.  For D = 1 the
    // outer loop's body IS the row scan (no recursion needed): the prefix is
    // empty, S_a = S_b = 0.

    double t_ring0 = get_wall_time();

    if (D == 1) {
        // Single-row case: prefix is empty.  Just process row.
        // OpenMP: contend on a single histogram via per-thread locals.
        #pragma omp parallel
        {
            long long *local_hist = (long long*)calloc(num_bins, sizeof(long long));
            Statistics local_stats = {0};
            int *Bmax = (int*)malloc((num_bins + 1) * sizeof(int));
            int *Bmin = (int*)malloc((num_bins + 1) * sizeof(int));
            int signed_disp[MAX_DIM] = {0};
            int prefix[MAX_DIM] = {0};
            long long r_count = 0, r_pairs = 0, s_count = 0;

            RingWorker w = {
                .D = D, .m = m, .ac_per_axis = ac_per_axis,
                .ac = NULL, .in_r = in_r, .norm = norm,
                .lg = lg, .bs = bs, .cell = cell, .Rk2 = Rk2,
                .local_hist = local_hist, .local_stats = &local_stats,
                .Bmax = Bmax, .Bmin = Bmin, .signed_disp = signed_disp,
                .r_count = &r_count, .r_pairs = &r_pairs, .s_count = &s_count,
            };
            // The all-zero prefix forces last-axis >= 0.
            #pragma omp single
            process_prefix_row(prefix, /*prefix_has_nonzero*/0,
                               0.0, 0.0, &w);

            #pragma omp critical
            {
                for (int b = 0; b < num_bins; b++) histogram[b] += local_hist[b];
                stats->pruned_pairs                += local_stats.pruned_pairs;
                stats->coarse_resolved_pairs       += local_stats.coarse_resolved_pairs;
                stats->prune_operations            += local_stats.prune_operations;
                stats->exact_distance_computations += local_stats.exact_distance_computations;
                stats->leaf_comparisons            += local_stats.leaf_comparisons;
                stats->displacement_distance_calls += local_stats.displacement_distance_calls;
                total_resolved_intervals += r_count;
                total_resolved_pairs     += r_pairs;
                total_straddler_cells    += s_count;
            }
            free(local_hist); free(Bmax); free(Bmin);
        }
    } else {
        // D >= 2: parallelize over dx_0 = 0..m-1 (canonical first-axis range).
        #pragma omp parallel
        {
            long long *local_hist = (long long*)calloc(num_bins, sizeof(long long));
            Statistics local_stats = {0};
            int *Bmax = (int*)malloc((num_bins + 1) * sizeof(int));
            int *Bmin = (int*)malloc((num_bins + 1) * sizeof(int));
            int signed_disp[MAX_DIM] = {0};
            int prefix[MAX_DIM] = {0};
            long long r_count = 0, r_pairs = 0, s_count = 0;

            RingWorker w = {
                .D = D, .m = m, .ac_per_axis = ac_per_axis,
                .ac = NULL, .in_r = in_r, .norm = norm,
                .lg = lg, .bs = bs, .cell = cell, .Rk2 = Rk2,
                .local_hist = local_hist, .local_stats = &local_stats,
                .Bmax = Bmax, .Bmin = Bmin, .signed_disp = signed_disp,
                .r_count = &r_count, .r_pairs = &r_pairs, .s_count = &s_count,
            };

            #pragma omp for schedule(dynamic, 1) nowait
            for (int dx0 = 0; dx0 < m; dx0++) {
                prefix[0] = dx0;
                int any_nz = (dx0 != 0);
                int settled = (dx0 != 0);
                double a = (dx0 > 0) ? (double)(dx0 - 1) : 0.0;
                double b = (double)(dx0 + 1);
                double S_a = a * a;
                double S_b = b * b;
                ring_recurse_prefix(1, prefix, any_nz, settled, S_a, S_b, &w);
            }

            #pragma omp critical
            {
                for (int b = 0; b < num_bins; b++) histogram[b] += local_hist[b];
                stats->pruned_pairs                += local_stats.pruned_pairs;
                stats->coarse_resolved_pairs       += local_stats.coarse_resolved_pairs;
                stats->prune_operations            += local_stats.prune_operations;
                stats->exact_distance_computations += local_stats.exact_distance_computations;
                stats->leaf_comparisons            += local_stats.leaf_comparisons;
                stats->displacement_distance_calls += local_stats.displacement_distance_calls;
                total_resolved_intervals += r_count;
                total_resolved_pairs     += r_pairs;
                total_straddler_cells    += s_count;
            }
            free(local_hist); free(Bmax); free(Bmin);
        }
    }

    double t_ring = get_wall_time() - t_ring0;

    free(Rk2);
    fftw_free(in_r);
    free_leaf_grid(lg);

    printf("   Ring enumeration: %lld resolved intervals (%lld pairs),"
           " %lld straddler cells\n",
           total_resolved_intervals, total_resolved_pairs,
           total_straddler_cells);
    printf("   [diag] leaf FFT: %.3f s   ring scan: %.3f s\n",
           time_autocorr, t_ring);
    printf("   nD Ring method total: %.3f s\n", get_wall_time() - t_start);
}
