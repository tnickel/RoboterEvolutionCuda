import cupy as cp
import numpy as np

CUDA_KERNEL = '''
extern "C" {
__device__ float my_tanh(float x) {
    return tanhf(x);
}
__device__ float my_exp(float x) {
    return expf(x);
}

__global__ void activate_kernel(
    const double* inputs_batch,
    const int* genome_start_nodes, const int* genome_num_nodes,
    const double* flat_node_biases, const double* flat_node_responses, const int* flat_node_act_types,
    const int* flat_value_indices,
    const int* flat_link_from_idx, const double* flat_link_weights,
    const int* flat_link_ranges_start, const int* flat_link_ranges_end,
    const int* genome_start_outputs, const int* genome_num_outputs,
    const int* flat_output_indices,
    double* values_batch,
    double* outputs_batch,
    int n_genomes,
    int n_inputs) 
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_genomes) return;

    // Inputs setzen
    for (int i = 0; i < n_inputs; ++i) {
        values_batch[g * %(max_values)d + i] = inputs_batch[g * n_inputs + i];
    }

    // Knoten evaluieren
    int start_node = genome_start_nodes[g];
    int n_nodes = genome_num_nodes[g];

    for (int i = 0; i < n_nodes; ++i) {
        int node_idx = start_node + i;
        
        // Gewichtete Summe
        double s = 0.0;
        int link_start = flat_link_ranges_start[node_idx];
        int link_end = flat_link_ranges_end[node_idx];
        
        for (int j = link_start; j < link_end; ++j) {
            int from_idx = flat_link_from_idx[j];
            s += values_batch[g * %(max_values)d + from_idx] * flat_link_weights[j];
        }
            
        double z = flat_node_biases[node_idx] + flat_node_responses[node_idx] * s;
        
        int act = flat_node_act_types[node_idx];
        double val = 0.0;
        if (act == 0) {  // tanh
            double z2 = 2.5 * z;
            if (z2 < -60.0) z2 = -60.0;
            else if (z2 > 60.0) z2 = 60.0;
            val = my_tanh((float)z2);
        } else if (act == 1) {  // sigmoid
            double z2 = 5.0 * z;
            if (z2 < -60.0) z2 = -60.0;
            else if (z2 > 60.0) z2 = 60.0;
            val = 1.0 / (1.0 + my_exp((float)(-z2)));
        } else if (act == 2) {  // relu
            val = z > 0.0 ? z : 0.0;
        } else {
            double z2 = 2.5 * z;
            if (z2 < -60.0) z2 = -60.0;
            else if (z2 > 60.0) z2 = 60.0;
            val = my_tanh((float)z2);
        }
            
        int v_idx = flat_value_indices[node_idx];
        values_batch[g * %(max_values)d + v_idx] = val;
    }

    // Outputs in result array kopieren
    int start_out = genome_start_outputs[g];
    int n_out = genome_num_outputs[g];
    for (int i = 0; i < n_out; ++i) {
        int out_idx = flat_output_indices[start_out + i];
        outputs_batch[g * %(n_outputs)d + i] = values_batch[g * %(max_values)d + out_idx];
    }
}
}
'''

class CudaBatchedNetwork:
    """Evaluiert eine Population von NEAT Netzwerken auf der GPU (CuPy)."""
    
    def __init__(self, fast_networks):
        """Initialisiert den Batch mit einer Liste von FastNetwork Objekten."""
        self.n_genomes = len(fast_networks)
        if self.n_genomes == 0:
            return
            
        self.n_inputs = fast_networks[0]._n_inputs
        self.n_outputs = len(fast_networks[0]._output_indices)
        self.max_values = max(net._n_total_values for net in fast_networks)
        
        # Arrays sammeln
        genome_start_nodes = np.zeros(self.n_genomes, dtype=np.int32)
        genome_num_nodes = np.zeros(self.n_genomes, dtype=np.int32)
        genome_start_outputs = np.zeros(self.n_genomes, dtype=np.int32)
        genome_num_outputs = np.zeros(self.n_genomes, dtype=np.int32)
        
        total_nodes = sum(net._n_nodes for net in fast_networks)
        total_links = sum(net._n_links for net in fast_networks)
        total_outputs = sum(len(net._output_indices) for net in fast_networks)
        
        flat_node_biases = np.zeros(total_nodes, dtype=np.float64)
        flat_node_responses = np.zeros(total_nodes, dtype=np.float64)
        flat_node_act_types = np.zeros(total_nodes, dtype=np.int32)
        flat_value_indices = np.zeros(total_nodes, dtype=np.int32)
        
        flat_link_from_idx = np.zeros(total_links, dtype=np.int32)
        flat_link_weights = np.zeros(total_links, dtype=np.float64)
        flat_link_ranges_start = np.zeros(total_nodes, dtype=np.int32)
        flat_link_ranges_end = np.zeros(total_nodes, dtype=np.int32)
        
        flat_output_indices = np.zeros(total_outputs, dtype=np.int32)
        
        node_offset = 0
        link_offset = 0
        out_offset = 0
        
        for i, net in enumerate(fast_networks):
            n_nodes = net._n_nodes
            n_links = net._n_links
            n_out = len(net._output_indices)
            
            genome_start_nodes[i] = node_offset
            genome_num_nodes[i] = n_nodes
            genome_start_outputs[i] = out_offset
            genome_num_outputs[i] = n_out
            
            flat_node_biases[node_offset : node_offset + n_nodes] = net._node_biases
            flat_node_responses[node_offset : node_offset + n_nodes] = net._node_responses
            flat_node_act_types[node_offset : node_offset + n_nodes] = net._node_act_types
            flat_value_indices[node_offset : node_offset + n_nodes] = net._value_indices
            
            flat_link_from_idx[link_offset : link_offset + n_links] = net._link_from_idx
            flat_link_weights[link_offset : link_offset + n_links] = net._link_weights
            
            flat_link_ranges_start[node_offset : node_offset + n_nodes] = net._link_ranges_start + link_offset
            flat_link_ranges_end[node_offset : node_offset + n_nodes] = net._link_ranges_end + link_offset
            
            flat_output_indices[out_offset : out_offset + n_out] = net._output_indices
            
            node_offset += n_nodes
            link_offset += n_links
            out_offset += n_out
            
        # Auf die GPU kopieren
        self.d_genome_start_nodes = cp.asarray(genome_start_nodes)
        self.d_genome_num_nodes = cp.asarray(genome_num_nodes)
        
        self.d_flat_node_biases = cp.asarray(flat_node_biases)
        self.d_flat_node_responses = cp.asarray(flat_node_responses)
        self.d_flat_node_act_types = cp.asarray(flat_node_act_types)
        self.d_flat_value_indices = cp.asarray(flat_value_indices)
        
        self.d_flat_link_from_idx = cp.asarray(flat_link_from_idx)
        self.d_flat_link_weights = cp.asarray(flat_link_weights)
        self.d_flat_link_ranges_start = cp.asarray(flat_link_ranges_start)
        self.d_flat_link_ranges_end = cp.asarray(flat_link_ranges_end)
        
        self.d_genome_start_outputs = cp.asarray(genome_start_outputs)
        self.d_genome_num_outputs = cp.asarray(genome_num_outputs)
        self.d_flat_output_indices = cp.asarray(flat_output_indices)
        
        self.d_values_batch = cp.zeros((self.n_genomes * self.max_values,), dtype=cp.float64)
        self.d_outputs_batch = cp.zeros((self.n_genomes * self.n_outputs,), dtype=cp.float64)
        
        self.threads_per_block = 256
        self.blocks_per_grid = (self.n_genomes + self.threads_per_block - 1) // self.threads_per_block
        
        # Kernel kompilieren
        code = CUDA_KERNEL % {'max_values': self.max_values, 'n_outputs': self.n_outputs}
        self.kernel = cp.RawKernel(code, 'activate_kernel')
        
    def activate_batch(self, inputs_batch):
        """Evaluiert alle Netzwerke für einen Batch von Inputs.
        
        Args:
            inputs_batch: (N, n_inputs) numpy array
            
        Returns:
            (N, n_outputs) numpy array
        """
        d_inputs = cp.asarray(inputs_batch, dtype=cp.float64)
        
        self.kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (d_inputs,
             self.d_genome_start_nodes, self.d_genome_num_nodes,
             self.d_flat_node_biases, self.d_flat_node_responses, self.d_flat_node_act_types,
             self.d_flat_value_indices,
             self.d_flat_link_from_idx, self.d_flat_link_weights,
             self.d_flat_link_ranges_start, self.d_flat_link_ranges_end,
             self.d_genome_start_outputs, self.d_genome_num_outputs,
             self.d_flat_output_indices,
             self.d_values_batch,
             self.d_outputs_batch,
             self.n_genomes,
             self.n_inputs)
        )
        
        # Zurueckkopieren und Reshapen
        return self.d_outputs_batch.get().reshape((self.n_genomes, self.n_outputs))

    def activate_batch_device(self, d_inputs_batch):
        """Evaluiert alle Netzwerke für einen Batch von Inputs (alles auf GPU).
        
        Args:
            d_inputs_batch: cupy array containing inputs
            
        Returns:
            cupy array containing outputs
        """
        self.kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (d_inputs_batch,
             self.d_genome_start_nodes, self.d_genome_num_nodes,
             self.d_flat_node_biases, self.d_flat_node_responses, self.d_flat_node_act_types,
             self.d_flat_value_indices,
             self.d_flat_link_from_idx, self.d_flat_link_weights,
             self.d_flat_link_ranges_start, self.d_flat_link_ranges_end,
             self.d_genome_start_outputs, self.d_genome_num_outputs,
             self.d_flat_output_indices,
             self.d_values_batch,
             self.d_outputs_batch,
             self.n_genomes,
             self.n_inputs)
        )
        return self.d_outputs_batch
