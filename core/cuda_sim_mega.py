import math
import cupy as cp
import numpy as np
from config.config_manager import SimConfig

CUDA_SIM_KERNEL = r'''

extern "C" {

#define OBJ_NONE 0
#define OBJ_WALL 1
#define OBJ_BATTERY 2
#define OBJ_COLLECTOR 3
#define OBJ_HUNTER 4

__device__ float ray_segment_intersection(float ox, float oy, float dx, float dy, float x1, float y1, float x2, float y2) {
    float sx = x2 - x1;
    float sy = y2 - y1;
    float denom = dx * sy - dy * sx;
    if (abs(denom) < 1e-10f) return -1.0f;
    float t_num = (x1 - ox) * sy - (y1 - oy) * sx;
    float u_num = (x1 - ox) * dy - (y1 - oy) * dx;
    float t = t_num / denom;
    float u = u_num / denom;
    if (t >= 0.0f && u >= 0.0f && u <= 1.0f) return t;
    return -1.0f;
}

__device__ float ray_circle_intersection(float ox, float oy, float dx, float dy, float cx, float cy, float cr) {
    float fx = ox - cx;
    float fy = oy - cy;
    float a = dx * dx + dy * dy;
    float b = 2.0f * (fx * dx + fy * dy);
    float c = fx * fx + fy * fy - cr * cr;
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f) return -1.0f;
    float sqrt_disc = sqrtf(discriminant);
    float inv_2a = 1.0f / (2.0f * a);
    float t1 = (-b - sqrt_disc) * inv_2a;
    float t2 = (-b + sqrt_disc) * inv_2a;
    if (t1 >= 0.0f) return t1;
    if (t2 >= 0.0f) return t2;
    return -1.0f;
}

__device__ float my_tanh(float x) { return tanhf(x); }
__device__ float my_exp(float x) { return expf(x); }

__device__ void do_raycast(
    int g, int n_robots, float* r_x, float* r_y, float* r_angle, int* r_alive, int* r_type, float* r_radius,
    int n_bats, float* b_x, float* b_y, int* b_active, float b_radius,
    int n_walls, float* w_x1, float* w_y1, float* w_x2, float* w_y2,
    double* sensor_inputs, int n_sensor_rays, float c_ray_len, float c_fov, float h_ray_len, float h_fov
) {
    if (g >= n_robots || r_alive[g] == 0) return;
    float rx = r_x[g];
    float ry = r_y[g];
    float ra = r_angle[g];
    int rtype = r_type[g];
    float max_length = (rtype == OBJ_HUNTER) ? h_ray_len : c_ray_len;
    float fov_rad = (rtype == OBJ_HUNTER) ? h_fov : c_fov;

    for (int i = 0; i < n_sensor_rays; ++i) {
        float angle_offset = 0.0f;
        if (n_sensor_rays > 1) {
            angle_offset = -fov_rad / 2.0f + (i / (float)(n_sensor_rays - 1)) * fov_rad;
        }
        float ray_angle = ra + angle_offset;
        float ray_dx = cosf(ray_angle);
        float ray_dy = sinf(ray_angle);
        float closest_dist = max_length;
        float closest_type = 0.0f;

        for (int w = 0; w < n_walls; ++w) {
            float t = ray_segment_intersection(rx, ry, ray_dx, ray_dy, w_x1[w], w_y1[w], w_x2[w], w_y2[w]);
            if (t >= 0.0f && t < closest_dist) { closest_dist = t; closest_type = 1.0f; }
        }
        for (int b = 0; b < n_bats; ++b) {
            if (b_active[b]) {
                float t = ray_circle_intersection(rx, ry, ray_dx, ray_dy, b_x[b], b_y[b], b_radius);
                if (t >= 0.0f && t < closest_dist) { closest_dist = t; closest_type = 2.0f; }
            }
        }
        for (int ro = 0; ro < n_robots; ++ro) {
            if (ro != g && r_alive[ro]) {
                float t = ray_circle_intersection(rx, ry, ray_dx, ray_dy, r_x[ro], r_y[ro], r_radius[ro]);
                if (t >= 0.0f && t < closest_dist) { closest_dist = t; closest_type = (float)r_type[ro]; }
            }
        }
        float dist_norm = closest_dist / max_length;
        int base_idx = g * (n_sensor_rays * 5 + 2) + i * 5;
        sensor_inputs[base_idx + 0] = dist_norm;
        sensor_inputs[base_idx + 1] = (closest_type == 2.0f) ? 1.0 : 0.0;
        sensor_inputs[base_idx + 2] = (closest_type == 4.0f) ? 1.0 : 0.0;
        sensor_inputs[base_idx + 3] = (closest_type == 1.0f) ? 1.0 : 0.0;
        sensor_inputs[base_idx + 4] = (closest_type == 3.0f) ? 1.0 : 0.0;
    }
    
    float nearest_dist = max_length;
    float dx_norm = 0.0f;
    float dy_norm = 0.0f;
    for (int ro = 0; ro < n_robots; ++ro) {
        if (ro != g && r_alive[ro]) {
            int target_type = r_type[ro];
            if ((rtype == OBJ_COLLECTOR && target_type == OBJ_HUNTER) || 
                (rtype == OBJ_HUNTER && target_type == OBJ_COLLECTOR)) {
                float dx = r_x[ro] - rx;
                float dy = r_y[ro] - ry;
                float dsq = dx*dx + dy*dy;
                if (dsq < nearest_dist * nearest_dist) {
                    nearest_dist = sqrtf(dsq);
                    float dist_norm = 1.0f - (nearest_dist / max_length);
                    float angle_to_target = atan2f(dy, dx);
                    float rel_angle = angle_to_target - ra;
                    dx_norm = cosf(rel_angle) * dist_norm;
                    dy_norm = sinf(rel_angle) * dist_norm;
                }
            }
        }
    }
    int end_idx = g * (n_sensor_rays * 5 + 2) + n_sensor_rays * 5;
    sensor_inputs[end_idx + 0] = dx_norm;
    sensor_inputs[end_idx + 1] = dy_norm;
}

__device__ void eval_network(
    int g_idx, int global_robot_idx, double* sensor_inputs, double* motor_outputs,
    const int* genome_start_nodes, const int* genome_num_nodes,
    const double* flat_node_biases, const double* flat_node_responses, const int* flat_node_act_types,
    const int* flat_value_indices, const int* flat_link_from_idx, const double* flat_link_weights,
    const int* flat_link_ranges_start, const int* flat_link_ranges_end,
    const int* genome_start_outputs, const int* genome_num_outputs,
    const int* flat_output_indices, double* values_batch,
    int n_inputs, int max_values, int n_outputs
) {
    for (int i = 0; i < n_inputs; ++i) {
        values_batch[g_idx * max_values + i] = sensor_inputs[global_robot_idx * n_inputs + i];
    }
    int start_node = genome_start_nodes[g_idx];
    int n_nodes = genome_num_nodes[g_idx];
    for (int i = 0; i < n_nodes; ++i) {
        int node_idx = start_node + i;
        double s = 0.0;
        int link_start = flat_link_ranges_start[node_idx];
        int link_end = flat_link_ranges_end[node_idx];
        for (int j = link_start; j < link_end; ++j) {
            int from_idx = flat_link_from_idx[j];
            s += values_batch[g_idx * max_values + from_idx] * flat_link_weights[j];
        }
        double z = flat_node_biases[node_idx] + flat_node_responses[node_idx] * s;
        int act = flat_node_act_types[node_idx];
        double val = 0.0;
        if (act == 0) {
            double z2 = 2.5 * z; if (z2 < -60.0) z2 = -60.0; else if (z2 > 60.0) z2 = 60.0;
            val = my_tanh((float)z2);
        } else if (act == 1) {
            double z2 = 5.0 * z; if (z2 < -60.0) z2 = -60.0; else if (z2 > 60.0) z2 = 60.0;
            val = 1.0 / (1.0 + my_exp((float)(-z2)));
        } else if (act == 2) {
            val = z > 0.0 ? z : 0.0;
        } else {
            double z2 = 2.5 * z; if (z2 < -60.0) z2 = -60.0; else if (z2 > 60.0) z2 = 60.0;
            val = my_tanh((float)z2);
        }
        int v_idx = flat_value_indices[node_idx];
        values_batch[g_idx * max_values + v_idx] = val;
    }
    int start_out = genome_start_outputs[g_idx];
    int n_out = genome_num_outputs[g_idx];
    for (int i = 0; i < n_out; ++i) {
        int out_idx = flat_output_indices[start_out + i];
        motor_outputs[global_robot_idx * n_outputs + i] = values_batch[g_idx * max_values + out_idx];
    }
}

__device__ void do_physics(
    int g, int n_robots, float* r_x, float* r_y, float* r_angle, int* r_alive, int* r_type, float* r_radius,
    float* r_energy, float* r_fitness, double* motor_outputs,
    int n_bats, float* b_x, float* b_y, int* b_active, float b_radius,
    int n_walls, float* w_x1, float* w_y1, float* w_x2, float* w_y2,
    int n_obstacles, float* obs_left, float* obs_top, float* obs_right, float* obs_bottom,
    float* prev_hunter_dists, float c_speed, float h_speed, float w_width, float w_height,
    float fit_idle, float fit_surv, float fit_bat, float fit_prox, 
    float fit_danger, float fit_appr, float fit_kill, float fit_eaten,
    float energy_drain, float bat_energy, float energy_start,
    float danger_zone, float c_prox_zone, float h_prox_zone,
    int* stats_kills, int* stats_bats, int bat_respawn_delay, int* b_timer, int* r_eaten, int* indiv_kills, int* indiv_bats
) {
    if (g >= n_robots || r_alive[g] == 0) return;
    float rx = r_x[g]; float ry = r_y[g]; float ra = r_angle[g];
    float rrad = r_radius[g]; int rtype = r_type[g];

    float m_left = (float)motor_outputs[g * 2 + 0];
    float m_right = (float)motor_outputs[g * 2 + 1];
    if (m_left < -1.0f) m_left = -1.0f; if (m_left > 1.0f) m_left = 1.0f;
    if (m_right < -1.0f) m_right = -1.0f; if (m_right > 1.0f) m_right = 1.0f;

    float speed_mult = (rtype == OBJ_COLLECTOR) ? c_speed : h_speed;
    float forward_speed = (m_left + m_right) / 2.0f;
    float turn_speed = (m_right - m_left) / 20.0f;
    float actual_fwd = forward_speed * speed_mult;
    float actual_turn = turn_speed * speed_mult;
    ra += actual_turn;
    
    float TWO_PI = 2.0f * 3.14159265f;
    ra = fmodf(ra, TWO_PI); if (ra < 0.0f) ra += TWO_PI;
    
    float nx = rx + cosf(ra) * actual_fwd; float ny = ry + sinf(ra) * actual_fwd;
    if (nx < rrad) nx = rrad; if (nx > w_width - rrad) nx = w_width - rrad;
    if (ny < rrad) ny = rrad; if (ny > w_height - rrad) ny = w_height - rrad;

    float r_sq = rrad * rrad;
    for (int obs_i = 0; obs_i < n_obstacles; ++obs_i) {
        float left = obs_left[obs_i]; float top = obs_top[obs_i]; float right = obs_right[obs_i]; float bottom = obs_bottom[obs_i];
        float closest_x = (nx < left) ? left : ((nx > right) ? right : nx);
        float closest_y = (ny < top) ? top : ((ny > bottom) ? bottom : ny);
        float ddx = nx - closest_x; float ddy = ny - closest_y;
        float dist_sq = ddx * ddx + ddy * ddy;
        if (dist_sq < r_sq) {
            float dist = (dist_sq > 0.0f) ? sqrtf(dist_sq) : 0.01f;
            float overlap = rrad - dist;
            if (dist > 0.0f) { nx += (ddx / dist) * overlap; ny += (ddy / dist) * overlap; }
            else { nx += overlap; ny += overlap; }
        }
    }
    rx = nx; ry = ny;
    
    float ren = r_energy[g];
    ren -= energy_drain;
    if (ren <= 0.0f) {
        if (rtype == OBJ_COLLECTOR) r_alive[g] = 0;
        else ren = 1.0f;
    }
    
    float rfit = r_fitness[g];
    if (forward_speed < 0.1f) rfit += fit_idle;
    rfit += fit_surv;
    
    if (rtype == OBJ_COLLECTOR) {
        float nearest_bat_sq = c_prox_zone * c_prox_zone;
        float collect_dist_sq = (rrad + 15.0f) * (rrad + 15.0f);
        for (int b = 0; b < n_bats; ++b) {
            if (b_active[b]) {
                float dx = b_x[b] - rx; float dy = b_y[b] - ry;
                float dsq = dx*dx + dy*dy;
                if (dsq < collect_dist_sq) {
                    int old = atomicExch(&b_active[b], 0);
                    if (old == 1) {
                        ren += bat_energy; if (ren > energy_start) ren = energy_start;
                        rfit += fit_bat;
                        atomicAdd(stats_bats, 1); atomicAdd(&indiv_bats[g], 1);
                        int delay = bat_respawn_delay; if (delay <= 0) delay = 1;
                        b_timer[b] = delay;
                    }
                } else if (dsq < nearest_bat_sq) nearest_bat_sq = dsq;
            }
        }
        if (nearest_bat_sq < c_prox_zone * c_prox_zone) {
            float dist = sqrtf(nearest_bat_sq);
            rfit += fit_prox * (1.0f - dist / c_prox_zone);
        }
        float danger_zone_sq = danger_zone * danger_zone;
        for (int ro = 0; ro < n_robots; ++ro) {
            if (ro != g && r_type[ro] == OBJ_HUNTER && r_alive[ro]) {
                float dx = r_x[ro] - rx; float dy = r_y[ro] - ry;
                float dsq = dx*dx + dy*dy;
                if (dsq < danger_zone_sq) {
                    float dist = sqrtf(dsq);
                    float penalty = fit_danger * (1.0f - dist / danger_zone);
                    if (penalty > 0.0f) rfit -= penalty;
                    float prev_d = prev_hunter_dists[g * n_robots + ro];
                    if (prev_d > 0.0f) {
                        float delta = dist - prev_d;
                        if (delta > 0.0f) rfit += (delta / c_speed) * 15.0f;
                        else if (delta < 0.0f) rfit -= (-delta / c_speed) * fit_appr;
                    }
                    prev_hunter_dists[g * n_robots + ro] = dist;
                } else prev_hunter_dists[g * n_robots + ro] = 0.0f;
            }
        }
    } else if (rtype == OBJ_HUNTER) {
        for (int ro = 0; ro < n_robots; ++ro) {
            if (ro != g && r_type[ro] == OBJ_COLLECTOR && r_alive[ro]) {
                float dx = r_x[ro] - rx; float dy = r_y[ro] - ry;
                float dsq = dx*dx + dy*dy;
                float catch_dist = rrad + r_radius[ro];
                if (dsq < catch_dist * catch_dist) {
                    int old = atomicExch(&r_alive[ro], 0);
                    if (old == 1) {
                        ren += energy_start; if (ren > energy_start) ren = energy_start;
                        rfit += fit_kill;
                        atomicAdd(stats_kills, 1); atomicAdd(&indiv_kills[g], 1);
                        atomicExch(&r_eaten[ro], 1);
                    }
                }
            }
        }
        float nearest_prey_sq = h_prox_zone * h_prox_zone;
        for (int ro = 0; ro < n_robots; ++ro) {
            if (ro != g && r_type[ro] == OBJ_COLLECTOR && r_alive[ro]) {
                float dx = r_x[ro] - rx; float dy = r_y[ro] - ry;
                float dsq = dx*dx + dy*dy;
                if (dsq < nearest_prey_sq) nearest_prey_sq = dsq;
            }
        }
        if (nearest_prey_sq < h_prox_zone * h_prox_zone) {
            float dist = sqrtf(nearest_prey_sq);
            rfit += 0.1f * (1.0f - dist / h_prox_zone);
        }
    }

    r_x[g] = rx; r_y[g] = ry; r_angle[g] = ra;
    r_energy[g] = ren; r_fitness[g] = rfit;
}

__device__ void do_battery(
    int b, int n_bats, float* b_x, float* b_y, int* b_active, int* b_timer,
    float* rnd_x, float* rnd_y, int max_rnd, int* rnd_counter
) {
    if (b >= n_bats) return;
    if (b_active[b] == 0) {
        if (b_timer[b] > 0) b_timer[b] -= 1;
        if (b_timer[b] <= 0) {
            int r_idx = atomicAdd(rnd_counter, 1);
            int idx = r_idx % max_rnd;
            b_x[b] = rnd_x[idx];
            b_y[b] = rnd_y[idx];
            b_active[b] = 1;
        }
    }
}

__global__ void mega_kernel(
    int steps,
    int n_robots, float* r_x, float* r_y, float* r_angle, int* r_alive, int* r_type, float* r_radius, float* r_energy, float* r_fitness, double* motor_outputs,
    int n_bats, float* b_x, float* b_y, int* b_active, float b_radius,
    int n_walls, float* w_x1, float* w_y1, float* w_x2, float* w_y2,
    int n_obstacles, float* obs_left, float* obs_top, float* obs_right, float* obs_bottom,
    double* sensor_inputs, int n_sensor_rays, float c_ray_len, float c_fov, float h_ray_len, float h_fov,
    float* prev_hunter_dists,
    float c_speed, float h_speed, float w_width, float w_height,
    float fit_idle, float fit_surv, float fit_bat, float fit_prox, float fit_danger, float fit_appr, float fit_kill, float fit_eaten,
    float energy_drain, float bat_energy, float energy_start, float danger_zone, float c_prox_zone, float h_prox_zone,
    float* rnd_x, float* rnd_y, int max_rnd, int* rnd_counter,
    int* stats_kills, int* stats_bats, int bat_respawn_delay, int* b_timer, int* r_eaten, int* indiv_kills, int* indiv_bats,
    
    int c_n_genomes, const int* c_genome_start_nodes, const int* c_genome_num_nodes, const double* c_flat_node_biases, const double* c_flat_node_responses, const int* c_flat_node_act_types, const int* c_flat_value_indices, const int* c_flat_link_from_idx, const double* c_flat_link_weights, const int* c_flat_link_ranges_start, const int* c_flat_link_ranges_end, const int* c_genome_start_outputs, const int* c_genome_num_outputs, const int* c_flat_output_indices, double* c_values_batch, int c_max_values,
    
    int h_n_genomes, const int* h_genome_start_nodes, const int* h_genome_num_nodes, const double* h_flat_node_biases, const double* h_flat_node_responses, const int* h_flat_node_act_types, const int* h_flat_value_indices, const int* h_flat_link_from_idx, const double* h_flat_link_weights, const int* h_flat_link_ranges_start, const int* h_flat_link_ranges_end, const int* h_genome_start_outputs, const int* h_genome_num_outputs, const int* h_flat_output_indices, double* h_values_batch, int h_max_values,
    
    int n_inputs, int n_outputs
) {
    int g = threadIdx.x;

    for (int step = 0; step < steps; ++step) {
        if (g < n_robots) {
            do_raycast(g, n_robots, r_x, r_y, r_angle, r_alive, r_type, r_radius,
                       n_bats, b_x, b_y, b_active, b_radius,
                       n_walls, w_x1, w_y1, w_x2, w_y2,
                       sensor_inputs, n_sensor_rays, c_ray_len, c_fov, h_ray_len, h_fov);
        }
        __syncthreads();

        if (g < c_n_genomes) {
            eval_network(g, g, sensor_inputs, motor_outputs,
                         c_genome_start_nodes, c_genome_num_nodes, c_flat_node_biases, c_flat_node_responses, c_flat_node_act_types, c_flat_value_indices, c_flat_link_from_idx, c_flat_link_weights, c_flat_link_ranges_start, c_flat_link_ranges_end, c_genome_start_outputs, c_genome_num_outputs, c_flat_output_indices, c_values_batch, n_inputs, c_max_values, n_outputs);
        } else if (g >= c_n_genomes && g < c_n_genomes + h_n_genomes) {
            int h_g = g - c_n_genomes;
            eval_network(h_g, g, sensor_inputs, motor_outputs,
                         h_genome_start_nodes, h_genome_num_nodes, h_flat_node_biases, h_flat_node_responses, h_flat_node_act_types, h_flat_value_indices, h_flat_link_from_idx, h_flat_link_weights, h_flat_link_ranges_start, h_flat_link_ranges_end, h_genome_start_outputs, h_genome_num_outputs, h_flat_output_indices, h_values_batch, n_inputs, h_max_values, n_outputs);
        }
        __syncthreads();

        if (g < n_robots) {
            do_physics(g, n_robots, r_x, r_y, r_angle, r_alive, r_type, r_radius, r_energy, r_fitness, motor_outputs,
                       n_bats, b_x, b_y, b_active, b_radius,
                       n_walls, w_x1, w_y1, w_x2, w_y2,
                       n_obstacles, obs_left, obs_top, obs_right, obs_bottom,
                       prev_hunter_dists, c_speed, h_speed, w_width, w_height,
                       fit_idle, fit_surv, fit_bat, fit_prox, fit_danger, fit_appr, fit_kill, fit_eaten,
                       energy_drain, bat_energy, energy_start, danger_zone, c_prox_zone, h_prox_zone,
                       stats_kills, stats_bats, bat_respawn_delay, b_timer, r_eaten, indiv_kills, indiv_bats);
        }
        __syncthreads();

        if (g < n_bats) {
            do_battery(g, n_bats, b_x, b_y, b_active, b_timer, rnd_x, rnd_y, max_rnd, rnd_counter);
        }
        __syncthreads();
    }
}

// Wrappers for classic generation:
__global__ void raycast_kernel(
    int n_robots, float* r_x, float* r_y, float* r_angle, int* r_alive, int* r_type, float* r_radius,
    int n_bats, float* b_x, float* b_y, int* b_active, float b_radius,
    int n_walls, float* w_x1, float* w_y1, float* w_x2, float* w_y2,
    double* sensor_inputs, int n_sensor_rays, float c_ray_len, float c_fov, float h_ray_len, float h_fov
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    do_raycast(g, n_robots, r_x, r_y, r_angle, r_alive, r_type, r_radius, n_bats, b_x, b_y, b_active, b_radius, n_walls, w_x1, w_y1, w_x2, w_y2, sensor_inputs, n_sensor_rays, c_ray_len, c_fov, h_ray_len, h_fov);
}

__global__ void physics_kernel(
    int n_robots, float* r_x, float* r_y, float* r_angle, int* r_alive, int* r_type, float* r_radius, float* r_energy, float* r_fitness, double* motor_outputs,
    int n_bats, float* b_x, float* b_y, int* b_active, float b_radius,
    int n_walls, float* w_x1, float* w_y1, float* w_x2, float* w_y2,
    int n_obstacles, float* obs_left, float* obs_top, float* obs_right, float* obs_bottom,
    float* prev_hunter_dists, float c_speed, float h_speed, float w_width, float w_height,
    float fit_idle, float fit_surv, float fit_bat, float fit_prox, float fit_danger, float fit_appr, float fit_kill, float fit_eaten,
    float energy_drain, float bat_energy, float energy_start, float danger_zone, float c_prox_zone, float h_prox_zone,
    float* rnd_x, float* rnd_y, int max_rnd, int* rnd_counter,
    int* stats_kills, int* stats_bats, int bat_respawn_delay, int* b_timer, int* r_eaten, int* indiv_kills, int* indiv_bats
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    do_physics(g, n_robots, r_x, r_y, r_angle, r_alive, r_type, r_radius, r_energy, r_fitness, motor_outputs, n_bats, b_x, b_y, b_active, b_radius, n_walls, w_x1, w_y1, w_x2, w_y2, n_obstacles, obs_left, obs_top, obs_right, obs_bottom, prev_hunter_dists, c_speed, h_speed, w_width, w_height, fit_idle, fit_surv, fit_bat, fit_prox, fit_danger, fit_appr, fit_kill, fit_eaten, energy_drain, bat_energy, energy_start, danger_zone, c_prox_zone, h_prox_zone, stats_kills, stats_bats, bat_respawn_delay, b_timer, r_eaten, indiv_kills, indiv_bats);
}

__global__ void battery_kernel(
    int n_bats, float* b_x, float* b_y, int* b_active, int* b_timer, float* rnd_x, float* rnd_y, int max_rnd, int* rnd_counter
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    do_battery(b, n_bats, b_x, b_y, b_active, b_timer, rnd_x, rnd_y, max_rnd, rnd_counter);
}

}

'''


class CudaSimulation:
    def __init__(self, collectors, hunters, batteries, walls, config: SimConfig, collector_net, hunter_net, obstacles=None):
        self.config = config
        self.collector_net = collector_net
        self.hunter_net = hunter_net
        
        # 1. State extraction
        n_col = len(collectors)
        n_hun = len(hunters)
        self.n_robots = n_col + n_hun
        
        r_x = np.zeros(self.n_robots, dtype=np.float32)
        r_y = np.zeros(self.n_robots, dtype=np.float32)
        r_angle = np.zeros(self.n_robots, dtype=np.float32)
        r_alive = np.ones(self.n_robots, dtype=np.int32)
        r_type = np.zeros(self.n_robots, dtype=np.int32)
        r_radius = np.zeros(self.n_robots, dtype=np.float32)
        r_energy = np.zeros(self.n_robots, dtype=np.float32)
        r_fitness = np.zeros(self.n_robots, dtype=np.float32)
        
        # We need a unified array of neural nets inputs.
        # However, collectors and hunters might have different input sizes depending on sensor rays!
        # Wait, the sensor structure is identical: (n_rays * 5) + 2
        # Yes, neat config has the same num inputs.
        self.n_inputs = config.sensor_ray_count * 5 + 2
        self.n_outputs = 3
        
        for i, c in enumerate(collectors):
            r_x[i] = c.x
            r_y[i] = c.y
            r_angle[i] = c.angle
            r_type[i] = 3 # OBJ_COLLECTOR
            r_radius[i] = c.radius
            r_energy[i] = config.energy_start
            
        for i, h in enumerate(hunters):
            idx = n_col + i
            r_x[idx] = h.x
            r_y[idx] = h.y
            r_angle[idx] = h.angle
            r_type[idx] = 4 # OBJ_HUNTER
            r_radius[idx] = h.radius
            r_energy[idx] = config.energy_start
            
        self.n_bats = len(batteries)
        b_x = np.zeros(self.n_bats, dtype=np.float32)
        b_y = np.zeros(self.n_bats, dtype=np.float32)
        b_active = np.ones(self.n_bats, dtype=np.int32)
        for i, b in enumerate(batteries):
            b_x[i] = b.x
            b_y[i] = b.y
            b_active[i] = 1 if b.active else 0
            
        self.n_walls = len(walls)
        w_x1 = np.zeros(self.n_walls, dtype=np.float32)
        w_y1 = np.zeros(self.n_walls, dtype=np.float32)
        w_x2 = np.zeros(self.n_walls, dtype=np.float32)
        w_y2 = np.zeros(self.n_walls, dtype=np.float32)
        for i, w in enumerate(walls):
            w_x1[i], w_y1[i], w_x2[i], w_y2[i] = w
            
        # 2. Upload to GPU
        self.d_r_x = cp.asarray(r_x)
        self.d_r_y = cp.asarray(r_y)
        self.d_r_angle = cp.asarray(r_angle)
        self.d_r_alive = cp.asarray(r_alive)
        self.d_r_type = cp.asarray(r_type)
        self.d_r_radius = cp.asarray(r_radius)
        self.d_r_energy = cp.asarray(r_energy)
        self.d_r_fitness = cp.asarray(r_fitness)
        
        self.d_b_x = cp.asarray(b_x)
        self.d_b_y = cp.asarray(b_y)
        self.d_b_active = cp.asarray(b_active)
        
        self.d_w_x1 = cp.asarray(w_x1)
        self.d_w_y1 = cp.asarray(w_y1)
        self.d_w_x2 = cp.asarray(w_x2)
        self.d_w_y2 = cp.asarray(w_y2)
        
        # Fix 5: Obstacle AABB data
        if obstacles and len(obstacles) > 0:
            self.n_obstacles = len(obstacles)
            o_left = np.array([float(obs.left) for obs in obstacles], dtype=np.float32)
            o_top = np.array([float(obs.top) for obs in obstacles], dtype=np.float32)
            o_right = np.array([float(obs.right) for obs in obstacles], dtype=np.float32)
            o_bottom = np.array([float(obs.bottom) for obs in obstacles], dtype=np.float32)
        else:
            self.n_obstacles = 0
            o_left = np.zeros(1, dtype=np.float32)
            o_top = np.zeros(1, dtype=np.float32)
            o_right = np.zeros(1, dtype=np.float32)
            o_bottom = np.zeros(1, dtype=np.float32)
        self.d_obs_left = cp.asarray(o_left)
        self.d_obs_top = cp.asarray(o_top)
        self.d_obs_right = cp.asarray(o_right)
        self.d_obs_bottom = cp.asarray(o_bottom)
        
        # Temp buffers
        self.d_sensor_inputs = cp.zeros((self.n_robots * self.n_inputs,), dtype=cp.float64)
        self.d_motor_outputs = cp.zeros((self.n_robots * self.n_outputs,), dtype=cp.float64)
        self.d_prev_hunter_dists = cp.zeros((self.n_robots * self.n_robots,), dtype=cp.float32)
        
        # Random respawn buffer (Hindernis-frei!)
        self.max_rnd = 10000
        pad = config.cell_pixel_size
        rnd_positions = []
        obs_rects = obstacles if obstacles else []
        bat_r = 12  # Battery.RADIUS
        attempts = 0
        while len(rnd_positions) < self.max_rnd and attempts < self.max_rnd * 10:
            rx = np.random.uniform(pad, config.window_width - pad)
            ry = np.random.uniform(pad, config.window_height - pad)
            valid = True
            for obs in obs_rects:
                if (obs.left - bat_r <= rx <= obs.right + bat_r and
                    obs.top - bat_r <= ry <= obs.bottom + bat_r):
                    valid = False
                    break
            if valid:
                rnd_positions.append((rx, ry))
            attempts += 1
        # Fallback falls nicht genug Positionen
        while len(rnd_positions) < self.max_rnd:
            rnd_positions.append((np.random.uniform(pad, config.window_width - pad),
                                  np.random.uniform(pad, config.window_height - pad)))
        rnd_x = np.array([p[0] for p in rnd_positions], dtype=np.float32)
        rnd_y = np.array([p[1] for p in rnd_positions], dtype=np.float32)
        self.d_rnd_x = cp.asarray(rnd_x)
        self.d_rnd_y = cp.asarray(rnd_y)
        self.d_rnd_counter = cp.zeros(1, dtype=cp.int32)
        
        # Battery respawn timer
        self.d_b_timer = cp.zeros(self.n_bats, dtype=cp.int32)
        
        # Eaten tracking und Individual Stats
        self.d_r_eaten = cp.zeros(self.n_robots, dtype=cp.int32)
        self.d_indiv_kills = cp.zeros(self.n_robots, dtype=cp.int32)
        self.d_indiv_bats = cp.zeros(self.n_robots, dtype=cp.int32)
        
        # Stats
        self.d_stats_kills = cp.zeros(1, dtype=cp.int32)
        self.d_stats_bats = cp.zeros(1, dtype=cp.int32)
        
        # 3. Compile Kernels
        module = cp.RawModule(code=CUDA_SIM_KERNEL)
        self.raycast_kernel = module.get_function('raycast_kernel')
        self.physics_kernel = module.get_function('physics_kernel')
        self.battery_kernel = module.get_function('battery_kernel')
        self.mega_kernel = module.get_function('mega_kernel')
        
        self.threads_per_block = 256
        self.blocks_per_grid = (self.n_robots + self.threads_per_block - 1) // self.threads_per_block
        self.bat_blocks_per_grid = (self.n_bats + self.threads_per_block - 1) // self.threads_per_block
        
        self.n_col = n_col
        self.n_hun = n_hun

    def run_generation(self, steps: int):
        import math
        c_ray_len = np.float32(self.config.collector_sensor_ray_length)
        c_fov = np.float32(math.radians(self.config.collector_sensor_fov))
        h_ray_len = np.float32(self.config.hunter_sensor_ray_length)
        h_fov = np.float32(math.radians(self.config.hunter_sensor_fov))
        
        bat_radius = np.float32(12.0) # Fix 4: Battery.RADIUS = 12
        
        c_speed = np.float32(self.config.collector_speed)
        h_speed = np.float32(self.config.hunter_speed)
        w_width = np.float32(self.config.window_width)
        w_height = np.float32(self.config.window_height)
        
        fit_idle = np.float32(self.config.fitness_idle_penalty)
        fit_surv = np.float32(self.config.fitness_survival_bonus)
        fit_bat = np.float32(self.config.fitness_battery_collected)
        fit_prox = np.float32(self.config.fitness_battery_proximity)
        fit_danger = np.float32(max(self.config.fitness_hunter_danger, 0.15))
        fit_appr = np.float32(max(getattr(self.config, 'fitness_hunter_approach_penalty', 15.0), 15.0))
        fit_kill = np.float32(self.config.fitness_hunter_kill)
        fit_eaten = np.float32(self.config.fitness_eaten_penalty)
        
        energy_drain = np.float32(self.config.energy_drain_per_frame)
        bat_energy = np.float32(self.config.battery_energy)
        energy_start = np.float32(self.config.energy_start)
        
        danger_zone = np.float32(self.config.fitness_danger_zone)
        c_prox_zone = np.float32(self.config.collector_sensor_ray_length)
        h_prox_zone = np.float32(self.config.hunter_sensor_ray_length)
        
        # Pre-slice arrays for networks
        d_c_inputs = self.d_sensor_inputs[:self.n_col * self.n_inputs]
        d_h_inputs = self.d_sensor_inputs[self.n_col * self.n_inputs : self.n_robots * self.n_inputs]
        
        for _ in range(steps):
            # 1. Raycast
            self.raycast_kernel(
                (self.blocks_per_grid,), (self.threads_per_block,),
                (np.int32(self.n_robots),
                 self.d_r_x, self.d_r_y, self.d_r_angle, self.d_r_alive, self.d_r_type, self.d_r_radius,
                 np.int32(self.n_bats),
                 self.d_b_x, self.d_b_y, self.d_b_active, bat_radius,
                 np.int32(self.n_walls),
                 self.d_w_x1, self.d_w_y1, self.d_w_x2, self.d_w_y2,
                 self.d_sensor_inputs,
                 np.int32(self.config.sensor_ray_count),
                 c_ray_len, c_fov, h_ray_len, h_fov)
            )
            
            # 2. Neural Nets
            if self.collector_net:
                c_out = self.collector_net.activate_batch_device(d_c_inputs)
                # Ensure the cupy operation finishes (actually it's queued in stream so it's fine)
                # c_out is a view of d_outputs_batch of collector_net. We need to copy it to our d_motor_outputs
                cp.copyto(self.d_motor_outputs[:self.n_col * self.n_outputs], c_out.flatten())
                
            if self.hunter_net:
                h_out = self.hunter_net.activate_batch_device(d_h_inputs)
                cp.copyto(self.d_motor_outputs[self.n_col * self.n_outputs:], h_out.flatten())
                
            # 3. Physics & Interaction
            self.physics_kernel(
                (self.blocks_per_grid,), (self.threads_per_block,),
                (np.int32(self.n_robots),
                 self.d_r_x, self.d_r_y, self.d_r_angle, self.d_r_alive, self.d_r_type, self.d_r_radius,
                 self.d_r_energy, self.d_r_fitness,
                 self.d_motor_outputs,
                 np.int32(self.n_bats),
                 self.d_b_x, self.d_b_y, self.d_b_active, bat_radius,
                 np.int32(self.n_walls),
                 self.d_w_x1, self.d_w_y1, self.d_w_x2, self.d_w_y2,
                 np.int32(self.n_obstacles),
                 self.d_obs_left, self.d_obs_top, self.d_obs_right, self.d_obs_bottom,
                 self.d_prev_hunter_dists,
                 c_speed, h_speed, w_width, w_height,
                 fit_idle, fit_surv, fit_bat, fit_prox,
                 fit_danger, fit_appr, fit_kill, fit_eaten,
                 energy_drain, bat_energy, energy_start,
                 danger_zone, c_prox_zone, h_prox_zone,
                 self.d_rnd_x, self.d_rnd_y, np.int32(self.max_rnd), self.d_rnd_counter,
                 self.d_stats_kills, self.d_stats_bats,
                 np.int32(self.config.battery_respawn_delay), self.d_b_timer,
                 self.d_r_eaten, self.d_indiv_kills, self.d_indiv_bats)
            )
            
            # 4. Battery Respawn Timer (GPU Kernel, keine Host-Syncs mehr!)
            if self.n_bats > 0:
                self.battery_kernel(
                    (self.bat_blocks_per_grid,), (self.threads_per_block,),
                    (np.int32(self.n_bats),
                     self.d_b_x, self.d_b_y, self.d_b_active, self.d_b_timer,
                     self.d_rnd_x, self.d_rnd_y, np.int32(self.max_rnd), self.d_rnd_counter)
                )
            
        # End of generation, return fitnesses
        final_fitness = self.d_r_fitness.get()
        final_alive = self.d_r_alive.get()
        final_bats_active = self.d_b_active.get()
        kills = int(self.d_stats_kills.get()[0])
        bats = int(self.d_stats_bats.get()[0])
        
        # Race-Condition Fix: eaten_penalty im CPU Post-Processing
        # Nur Collectors die TATSÄCHLICH gefressen wurden (r_eaten == 1) bekommen die Strafe!
        eaten_penalty_val = float(self.config.fitness_eaten_penalty)
        final_eaten = self.d_r_eaten.get()
        indiv_kills = self.d_indiv_kills.get()
        indiv_bats = self.d_indiv_bats.get()
        
        final_b_x = self.d_b_x.get()
        final_b_y = self.d_b_y.get()
        final_b_timer = self.d_b_timer.get()
        
        for i in range(self.n_col):
            if final_eaten[i] == 1:
                final_fitness[i] += eaten_penalty_val
        
        return (final_fitness[:self.n_col], final_fitness[self.n_col:], 
                final_alive, final_bats_active, kills, bats,
                indiv_kills, indiv_bats, final_b_x, final_b_y, final_b_timer)

    def run_generation_mega(self, steps: int):
        import cupy as cp
        
        c_net = self.collector_net
        h_net = self.hunter_net
        
        c_genome_start_nodes = c_net.d_genome_start_nodes
        c_genome_num_nodes = c_net.d_genome_num_nodes
        c_flat_node_biases = c_net.d_flat_node_biases
        c_flat_node_responses = c_net.d_flat_node_responses
        c_flat_node_act_types = c_net.d_flat_node_act_types
        c_flat_value_indices = c_net.d_flat_value_indices
        c_flat_link_from_idx = c_net.d_flat_link_from_idx
        c_flat_link_weights = c_net.d_flat_link_weights
        c_flat_link_ranges_start = c_net.d_flat_link_ranges_start
        c_flat_link_ranges_end = c_net.d_flat_link_ranges_end
        c_genome_start_outputs = c_net.d_genome_start_outputs
        c_genome_num_outputs = c_net.d_genome_num_outputs
        c_flat_output_indices = c_net.d_flat_output_indices
        c_values_batch = c_net.d_values_batch

        h_genome_start_nodes = h_net.d_genome_start_nodes
        h_genome_num_nodes = h_net.d_genome_num_nodes
        h_flat_node_biases = h_net.d_flat_node_biases
        h_flat_node_responses = h_net.d_flat_node_responses
        h_flat_node_act_types = h_net.d_flat_node_act_types
        h_flat_value_indices = h_net.d_flat_value_indices
        h_flat_link_from_idx = h_net.d_flat_link_from_idx
        h_flat_link_weights = h_net.d_flat_link_weights
        h_flat_link_ranges_start = h_net.d_flat_link_ranges_start
        h_flat_link_ranges_end = h_net.d_flat_link_ranges_end
        h_genome_start_outputs = h_net.d_genome_start_outputs
        h_genome_num_outputs = h_net.d_genome_num_outputs
        h_flat_output_indices = h_net.d_flat_output_indices
        h_values_batch = h_net.d_values_batch
        import math
        c_ray_len = cp.float32(self.config.collector_sensor_ray_length)
        c_fov = cp.float32(math.radians(self.config.collector_sensor_fov))
        h_ray_len = cp.float32(self.config.hunter_sensor_ray_length)
        h_fov = cp.float32(math.radians(self.config.hunter_sensor_fov))
        
        bat_radius = cp.float32(12.0)
        
        c_speed = cp.float32(self.config.collector_speed)
        h_speed = cp.float32(self.config.hunter_speed)
        w_width = cp.float32(self.config.window_width)
        w_height = cp.float32(self.config.window_height)
        
        fit_idle = cp.float32(self.config.fitness_idle_penalty)
        fit_surv = cp.float32(self.config.fitness_survival_bonus)
        fit_bat = cp.float32(self.config.fitness_battery_collected)
        fit_prox = cp.float32(self.config.fitness_battery_proximity)
        fit_danger = cp.float32(max(self.config.fitness_hunter_danger, 0.15))
        fit_appr = cp.float32(max(getattr(self.config, 'fitness_hunter_approach_penalty', 15.0), 15.0))
        fit_kill = cp.float32(self.config.fitness_hunter_kill)
        fit_eaten = cp.float32(self.config.fitness_eaten_penalty)
        
        energy_drain = cp.float32(self.config.energy_drain_per_frame)
        bat_energy = cp.float32(self.config.battery_energy)
        energy_start = cp.float32(self.config.energy_start)
        
        danger_zone = cp.float32(self.config.fitness_danger_zone)
        c_prox_zone = cp.float32(self.config.collector_sensor_ray_length)
        h_prox_zone = cp.float32(self.config.hunter_sensor_ray_length)

        block = (256, 1, 1)
        grid = (1, 1, 1)

        self.mega_kernel(
            grid, block,
            (
                cp.int32(steps),
                
                cp.int32(self.n_robots), self.d_r_x, self.d_r_y, self.d_r_angle, self.d_r_alive, self.d_r_type, self.d_r_radius, self.d_r_energy, self.d_r_fitness, self.d_motor_outputs,
                cp.int32(self.n_bats), self.d_b_x, self.d_b_y, self.d_b_active, bat_radius,
                cp.int32(self.n_walls), self.d_w_x1, self.d_w_y1, self.d_w_x2, self.d_w_y2,
                cp.int32(self.n_obstacles), self.d_obs_left, self.d_obs_top, self.d_obs_right, self.d_obs_bottom,
                self.d_sensor_inputs, cp.int32(self.config.sensor_ray_count), c_ray_len, c_fov, h_ray_len, h_fov,
                self.d_prev_hunter_dists,
                c_speed, h_speed, w_width, w_height,
                fit_idle, fit_surv, fit_bat, fit_prox, fit_danger, fit_appr, fit_kill, fit_eaten,
                energy_drain, bat_energy, energy_start, danger_zone, c_prox_zone, h_prox_zone,
                self.d_rnd_x, self.d_rnd_y, cp.int32(self.max_rnd), self.d_rnd_counter,
                self.d_stats_kills, self.d_stats_bats, cp.int32(self.config.battery_respawn_delay), self.d_b_timer, self.d_r_eaten, self.d_indiv_kills, self.d_indiv_bats,
                
                cp.int32(c_net.n_genomes), c_genome_start_nodes, c_genome_num_nodes, c_flat_node_biases, c_flat_node_responses, c_flat_node_act_types, c_flat_value_indices, c_flat_link_from_idx, c_flat_link_weights, c_flat_link_ranges_start, c_flat_link_ranges_end, c_genome_start_outputs, c_genome_num_outputs, c_flat_output_indices, c_values_batch, cp.int32(c_net.max_values),
                
                cp.int32(h_net.n_genomes), h_genome_start_nodes, h_genome_num_nodes, h_flat_node_biases, h_flat_node_responses, h_flat_node_act_types, h_flat_value_indices, h_flat_link_from_idx, h_flat_link_weights, h_flat_link_ranges_start, h_flat_link_ranges_end, h_genome_start_outputs, h_genome_num_outputs, h_flat_output_indices, h_values_batch, cp.int32(h_net.max_values),
                
                cp.int32(c_net.n_inputs), cp.int32(c_net.n_outputs)
            )
        )
        
        c_fits = self.d_r_fitness[:self.n_col].get()
        h_fits = self.d_r_fitness[self.n_col:].get()
        alive_arr = self.d_r_alive.get()
        bats_active = self.d_b_active.get()
        kills = int(self.d_stats_kills.get()[0])
        bats = int(self.d_stats_bats.get()[0])
        indiv_kills = self.d_indiv_kills.get()
        indiv_bats = self.d_indiv_bats.get()
        bat_x = self.d_b_x.get()
        bat_y = self.d_b_y.get()
        bat_timer = self.d_b_timer.get()

        eaten_penalty_val = float(self.config.fitness_eaten_penalty)
        final_eaten = self.d_r_eaten.get()
        for i in range(self.n_col):
            if final_eaten[i] == 1:
                c_fits[i] += eaten_penalty_val

        return c_fits, h_fits, alive_arr, bats_active, kills, bats, indiv_kills, indiv_bats, bat_x, bat_y, bat_timer
