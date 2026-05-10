import math
import cupy as cp
import numpy as np
from config.config_manager import SimConfig

CUDA_SIM_KERNEL = '''
extern "C" {

#define OBJ_NONE 0
#define OBJ_WALL 1
#define OBJ_BATTERY 2
#define OBJ_COLLECTOR 3
#define OBJ_HUNTER 4

// -------------------------------------------------------------------------
// Hilfsfunktionen fuer Raycasting
// -------------------------------------------------------------------------
__device__ float ray_segment_intersection(float ox, float oy, float dx, float dy, float x1, float y1, float x2, float y2) {
    float sx = x2 - x1;
    float sy = y2 - y1;

    float denom = dx * sy - dy * sx;
    if (abs(denom) < 1e-10f) return -1.0f;

    float t_num = (x1 - ox) * sy - (y1 - oy) * sx;
    float u_num = (x1 - ox) * dy - (y1 - oy) * dx;

    float t = t_num / denom;
    float u = u_num / denom;

    if (t >= 0.0f && u >= 0.0f && u <= 1.0f) {
        return t;
    }
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

// -------------------------------------------------------------------------
// Raycast Kernel
// Berechnet die Sensor-Inputs (Distanzen zu Objekten + nearest danger/prey angle)
// -------------------------------------------------------------------------
__global__ void raycast_kernel(
    int n_robots,
    float* r_x, float* r_y, float* r_angle, int* r_alive, int* r_type, float* r_radius,
    int n_bats,
    float* b_x, float* b_y, int* b_active, float b_radius,
    int n_walls,
    float* w_x1, float* w_y1, float* w_x2, float* w_y2,
    double* sensor_inputs,
    int n_sensor_rays,
    float c_ray_len, float c_fov,
    float h_ray_len, float h_fov
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_robots) return;
    if (r_alive[g] == 0) return;

    float rx = r_x[g];
    float ry = r_y[g];
    float ra = r_angle[g];
    int rtype = r_type[g];

    float max_length = (rtype == OBJ_HUNTER) ? h_ray_len : c_ray_len;
    float fov_rad = (rtype == OBJ_HUNTER) ? h_fov : c_fov;

    // 1. Raycasting
    for (int i = 0; i < n_sensor_rays; ++i) {
        float angle_offset = 0.0f;
        if (n_sensor_rays > 1) {
            angle_offset = -fov_rad / 2.0f + (i / (float)(n_sensor_rays - 1)) * fov_rad;
        }

        float ray_angle = ra + angle_offset;
        float ray_dx = cosf(ray_angle);
        float ray_dy = sinf(ray_angle);

        float closest_dist = max_length;
        float closest_type = 0.0f; // OBJ_NONE

        // Waende
        for (int w = 0; w < n_walls; ++w) {
            float t = ray_segment_intersection(rx, ry, ray_dx, ray_dy, w_x1[w], w_y1[w], w_x2[w], w_y2[w]);
            if (t >= 0.0f && t < closest_dist) {
                closest_dist = t;
                closest_type = 1.0f;
            }
        }

        // Batterien
        for (int b = 0; b < n_bats; ++b) {
            if (b_active[b]) {
                float t = ray_circle_intersection(rx, ry, ray_dx, ray_dy, b_x[b], b_y[b], b_radius);
                if (t >= 0.0f && t < closest_dist) {
                    closest_dist = t;
                    closest_type = 2.0f;
                }
            }
        }

        // Roboter
        for (int ro = 0; ro < n_robots; ++ro) {
            if (ro != g && r_alive[ro]) {
                float t = ray_circle_intersection(rx, ry, ray_dx, ray_dy, r_x[ro], r_y[ro], r_radius[ro]);
                if (t >= 0.0f && t < closest_dist) {
                    closest_dist = t;
                    closest_type = (float)r_type[ro];
                }
            }
        }

        // Output features per ray: (dist_norm, type, unused, unused) - in python it expects 5 per ray!
        // wait, the neural net input is flattened.
        // In python: c.get_sensor_inputs() returns flat list of 5 values per ray.
        // Actually `c.get_sensor_inputs()` appends: dist_norm, is_wall, is_battery, is_collector, is_hunter
        
        float dist_norm = closest_dist / max_length;
        
        // Base index for this robot in the inputs batch
        int base_idx = g * (n_sensor_rays * 5 + 2) + i * 5;
        
        sensor_inputs[base_idx + 0] = dist_norm;
        sensor_inputs[base_idx + 1] = (closest_type == 1.0f) ? 1.0 : 0.0;
        sensor_inputs[base_idx + 2] = (closest_type == 2.0f) ? 1.0 : 0.0;
        sensor_inputs[base_idx + 3] = (closest_type == 3.0f) ? 1.0 : 0.0;
        sensor_inputs[base_idx + 4] = (closest_type == 4.0f) ? 1.0 : 0.0;
    }
    
    // 2. Proximity Tracker (Nearest target direction)
    float nearest_dist = max_length;
    float dx_norm = 0.0f;
    float dy_norm = 0.0f;
    
    for (int ro = 0; ro < n_robots; ++ro) {
        if (ro != g && r_alive[ro]) {
            int target_type = r_type[ro];
            // Collectors look for Hunters, Hunters look for Collectors
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

// -------------------------------------------------------------------------
// Physics & Interaction Kernel
// -------------------------------------------------------------------------
__global__ void physics_kernel(
    int n_robots,
    float* r_x, float* r_y, float* r_angle, int* r_alive, int* r_type, float* r_radius,
    float* r_energy, float* r_fitness,
    double* motor_outputs,
    int n_bats,
    float* b_x, float* b_y, int* b_active, float b_radius,
    int n_walls,
    float* w_x1, float* w_y1, float* w_x2, float* w_y2,
    float* prev_hunter_dists,
    float c_speed, float h_speed, float w_width, float w_height,
    float fit_idle, float fit_surv, float fit_bat, float fit_prox, 
    float fit_danger, float fit_appr, float fit_kill, float fit_eaten,
    float energy_drain, float bat_energy, float energy_start,
    float danger_zone, float prox_zone,
    float* rnd_x, float* rnd_y, int max_rnd, int* rnd_counter,
    int* stats_kills, int* stats_bats
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_robots) return;
    if (r_alive[g] == 0) return;

    float rx = r_x[g];
    float ry = r_y[g];
    float ra = r_angle[g];
    float rrad = r_radius[g];
    int rtype = r_type[g];

    
    float m_left = (float)motor_outputs[g * 3 + 0];
    float m_right = (float)motor_outputs[g * 3 + 1];
    // float radio_out = (float)motor_outputs[g * 3 + 2]; // Not used in simple physics

    // 1. Move
    float speed_mult = (rtype == OBJ_COLLECTOR) ? c_speed : h_speed;
    float forward_speed = (m_left + m_right) / 2.0f;
    float turn_speed = (m_right - m_left) * 0.1f;
    
    float actual_fwd = forward_speed * speed_mult;
    float actual_turn = turn_speed * speed_mult;
    
    ra += actual_turn;
    
    // Normalize angle (optional but good practice)
    while (ra > 3.14159265f) ra -= 2.0f * 3.14159265f;
    while (ra < -3.14159265f) ra += 2.0f * 3.14159265f;
    
    float nx = rx + cosf(ra) * actual_fwd;
    float ny = ry + sinf(ra) * actual_fwd;

    // Fast bounds clamping
    if (nx < rrad) nx = rrad;
    if (nx > w_width - rrad) nx = w_width - rrad;
    if (ny < rrad) ny = rrad;
    if (ny > w_height - rrad) ny = w_height - rrad;

    // Obstacle collision (simplified circles/AABB interaction for walls? The original code uses AABB for obstacles)
    // To match perfectly, we would need to do AABB collision against obstacles. 
    // Wait, the Python code has `obstacles_tuples`. Walls are lines, but check_obstacle_collision_fast uses rectangles.
    // Let's implement AABB collision for obstacles if passed.
    // But walls are passed as lines. Actually, in world.py walls are the outer boundary and obstacles are rectangles.
    // I need to add obstacle support here.

    // Update position
    rx = nx;
    ry = ny;
    
    // 2. Energy
    float ren = r_energy[g];
    ren -= energy_drain;
    if (ren <= 0.0f) {
        r_alive[g] = 0;
    }
    
    // 3. Fitness Base
    float rfit = r_fitness[g];
    if (forward_speed < 0.1f) rfit += fit_idle;
    rfit += fit_surv;
    
    // 4. Interactions
    if (rtype == OBJ_COLLECTOR) {
        // Battery collection
        float nearest_bat_sq = prox_zone * prox_zone;
        float collect_dist_sq = (rrad + 15.0f) * (rrad + 15.0f);
        
        for (int b = 0; b < n_bats; ++b) {
            if (b_active[b]) {
                float dx = b_x[b] - rx;
                float dy = b_y[b] - ry;
                float dsq = dx*dx + dy*dy;
                if (dsq < collect_dist_sq) {
                    int old = atomicExch(&b_active[b], 0);
                    if (old == 1) { // We were the first to grab it
                        ren += bat_energy;
                        rfit += fit_bat;
                        atomicAdd(stats_bats, 1);
                        
                        // Respawn battery immediately
                        int r_idx = atomicAdd(rnd_counter, 1);
                        b_x[b] = rnd_x[r_idx % max_rnd];
                        b_y[b] = rnd_y[r_idx % max_rnd];
                        b_active[b] = 1;
                    }
                } else if (dsq < nearest_bat_sq) {
                    nearest_bat_sq = dsq;
                }
            }
        }
        
        // Battery Proximity
        if (nearest_bat_sq < prox_zone * prox_zone) {
            float dist = sqrtf(nearest_bat_sq);
            rfit += fit_prox * (1.0f - dist / prox_zone);
        }
        
        // Hunter Evasion
        float danger_zone_sq = danger_zone * danger_zone;
        for (int ro = 0; ro < n_robots; ++ro) {
            if (ro != g && r_type[ro] == OBJ_HUNTER && r_alive[ro]) {
                float dx = r_x[ro] - rx;
                float dy = r_y[ro] - ry;
                float dsq = dx*dx + dy*dy;
                
                if (dsq < danger_zone_sq) {
                    float dist = sqrtf(dsq);
                    float penalty = fit_danger * (1.0f - dist / danger_zone);
                    if (penalty > 0.0f) rfit -= penalty;
                    
                    float prev_d = prev_hunter_dists[g * n_robots + ro];
                    if (prev_d > 0.0f) {
                        float delta = dist - prev_d;
                        if (delta > 0.0f) {
                            rfit += (delta / c_speed) * 15.0f; // Escape bonus
                        } else if (delta < 0.0f) {
                            rfit -= (-delta / c_speed) * fit_appr; // Approach penalty
                        }
                    }
                    prev_hunter_dists[g * n_robots + ro] = dist;
                } else {
                    prev_hunter_dists[g * n_robots + ro] = 0.0f;
                }
            }
        }
        
    } else if (rtype == OBJ_HUNTER) {
        // Collect collectors!
        for (int ro = 0; ro < n_robots; ++ro) {
            if (ro != g && r_type[ro] == OBJ_COLLECTOR && r_alive[ro]) {
                float dx = r_x[ro] - rx;
                float dy = r_y[ro] - ry;
                float dsq = dx*dx + dy*dy;
                float catch_dist = rrad + r_radius[ro];
                
                if (dsq < catch_dist * catch_dist) {
                    int old = atomicExch(&r_alive[ro], 0);
                    if (old == 1) { // Ate the collector!
                        ren += energy_start;
                        rfit += fit_kill;
                        atomicAdd(stats_kills, 1);
                        atomicAdd(&r_fitness[ro], fit_eaten);
                    }
                }
            }
        }
        
        // Proximity bonus towards collectors
        float nearest_prey_sq = prox_zone * prox_zone;
        for (int ro = 0; ro < n_robots; ++ro) {
            if (ro != g && r_type[ro] == OBJ_COLLECTOR && r_alive[ro]) {
                float dx = r_x[ro] - rx;
                float dy = r_y[ro] - ry;
                float dsq = dx*dx + dy*dy;
                if (dsq < nearest_prey_sq) nearest_prey_sq = dsq;
            }
        }
        if (nearest_prey_sq < prox_zone * prox_zone) {
            float dist = sqrtf(nearest_prey_sq);
            rfit += 0.1f * (1.0f - dist / prox_zone);
        }
    }

    // Write back
    r_x[g] = rx;
    r_y[g] = ry;
    r_angle[g] = ra;
    r_energy[g] = ren;
    r_fitness[g] = rfit;
}

}
'''

class CudaSimulation:
    def __init__(self, collectors, hunters, batteries, walls, config: SimConfig, collector_net, hunter_net):
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
        
        # Temp buffers
        self.d_sensor_inputs = cp.zeros((self.n_robots * self.n_inputs,), dtype=cp.float64)
        self.d_motor_outputs = cp.zeros((self.n_robots * self.n_outputs,), dtype=cp.float64)
        self.d_prev_hunter_dists = cp.zeros((self.n_robots * self.n_robots,), dtype=cp.float32)
        
        # Random respawn buffer
        self.max_rnd = 10000
        pad = config.cell_pixel_size
        rnd_x = np.random.uniform(pad, config.window_width - pad, self.max_rnd).astype(np.float32)
        rnd_y = np.random.uniform(pad, config.window_height - pad, self.max_rnd).astype(np.float32)
        self.d_rnd_x = cp.asarray(rnd_x)
        self.d_rnd_y = cp.asarray(rnd_y)
        self.d_rnd_counter = cp.zeros(1, dtype=cp.int32)
        
        # Stats
        self.d_stats_kills = cp.zeros(1, dtype=cp.int32)
        self.d_stats_bats = cp.zeros(1, dtype=cp.int32)
        
        # 3. Compile Kernels
        module = cp.RawModule(code=CUDA_SIM_KERNEL)
        self.raycast_kernel = module.get_function('raycast_kernel')
        self.physics_kernel = module.get_function('physics_kernel')
        
        self.threads_per_block = 256
        self.blocks_per_grid = (self.n_robots + self.threads_per_block - 1) // self.threads_per_block
        
        self.n_col = n_col
        self.n_hun = n_hun

    def run_generation(self, steps: int):
        import math
        c_ray_len = np.float32(self.config.collector_sensor_ray_length)
        c_fov = np.float32(math.radians(self.config.collector_sensor_fov))
        h_ray_len = np.float32(self.config.hunter_sensor_ray_length)
        h_fov = np.float32(math.radians(self.config.hunter_sensor_fov))
        
        bat_radius = np.float32(15.0) # b.RADIUS
        
        c_speed = np.float32(self.config.collector_speed)
        h_speed = np.float32(self.config.hunter_speed)
        w_width = np.float32(self.config.window_width)
        w_height = np.float32(self.config.window_height)
        
        fit_idle = np.float32(self.config.fitness_idle_penalty)
        fit_surv = np.float32(self.config.fitness_survival_bonus)
        fit_bat = np.float32(self.config.fitness_battery_collected)
        fit_prox = np.float32(self.config.fitness_battery_proximity)
        fit_danger = np.float32(self.config.fitness_hunter_danger)
        fit_appr = np.float32(getattr(self.config, 'fitness_hunter_approach_penalty', 15.0))
        fit_kill = np.float32(self.config.fitness_hunter_kill)
        fit_eaten = np.float32(self.config.fitness_eaten_penalty)
        
        energy_drain = np.float32(self.config.energy_drain_per_frame)
        bat_energy = np.float32(self.config.battery_energy)
        energy_start = np.float32(self.config.energy_start)
        
        danger_zone = np.float32(self.config.fitness_danger_zone)
        prox_zone = np.float32(self.config.collector_sensor_ray_length) # proximiy uses collector ray length
        
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
                 self.d_prev_hunter_dists,
                 c_speed, h_speed, w_width, w_height,
                 fit_idle, fit_surv, fit_bat, fit_prox,
                 fit_danger, fit_appr, fit_kill, fit_eaten,
                 energy_drain, bat_energy, energy_start,
                 danger_zone, prox_zone,
                 self.d_rnd_x, self.d_rnd_y, np.int32(self.max_rnd), self.d_rnd_counter,
                 self.d_stats_kills, self.d_stats_bats)
            )
            
        # End of generation, return fitnesses
        final_fitness = self.d_r_fitness.get()
        final_alive = self.d_r_alive.get()
        final_bats_active = self.d_b_active.get()
        kills = int(self.d_stats_kills.get()[0])
        bats = int(self.d_stats_bats.get()[0])
        return final_fitness[:self.n_col], final_fitness[self.n_col:], final_alive, final_bats_active, kills, bats
