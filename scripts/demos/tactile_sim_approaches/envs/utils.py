import datetime
import json
import time
import traceback
from pathlib import Path

import carb
import numpy as np
import psutil
import pynvml
import torch
from envs.ball_roll_base_env import BallRollingEnv
from envs.ball_roll_base_env_cfg import BallRollingEnvCfg


def _get_utilization_percentages(reset: bool = False, max_values: list[float] = [0.0, 0.0, 0.0, 0.0]) -> list[float]:
    """System diagnosis: Get the maximum CPU, RAM, GPU utilization (processing), and
    GPU memory usage percentages since the last env reset.

    Adapted from benchmark_cameras.py script of IsaacLab
    """
    if reset:
        max_values[:] = [0, 0, 0, 0]  # Reset the max values

    # # CPU utilization
    # cpu_usage = psutil.cpu_percent(interval=0.1) # blocking slows down Isaac Sim a lot
    cpu_usage = psutil.cpu_percent(interval=None)
    max_values[0] = max(max_values[0], cpu_usage)

    # # RAM utilization
    memory_info = psutil.virtual_memory()
    ram_usage = memory_info.percent
    max_values[1] = max(max_values[1], ram_usage)

    # GPU utilization using pynvml
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # GPU Utilization
            gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_processing_utilization_percent = gpu_utilization.gpu  # GPU core utilization
            max_values[2] = max(max_values[2], gpu_processing_utilization_percent)

            # GPU Memory Usage
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_total = memory_info.total
            gpu_memory_used = memory_info.used
            gpu_memory_utilization_percent = (gpu_memory_used / gpu_memory_total) * 100
            max_values[3] = max(max_values[3], gpu_memory_utilization_percent)
    else:
        gpu_processing_utilization_percent = None
        gpu_memory_utilization_percent = None

    return max_values


def run_simulator(env: BallRollingEnv, simulation_app):
    """Runs the simulation loop."""

    # we directly turn on debug_vis for convenience
    if env.cfg.gsmini.debug_vis:
        for data_type in env.cfg.gsmini.data_types:
            env.gsmini._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)

    # for time measurements
    timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H_%M_%S}"
    output_dir = Path(__file__).parent.resolve()
    file_name = str(output_dir) + f"/envs_{env.num_envs}_{timestamp}.txt"

    frame_times_physics = []
    frame_times_tactile = []

    save_after_num_resets = 10
    current_num_resets = 0
    # GPU utilization using pynvml
    if torch.cuda.is_available():
        pynvml.nvmlInit()
    system_utilization_analytics = _get_utilization_percentages(reset=False)

    print(f"Starting simulation with {env.num_envs} envs")
    print("Number of steps till reset: ", len(env.pattern_offsets) * env.num_step_goal_change)

    total_sim_time = time.time()
    # Simulation loop
    while simulation_app.is_running():
        # get_utilization_percentages()
        # reset at the beginning of the simulation and after doing pattern 2 times
        if (
            env.step_count % (len(env.pattern_offsets) * env.num_step_goal_change) == 0
        ):  # reset after 900 steps, cause every 50 steps we change action -> pattern once = 450 steps
            print(f"[INFO]: Env reset num {current_num_resets}")
            env.reset()
            system_utilization_analytics = _get_utilization_percentages(reset=True)

            if len(frame_times_physics) != 0:
                print("Current total amount of 'in-contact' frames per env: ", len(frame_times_physics))
                print(f"Total sim time currently: {time.time() - total_sim_time:8.4f}ms")
                print(
                    f"Avg physics_sim time per env:    {np.mean(np.array(frame_times_physics) / env.num_envs):8.4f}ms"
                )
                print(
                    f"Avg tactile_sim time per env:    {np.mean(np.array(frame_times_tactile) / env.num_envs):8.4f}ms"
                )
                print(
                    f"| CPU:{system_utilization_analytics[0]}% | "
                    f"RAM:{system_utilization_analytics[1]}% | "
                    f"GPU Compute:{system_utilization_analytics[2]}% | "
                    f"GPU Memory: {system_utilization_analytics[3]:.2f}% |"
                )
                print()

                # write down the average simulation times
                if current_num_resets == save_after_num_resets:
                    print("*" * 15)
                    print("Writing performance data into ", file_name)
                    print("*" * 15)
                    with open(file_name, "a+") as f:
                        f.write("Sensor Config: \n")
                        f.write(json.dumps(env.cfg.gsmini.to_dict(), indent=2))
                        f.write("\n")
                        f.write("GPU Info: \n")
                        f.write(f"Name: {pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))} \n")
                        f.write(f"Driver: {pynvml.nvmlSystemGetDriverVersion()} \n")
                        f.write("\n")
                        f.write("Performance data: \n")
                        f.write(f"num_envs: {env.num_envs} \n")
                        f.write(
                            f"Total amount of 'in-contact' frames per env (ran pattern {current_num_resets} times):"
                            f" {len(frame_times_physics)}\n"
                        )
                        f.write(f"Total sim time: {time.time() - total_sim_time:8.4f}ms \n")
                        f.write(
                            "Avg physics_sim time for one frame per env:   "
                            f" {np.mean(np.array(frame_times_physics) / env.num_envs):8.4f}ms \n"
                        )
                        f.write(
                            "Avg tactile_sim time for one frame per env:   "
                            f" {np.mean(np.array(frame_times_tactile) / env.num_envs):8.4f}ms \n"
                        )
                        f.write("\n")
                        f.write(
                            f"| CPU:{system_utilization_analytics[0]}% | "
                            f"RAM:{system_utilization_analytics[1]}% | "
                            f"GPU Compute:{system_utilization_analytics[2]}% | "
                            f"GPU Memory: {system_utilization_analytics[3]:.2f}% |"
                        )
                        f.write("\n")
                    frame_times_physics = []
                    frame_times_tactile = []
                    break
            current_num_resets += 1

        # perform physics step
        physics_start = time.time()
        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        # render scene for cameras (used by sensor)
        env.sim.render()
        # update isaac buffers() -> also updates sensors, if lazy_sensor_update is false
        env.scene.update(dt=env.physics_dt)
        physics_end = time.time()

        # update sensors
        tactile_sim_start = time.time()
        env.gsmini.update(dt=env.physics_dt, force_recompute=True)
        # tactile_rgb = env.gsmini.data.output["tactile_rgb"]
        tactile_sim_end = time.time()

        # - add frame times, if sensor was in contact
        # todo currently assumed that every env was in contact. Need to filter out the envs where no contact
        (contact_idx,) = torch.where(env.gsmini._indentation_depth > 0)
        if contact_idx.shape[0] != 0:
            frame_times_physics.append(1000 * (physics_end - physics_start))
            frame_times_tactile.append(1000 * (tactile_sim_end - tactile_sim_start))
    env.close()

    pynvml.nvmlShutdown()
