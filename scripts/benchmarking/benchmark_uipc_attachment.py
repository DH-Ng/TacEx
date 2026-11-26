"""Benchmark: UIPC gelpad attachment via rigid carrier body vs. soft position constraint, in a gripping task.

A Franka with two GelSight Minis on its gripper (UIPC gelpads) grips a 30 mm cube and lifts it. Compared:

- attachments: ``carrier`` (gel stitched to a UIPC affine body moved by the robot, ``UipcIsaacCarrierAttachments``)
  vs. ``soft_position`` (soft position constraint on the gel vertices, ``UipcIsaacAttachments``)
- objects: ``rigid`` (UIPC affine body) vs. ``soft`` (UIPC FEM body, StableNeoHookean, ``--soft_youngs_modulus``)

Own scene (no demo envs): the cube rests on the UIPC ground 5 cm below the open gripper. Phases: settle (the robot
moves smoothly from the zero joint configuration, in which the gelpads are authored in USD, to its start pose; the gels
follow without jumping, which would knock the cube away) -> approach (descend 5 cm, differential IK) -> close -> hold on the ground -> lift 5 cm -> hold in the air.

Every (attachment, object) runs in a fresh Isaac Sim process. Measured per run:

- runtime per step (physics + UIPC): mean / p95 / max, overall and per phase; setup time
- grasp: object lifted (object rise / gripper rise), slip of the object relative to the gripper during lift and
  hold [mm]
- attachment: max displacement of the gel undersides relative to their sensor case under load (they are bonded to the
  sensor and should not move), max attachment error (attached gel vertices vs. their targets)

Usage (Isaac Lab python):
    python scripts/benchmarking/benchmark_uipc_attachment.py
    python scripts/benchmarking/benchmark_uipc_attachment.py --attachments carrier --objects soft
    # watch one run live (GUI; carrier body shown in orange; timings not representative, no result files)
    python scripts/benchmarking/benchmark_uipc_attachment.py --debug_vis \
        --attachments carrier --objects soft

Results: ``scripts/benchmarking/tactile_sim_performance/logs/uipc_attachment/<timestamp>/``
(results.csv, results.json, summary.md, runs/*.log).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ATTACHMENTS = ["carrier", "soft_position"]
OBJECTS = ["rigid", "soft"]

DT = 0.01
CUBE_SIZE = 0.03
APPROACH_HEIGHT = 0.05
# Gel contact surfaces of the open gripper in the robot's start pose (from the case poses): centered at
# (0.4878, 0, 0.3615), lower gel edges at ~0.344. After the approach, the cube sits 8 mm below the gel centers, so
# the gels clear the ground (which would otherwise hold them back by friction) and still cover the cube's upper part.
CUBE_CENTER = (0.4878, 0.0, 0.3535 - APPROACH_HEIGHT)
GROUND_Z = CUBE_CENTER[2] - CUBE_SIZE / 2 - 5e-4  # gap: UIPC rejects initial contact with the ground
FINGER_OPEN = 0.04
FINGER_CLOSED = 0.012  # contact at ~0.0147 (30 mm cube, 80.6 mm gel gap) -> ~2.7 mm squeeze per finger
LIFT_HEIGHT = 0.05
SETTLE_MOVE_STEPS = 40  # of the settle phase: zero joint config -> start pose; the rest holds
PHASES = [("settle", 60), ("approach", 50), ("close", 40), ("hold_ground", 10), ("lift", 60), ("hold_air", 40)]
GEL_PRIMS = {"left": "/World/Robot/gsmini_gelpad_left", "right": "/World/Robot/gsmini_gelpad_right"}
CASE_BODIES = {"left": "gelsight_mini_case_left", "right": "gelsight_mini_case_right"}
UNDERSIDE_Z = 0.02425  # case frame: gel underside at 24 mm (contact face up to 28 mm)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attachments", nargs="+", default=ATTACHMENTS, choices=ATTACHMENTS)
    parser.add_argument("--objects", nargs="+", default=OBJECTS, choices=OBJECTS)
    parser.add_argument("--soft_youngs_modulus", type=float, default=0.05, help="Soft cube Young's modulus [MPa].")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per run before it is killed.")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "--debug_vis",
        action="store_true",
        help="Watch one run live in the GUI (one --attachments and one --objects value); no benchmark results.",
    )
    # worker mode (internal)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attachment", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--object", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=str, help=argparse.SUPPRESS)
    return parser.parse_args()


"""
Worker: one (attachment, object) run in its own Isaac Sim process.
"""


def run_worker(args):
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(
        headless=not args.debug_vis, enable_cameras=True
    ).app  # cameras: loads omni.ui (tacex_uipc)

    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils
    from isaaclab.assets import AssetBaseCfg

    from tacex import TacExRobot, TacExRobotCfg
    from tacex_assets.robots.franka.franka_gsmini_gripper_uipc import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_UIPC_CFG
    from tacex_uipc import UipcSim, UipcSimCfg
    from tacex_uipc.objects import (
        UipcDeformableObject,
        UipcDeformableObjectCfg,
        UipcIsaacAttachmentsCfg,
        UipcRigidObject,
        UipcRigidObjectCfg,
    )
    from tacex_uipc.objects.constraints.uipc_carrier_attachment import (
        UipcIsaacCarrierAttachments,
        UipcIsaacCarrierAttachmentsCfg,
    )
    from tacex_uipc.utils import TetMeshCfg

    result = {"attachment": args.attachment, "object": args.object, "status": "ok"}
    try:
        t_setup = time.perf_counter()
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cuda"))
        uipc_sim = UipcSim(
            UipcSimCfg(
                dt=DT,
                ground_height=GROUND_Z,
                contact=UipcSimCfg.Contact(d_hat=1e-4, default_friction_ratio=0.95),
                update_render_meshes=args.debug_vis,  # show the UIPC meshes (gels, cube) in the GUI
            )
        )
        robot = TacExRobot(
            TacExRobotCfg.from_articulation_cfg(
                FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_UIPC_CFG,
                prim_path="/World/Robot",
                ik_body_name="TCP",
                ik_joint_names=["panda_joint[1-7]"],
            )
        )

        # -- UIPC gelpads (attachment: soft position constraint on the gel, or carrier created after all objects)
        gel_mesh_cfg = TetMeshCfg(stop_quality=8, max_its=100, edge_length_r=1 / 20)
        gels = {}
        for side, prim in GEL_PRIMS.items():
            cfg = UipcDeformableObjectCfg(
                prim_path=prim, mesh_cfg=gel_mesh_cfg, constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg()
            )
            if args.attachment == "soft_position":
                cfg = cfg.replace(
                    constraint_cfg=UipcIsaacAttachmentsCfg(
                        constraint_strength_ratio=100.0,
                        body_name=CASE_BODIES[side],
                        isaaclab_rigid_body_prim_path="/World/Robot",
                    )
                )
            gels[side] = UipcDeformableObject(cfg)

        # -- cube on the UIPC ground at gripper height
        cube_color = (0.15, 0.35, 0.8) if args.object == "rigid" else (0.2, 0.65, 0.3)  # blue: rigid, green: soft
        cube_spawn = sim_utils.MeshCuboidCfg(
            size=(CUBE_SIZE,) * 3, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cube_color)
        )
        cube_init = AssetBaseCfg.InitialStateCfg(pos=CUBE_CENTER)
        cube_mesh_cfg = TetMeshCfg(stop_quality=8, max_its=100, edge_length_r=1 / 10)
        if args.object == "rigid":
            cube = UipcRigidObject(
                UipcRigidObjectCfg(
                    prim_path="/World/Cube", spawn=cube_spawn, init_state=cube_init, mesh_cfg=cube_mesh_cfg,
                    constitution_cfg=UipcRigidObjectCfg.AffineBodyConstitutionCfg(),
                ),
            )  # fmt: skip
        else:
            cube = UipcDeformableObject(
                UipcDeformableObjectCfg(
                    prim_path="/World/Cube", spawn=cube_spawn, init_state=cube_init, mesh_cfg=cube_mesh_cfg,
                    constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(youngs_modulus=args.soft_youngs_modulus),
                ),
            )  # fmt: skip

        # -- attachments
        attachments = {}
        for side, gel in gels.items():
            if args.attachment == "soft_position":
                gel.constraint.isaaclab_rigid_object = robot
                attachments[side] = gel.constraint
            else:  # carrier: created after all other UIPC objects
                attachments[side] = UipcIsaacCarrierAttachments(
                    UipcIsaacCarrierAttachmentsCfg(
                        isaaclab_rigid_body_prim_path="/World/Robot", body_name=CASE_BODIES[side], debug_vis=args.debug_vis
                    ),
                    gel, robot,
                )  # fmt: skip

        if args.debug_vis:
            # visuals only: Isaac's default ground at the robot base, a table block up to the UIPC ground (which is
            # what the cube rests on), soft grey lighting
            sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
            table = sim_utils.MeshCuboidCfg(
                size=(0.3, 0.4, GROUND_Z),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.3, 0.25), roughness=0.8),
            )
            table.func("/World/Table", table, translation=(CUBE_CENTER[0], 0.0, GROUND_Z / 2))
            dome = sim_utils.DomeLightCfg(intensity=800.0, color=(0.75, 0.75, 0.75))
            dome.func("/World/DebugVisDomeLight", dome)
            sun = sim_utils.DistantLightCfg(intensity=1500.0, color=(1.0, 0.97, 0.9), angle=1.0)
            sun.func("/World/DebugVisSunLight", sun, orientation=(0.88, 0.25, 0.35, 0.2))

        uipc_sim.setup_sim()
        sim.reset()
        if args.debug_vis:
            sim.set_camera_view(eye=(0.8, -0.32, 0.47), target=(CUBE_CENTER[0], 0.0, CUBE_CENTER[2] + 0.02))
        # start in the zero arm configuration (gelpads' USD pose), fingers open; settle moves to the start pose
        joint_pos = robot.data.default_joint_pos.clone()
        arm_ids = robot.find_joints("panda_joint[1-7]")[0]
        q_start = joint_pos[:, arm_ids].clone()
        joint_pos[:, arm_ids] = 0.0
        robot.write_joint_state_to_sim(joint_pos, robot.data.default_joint_vel.clone())
        result["setup_s"] = time.perf_counter() - t_setup

        # -- control: differential IK for the arm (TCP), position targets for the fingers
        tcp_id = robot.find_bodies("TCP")[0][0]
        finger_ids = robot.find_joints("panda_finger_joint.*")[0]
        case_ids = {side: robot.find_bodies(name)[0][0] for side, name in CASE_BODIES.items()}

        def tcp_pose():
            return robot.data.body_pos_w[:, tcp_id].clone(), robot.data.body_quat_w[:, tcp_id].clone()

        def case_frame(points_w: np.ndarray, side: str) -> np.ndarray:
            pos = robot.data.body_pos_w[0, case_ids[side]].cpu().numpy()
            R = math_utils.matrix_from_quat(robot.data.body_quat_w[0, case_ids[side]][None])[0].cpu().numpy()
            return (points_w - pos) @ R

        def attachment_error(side: str) -> float:
            att = attachments[side]
            nodes = gels[side]._data.nodal_pos_w.reshape(-1, 3).cpu().numpy()
            idx = np.asarray(att.attachment_points_idx)
            target = att.aim_positions() if callable(att.aim_positions) else att.aim_positions
            return float(np.linalg.norm(nodes[idx] - target, axis=1).max())

        def object_pos() -> np.ndarray:
            return cube.data.root_pos_w[0].cpu().numpy().copy()

        steps = []
        ref_case = {}  # gel surfaces in the case frame at the end of settling (no contact yet)
        tcp0_pos, tcp0_quat = None, None  # TCP pose at the start of the approach
        finger_target = FINGER_OPEN
        for phase, n in PHASES:
            if args.debug_vis:
                print(f"[debug_vis] phase: {phase} ({n} steps)")
            for k in range(n):
                # targets
                if phase == "settle":
                    a = min(1.0, (k + 1) / SETTLE_MOVE_STEPS)
                    joint_pos[:, arm_ids] = q_start * (0.5 - 0.5 * np.cos(np.pi * a))  # smooth ramp
                    # kinematic: the joint drives are too slow for this move; the gels follow in UIPC
                    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
                if phase == "close":
                    finger_target = FINGER_OPEN + (FINGER_CLOSED - FINGER_OPEN) * (k + 1) / n
                if phase != "settle":
                    if tcp0_pos is None:
                        tcp0_pos, tcp0_quat = tcp_pose()
                    dz = {
                        "approach": -APPROACH_HEIGHT * (k + 1) / n,
                        "lift": -APPROACH_HEIGHT + LIFT_HEIGHT * (k + 1) / n,
                        "hold_air": -APPROACH_HEIGHT + LIFT_HEIGHT,
                    }.get(phase, -APPROACH_HEIGHT)
                    root = robot.data.root_pose_w
                    target_b = math_utils.subtract_frame_transforms(
                        root[:, :3], root[:, 3:7], tcp0_pos + torch.tensor([[0.0, 0.0, dz]], device="cuda"), tcp0_quat
                    )
                    # arm joint targets are computed by IK in robot.write_data_to_sim()
                    robot.set_ee_pose_target(*target_b)
                joint_pos[:, finger_ids] = finger_target
                if phase == "settle":
                    robot.set_joint_position_target(joint_pos)
                else:
                    # only the fingers: the arm joint targets come from IK (set_ee_pose_target)
                    robot.set_joint_position_target(joint_pos[:, finger_ids], joint_ids=finger_ids)
                robot.write_data_to_sim()

                # step (physics + UIPC via physics callback)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                sim.step(render=args.debug_vis)
                torch.cuda.synchronize()
                step_s = time.perf_counter() - t0

                robot.update(DT)
                for gel in gels.values():
                    gel.update(DT)
                cube.update(DT)

                rec = {"phase": phase, "step_s": step_s, "object": object_pos(), "tcp": tcp_pose()[0][0].cpu().numpy()}
                for side, gel in gels.items():
                    rec[f"gel_min_z_{side}"] = float(gel._data.surf_nodal_pos_w.reshape(-1, 3)[:, 2].min())
                    surf = case_frame(gel._data.surf_nodal_pos_w.reshape(-1, 3).cpu().numpy(), side)
                    if phase == "settle" and k == n - 1:
                        ref_case[side] = surf
                    if side in ref_case and phase != "settle":
                        under = ref_case[side][:, 2] < UNDERSIDE_Z
                        rec[f"underside_{side}"] = float(
                            np.linalg.norm(surf[under] - ref_case[side][under], axis=1).max()
                        )
                    rec[f"att_err_{side}"] = attachment_error(side)
                steps.append(rec)

        # -- metrics
        def phase_steps(*names):
            return [s for s in steps if s["phase"] in names]

        loaded = phase_steps("close", "hold_ground", "lift", "hold_air")
        # runtime over the grasp; settling (robot moving to its start pose) is reported separately
        step_ms = np.array([s["step_s"] for s in loaded]) * 1e3
        lift_start = phase_steps("lift")[0]
        end = steps[-1]
        tcp_rise = end["tcp"][2] - lift_start["tcp"][2]
        obj_rise = end["object"][2] - lift_start["object"][2]
        rel0 = lift_start["object"] - lift_start["tcp"]
        slip = np.linalg.norm((end["object"] - end["tcp"]) - rel0)
        result.update(
            step_ms_mean=float(step_ms.mean()),
            step_ms_p95=float(np.percentile(step_ms, 95)),
            step_ms_max=float(step_ms.max()),
            total_s=float(step_ms.sum() / 1e3),
            **{f"step_ms_mean_{p}": float(np.mean([s["step_s"] for s in phase_steps(p)]) * 1e3) for p, _ in PHASES},
            tcp_rise_mm=float(tcp_rise * 1e3),
            object_rise_mm=float(obj_rise * 1e3),
            lifted=bool(obj_rise > 0.8 * tcp_rise),
            slip_mm=float(slip * 1e3),
            underside_max_mm=float(max(max(s[f"underside_{side}"] for side in gels) for s in loaded) * 1e3),
            attachment_err_max_mm=float(max(max(s[f"att_err_{side}"] for side in gels) for s in loaded) * 1e3),
            # during the grasp, the gels must not touch the ground (< d_hat): it would hold them back
            gel_ground_clearance_mm=float(
                (min(min(s[f"gel_min_z_{side}"] for side in gels) for s in loaded) - GROUND_Z) * 1e3
            ),
        )
        # per-step series (only in runs/<name>.json)
        if args.debug_vis:
            print(
                "[debug_vis] "
                + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in result.items())
            )
            print("[debug_vis] done; holding the last frame until the window is closed (timings include GUI rendering)")
            while simulation_app.is_running():
                sim.render()
        result["series"] = [
            {k: (v.round(5).tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()} for s in steps
        ]
    except Exception as e:  # report the failure instead of crashing without a result
        import traceback

        result.update(status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc())

    if args.debug_vis and result["status"] != "ok":
        print(result.get("trace"))
    if args.result:
        Path(args.result).write_text(json.dumps(result, indent=2))
    sys.stdout.flush()
    os._exit(0)  # Kit's close() can hang after errors; the process exit frees the GPU


"""
Driver: runs every configuration in its own process, then writes CSV/JSON/summary.
"""

COLUMNS = [
    "attachment", "object", "status", "step_ms_mean", "step_ms_p95", "step_ms_max", "total_s", "setup_s",
    "step_ms_mean_settle", "step_ms_mean_approach", "step_ms_mean_close", "step_ms_mean_hold_ground", "step_ms_mean_lift",
    "step_ms_mean_hold_air", "lifted", "object_rise_mm", "tcp_rise_mm", "slip_mm", "underside_max_mm",
    "attachment_err_max_mm", "gel_ground_clearance_mm", "error",
]  # fmt: skip


def run_driver(args):
    if args.debug_vis:
        if len(args.attachments) != 1 or len(args.objects) != 1:
            sys.exit("--debug_vis watches one run: pass exactly one --attachments and one --objects value.")
        args.attachment, args.object = args.attachments[0], args.objects[0]
        run_worker(args)
        return
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).resolve().parent
        / "logs"
        / "uipc_attachment"
        / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)

    results = []
    for obj in args.objects:
        for attachment in args.attachments:
            name = f"{attachment}_{obj}"
            result_file, log_file = out_dir / "runs" / f"{name}.json", out_dir / "runs" / f"{name}.log"
            cmd = [
                sys.executable, os.path.abspath(__file__), "--worker", "--attachment", attachment, "--object", obj,
                "--soft_youngs_modulus", str(args.soft_youngs_modulus), "--result", str(result_file),
            ]  # fmt: skip
            print(f"[{attachment:>13} | {obj:>5}] running ...", end=" ", flush=True)
            t0 = time.perf_counter()
            with open(log_file, "w") as log:
                try:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    pass
            if result_file.exists():
                result = json.loads(result_file.read_text())
            else:
                result = {"attachment": attachment, "object": obj, "status": "timeout/crash",
                          "error": f"no result, see {log_file.name}"}  # fmt: skip
            result["wall_s"] = time.perf_counter() - t0
            results.append(result)
            if result["status"] == "ok":
                print(f"{result['step_ms_mean']:6.1f} ms/step (max {result['step_ms_max']:6.0f}), lifted {result['lifted']},"
                      f" slip {result['slip_mm']:.2f} mm, underside {result['underside_max_mm']:.3f} mm")  # fmt: skip
            else:
                print(f"{result['status']}: {result.get('error', '')[:150]}")

    (out_dir / "results.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    _write_csv(out_dir / "results.csv", results)
    _write_summary(out_dir / "summary.md", args, results)
    print(f"\nResults: {out_dir}")


def _write_csv(path: Path, results: list[dict]):
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _write_summary(path: Path, args, results: list[dict]):
    lines = [
        "# UIPC gelpad attachment: carrier vs. soft position constraint (gripping)",
        "",
        f"Franka with two GelSight Minis grips and lifts a {CUBE_SIZE * 1e3:.0f} mm cube ({LIFT_HEIGHT * 1e3:.0f} mm);"
        f" soft cube: StableNeoHookean, E = {args.soft_youngs_modulus} MPa. Steps of {DT * 1e3:.0f} ms:"
        f" {', '.join(f'{p} {n}' for p, n in PHASES)}.",
        "",
        "Step time = physics + UIPC per step, over the grasp (close to hold_air; settle only in the per-phase table). Slip = change of the object position relative to the gripper (TCP) during"
        " lift and hold. Underside = max displacement of the gel undersides relative to their sensor case under load"
        " (bonded to the sensor, should stay ~0). Attachment error = attached gel vertices vs. their targets.",
        "",
        "| object | attachment | status | step mean [ms] | p95 | max | total [s] | lifted | object rise [mm] | slip [mm]"
        " | underside [mm] | attachment err [mm] |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda r: (r["object"], r["attachment"])):
        if r["status"] != "ok":
            lines.append(f"| {r['object']} | {r['attachment']} | {r['status']} |" + " – |" * 9)
            continue
        lines.append(
            f"| {r['object']} | {r['attachment']} | ok | {r['step_ms_mean']:.1f} | {r['step_ms_p95']:.1f} |"
            f" {r['step_ms_max']:.0f} | {r['total_s']:.1f} | {r['lifted']} | {r['object_rise_mm']:.1f} |"
            f" {r['slip_mm']:.2f} | {r['underside_max_mm']:.3f} | {r['attachment_err_max_mm']:.3f} |"
        )
    lines += ["", "Mean step time per phase [ms]:", "", "| object | attachment | " + " | ".join(p for p, _ in PHASES) + " |",
              "|---|---|" + "---:|" * len(PHASES)]  # fmt: skip
    for r in sorted(results, key=lambda r: (r["object"], r["attachment"])):
        if r["status"] == "ok":
            lines.append(f"| {r['object']} | {r['attachment']} | "
                         + " | ".join(f"{r[f'step_ms_mean_{p}']:.1f}" for p, _ in PHASES) + " |")  # fmt: skip
    failed = [r for r in results if r["status"] != "ok"]
    if failed:
        lines += ["", "## Failed runs", ""] + [
            f"- {r['attachment']}, {r['object']}: {r.get('error', '')[:300]}" for r in failed
        ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.worker:
        run_worker(cli_args)
    else:
        run_driver(cli_args)
