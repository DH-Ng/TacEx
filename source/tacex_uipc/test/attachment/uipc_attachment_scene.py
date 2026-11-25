"""Shared scene and checks for the UIPC attachment tests (pure UIPC, no tactile simulation).

Scene: an Isaac rigid body with the GelSight Mini case geometry (``Case.usd``: case + plate collision meshes) whose
pose is scripted, a UIPC gelpad (``Gelpad_low_res.usd``, StableNeoHookean) attached to it at the same relative pose as
in the GelSight Mini asset, and a UIPC ball (affine body) resting on the UIPC ground. The case points down (rotated
180 deg about x) and follows the trajectory:

    settle (gel 3 mm above the ball) -> press 1.5 mm into the ball -> hold -> move 2 mm sideways (shear)

All geometry is evaluated in the case frame: gel underside at z = 24 mm, domed contact face up to z = 28 mm.
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_uipc import UipcSim, UipcSimCfg
from tacex_uipc.objects import (
    UipcDeformableObject,
    UipcDeformableObjectCfg,
    UipcIsaacAttachmentsCfg,
    UipcRigidObject,
    UipcRigidObjectCfg,
)
from tacex_uipc.utils import TetMeshCfg

import debug_vis_helper as dv

DT = 0.01
D_HAT = 1e-4
"""IPC contact distance: bodies in contact stay up to about this far apart."""
GROUND_Z = 0.005
BALL_RADIUS = 0.005
GELPAD_OFFSET = 0.024
"""Gelpad underside in the case frame (as in Sensor.usd)."""
GEL_HEIGHT = 0.004
CONTACT_FACE_Z = 0.027
"""Rest z (case frame) above which a gel surface vertex belongs to the domed contact face (dome top: 28 mm)."""
UNDERSIDE_Z = 0.02425
"""Rest z (case frame) below which a gel surface vertex belongs to the underside (24 mm), attached to the case."""
DENT_SPREAD = 2 * GEL_HEIGHT
"""Lateral spreading of the indentation beyond the ball's footprint: a thick gel deforms over an area wider than the
contact (about two gel thicknesses)."""
DENT_RADIUS = BALL_RADIUS + DENT_SPREAD
"""Allowed lateral extent of the dent around its deepest point (ball footprint + spreading)."""
GEL_MESH_SPACING = 2e-3
"""Approximate vertex spacing of the gel's tet mesh. IPC contact acts on the gel's triangles, so the ball touches
triangle faces and the nearest vertices can be up to about one spacing away from the contact point."""

SETTLE_STEPS = 20
START_GAP = 0.003
PRESS_DEPTH = 0.0015
PRESS_STEPS = 45  # 0.1 mm per step
HOLD_STEPS = 10
SHEAR = 0.002
SHEAR_STEPS = 20

CASE_QUAT = (0.0, 1.0, 0.0, 0.0)  # (w, x, y, z): 180 deg about x, gel facing down
SENSOR_DIR = f"{TACEX_ASSETS_DATA_DIR}/Sensors/GelSight_Mini"


def case_trajectory() -> list[tuple[str, np.ndarray]]:
    """(phase, case position) per step."""
    ball_top = GROUND_Z + 2 * BALL_RADIUS
    z0 = ball_top + START_GAP + GELPAD_OFFSET + GEL_HEIGHT  # case origin, gel surface START_GAP above the ball
    traj = [("settle", np.array([0.0, 0.0, z0]))] * SETTLE_STEPS
    for i in range(1, PRESS_STEPS + 1):
        traj.append(("press", np.array([0.0, 0.0, z0 - (START_GAP + PRESS_DEPTH) * i / PRESS_STEPS])))
    z1 = traj[-1][1][2]
    traj += [("hold", np.array([0.0, 0.0, z1]))] * HOLD_STEPS
    for i in range(1, SHEAR_STEPS + 1):
        traj.append(("shear", np.array([SHEAR * i / SHEAR_STEPS, 0.0, z1])))
    return traj


def build_scene(attachment: str):
    """Creates sim, case, gel, ball and the attachment ("soft_position" or "carrier"). Returns a dict of handles."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cuda"))
    uipc_sim = UipcSim(
        UipcSimCfg(
            dt=DT,
            ground_height=GROUND_Z,
            contact=UipcSimCfg.Contact(d_hat=D_HAT, default_friction_ratio=0.95),
            update_render_meshes=dv.DEBUG_VIS,
        )
    )
    z0 = case_trajectory()[0][1][2]

    # Isaac body the gel is attached to; pose written every step (dynamic without gravity, see benchmark notes)
    case = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Case",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{SENSOR_DIR}/Case.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0), rot=CASE_QUAT),
        )
    )

    # gel at the case pose, shifted by GELPAD_OFFSET along the case's z (pointing down after the rotation)
    gel_cfg = UipcDeformableObjectCfg(
        prim_path="/World/Gelpad",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{SENSOR_DIR}/Gelpad_low_res.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, z0 - GELPAD_OFFSET), rot=CASE_QUAT),
        mesh_cfg=TetMeshCfg(stop_quality=8, max_its=100, edge_length_r=1 / 20),
        constitution_cfg=UipcDeformableObjectCfg.StableNeoHookeanCfg(),
        debug_vis=dv.DEBUG_VIS,
    )
    if attachment == "soft_position":
        gel_cfg = gel_cfg.replace(
            constraint_cfg=UipcIsaacAttachmentsCfg(
                constraint_strength_ratio=100.0, isaaclab_rigid_body_prim_path="/World/Case"
            )
        )
    gel = UipcDeformableObject(gel_cfg)

    ball = UipcRigidObject(
        UipcRigidObjectCfg(
            prim_path="/World/Ball",
            spawn=sim_utils.UsdFileCfg(usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd"),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, GROUND_Z + BALL_RADIUS + 0.0005)),
            mesh_cfg=TetMeshCfg(stop_quality=8, max_its=100, edge_length_r=1 / 5),
            constitution_cfg=UipcRigidObjectCfg.AffineBodyConstitutionCfg(),
        ),
    )

    if attachment == "soft_position":
        gel.constraint.isaaclab_rigid_object = case
        attach = gel.constraint
    elif attachment == "carrier":
        # imported here: only needed (and only present) for the carrier variant
        from tacex_uipc.objects.constraints.uipc_carrier_attachment import (
            UipcIsaacCarrierAttachments,
            UipcIsaacCarrierAttachmentsCfg,
        )

        attach = UipcIsaacCarrierAttachments(
            UipcIsaacCarrierAttachmentsCfg(isaaclab_rigid_body_prim_path="/World/Case", debug_vis=dv.DEBUG_VIS),
            gel,
            isaaclab_rigid_object=case,
        )
    else:
        raise ValueError(f"Unknown attachment '{attachment}'.")

    uipc_sim.setup_sim()
    sim.reset()
    dv.setup_view(sim, eye=(0.05, 0.05, 0.04), target=(0.0, 0.0, 0.01))
    return {"sim": sim, "uipc_sim": uipc_sim, "case": case, "gel": gel, "ball": ball, "attachment": attach}


def _attachment_targets(attach) -> np.ndarray:
    """Where the attached gel vertices should be (world)."""
    if hasattr(attach, "aim_positions") and callable(attach.aim_positions):
        return attach.aim_positions()  # carrier
    return attach.aim_positions  # soft position constraint (property)


def run(scene) -> dict:
    """Runs the trajectory and records gel, ball and attachment state (case frame where noted)."""
    sim, case, gel, ball, attach = (scene[k] for k in ("sim", "case", "gel", "ball", "attachment"))
    quat = torch.tensor([CASE_QUAT], device="cuda")
    R = math_utils.matrix_from_quat(quat)[0].cpu().numpy()
    steps = []
    for phase, pos in case_trajectory():
        pose = torch.cat([torch.tensor([pos], device="cuda", dtype=torch.float32), quat], dim=-1)
        case.write_root_pose_to_sim(pose)
        case.write_root_velocity_to_sim(torch.zeros(1, 6, device="cuda"))
        dv.step(sim)
        case.update(DT)
        gel.update(DT)
        ball.update(DT)

        nodes = gel._data.nodal_pos_w.reshape(-1, 3).cpu().numpy()
        surf_w = gel._data.surf_nodal_pos_w.reshape(-1, 3).cpu().numpy()
        ball_center = ball.data.root_pos_w[0].cpu().numpy()
        idx = np.asarray(attach.attachment_points_idx)
        steps.append({
            "phase": phase,
            "surf_case": (surf_w - pos) @ R,  # world -> case frame
            "ball_dist": np.linalg.norm(surf_w - ball_center, axis=1) - BALL_RADIUS,
            "ball_center": ball_center,
            "ball_case": (ball_center - pos) @ R,
            "attachment_err": np.linalg.norm(nodes[idx] - _attachment_targets(attach), axis=1).max(),
            "attached_case": (nodes[idx] - pos) @ R,
        })  # fmt: skip
    # reference: undeformed gel, attached and settled, before any contact
    dv.hold(sim)
    ref = steps[SETTLE_STEPS - 1]["surf_case"]
    return {"steps": steps, "ref": ref, "num_attached": len(np.asarray(attach.attachment_points_idx))}


"""
Analysis.
"""


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform (R, t) with R @ src_i + t ~ dst_i (least squares)."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cd - R @ cs


def deepest_contact(result: dict):
    """Step of the press/hold phases with the deepest indentation of the gel's contact face, after removing the gel's
    rigid motion. Shear is excluded: there the ball rolls on and the gel lags behind dynamically, so the dent is not at
    the ball (shear is covered by the underside check).

    The rigid motion (fitted on the underside, which should stay on the case) is removed so that the indentation
    measures only the gel's own deformation; the underside's motion is checked separately. Indentation = rest z -
    current z in the case frame (how far a vertex is pressed towards the case).
    """
    ref = result["ref"]
    contact_face, underside = ref[:, 2] > CONTACT_FACE_Z, ref[:, 2] < UNDERSIDE_Z
    best = None
    for k, s in enumerate(result["steps"]):
        if s["phase"] not in ("press", "hold"):
            continue
        R, t = kabsch(s["surf_case"][underside], ref[underside])
        deformed = s["surf_case"] @ R.T + t  # rigid motion of the gel removed
        indent = ref[:, 2] - deformed[:, 2]
        if best is None or indent[contact_face].max() > best[2][contact_face].max():
            best = (k, deformed, indent)
    k, deformed, indent = best
    return result["steps"][k], deformed, indent, contact_face, underside


"""
Checks (shared by the attachment test modules).
"""

CONTACT_DIST = 2 * D_HAT
"""A gel surface vertex closer than this to the ball surface counts as touching it (IPC keeps a gap of up to d_hat)."""


def check_attachment_points(result: dict):
    """The attachment holds the gel by its underside: enough vertices, none on the contact face. (Attached vertices
    may reach up to ~26 mm in the case frame: the plate's collision hull extends to 25.4 mm, into the gel.)"""
    attached = result["steps"][SETTLE_STEPS - 1]["attached_case"]
    print(f"\n[attachment] {result['num_attached']} attached vertices, case-frame z"
          f" [{attached[:, 2].min() * 1e3:.2f}, {attached[:, 2].max() * 1e3:.2f}] mm")  # fmt: skip
    assert result["num_attached"] >= 100, "too few attached gel vertices"
    assert attached[:, 2].max() < CONTACT_FACE_Z, "contact face vertices are attached to the case"


def check_gel_follows_case(result: dict):
    """Before the first contact (case approaching the ball), the gel keeps its shape relative to the moving case and
    the attached vertices stay on target. Steps after a contact are excluded: the gel may still be deformed then (e.g.
    when the ball rolls away during the shear)."""
    ref = result["ref"]
    free = []
    for s in result["steps"][SETTLE_STEPS:]:
        if s["ball_dist"].min() <= CONTACT_DIST:
            break
        free.append(s)
    shape_err = max(np.linalg.norm(s["surf_case"] - ref, axis=1).max() for s in free)
    att_err = max(s["attachment_err"] for s in free)
    print(f"\n[follow] {len(free)} steps before contact: max gel deviation from rest (case frame)"
          f" {shape_err * 1e3:.3f} mm, max attachment error {att_err * 1e3:.3f} mm")  # fmt: skip
    assert len(free) >= 20, "expected a phase without contact"
    assert shape_err < 2e-4, "gel does not keep its shape relative to the moving case"
    assert att_err < 3e-4, "attached vertices do not follow the case"


def check_gel_deforms_like_ball(result: dict):
    """The ball dents the gel: deep enough, the ball touches the gel, the deepest point lies under the ball, the dent
    is about the ball's size (plus the spreading of a thick gel) and local, no penetration. Rigid motion of the gel is
    removed (see ``deepest_contact``).

    "Under the ball" is checked laterally (case xy plane) against the contact radius sqrt(2 r depth) plus the mesh
    spacing, not via the deepest vertex's distance to the ball: contact acts on the gel's triangles, so the vertices
    around the contact point can be up to ~e^2 / (6 r) (+ d_hat) away from the ball surface (e = mesh spacing)."""
    step, deformed, indent, contact_face, _ = deepest_contact(result)
    dist = step["ball_dist"]
    face = indent[contact_face]
    max_indent = face.max()
    deepest = np.flatnonzero(contact_face)[np.argmax(face)]
    lateral = np.linalg.norm(deformed[:, :2] - deformed[deepest, :2], axis=1)[contact_face]
    dent = face > 0.5 * max_indent
    outside = lateral > DENT_RADIUS
    contact_radius = np.sqrt(2 * BALL_RADIUS * max_indent)
    under_ball = np.linalg.norm(deformed[deepest, :2] - step["ball_case"][:2])  # deepest point <-> ball center
    print(f"\n[deform] {step['phase']}: max indentation {max_indent * 1e3:.2f} mm, deepest vertex"
          f" {under_ball * 1e3:.2f} mm laterally from the ball center (allowed {(contact_radius + GEL_MESH_SPACING) * 1e3:.2f}"
          f" mm) and {dist[deepest] * 1e3:.3f} mm from its surface, closest vertex {dist.min() * 1e3:.3f} mm (touching:"
          f" < {CONTACT_DIST * 1e3:.2f} mm), dent ({dent.sum()} vertices) within {lateral[dent].max() * 1e3:.2f} mm"
          f" (allowed {DENT_RADIUS * 1e3:.0f} mm), max indentation outside {face[outside].max() * 1e3:.3f} mm")  # fmt: skip
    assert max_indent > 0.5e-3, "no substantial dent"
    assert dist.min() > -1e-5, "ball penetrates the gel"
    assert dist.min() < CONTACT_DIST, "the ball does not touch the gel"
    assert under_ball < contact_radius + GEL_MESH_SPACING, "the deepest point of the dent is not under the ball"
    assert lateral[dent].max() < DENT_RADIUS, "the dent is wider than the ball plus the gel's spreading"
    assert face[outside].max() < 0.3 * max_indent, "deformation is not local to the contact"


def check_underside_stays_on_case(result: dict):
    """Under load (press, hold, shear), the gel's underside stays on the case: the ball dents the gel, not moves it."""
    ref = result["ref"]
    underside = ref[:, 2] < UNDERSIDE_Z
    loaded = [s for s in result["steps"][SETTLE_STEPS:] if s["ball_dist"].min() < CONTACT_DIST]
    disp = max(np.linalg.norm(s["surf_case"][underside] - ref[underside], axis=1).max() for s in loaded)
    att_err = max(s["attachment_err"] for s in loaded)
    print(f"\n[underside] {len(loaded)} steps with contact: max underside displacement {disp * 1e3:.3f} mm,"
          f" max attachment error {att_err * 1e3:.3f} mm")  # fmt: skip
    assert len(loaded) >= 10, "expected a phase with contact"
    assert disp < 3e-4, "the gel's underside moves under load"
