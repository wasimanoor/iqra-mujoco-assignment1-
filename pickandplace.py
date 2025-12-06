# -*- coding: utf-8 -*-
import time
import threading
from threading import Thread
import tkinter as tk
from tkinter import ttk, messagebox

import glfw
import mujoco
import numpy as np


class Demo:
    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    K = np.array([900.0, 900.0, 900.0, 40.0, 40.0, 40.0])
    height, width = 480, 640
    fps = 30

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path("world.xml")
        self.data = mujoco.MjData(self.model)

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 0
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)

        self.run = True
        self.stop_flag = threading.Event()
        self.seq_thread = None

        # open gripper + home joints
        self.gripper(True)
        for i in range(1, 8):
            self.data.joint(f"panda_joint{i}").qpos = self.qpos0[i - 1]
        mujoco.mj_forward(self.model, self.data)

        # hold targets (keep pose fixed when idle)
        hand = self.data.body("panda_hand")
        self.target_pos = hand.xpos.copy()
        self.target_quat = hand.xquat.copy()
        self._hold_running = True

    # ---------------- controller ----------------
    def gripper(self, open=True):
        self.data.actuator("pos_panda_finger_joint1").ctrl = 0.04 if open else 0.0
        self.data.actuator("pos_panda_finger_joint2").ctrl = 0.04 if open else 0.0

    def control(self, xpos_d, xquat_d):
        xpos = self.data.body("panda_hand").xpos
        xquat = self.data.body("panda_hand").xquat
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)

        error = np.zeros(6)
        error[:3] = xpos_d - xpos
        res = np.zeros(3)
        mujoco.mju_subQuat(res, xquat, xquat_d)
        mujoco.mju_rotVecQuat(res, res, xquat)
        error[3:] = -res

        J = np.concatenate((jacp, jacr))
        v = J @ self.data.qvel
        for i in range(1, 8):
            dofadr = self.model.joint(f"panda_joint{i}").dofadr
            self.data.actuator(f"panda_joint{i}").ctrl = self.data.joint(
                f"panda_joint{i}"
            ).qfrc_bias
            self.data.actuator(f"panda_joint{i}").ctrl += (
                J[:, dofadr].T @ np.diag(self.K) @ error
            )
            self.data.actuator(f"panda_joint{i}").ctrl -= (
                J[:, dofadr].T @ np.diag(2 * np.sqrt(self.K)) @ v
            )

    # --------------- hold loop ----------------
    def _hold_loop(self):
        while self.run and self._hold_running:
            self.control(self.target_pos, self.target_quat)
            mujoco.mj_step(self.model, self.data)
            time.sleep(1 / 500)

    def get_yaw_to_target(self, target_xy):
     """Compute yaw so gripper faces the target from current position."""
     hand_xy = self.data.body("panda_hand").xpos[:2]
     dx = target_xy[0] - hand_xy[0]
     dy = target_xy[1] - hand_xy[1]
     return np.arctan2(dy, dx)
    
    
    def _quat_err(self, q, r):
    # smallest angle between unit quats q,r
     dot = abs(float(np.dot(q, r)))
     dot = max(min(dot, 1.0), -1.0)
     return 2.0 * np.arccos(dot)
   
    
    def _reach_pose(self, pos_goal, quat_goal, pos_tol=0.003, ang_tol=0.03, timeout=2.0):
     """Hold until we are within tolerance or timeout (seconds)."""
     t0 = time.time()
     self.target_pos = pos_goal.copy()
     self.target_quat = quat_goal.copy()
     while time.time() - t0 < timeout and not self.stop_flag.is_set():
        hand = self.data.body("panda_hand")
        p_err = np.linalg.norm(pos_goal - hand.xpos)
        a_err = self._quat_err(quat_goal, hand.xquat)
        if p_err < pos_tol and a_err < ang_tol:
            return True
        time.sleep(1/400)
     return False
    # --------------- helpers ------------------
    def _move_linear(self, target_pos, xquat_ref, duration_s):
        start = self.data.body("panda_hand").xpos.copy()
        steps = max(1, int(duration_s * 400))
        self.target_quat = xquat_ref
        for k in range(steps):
            if self.stop_flag.is_set():
                return
            a = (k + 1) / steps
            self.target_pos = (1.0 - a) * start + a * target_pos
            time.sleep(1 / 400)

    def move_z_relative(self, dz, duration_s):
        hand = self.data.body("panda_hand")
        xquat_ref = hand.xquat.copy()
        target = hand.xpos.copy() + np.array([0.0, 0.0, dz])
        self._move_linear(target, xquat_ref, duration_s)
    
    def wait(self, seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            if self.stop_flag.is_set():
                return
            time.sleep(1 / 400)

    def _find_body_and_top_z(self, names=("box", "cube", "object"), allow_auto=True):
        """Return (name, xy, z_top). Named lookup first; optionally auto-detect a small object."""
        # 1) Try provided names
        for nm in names:
            try:
                b = self.data.body(nm)
                xy = b.xpos[:2].copy()
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm)
                z_top = float(b.xpos[2])
                for g in range(self.model.ngeom):
                    if self.model.geom_bodyid[g] == bid and self.model.geom_size.shape[1] >= 3:
                        z_top = float(b.xpos[2] + self.model.geom_size[g][2])
                        break
                return nm, xy, z_top
            except KeyError:
                pass

        if not allow_auto:
            raise KeyError("Target body not found. Tried: " + ", ".join(names))

        # 2) Auto-detect: pick the smallest non-robot box/cylinder body
        skip_keywords = ("panda", "floor", "world", "table", "ground", "base")
        candidate = None
        best_score = 1e9
        for bid in range(self.model.nbody):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if any(k in nm.lower() for k in skip_keywords):
                continue
            # first geom on this body
            g_idx = None
            for g in range(self.model.ngeom):
                if self.model.geom_bodyid[g] == bid:
                    g_idx = g
                    break
            if g_idx is None:
                continue
            gtype = int(self.model.geom_type[g_idx])
            size = np.array(self.model.geom_size[g_idx])
            if gtype not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER):
                continue
            s = float(np.linalg.norm(size))
            if s < best_score:
                best_score = s
                xy = self.data.xpos[bid][:2].copy()
                z_top = float(self.data.xpos[bid][2] + (size[2] if size.size >= 3 else 0.0))
                candidate = (nm, xy, z_top)

        if candidate:
            return candidate

        raise KeyError("Could not auto-detect a target body; please enter its exact name in the GUI.")

    def move_xy_to(self, xy_target, z_hold=None, duration_s=2.0):
        hand = self.data.body("panda_hand")
        if z_hold is None:
            z_hold = float(hand.xpos[2])
        xquat = hand.xquat.copy()
        goal = np.array([xy_target[0], xy_target[1], z_hold], dtype=float)
        self._move_linear(goal, xquat, duration_s)

    # --------------- targeted Pick & Place ---------------
    def pick_and_place(self, down_m=0.15, down_time=1.0, settle_s=0.3, up_m=0.15, up_time=1.0, target_hint="box"):
     self.stop_flag.clear()
  
     # locate object (uses your robust finder)
     names = (target_hint,) if target_hint else ()
     _, xy, z_top = self._find_body_and_top_z(names or ("box","cube","object"), allow_auto=True)

     hand = self.data.body("panda_hand")
     xquat_ref = hand.xquat.copy()

     approach_clearance = float(abs(up_m))              # hover height above top
     grasp_depth = float(abs(min(down_m, up_m)) / 3.0)  # shallow dip below top

     hover = np.array([xy[0], xy[1], z_top + approach_clearance], dtype=float)
     pregrasp = np.array([xy[0], xy[1], z_top + 0.005], dtype=float)   # 5 mm above top
     grasp   = np.array([xy[0], xy[1], z_top - grasp_depth], dtype=float)

    # Open and give the scene a brief settle (important on the first attempt)
     self.gripper(True)
     self.wait(0.20)

    # Move to hover and WAIT UNTIL CONVERGED
     self._move_linear(hover, xquat_ref, max(1.0, up_time))
     self._reach_pose(hover, xquat_ref, pos_tol=0.003, ang_tol=0.03, timeout=2.0)
     self.wait(0.10)  # tiny settle

    # Move to just above top, converge, then final short descent
     self._move_linear(pregrasp, xquat_ref, max(0.5, down_time*0.5))
     self._reach_pose(pregrasp, xquat_ref, pos_tol=0.002, ang_tol=0.03, timeout=1.0)
     self._move_linear(grasp, xquat_ref, max(0.5, down_time*0.5))
     self._reach_pose(grasp, xquat_ref, pos_tol=0.002, ang_tol=0.03, timeout=1.0)

    # Close + settle
     self.wait(settle_s)
     self.gripper(False)
     self.wait(0.20)

    # Lift and converge at hover
     self._move_linear(hover, xquat_ref, max(0.8, up_time))
     self._reach_pose(hover, xquat_ref, pos_tol=0.003, ang_tol=0.03, timeout=2.0)
    def get_yaw_to_target(self, target_xy):
      """Compute yaw so gripper faces the target from current position."""
      hand_xy = self.data.body("panda_hand").xpos[:2]
      dx = target_xy[0] - hand_xy[0]
      dy = target_xy[1] - hand_xy[1]
      return np.arctan2(dy, dx)
    
    

    def pick_only(self, down_m=0.15, down_time=1.0, settle_s=0.3, up_m=0.15, up_time=1.0, target_hint="box"):
     self.stop_flag.clear()
     names = (target_hint,) if target_hint else ()
     _, xy, z_top = self._find_body_and_top_z(names or ("box", "cube", "object"), allow_auto=True)

     hand = self.data.body("panda_hand")
     xquat_ref = hand.xquat.copy()

     approach_clearance = float(abs(up_m))
     grasp_depth = float(abs(min(down_m, up_m)) / 3.0)

     hover = np.array([xy[0], xy[1], z_top + approach_clearance], dtype=float)
     pregrasp = np.array([xy[0], xy[1], z_top + 0.005], dtype=float)
     grasp   = np.array([xy[0], xy[1], z_top - grasp_depth], dtype=float)

     self.gripper(True)
     self.wait(0.20)

     self._move_linear(hover, xquat_ref, max(1.0, up_time))
     self._reach_pose(hover, xquat_ref, 0.003, 0.03, 2.0)
     self.wait(0.10)

     self._move_linear(pregrasp, xquat_ref, max(0.5, down_time*0.5))
     self._reach_pose(pregrasp, xquat_ref, 0.002, 0.03, 1.0)
     self._move_linear(grasp, xquat_ref, max(0.5, down_time*0.5))
     self._reach_pose(grasp, xquat_ref, 0.002, 0.03, 1.0)

     self.wait(settle_s)
     self.gripper(False)  # close
     self.wait(0.20)

     self._move_linear(hover, xquat_ref, max(0.8, up_time))
     self._reach_pose(hover, xquat_ref, 0.003, 0.03, 2.0)

    def place_only(self, place_xy, down_m=0.15, down_time=1.0, up_m=0.15, up_time=1.0):
     self.stop_flag.clear()
     hand = self.data.body("panda_hand")
     xquat_ref = hand.xquat.copy()

     z_top = hand.xpos[2] - 0.05  # adjust to match object bottom height
     approach_clearance = float(abs(up_m))
     hover = np.array([place_xy[0], place_xy[1], z_top + approach_clearance], dtype=float)
     place  = np.array([place_xy[0], place_xy[1], z_top], dtype=float)

     self._move_linear(hover, xquat_ref, max(1.0, up_time))
     self._reach_pose(hover, xquat_ref, 0.003, 0.03, 2.0)
     self.wait(0.1)

     self._move_linear(place, xquat_ref, max(0.5, down_time))
     self._reach_pose(place, xquat_ref, 0.003, 0.03, 1.0)

     self.gripper(True)  # open to release
     self.wait(0.2)

     self._move_linear(hover, xquat_ref, max(0.8, up_time))
     self._reach_pose(hover, xquat_ref, 0.003, 0.03, 2.0)

    # ---------------- viewer ----------------
    def render(self) -> None:
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 8)
        window = glfw.create_window(self.width, self.height, "Panda Demo", None, None)
        glfw.make_context_current(window)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        while not glfw.window_should_close(window):
            w, h = glfw.get_framebuffer_size(window)
            viewport.width, viewport.height = w, h
            mujoco.mjv_updateScene(
                self.model, self.data, opt, pert, self.cam,
                mujoco.mjtCatBit.mjCAT_ALL, self.scene
            )
            mujoco.mjr_render(viewport, self.scene, self.context)
            time.sleep(1.0 / self.fps)
            glfw.swap_buffers(window)
            glfw.poll_events()
        self.run = False
        self._hold_running = False
        self.stop_flag.set()
        glfw.terminate()

    def start(self) -> None:
        Thread(target=self._hold_loop, daemon=True).start()
        self.render()


# ---------------- GUI ----------------
def launch_gui(demo: Demo):
    root = tk.Tk()
    root.title("Panda: Pick & Place")
    root.geometry("520x170")

    target_name = tk.StringVar(value="box")
    ttk.Label(root, text="Target body:").grid(row=0, column=0, padx=6, sticky="e")
    ttk.Entry(root, width=16, textvariable=target_name).grid(row=0, column=1, padx=4, sticky="w")
    row = 1  # ✅ row now defined
    down_m = tk.StringVar(value="0.15")
    down_t = tk.StringVar(value="1.0")
    settle = tk.StringVar(value="0.3")
    up_m = tk.StringVar(value="0.15")
    up_t = tk.StringVar(value="1.0")
    place_x = tk.StringVar(value="0.4")
    place_y = tk.StringVar(value="0.0")

    ttk.Label(root, text="Place X:").grid(row=row+2, column=0, sticky="e", padx=6)
    ttk.Entry(root, width=8, textvariable=place_x).grid(row=row+2, column=1, padx=4)

    ttk.Label(root, text="Place Y:").grid(row=row+2, column=2, sticky="e", padx=6)
    ttk.Entry(root, width=8, textvariable=place_y).grid(row=row+2, column=3, padx=4)

    ttk.Button(root, text="Pick", command=lambda: demo.pick_only(
     down_m=float(down_m.get()), down_time=float(down_t.get()),
     settle_s=float(settle.get()), up_m=float(up_m.get()),
     up_time=float(up_t.get()), target_hint=target_name.get().strip()
    )).grid(row=row+3, column=0, padx=8, pady=8, sticky="w")

    ttk.Button(root, text="Place", command=lambda: demo.place_only(
     place_xy=(float(place_x.get()), float(place_y.get())),
     down_m=float(down_m.get()), down_time=float(down_t.get()),
     up_m=float(up_m.get()), up_time=float(up_t.get())
    )).grid(row=row+3, column=1, padx=8, pady=8, sticky="w")

    row = 1
    ttk.Label(root, text="Down (m):").grid(row=row, column=0, sticky="e", padx=6)
    ttk.Entry(root, width=8, textvariable=down_m).grid(row=row, column=1, padx=4)
    ttk.Label(root, text="Down time (s):").grid(row=row, column=2, sticky="e", padx=6)
    ttk.Entry(root, width=8, textvariable=down_t).grid(row=row, column=3, padx=4)

    row += 1
    ttk.Label(root, text="Settle (s):").grid(row=row, column=0, sticky="e", padx=6)
    ttk.Entry(root, width=8, textvariable=settle).grid(row=row, column=1, padx=4)
    ttk.Label(root, text="Up (m):").grid(row=row, column=2, sticky="e", padx=6)
    ttk.Entry(root, width=8, textvariable=up_m).grid(row=row, column=3, padx=4)
    ttk.Label(root, text="Up time (s):").grid(row=row, column=4, sticky="e", padx=6)
    ttk.Entry(root, width=8, textvariable=up_t).grid(row=row, column=5, padx=4)

    def run_pickplace_now():
        try:
            demo.stop_flag.clear()
            demo.pick_and_place(
                down_m=float(down_m.get()),
                down_time=float(down_t.get()),
                settle_s=float(settle.get()),
                up_m=float(up_m.get()),
                up_time=float(up_t.get()),
                target_hint=target_name.get().strip(),
            )
        except ValueError:
            messagebox.showerror("Invalid", "Please enter numeric values.")
        except KeyError as e:
            messagebox.showerror("Target not found", str(e))

    ttk.Button(root, text="Run Pick & Place", command=run_pickplace_now)\
        .grid(row=row+1, column=0, padx=8, pady=8, sticky="w")

    root.mainloop()


if __name__ == "__main__":
    demo = Demo()
    Thread(target=launch_gui, args=(demo,), daemon=True).start()
    demo.start()
