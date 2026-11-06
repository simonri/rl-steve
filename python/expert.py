"""A hand-written player the model can copy. Handles straight bridges (tapping shift) and jumps."""
import math

from skill_env import JUMP, PITCH_STEPS, TICKS_PER_DECISION, YAW_STEPS

TAP_EDGE = 0.45  # blocks before the edge where shift goes down
BRIDGE_PITCH = 80.0
JUMP_EDGE = 0.5


def _step_toward(error, steps):
    """The action index whose turn gets closest to the error, without overshooting much."""
    return min(range(len(steps)), key=lambda i: abs(error - steps[i]))


def _wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


def act(env):
    """Chooses the next action from the true game state."""
    o = env.obs
    x, _, z = o["pos"]
    jumping = env.extras == JUMP
    # Stick to the main axis of the bridge.
    along_x = abs(env.goal_x) >= abs(env.goal_z)
    ux, uz = (math.copysign(1, env.goal_x), 0.0) if along_x else (0.0, math.copysign(1, env.goal_z))
    to_goal = math.degrees(math.atan2(-ux, uz))
    target_yaw = to_goal if jumping else to_goal + 180.0
    target_pitch = 5.0 if jumping else BRIDGE_PITCH
    yaw_error = _wrap(target_yaw - o["yaw"])
    pitch_error = target_pitch - o["pitch"]
    # Each decision lasts two ticks, and a turn applies on both of them.
    yaw_i = _step_toward(yaw_error / TICKS_PER_DECISION, YAW_STEPS)
    pitch_i = _step_toward(pitch_error / TICKS_PER_DECISION, PITCH_STEPS)
    aligned = abs(yaw_error) < 4 and abs(pitch_error) < 4
    if not aligned:
        return [0, 0, 1, 0, 0, 0, yaw_i, pitch_i]  # turn first, on the spot and with shift for safety

    # Keep to the middle of the line with small sideways steps.
    side = (0.5 - z) if along_x else (0.5 - x)
    yaw = math.radians(o["yaw"])
    right_x, right_z = -math.cos(yaw), -math.sin(yaw)
    want_x, want_z = (0.0, side) if along_x else (side, 0.0)
    drift = want_x * right_x + want_z * right_z
    strafe = 2 if drift > 0.08 else 1 if drift < -0.08 else 0

    edge = env._edge_left(o) * 2  # blocks to the edge along the way to go
    if jumping:
        return [1, 0 if abs(drift) < 0.15 else strafe, 0, 0, int(edge < JUMP_EDGE), 1, yaw_i, pitch_i]
    return [2, strafe, int(edge < TAP_EDGE), 2, 0, 0, yaw_i, pitch_i]
