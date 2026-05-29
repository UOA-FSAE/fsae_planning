import math

import numpy as np

# FSDS: max steering angle 25 degrees per ros-bridge.md
MAX_STEER_RAD = math.radians(25.0)


def _heading_error(car_pos, car_yaw, target_global) -> float:
    """Return heading error in radians: positive when target is left of car."""
    dx = target_global[0] - car_pos[0]
    dy = target_global[1] - car_pos[1]
    cos_y, sin_y = math.cos(car_yaw), math.sin(car_yaw)
    x_car =  dx * cos_y + dy * sin_y
    y_car = -dx * sin_y + dy * cos_y
    return math.atan2(y_car, x_car)


def compute_steering(car_pos, car_yaw, target_global) -> float:
    """
    Pure-proportional steering (legacy helper, kept for reference).
    Prefer SteeringPID for closed-loop driving.
    """
    return float(np.clip(-_heading_error(car_pos, car_yaw, target_global)
                         / MAX_STEER_RAD, -1.0, 1.0))


class SteeringPID:
    """
    PID controller for heading-error steering.

    Sign convention (FSDS ENU: x forward, y left):
      heading_error > 0  → target is left  → need negative steering (steer left)
      heading_error < 0  → target is right → need positive steering (steer right)
      steering output ∈ [-1, 1]

    Gains:
      Kp — proportional.  Default 1/MAX_STEER_RAD keeps the same proportional
           response as the original pure-pursuit controller.
      Ki — integral.      Corrects steady-state drift from road camber, etc.
           Accumulated error is clamped to ±integral_limit to prevent windup.
      Kd — derivative.    The primary anti-sway term: damps oscillation by
           counter-steering when the heading error is changing rapidly.
           A simple EMA filter (d_alpha) reduces differentiation noise.
    """

    def __init__(
        self,
        kp: float = 1.0 / MAX_STEER_RAD,
        ki: float = 0.2,
        kd: float = 0.15,
        integral_limit: float = 0.5,
        d_alpha: float = 0.3,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral_limit = integral_limit
        self._d_alpha = d_alpha          # EMA weight for previous derivative

        self._integral  = 0.0
        self._prev_err  = 0.0
        self._prev_d    = 0.0            # filtered derivative

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, car_pos, car_yaw, target_global, dt: float) -> float:
        """
        Compute a steering command in [-1, 1].

        car_pos        — (2,) array, car position in map frame
        car_yaw        — float, car heading in radians
        target_global  — (2,) array, lookahead target in map frame
        dt             — seconds since the last call (must be > 0)
        """
        dt  = max(dt, 1e-4)
        err = _heading_error(car_pos, car_yaw, target_global)

        # --- P ---
        p = self.kp * err

        # --- I  (with anti-windup clamp) ---
        self._integral = float(np.clip(
            self._integral + err * dt,
            -self._integral_limit,
            self._integral_limit,
        ))
        i = self.ki * self._integral

        # --- D  (EMA-filtered finite difference) ---
        d_raw         = (err - self._prev_err) / dt
        self._prev_d  = self._d_alpha * self._prev_d + (1.0 - self._d_alpha) * d_raw
        self._prev_err = err
        d = self.kd * self._prev_d

        return float(np.clip(-(p + i + d), -1.0, 1.0))

    def reset(self) -> None:
        """Reset integrator and derivative state (call when car stops)."""
        self._integral = 0.0
        self._prev_err = 0.0
        self._prev_d   = 0.0
