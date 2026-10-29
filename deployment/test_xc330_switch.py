"""Offline tests for the switch gripper used by test/test_model_inference.py."""

import types
import unittest

from driver.xc330 import XC330Error
from driver.xc330_switch import XC330SwitchGripper


class FakeMotors:
    """Stands in for driver.gripper_torque_control (same names and constants)."""

    MOTOR_DIRECTION = {1: 1, 2: -1}
    GRIP_SPEED = 30
    RELEASE_SPEED = 30
    VELOCITY_THRESHOLD = 25
    STALL_TIME = 1
    MAX_GRIP_TIME = 5.0
    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0

    def __init__(self):
        self.velocity = {1: 0, 2: 0}      # last goal velocity
        self.present = {1: 0, 2: 0}       # reported present velocity
        self.torque = {1: 0, 2: 0}

    def set_velocity(self, packet, port, motor_id, velocity):
        self.velocity[motor_id] = velocity
        return True

    def write_torque(self, packet, port, motor_id, enable):
        self.torque[motor_id] = enable
        return True

    def read_current_velocity(self, packet, port, motor_id):
        return 300, self.present[motor_id]


def make(close=0.088, open_=0.06, at_floor=None):
    gripper = XC330SwitchGripper.__new__(XC330SwitchGripper)   # skip opening the serial port
    clock = types.SimpleNamespace(t=0.0)
    gripper.__dict__.update(
        gtc=FakeMotors(), packet=None, port=None, ids=(1, 2), close_below_m=close, open_above_m=open_,
        close_below_at_floor_m=at_floor, at_floor_until=float("-inf"),
        clock=lambda: clock.t, state="open", state_since=0.0, stall_since=None, events=[], torque_enabled=True,
    )
    gripper._read4_signed = lambda motor_id, address: gripper.gtc.present[motor_id]   # present velocity
    return gripper, clock


class SwitchGripperTest(unittest.TestCase):
    def test_full_cycle_matches_gripper_torque_control(self):
        g, clock = make()
        m = g.gtc
        g.command_width(0.095)
        self.assertEqual(g.state, "open")
        self.assertEqual(m.velocity, {1: 0, 2: 0})

        clock.t = 1.0
        g.command_width(0.085)                       # below 88 mm: grip
        self.assertEqual(g.state, "closing")
        self.assertEqual(m.velocity, {1: 30, 2: -30})  # same direction/speed as grip()
        self.assertEqual(m.torque, {1: 1, 2: 1})

        m.present = {1: 30, 2: -30}                   # still moving
        clock.t = 1.5
        g.command_width(0.05)
        self.assertEqual(g.state, "closing")
        m.present = {1: 3, 2: -2}                     # both stalled on the object
        clock.t = 2.0
        g.command_width(0.03)
        clock.t = 2.9
        g.command_width(0.03)
        self.assertEqual(g.state, "closing")          # stalled < STALL_TIME
        clock.t = 3.0
        g.command_width(0.03)
        self.assertEqual(g.state, "closed")
        self.assertEqual(m.velocity, {1: 0, 2: 0})    # hold, torque stays on
        self.assertEqual(m.torque, {1: 1, 2: 1})

        clock.t = 4.0
        g.command_width(0.055)                        # between thresholds: keep holding
        self.assertEqual(g.state, "closed")
        g.command_width(0.07)                         # above 60 mm: release
        self.assertEqual(g.state, "releasing")
        self.assertEqual(m.velocity, {1: -30, 2: 30})
        clock.t = 5.0
        g.command_width(0.09)
        self.assertEqual(g.state, "release_settling")
        self.assertEqual(m.velocity, {1: 0, 2: 0})
        clock.t = 5.15
        g.command_width(0.09)
        self.assertEqual(g.state, "open")
        self.assertEqual(m.torque, {1: 0, 2: 0})      # release() turns torque off
        self.assertFalse(g.torque_enabled)

        clock.t = 6.0
        g.command_width(0.02)                         # a second grip re-enables torque
        self.assertEqual(g.state, "closing")
        self.assertEqual(m.torque, {1: 1, 2: 1})
        self.assertEqual([e["state"] for e in g.events],
                         ["closing", "closed", "releasing", "release_settling", "open", "closing"])

    def test_one_motor_still_moving_is_not_a_stall(self):
        g, clock = make()
        g.command_width(0.08)
        g.gtc.present = {1: 2, 2: -40}
        for t in (1.0, 2.0, 3.0):
            clock.t = t
            g.command_width(0.02)
        self.assertEqual(g.state, "closing")

    def test_max_grip_time_holds_anyway(self):
        g, clock = make()
        g.command_width(0.08)
        g.gtc.present = {1: 30, 2: -30}
        clock.t = 5.01
        g.command_width(0.02)
        self.assertEqual(g.state, "closed")
        self.assertEqual(g.events[-1]["note"], "max grip time")

    def test_busy_only_while_moving(self):
        g, clock = make()
        self.assertFalse(g.is_busy())
        g.command_width(0.08)
        self.assertTrue(g.is_busy())
        g.state = "closed"
        self.assertFalse(g.is_busy())
        g.state = "releasing"
        self.assertTrue(g.is_busy())

    def test_looser_threshold_only_while_on_the_floor(self):
        g, clock = make(at_floor=0.093)
        g.command_width(0.091)                       # descending: 91 mm does not trigger (88 mm rule)
        self.assertEqual(g.state, "open")
        g.mark_at_floor()
        g.command_width(0.0915)                      # resting on the floor: below 93 mm triggers
        self.assertEqual(g.state, "closing")
        self.assertIn("at floor", g.events[-1]["note"])

    def test_floor_mark_expires(self):
        g, clock = make(at_floor=0.093)
        g.mark_at_floor()
        clock.t = 0.6                                # more than AT_FLOOR_HOLD_S without floor contact
        g.command_width(0.091)
        self.assertEqual(g.state, "open")
        clock.t = 0.7
        g.command_width(0.087)                       # normal threshold still works
        self.assertEqual(g.state, "closing")

    def test_no_decision_while_the_arm_holds(self):
        g, clock = make()
        g.command_width(None)                        # open: no decision -> stays open
        self.assertEqual(g.state, "open")
        g.command_width(0.08)
        g.gtc.present = {1: 0, 2: 0}
        clock.t = 0.1; g.command_width(None)         # closing keeps advancing without a width
        clock.t = 1.2; g.command_width(None)
        self.assertEqual(g.state, "closed")
        clock.t = 2.0; g.command_width(None)         # closed: the stale width cannot release it
        self.assertEqual(g.state, "closed")
        g.command_width(0.022)                       # fresh post-grip prediction keeps holding
        self.assertEqual(g.state, "closed")

    def test_failed_write_raises(self):
        g, _ = make()
        g.gtc.set_velocity = lambda *a: False
        with self.assertRaises(XC330Error):
            g.command_width(0.08)

    def test_thresholds_are_validated(self):
        with self.assertRaises(ValueError):
            XC330SwitchGripper("p", 57600, None, (1, 2), 600, 80, close_below_m=0.05, open_above_m=0.06)
        with self.assertRaises(ValueError):
            XC330SwitchGripper("p", 57600, None, (1, 2), 600, 80, close_below_m=0.088, open_above_m=0.06,
                               close_below_at_floor_m=0.08)


if __name__ == "__main__":
    unittest.main()
