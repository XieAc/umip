"""Collection lifecycle tests with mocked hardware and camera process."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import gripper_grasp_control as collection


class CollectionTest(unittest.TestCase):
    def make_collector(self):
        with patch.object(collection, "PortHandler"), patch.object(collection, "PacketHandler"):
            return collection.Collector()

    def test_camera_ready_and_clean_stop(self):
        collector = self.make_collector()
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            camera_dir = session / "d435i"
            camera_dir.mkdir()
            (camera_dir / "recording_events.json").write_text(json.dumps({"events": {"camera_ready": {}}}))
            camera = Mock()
            camera.poll.return_value = None
            with patch.object(collector, "new_session", return_value=session), \
                 patch.object(collection, "load_width_calibration", return_value={}), \
                 patch.object(collection.subprocess, "Popen", return_value=camera):
                collector.start_recording()
            self.assertTrue(collector.recording)
            collector.stop_recording()
            self.assertFalse(collector.recording)
            self.assertIsNone(collector.log_writer)
            self.assertIsNone(collector.camera)
            camera.stdin.write.assert_called_once_with(b"stop\n")
            camera.stdin.close.assert_called_once()

    def test_camera_start_failure_closes_log(self):
        collector = self.make_collector()
        with tempfile.TemporaryDirectory() as directory:
            camera = Mock()
            camera.poll.return_value = 1
            camera.returncode = 1
            with patch.object(collector, "new_session", return_value=Path(directory)), \
                 patch.object(collection, "load_width_calibration", return_value={}), \
                 patch.object(collection.subprocess, "Popen", return_value=camera), \
                 self.assertRaisesRegex(RuntimeError, "recorder exited"):
                collector.start_recording()
            self.assertFalse(collector.recording)
            self.assertIsNone(collector.log_stream)
            self.assertIsNone(collector.camera)
            camera.wait.assert_called_once()

    def test_serial_error_stops_recording_loop(self):
        collector = self.make_collector()
        collector.recording = True
        collector.log_writer = Mock()
        collector.read4 = Mock(side_effect=RuntimeError("serial failure"))
        collector.log_loop()
        self.assertFalse(collector.running)
        self.assertIsInstance(collector.log_error, RuntimeError)
        collector.log_writer.writerow.assert_not_called()


if __name__ == "__main__":
    unittest.main()
