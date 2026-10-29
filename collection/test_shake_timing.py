import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from shake_timing import find_shake_pause, refine_burst_offset


def motion(times, values):
    return SimpleNamespace(timestamps_s=times, smoothed_signal=values, signal_kind='gyro')


def burst(t):
    return .05 + sum(a*np.exp(-((t-c)/w)**2) for a,c,w in
        ((5.,1.,.12),(8.,1.7,.15),(6.,2.35,.12)))


class ShakeTimingTest(unittest.TestCase):
    def test_whole_burst_corrects_wrong_oscillation_peak(self):
        dtime = np.arange(0,5,.1)
        ptime = np.arange(0,6,.01)
        actual_offset = .65
        offset, report = refine_burst_offset(motion(dtime,burst(dtime)),
            motion(ptime,burst(ptime-actual_offset)), 1.35, .4, 4.5)
        self.assertAlmostEqual(offset,actual_offset,delta=.02)
        self.assertGreater(report['best_correlation'],.99)

    def test_complete_repeated_shake_and_pause_are_retained(self):
        times=np.arange(0,5,.01)
        signal=motion(times,burst(times))
        result=find_shake_pause(signal,signal,0.,1.,.4,4.5,8.,8.,.4)
        self.assertGreater(result['last_strong_motion_d435i_time_s'],2.35)
        self.assertGreater(result['dataset_start_time_s'],2.8)
        self.assertLess(result['dataset_start_time_s'],3.1)
        self.assertAlmostEqual(result['dataset_start_time_s']-
            result['quiet_start_d435i_time_s'],.4)

    def test_no_quiet_pause_requires_review(self):
        times=np.arange(0,5,.01)
        signal=motion(times,burst(times)+.6)
        with self.assertRaisesRegex(ValueError,'quiet pause'):
            find_shake_pause(signal,signal,0.,1.,.4,4.5,8.,8.,.4)

    def test_weak_or_flat_signal_does_not_guess_sync(self):
        times=np.arange(0,5,.1)
        signal=motion(times,np.zeros_like(times))
        with self.assertRaises(ValueError):
            refine_burst_offset(signal,signal,0.,0.,4.)

    def test_incomplete_recording_stays_excluded(self):
        from build_dataset import prepare_shake_timing
        with patch('build_dataset.load_realtime_recording_events', return_value={'events': {}}):
            with self.assertRaisesRegex(RuntimeError, '未完成'):
                prepare_shake_timing(None, {}, 'shake-pause', 'burst-correlation', .4)

    def test_explicit_legacy_mode_preserves_stored_timing(self):
        from build_dataset import prepare_shake_timing
        original={'offset_iphone_minus_d435i': 12.3}
        with patch('build_dataset.load_realtime_recording_events', return_value={
                'events': {'valid_data_started': {'system_time_s': 100.}}}):
            prepare_shake_timing(None, original, 'realtime-valid', 'single-peak', .4)
        self.assertEqual(original, {'offset_iphone_minus_d435i': 12.3})


if __name__=='__main__':
    unittest.main()
