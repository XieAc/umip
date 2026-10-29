"""Offline timing of a complete synchronization shake and its short pause."""
from __future__ import annotations

import numpy as np


def refine_burst_offset(d435i, iphone, initial_offset, window_start, window_end):
    """Match the whole gyro burst, avoiding a different oscillation's peak."""
    if d435i.signal_kind != 'gyro' or iphone.signal_kind != 'gyro':
        raise ValueError('Full-burst timing requires gyro signals on both devices')
    selected = (d435i.timestamps_s >= window_start) & (d435i.timestamps_s <= window_end)
    times = d435i.timestamps_s[selected]
    values = d435i.smoothed_signal[selected]
    if len(times) < 12 or np.ptp(values) < 1:
        raise ValueError('Too few samples or no distinct gyro burst for synchronization')
    shifts = np.arange(-1.5, 1.5001, .005)
    scores = []
    for shift in shifts:
        other = np.interp(times + initial_offset + shift, iphone.timestamps_s,
                          iphone.smoothed_signal, left=np.nan, right=np.nan)
        valid = np.isfinite(other)
        if valid.sum() < 12 or valid.mean() < .85 or np.std(other[valid]) < 1e-6:
            scores.append(-1.)
        else:
            scores.append(float(np.corrcoef(values[valid], other[valid])[0, 1]))
    best = int(np.argmax(scores))
    if scores[best] < .75 or best in (0, len(shifts)-1):
        raise ValueError(f'Ambiguous gyro synchronization; best correlation={scores[best]:.3f}')
    initial_score = scores[len(shifts)//2]
    # Do not introduce a tiny new offset solely to chase correlation noise.
    correction = float(shifts[best]) if scores[best] - initial_score > .03 else 0.
    return float(initial_offset + correction), {
        'method': 'gyro_burst_correlation', 'initial_peak_offset_seconds': float(initial_offset),
        'correction_seconds': correction, 'initial_correlation': initial_score,
        'best_correlation': scores[best], 'search_radius_seconds': 1.5,
        'search_step_seconds': .005, 'd435i_samples': len(times),
        'window_start_d435i_time_s': float(window_start),
        'window_end_d435i_time_s': float(window_end),
    }


def find_shake_pause(d435i, iphone, offset, anchor, window_start, window_end,
                     d435i_peak, iphone_peak, pause_seconds=.4):
    """Include repeated oscillations, then require a jointly quiet interval.

    The .3 s confirmation lies within the requested pause; it is not added
    twice. The event-file bound prevents scanning later task movements.
    """
    if not np.isfinite(pause_seconds) or pause_seconds < .3:
        raise ValueError('shake-pause requires at least 0.3 seconds of pause')
    start = max(window_start, d435i.timestamps_s[0], iphone.timestamps_s[0]-offset)
    stop = min(window_end, d435i.timestamps_s[-1], iphone.timestamps_s[-1]-offset)
    times = np.arange(start, stop, .01)
    if len(times) < 32:
        raise ValueError('No shared gyro coverage for shake end detection')
    left = np.interp(times, d435i.timestamps_s, d435i.smoothed_signal)
    right = np.interp(times + offset, iphone.timestamps_s, iphone.smoothed_signal)
    d_threshold, i_threshold = max(1., .2*d435i_peak), max(1., .2*iphone_peak)
    strong = (left >= d_threshold) | (right >= i_threshold)
    indices = np.flatnonzero(strong)
    if not len(indices):
        raise ValueError('No strong gyro burst in synchronization window')
    groups = np.split(indices, np.flatnonzero(np.diff(times[indices]) > 1.) + 1)
    matching = [g for g in groups if times[g[0]]-.3 <= anchor <= times[g[-1]]+.3]
    if len(matching) != 1:
        raise ValueError('Cannot associate synchronization anchor with a unique shake burst')
    burst = matching[0]
    quiet_index = None
    # Include the full pause, not only its first low sample.
    pause_steps = int(np.ceil(pause_seconds/.01))
    for i in range(int(burst[-1]), len(times)-pause_steps):
        if np.all(np.maximum(left, right)[i:i+pause_steps+1] < .5):
            quiet_index = i
            break
    if quiet_index is None:
        raise ValueError('No jointly quiet pause after the complete shake; review this session')
    quiet_start = float(times[quiet_index])
    return {
        'method': 'joint_gyro_complete_burst_then_pause',
        'burst_start_d435i_time_s': float(times[burst[0]]),
        'last_strong_motion_d435i_time_s': float(times[burst[-1]]),
        'quiet_start_d435i_time_s': quiet_start,
        'dataset_start_time_s': quiet_start + pause_seconds,
        'pause_seconds': float(pause_seconds), 'quiet_threshold_rad_s': .5,
        'confirmation_minimum_seconds': .3, 'burst_merge_gap_seconds': 1.,
        'd435i_strong_threshold_rad_s': d_threshold,
        'iphone_strong_threshold_rad_s': i_threshold, 'analysis_grid_seconds': .01,
        'window_end_d435i_time_s': float(window_end),
    }


def refine_task_offset(d435i, iphone, offset, window_start, window_end,
                       radius=.3, min_gain=.03, min_score=.8):
    """Second-stage sync check on the slow task motion after the shake.

    The D435i gyro is logged at only ~9 Hz, which can alias a fast hand shake.
    Task motion is slow, so the rigidly mounted camera and iPhone must show the
    same angular-speed profile there.  Apply a small correction only when it
    clearly improves the match and the peak is not at the search boundary.
    """
    selected = (d435i.timestamps_s >= window_start) & (d435i.timestamps_s <= window_end)
    times = d435i.timestamps_s[selected]
    values = d435i.smoothed_signal[selected]
    details = {'method': 'gyro_task_correlation', 'search_radius_seconds': radius,
               'window_start_d435i_time_s': float(window_start),
               'window_end_d435i_time_s': float(window_end), 'd435i_samples': int(len(times))}
    if len(times) < 20 or np.ptp(values) < .2:
        return float(offset), {**details, 'correction_seconds': 0., 'applied': False,
                               'reason': 'too few samples or no task rotation'}
    shifts = np.arange(-radius, radius + 1e-9, .005)
    scores = []
    for shift in shifts:
        other = np.interp(times + offset + shift, iphone.timestamps_s, iphone.smoothed_signal,
                          left=np.nan, right=np.nan)
        valid = np.isfinite(other)
        if valid.mean() < .9 or np.std(other[valid]) < 1e-6:
            scores.append(-1.)
        else:
            scores.append(float(np.corrcoef(values[valid], other[valid])[0, 1]))
    scores = np.asarray(scores)
    best = int(np.argmax(scores))
    initial = float(scores[len(shifts) // 2])
    apply = (scores[best] >= min_score and scores[best] - initial >= min_gain
             and 0 < best < len(shifts) - 1)
    correction = float(shifts[best]) if apply else 0.
    return float(offset + correction), {
        **details, 'correction_seconds': correction, 'applied': bool(apply),
        'initial_correlation': initial, 'best_correlation': float(scores[best]),
        'best_shift_seconds': float(shifts[best]),
    }
