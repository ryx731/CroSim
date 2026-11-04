import time
import unittest
from unittest.mock import patch

import numpy as np

from crosim_agent.timer import MatchClock,TimerReader,read_match_timer
from tests.test_agent import live_fixture


class MatchTimerTests(unittest.TestCase):
    def test_end_requires_separate_zero_reads_and_a_confirmation_interval(self):
        clock = MatchClock()
        clock.observe(1,10,now=10)
        clock.observe(0,10.2,now=10.2)
        self.assertFalse(clock.can_restart(10.2))
        clock.observe(0,10.8,now=10.8)
        self.assertFalse(clock.can_restart(10.8))
        clock.observe(0,11.3,now=11.3)
        self.assertFalse(clock.can_restart(11.3))
        self.assertTrue(clock.can_restart(13.3))

    def test_missing_stale_or_duplicate_read_cannot_authorize_restart(self):
        clock = MatchClock()
        for _ in range(20):
            clock.observe(0,10,now=10)
        self.assertFalse(clock.can_restart(10))
        clock.observe(0,10.5,now=20)
        clock.observe(None,20,now=20)
        self.assertTrue(clock.can_restart(20))
        fresh = MatchClock()
        fresh.observe(0,10,now=20)
        self.assertFalse(fresh.can_restart(20))

    def test_positive_clock_cancels_an_end_candidate(self):
        clock = MatchClock()
        clock.observe(0,10,now=10)
        clock.observe(0,10.5,now=10.5)
        clock.observe(120,11,now=11)
        self.assertFalse(clock.can_restart(11))
        self.assertEqual(clock.remaining(11),120)

    def test_each_end_can_restart_only_once_until_new_positive_clock(self):
        clock = MatchClock()
        for at in (10,10.5,11.1):
            clock.observe(0,at,now=at)
        self.assertTrue(clock.can_restart(13.1))
        clock.restarted()
        for at in (12,12.5,13.1):
            clock.observe(0,at,now=at)
        self.assertFalse(clock.can_restart(13.1))
        clock.observe(160,14,now=14)
        self.assertFalse(clock.awaiting_positive)

    def test_timer_reader_rejects_a_false_zero_jump_and_supports_manual_restart(self):
        reader = TimerReader()
        crop = np.zeros((96,320,3),np.uint8)
        with patch("crosim_agent.timer.read_match_timer",side_effect=[136,136,0,0,135,160,160]):
            for at in (0,.3,.6,.9):reader.update(crop,at)
            self.assertEqual(reader.seconds,0)
            reader.update(crop,1.2)
            self.assertEqual(reader.seconds,0)
            reader.update(crop,1.5);reader.update(crop,1.8)
            self.assertEqual(reader.seconds,160)

    def test_blank_timer_crop_is_unknown(self):
        self.assertIsNone(read_match_timer(np.zeros((96,320,3),np.uint8)))

    def test_bad_minute_read_does_not_lock_out_zero_and_reset_waits_three_seconds(self):
        reader = TimerReader()
        reader.seconds,reader.at = 267,0
        with patch("crosim_agent.timer.read_match_timer",side_effect=[0,0]):
            reader.update(np.zeros((10,10,3),np.uint8),.3)
            reader.update(np.zeros((10,10,3),np.uint8),.6)
        self.assertEqual(reader.seconds,0)
        clock = MatchClock()
        clock.observe(reader.seconds,reader.at,now=.6)
        self.assertFalse(clock.can_restart(3.5))
        self.assertTrue(clock.can_restart(3.61))

    def test_initial_attach_and_control_gap_do_not_send_restart(self):
        with live_fixture() as (env,pdi):
            env.reset()
            self.assertNotIn(("down","r"),pdi.events)
            env._live_start = time.perf_counter()-150
            env.reward_engine.idle_seconds = 100
            self.assertFalse(env.step(0)[2])
            env._last_control_end = time.perf_counter()-10
            self.assertTrue(env.step(1)[3])
            env.reset()
            self.assertNotIn(("down","r"),pdi.events)

    def test_live_reset_is_authorized_only_after_confirmed_zero(self):
        with live_fixture() as (env,pdi):
            env.reset()
            now = time.perf_counter()
            for at in (now-1.1,now-.5,now):env.match_clock.observe(0,at,now=now)
            self.assertTrue(env.step(0)[2])
            env.reset()
            self.assertEqual(pdi.events.count(("down","r")),1)
            env.reset()
            self.assertEqual(pdi.events.count(("down","r")),1)
