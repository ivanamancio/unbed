"""Unit tests for pipeline helpers that do not need ffmpeg, torch, or Demucs."""
import os
import unittest

import numpy as np

from unbed import pipeline


def _restore(name, old):
    if old is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old


class DemucsCommandTest(unittest.TestCase):
    def test_cpu_passes_jobs_as_its_own_arguments(self):
        cmd = pipeline._demucs_cmd("in.wav", "out", "cpu")
        self.assertEqual(cmd[cmd.index("-j") + 1], "2")
        self.assertLess(cmd.index("-j"), cmd.index("-o"))
        self.assertEqual(cmd[-2:], ["out", "in.wav"])

    def test_jobs_and_segment_are_read_when_the_command_is_built(self):
        old_jobs, old_seg = os.environ.get("UNBED_DEMUCS_JOBS"), os.environ.get("UNBED_DEMUCS_SEGMENT")
        os.environ["UNBED_DEMUCS_JOBS"] = "4"
        os.environ["UNBED_DEMUCS_SEGMENT"] = "5"
        try:
            cmd = pipeline._demucs_cmd("in.wav", "out", "cpu")
        finally:
            _restore("UNBED_DEMUCS_JOBS", old_jobs)
            _restore("UNBED_DEMUCS_SEGMENT", old_seg)
        self.assertEqual(cmd[cmd.index("-j") + 1], "4")
        self.assertEqual(cmd[cmd.index("--segment") + 1], "5")

    def test_bad_attenuation_falls_back_instead_of_crashing(self):
        old = os.environ.get("UNBED_ALT_DB")
        os.environ["UNBED_ALT_DB"] = "loud"
        try:
            self.assertEqual(pipeline._env_float("UNBED_ALT_DB", -25.0), -25.0)
        finally:
            _restore("UNBED_ALT_DB", old)

    def test_gpu_does_not_pass_jobs(self):
        cmd = pipeline._demucs_cmd("in.wav", "out", "cuda")
        self.assertNotIn("-j", cmd)
        self.assertEqual(cmd[-2:], ["out", "in.wav"])


class FidelityTest(unittest.TestCase):
    def setUp(self):
        sr = 16000
        t = np.linspace(0, 1, sr, endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        self.sr = sr
        self.orig = np.stack([tone, tone], axis=1)

    def test_clean_stem_is_not_damaged(self):
        fid = pipeline.fidelity(self.orig, self.orig.copy(), np.zeros_like(self.orig), self.sr)
        self.assertFalse(fid["damaged"])

    def test_silent_stem_is_damaged(self):
        fid = pipeline.fidelity(self.orig, np.zeros_like(self.orig), np.zeros_like(self.orig), self.sr)
        self.assertTrue(fid["damaged"])
        self.assertTrue(any("voice" in reason for reason in fid["damage_reasons"]))


class LoginPatternTest(unittest.TestCase):
    def test_unrelated_cookie_text_is_not_a_login_wall(self):
        self.assertIsNone(pipeline.LOGIN_PATTERNS.search("saved a copy of cookies.txt.bak"))
        self.assertIsNotNone(pipeline.LOGIN_PATTERNS.search("Sign in to confirm you're not a bot"))
        self.assertIsNotNone(pipeline.LOGIN_PATTERNS.search("ERROR: Use --cookies-from-browser or --cookies"))


class CreditDateTest(unittest.TestCase):
    def test_bad_upload_date_does_not_abort(self):
        import tempfile
        info = {"title": "Talk", "upload_date": "not-a-date", "duration": 3, "webpage_url": "https://example.com/v"}
        fid = {"duration": 3.0}
        with tempfile.TemporaryDirectory() as work:
            path = os.path.join(work, "credit.md")
            pipeline.write_credit(path, info, info["webpage_url"], fid, None, "en")
            with open(path, encoding="utf-8") as f:
                text = f.read()
        self.assertIn("Published: ?", text)


class CompactCopyTest(unittest.TestCase):
    def test_skips_a_file_that_is_already_under_the_limit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as work:
            src = os.path.join(work, "main.mp4")
            with open(src, "wb") as f:
                f.write(b"x" * 100)
            out = os.path.join(work, "small.mp4")
            self.assertIsNone(pipeline.compact_copy(src, out, 10))
            self.assertFalse(os.path.exists(out))


class InfoJsonTest(unittest.TestCase):
    def test_falls_back_to_the_single_info_json(self):
        import tempfile
        with tempfile.TemporaryDirectory() as work:
            path = os.path.join(work, "src.mp4.info.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"id": "abc"}')
            self.assertEqual(pipeline._load_info_json(work)["id"], "abc")
            self.assertTrue(os.path.isfile(os.path.join(work, "src.info.json")))

    def test_broken_info_json_is_a_pipeline_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as work:
            with open(os.path.join(work, "src.info.json"), "w", encoding="utf-8") as f:
                f.write("{")
            with self.assertRaises(pipeline.PipelineError):
                pipeline._load_info_json(work)

    def test_missing_info_json_is_a_pipeline_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as work:
            with self.assertRaises(pipeline.PipelineError):
                pipeline._load_info_json(work)


class LoudnormStatsTest(unittest.TestCase):
    def test_parses_the_stats_object_when_later_text_contains_a_brace(self):
        text = """
[Parsed_loudnorm_0 @ 0x1]
{
    "input_i" : "-23.0",
    "input_tp" : "-3.0",
    "input_lra" : "5.0",
    "input_thresh" : "-33.0",
    "target_offset" : "0.50"
}
frame= 1 fps=0.0 q=-0.0 size=N/A time=00:00:01.00 {not-json
"""
        stats = pipeline._loudnorm_stats(text, "input_i")
        self.assertEqual(pipeline._loudnorm_floats(stats, ("input_i", "target_offset")), (-23.0, 0.50))

    def test_missing_stats_are_a_pipeline_error(self):
        with self.assertRaises(pipeline.PipelineError):
            pipeline._loudnorm_stats("no json here {still not", "input_i")


class ToolErrorTest(unittest.TestCase):
    def test_missing_program_is_a_pipeline_error(self):
        with self.assertRaises(pipeline.PipelineError) as caught:
            pipeline.run(["/no/such/program"])
        self.assertIn("Could not run", caught.exception.pt)

    def test_timeout_is_a_pipeline_error(self):
        with self.assertRaises(pipeline.PipelineError) as caught:
            pipeline.run([os.environ.get("PYTHON", "python3"), "-c", "import time; time.sleep(5)"], timeout=0.2)
        self.assertIn("too long", caught.exception.pt)


class NamingTest(unittest.TestCase):
    def test_is_url(self):
        self.assertTrue(pipeline.is_url("https://example.com/a"))
        self.assertFalse(pipeline.is_url("/tmp/clip.mp4"))

    def test_video_key_prefixes_non_youtube_ids(self):
        self.assertEqual(pipeline.video_key({"extractor_key": "youtube", "id": "abc"}), "abc")
        self.assertEqual(pipeline.video_key({"extractor_key": "twitter", "id": "a/b"}), "twitter_a_b")


if __name__ == "__main__":
    unittest.main()
