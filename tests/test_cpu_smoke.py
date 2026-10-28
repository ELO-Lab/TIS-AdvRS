from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from temporal_interval_selection.results import rebuild_fidelity_report
from temporal_interval_selection.scheduled_attack import PGDConfig, run_scheduled_pgd
from temporal_interval_selection.selectors import analyze_intervals
from temporal_interval_selection.temporal import temporal_regularizers


class TinyRuntime:
    @staticmethod
    def build_sliding_windows(frame_count, *, window_size, stride):
        return tuple(
            tuple(range(start, start + window_size))
            for start in range(0, frame_count - window_size + 1, stride)
        )

    @classmethod
    def overlapping_window_indices(cls, frame_count, changed_positions, *, window_size, stride):
        changed = set(changed_positions)
        windows = cls.build_sliding_windows(frame_count, window_size=window_size, stride=stride)
        return tuple(index for index, window in enumerate(windows) if changed.intersection(window))

    @staticmethod
    def compute_ssim(clean, adversarial):
        return float((1.0 - (clean - adversarial).square().mean()).clamp(-1.0, 1.0))


class TinyVideoAdapter:
    @staticmethod
    def get_video_embedding(clip):
        mean = clip.mean(dim=(1, 2, 3, 4))
        return torch.stack((mean, 1.0 - mean), dim=1)


class CpuSmokeTests(unittest.TestCase):
    def test_interval_selection_requires_complete_stride_one_windows(self):
        with self.assertRaisesRegex(ValueError, "every stride-1 interval"):
            analyze_intervals(
                [0.1, 0.2, 0.3, 0.4],
                ((0, 1), (1, 2), (2, 3), (2, 3)),
                frame_count=5,
                target_positions=(0, 1),
                window_size=2,
            )

    def test_default_attack_settings_match_the_manuscript_budget(self):
        config = PGDConfig(device="cpu")

        self.assertEqual(config.epsilon, 8.0 / 255.0)
        self.assertEqual(config.alpha, 2.0 / 255.0)
        self.assertEqual(config.steps, 15)
        self.assertEqual(config.lambda1, 0.5)
        self.assertEqual(config.lambda2, 0.0)
        self.assertEqual(config.final_step_size_ratio, 0.1)

    def test_temporal_losses_match_zero_extended_full_video_formula(self):
        generator = torch.Generator().manual_seed(7)
        edited = torch.randn((3, 2, 2, 2), generator=generator, dtype=torch.float64)
        total_frames = 9
        changed_positions = (3, 4, 5)
        full = torch.zeros((total_frames, 2, 2, 2), dtype=torch.float64)
        full[list(changed_positions)] = edited

        actual = temporal_regularizers(
            edited,
            total_frames=total_frames,
            changed_positions=changed_positions,
        )
        expected_flicker = (full[1:] - full[:-1]).square().mean()
        expected_acceleration = (full[2:] - 2.0 * full[1:-1] + full[:-2]).square().mean()

        torch.testing.assert_close(actual.flicker, expected_flicker)
        torch.testing.assert_close(actual.acceleration, expected_acceleration)

    def test_scheduled_attack_runs_on_cpu_and_only_changes_selected_frames(self):
        frames = torch.linspace(0.2, 0.8, 12).view(12, 1, 1, 1).expand(12, 3, 4, 4).clone()
        changed_positions = (4, 5, 6, 7)
        config = PGDConfig(device="cpu", steps=2, lambda1=0.5, lambda2=0.0)

        adversarial, metrics = run_scheduled_pgd(
            TinyRuntime(),
            TinyVideoAdapter(),
            frames,
            torch.tensor([1.0, 0.0]),
            config,
            window_size=4,
            stride=1,
            changed_positions=changed_positions,
        )

        unchanged = [
            position for position in range(len(frames)) if position not in changed_positions
        ]
        self.assertTrue(torch.equal(adversarial[unchanged], frames[unchanged]))
        self.assertTrue(
            torch.any(adversarial[list(changed_positions)] != frames[list(changed_positions)])
        )
        self.assertLessEqual((adversarial - frames).abs().max().item(), config.epsilon + 1e-7)
        self.assertEqual(metrics["iterations"], 2)
        self.assertEqual(metrics["changed_frame_count"], len(changed_positions))
        self.assertIsNotNone(metrics["psnr"])

    def test_fidelity_is_recovered_per_query_before_averaging(self):
        edited_psnr = (40.0, 45.0)
        edited_ssim = (0.8, 0.9)
        frame_counts = (20, 32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interval_path = root / "intervals.jsonl"
            method_path = root / "results.jsonl"
            interval_rows = []
            result_rows = []
            for index, (psnr, ssim, total_frames) in enumerate(
                zip(edited_psnr, edited_ssim, frame_counts, strict=True)
            ):
                query_id = f"q-{index}"
                interval_rows.append(
                    {
                        "query_id": query_id,
                        "analysis": {"intervals_evaluated": total_frames - 15},
                    }
                )
                result_rows.append(
                    {
                        "query_id": query_id,
                        "status": "success",
                        "measurements": {
                            "changed_frame_count": 16,
                            "psnr": psnr + 10.0 * math.log10(total_frames / 16),
                            "ssim": 1.0 + (ssim - 1.0) * 16 / total_frames,
                            "uses_mock": False,
                        },
                    }
                )
            interval_path.write_text(
                "".join(json.dumps(row) + "\n" for row in interval_rows), encoding="utf-8"
            )
            method_path.write_text(
                "".join(json.dumps(row) + "\n" for row in result_rows), encoding="utf-8"
            )

            report = rebuild_fidelity_report(
                interval_path,
                {"method": method_path},
            )

        method = report["methods"]["method"]
        self.assertAlmostEqual(method["mean_edited_psnr_db_finite_only"], 42.5)
        self.assertAlmostEqual(method["mean_edited_ssim"], 0.85)
        self.assertEqual(report["aggregation"], "recover per query, then take the arithmetic mean")


if __name__ == "__main__":
    unittest.main()
