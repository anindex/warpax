from __future__ import annotations

import os as _os
import sys as _sys

if "jax" not in _sys.modules:
    _os.environ.setdefault("JAX_PLATFORMS", "cpu")
    _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _os.environ.setdefault("MKL_NUM_THREADS", "1")
    _os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "1")
    if "xla_cpu_multi_thread_eigen" not in _os.environ.get("XLA_FLAGS", ""):
        _os.environ["XLA_FLAGS"] = (
            _os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
        ).strip()

import numpy as np
from manim import (
    DEGREES,
    DOWN,
    UP,
    Group,
    ImageMobject,
    MathTex,
    Text,
    ThreeDScene,
    ValueTracker,
    VGroup,
    VMobject,
    linear,
)

from warpax.benchmarks import AlcubierreMetric
from warpax.geometry import GridSpec
from warpax.visualization.common._physics import build_ec_frame_sequence, make_velocity_sweep
from warpax.visualization.manim._image_utils import frame_to_rgba
from warpax.visualization.manim._scene_utils import (
    compute_auto_exaggeration,
    compute_global_clim,
    format_metric_equation,
    make_axes_for_frames,
)
from warpax.visualization.manim._surface import auto_linthresh, symlog_height


class WallAndVelocitySweep(ThreeDScene):
    GRID_N = 321

    def construct(self) -> None:
        grid_spec = GridSpec(
            bounds=[(-2, 2), (-2, 2), (0, 0)],
            shape=(self.GRID_N, self.GRID_N, 1),
        )
        sigma_frames = self._build_sigma_frames(grid_spec)
        speed_frames = self._build_rampdown_frames(grid_spec)
        frames = sigma_frames + speed_frames[1:]
        sigmas = np.concatenate(
            [np.linspace(1.0, 16.0, len(sigma_frames)), np.full(len(speed_frames) - 1, 16.0)]
        )
        self.camera.background_color = "#10121B"
        self.set_camera_orientation(phi=65 * DEGREES, theta=-60 * DEGREES)
        title = Text("Alcubierre bubble: expansion and energy", font_size=30)
        title.to_edge(UP, buff=0.22)
        equation = format_metric_equation("Alcubierre").scale(0.7)
        equation.move_to([0, 3.08, 0])
        subtitle = Text(
            "Fixed-center parameter sweep  |  motion direction: +x",
            font_size=17,
            color="#BAC4D5",
        ).move_to([0, 2.62, 0])
        self.add_fixed_in_frame_mobjects(title, equation, subtitle)

        labels = [
            (
                "Expansion",
                r"\theta = \nabla_a n^a = -\mathrm{Tr}\,K",
                "Positive behind; negative ahead",
                -3.5,
                "#EF8A62",
            ),
            (
                "Eulerian energy density",
                r"\rho_{\rm Eul} = T_{ab}n^a n^b \leq 0",
                "Non-positive; zero on the motion axis",
                3.5,
                "#67A9CF",
            ),
        ]
        for name, formula, detail, x, color in labels:
            heading = Text(name, font_size=24, color=color).move_to([x, 1.8, 0])
            definition = MathTex(formula, font_size=24, color=color).move_to([x, 1.33, 0])
            explanation = Text(detail, font_size=17).move_to([x, -2.05, 0])
            self.add_fixed_in_frame_mobjects(heading, definition, explanation)

        axes_list = []
        scales = []
        thresholds = []
        screen_right = np.array([np.sqrt(3) / 2, 0.5, 0.0])
        screen_up = np.array(
            [
                -0.5 * np.cos(65 * DEGREES),
                np.sqrt(3) / 2 * np.cos(65 * DEGREES),
                np.sin(65 * DEGREES),
            ]
        )
        for field, side in [("theta", -1), ("energy_density", 1)]:
            maximum = max(abs(v) for v in compute_global_clim(frames, field))
            threshold = auto_linthresh(maximum)
            axes = make_axes_for_frames(frames, field, coord_range=(-2, 2), linthresh=threshold)
            axes.scale(0.68).shift(side * 3.45 * screen_right - 0.4 * screen_up)
            axes_list.append(axes)
            scales.append(compute_auto_exaggeration(frames, field, linthresh=threshold))
            thresholds.append(threshold)
            x_label = Text("+x", font_size=19).move_to(axes.c2p(2.3, 0, 0))
            self.add_fixed_orientation_mobjects(x_label)
            self.add(axes)

        nec_min = compute_global_clim(frames, "nec_margin_sweep")[0]
        nec_threshold = auto_linthresh(abs(nec_min))
        nec_images = [
            frame_to_rgba(
                frame,
                "nec_margin_sweep",
                (nec_min, 0.0, nec_threshold),
                cmap_name="nec_depth",
            )
            for frame in frames
        ]
        nec_image = ImageMobject(nec_images[0]).set(height=1.25).move_to([0, -2.13, 0])
        nec_label = MathTex(
            r"m_{\rm NEC} = \min_{1\leq j\leq312} T_{ab}k_j^a k_j^b,"
            r"\quad k_j\!\cdot n=-1",
            font_size=22,
        ).move_to([0, -3.06, 0])
        nec_key = Text("NEC: brighter = more negative", font_size=16).move_to([3.5, -2.62, 0])
        self.add_fixed_in_frame_mobjects(nec_image, nec_label, nec_key)

        footer = Text(
            "Scalar heights, signed log scales; not a spatial embedding or time evolution",
            font_size=18,
            color="#BAC4D5",
        ).to_edge(DOWN, buff=0.12)
        units = Text("G = c = R = 1; z = 0; signature (−,+,+,+)", font_size=15)
        units.next_to(footer, UP, buff=0.12)
        self.add_fixed_in_frame_mobjects(footer, units)

        tracker = ValueTracker(0)
        meshes = []
        for field, axes, scale, threshold in zip(
            ["theta", "energy_density"], axes_list, scales, thresholds, strict=True
        ):
            mesh = self._wireframe(frames[0], field, axes, scale, threshold)
            meshes.append(mesh)

        counters = MathTex(r"\sigma=1.0\quad v_s=0.50", font_size=25, color="#FFDB70")
        self.add_fixed_in_frame_mobjects(counters)
        current_index = -1

        def update_frame(_mob):
            nonlocal current_index
            index = min(int(tracker.get_value()), len(frames) - 1)
            if index == current_index:
                return
            current_index = index
            frame = frames[index]
            for mesh, field, axes, scale, threshold in zip(
                meshes,
                ["theta", "energy_density"],
                axes_list,
                scales,
                thresholds,
                strict=True,
            ):
                mesh.become(self._wireframe(frame, field, axes, scale, threshold))
            nec_image.become(
                ImageMobject(nec_images[index]).set(height=1.25).move_to([0, -2.13, 0])
            )
            status = r"\quad\text{flat spacetime}" if frame.v_s == 0 else ""
            counters.become(
                MathTex(
                    rf"\sigma = {sigmas[index]:.1f}\quad v_s = {frame.v_s:.2f}{status}",
                    font_size=25,
                    color="#FFDB70",
                ).move_to([0, 2.2, 0])
            )
            self.camera.add_fixed_in_frame_mobjects(counters, nec_image)

        plots = Group(*meshes, counters, nec_image)
        plots.add_updater(update_frame)
        self.add(plots, tracker)
        update_frame(tracker)
        self.wait(1)
        self.play(tracker.animate.set_value(len(frames) - 1), run_time=18, rate_func=linear)
        self.wait(2)

    @staticmethod
    def _wireframe(frame, field, axes, scale, threshold):
        values = frame.scalar_fields[field][:, :, 0]
        height = symlog_height(values, threshold) * scale
        origin = axes.c2p(0, 0, 0)
        points = (
            origin
            + frame.x[:, :, 0, None] * (axes.c2p(1, 0, 0) - origin)
            + frame.y[:, :, 0, None] * (axes.c2p(0, 1, 0) - origin)
            + height[:, :, None] * (axes.c2p(0, 0, 1) - origin)
        )
        mesh = VGroup()
        indices = np.linspace(0, values.shape[0] - 1, 41, dtype=int)
        middle = values.shape[0] // 2
        for index in indices:
            for line, samples in [
                (points[index], values[index]),
                (points[:, index], values[:, index]),
            ]:
                for part in [slice(0, middle + 1), slice(middle, None)]:
                    color = "#EF8A62" if field == "theta" and samples[part].sum() > 0 else "#67A9CF"
                    mesh.add(
                        VMobject(
                            stroke_color=color, stroke_width=0.8, stroke_opacity=0.85
                        ).set_points_as_corners(line[part])
                    )
        return mesh

    @staticmethod
    def _build_sigma_frames(grid_spec, n_frames: int = 25):
        frames = []
        for sigma in np.linspace(1.0, 16.0, n_frames):
            frames.extend(
                build_ec_frame_sequence(
                    AlcubierreMetric(v_s=0.5, sigma=float(sigma)),
                    grid_spec,
                    v_s_values=[0.5],
                    batch_size=2048,
                    progress=False,
                )
            )
        return frames

    @staticmethod
    def _build_rampdown_frames(grid_spec, n_frames: int = 25):
        return build_ec_frame_sequence(
            AlcubierreMetric(v_s=0.5, sigma=16.0),
            grid_spec,
            v_s_values=make_velocity_sweep(0.5, 0.0, n_frames),
            batch_size=2048,
            progress=True,
        )
