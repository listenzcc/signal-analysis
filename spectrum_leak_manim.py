from manim import *
import numpy as np
import os


class SpectrumLeak(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        title = Text("Spectrum Leak Analysis",
                     font="Comic Sans MS", color=WHITE).scale(0.75)
        self.play(Write(title), subcaption_duration=0.1)
        self.wait(0.1)
        self.play(title.animate.to_edge(UP))

        sampling_rate = 1000  # 1000 Hz
        freq = 50  # Hz
        t = np.linspace(0, 1, sampling_rate, endpoint=False)

        signal = np.sin(2 * np.pi * freq * t)

        self.clear()
        self.add(title)

        axes = Axes(
            x_range=[-100, 100, 20],
            y_range=[0, 50, 10],
            axis_config={"color": WHITE},
            x_axis_config={"include_numbers": True, "color": WHITE},
            y_axis_config={"include_numbers": True, "color": WHITE},
        ).scale(0.75).to_edge(DOWN)

        self.play(Create(axes))

        length_label = Text("--", font="Comic Sans MS",
                            color=WHITE).scale(0.5)
        self.add(length_label)
        length_label.to_edge(UP)

        for length in range(0, 20):
            color = interpolate_color(BLUE, RED, length / 20)
            new_length_label = Text(
                f"Length: {length}", font="Comic Sans MS", color=color).scale(0.5)
            graph = self.plot_spectrum_leak(
                axes, signal, sampling_rate, length, color)

            graph2 = self.plot_spectrum_leak(
                axes, signal, sampling_rate, length, color, 800, reverse=True)

            self.play((Transform(length_label, new_length_label),
                       Create(graph),
                       Create(graph2)),
                      subcaption_duration=0.02)

            self.wait(0.01)

            # Change graph color to white and reduce linewidth
            self.play((graph.animate.set_stroke(color=GRAY, width=1, opacity=0.5),
                       graph2.animate.set_stroke(color=GRAY, width=1, opacity=0.5)), subcaption_duration=0.1)

    def plot_spectrum_leak(self, axes, signal, sampling_rate, length, color, offset=200, reverse=False):
        length = length + offset
        signal = signal[:length]
        fft_result = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal), d=1/sampling_rate)

        fft_result = np.fft.fftshift(fft_result)
        fft_freq = np.fft.fftshift(fft_freq)

        fft_result = fft_result[np.abs(fft_freq) < 100]
        fft_freq = fft_freq[np.abs(fft_freq) < 100]

        if reverse:
            fft_freq = fft_freq[::-1]

        graph = axes.plot_line_graph(
            x_values=fft_freq,
            y_values=np.abs(fft_result),
            line_color=color,
            add_vertex_dots=False,
            stroke_width=5,  # Initial linewidth
        )

        return graph


if __name__ == "__main__":
    from manim import config
    config.media_width = "100%"
    config.verbosity = "WARNING"
    scene = SpectrumLeak()
    scene.render()
