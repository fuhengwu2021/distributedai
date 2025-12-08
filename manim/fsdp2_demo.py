from manim import *

class FSDP2Flow(Scene):
    def create_gpu_box(self, label, color=BLUE):
        box = Rectangle(width=3, height=1.2, color=color)
        text = Text(label, font_size=28)
        group = VGroup(box, text).arrange(DOWN, buff=0.1)
        return group

    def construct(self):
        title = Text("PyTorch FSDP2 (fully_shard) Execution Flow", font_size=40)
        self.play(FadeIn(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))

        # GPU boxes
        gpu0 = self.create_gpu_box("GPU0: shard0")
        gpu1 = self.create_gpu_box("GPU1: shard1")
        gpu2 = self.create_gpu_box("GPU2: shard2")
        gpu3 = self.create_gpu_box("GPU3: shard3")

        gpus = VGroup(gpu0, gpu1, gpu2, gpu3).arrange(RIGHT, buff=1)
        self.play(FadeIn(gpus))
        self.wait(1)

        # Step 1: All-Gather
        step1 = Text("Step 1: Pre-forward: AllGather → Full Params on Each GPU",
                     font_size=28).next_to(gpus, DOWN * 2)
        self.play(Write(step1))

        ag_arrows = VGroup(*[
            Arrow(gpu0.get_bottom(), gpu1.get_bottom()),
            Arrow(gpu1.get_bottom(), gpu2.get_bottom()),
            Arrow(gpu2.get_bottom(), gpu3.get_bottom()),
            Arrow(gpu3.get_bottom(), gpu0.get_bottom()),
        ])
        self.play(Create(ag_arrows))
        self.wait(1)

        for gpu in gpus:
            new_label = Text(gpu[1].text.replace("shard", "FULL param"), font_size=28)
            self.play(Transform(gpu[1], new_label))
        self.wait(1)

        # Step 2: Forward compute
        step2 = Text("Step 2: Forward Compute", font_size=28)
        self.play(ReplacementTransform(step1, step2))
        glow = SurroundingRectangle(gpus, color=YELLOW)
        self.play(Create(glow))
        self.wait(1)
        self.play(FadeOut(glow))

        # Step 3: Reshard after forward
        step3 = Text("Step 3: Post-forward: Reshard → Only keep local shard",
                     font_size=28)
        self.play(ReplacementTransform(step2, step3))

        shard_labels = ["shard0", "shard1", "shard2", "shard3"]
        for gpu, lbl in zip(gpus, shard_labels):
            new_label = Text(f"GPU: {lbl}", font_size=28)
            self.play(Transform(gpu[1], new_label))
        self.wait(1)

        # Step 4: Pre-backward AllGather
        step4 = Text("Step 4: Pre-backward: AllGather (same as forward)",
                     font_size=28)
        self.play(ReplacementTransform(step3, step4))
        self.play(Create(ag_arrows))
        self.wait(1)

        for gpu in gpus:
            new_label = Text("GPU: FULL param", font_size=28)
            self.play(Transform(gpu[1], new_label))
        self.wait(1)

        # Step 5: Backward compute
        step5 = Text("Step 5: Backward Compute → Full gradients",
                     font_size=28)
        self.play(ReplacementTransform(step4, step5))
        glow2 = SurroundingRectangle(gpus, color=GREEN)
        self.play(Create(glow2))
        self.wait(1)
        self.play(FadeOut(glow2))

        # Step 6: ReduceScatter gradients
        step6 = Text("Step 6: Post-backward: ReduceScatter → Gradient shards",
                     font_size=28)
        self.play(ReplacementTransform(step5, step6))

        grad_labels = ["grad0", "grad1", "grad2", "grad3"]
        for gpu, lbl in zip(gpus, grad_labels):
            new_label = Text(f"GPU: {lbl}", font_size=28)
            self.play(Transform(gpu[1], new_label))
        self.wait(1)

        final = Text("FSDP2 complete: Each GPU holds only its shard of params & grads",
                     font_size=30).next_to(gpus, DOWN * 2)
        self.play(Write(final))
        self.wait(2)

