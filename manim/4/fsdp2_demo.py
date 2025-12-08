from manim import *


class FSDP2Flow(Scene):
    """
    Visualizes PyTorch FSDP2 (fully_shard) execution flow across GPUs.
    
    Shows the complete lifecycle:
    1. Pre-forward: AllGather parameters
    2. Forward compute with full parameters
    3. Post-forward: Reshard (free full params, keep shards)
    4. Pre-backward: AllGather parameters again
    5. Backward compute with full parameters
    6. Post-backward: ReduceScatter gradients
    """
    
    # Configuration constants
    GPU_BOX_WIDTH = 2.8
    GPU_BOX_HEIGHT = 1.1
    BUFF_BETWEEN_GPUS = 0.8
    STEP_FONT_SIZE = 26
    LABEL_FONT_SIZE = 22
    NUM_GPUS = 4
    
    def create_gpu_box(self, label: str, color=BLUE) -> VGroup:
        """Create a GPU box with label."""
        box = Rectangle(width=self.GPU_BOX_WIDTH, height=self.GPU_BOX_HEIGHT, 
                       color=color, stroke_width=2)
        text = Text(label, font_size=self.LABEL_FONT_SIZE)
        group = VGroup(box, text).arrange(DOWN, buff=0.08)
        return group
    
    def create_gpu_grid(self) -> VGroup:
        """Create initial GPU grid with parameter shards."""
        gpus = VGroup()
        for i in range(self.NUM_GPUS):
            gpu = self.create_gpu_box(f"GPU{i}: shard{i}", color=BLUE)
            gpus.add(gpu)
        return gpus.arrange(RIGHT, buff=self.BUFF_BETWEEN_GPUS)
    
    def update_gpu_labels(self, gpus: VGroup, labels: list, color=None) -> None:
        """Update GPU labels with new text."""
        animations = []
        for gpu, label in zip(gpus, labels):
            # gpu[0] is the box, gpu[1] is the text
            new_text = Text(label, font_size=self.LABEL_FONT_SIZE)
            new_text.move_to(gpu[1].get_center())
            if color:
                gpu[0].set_color(color)
            animations.append(Transform(gpu[1], new_text))
        self.play(*animations, run_time=0.5)
    
    def highlight_gpus(self, gpus: VGroup, color: str = YELLOW) -> SurroundingRectangle:
        """Create and show highlight around GPU grid."""
        highlight = SurroundingRectangle(gpus, color=color, buff=0.2, stroke_width=3)
        self.play(Create(highlight), run_time=0.5)
        return highlight
    
    def create_communication_arrows(self, gpus: VGroup, mode: str = "allgather") -> VGroup:
        """
        Create arrows showing communication pattern.
        
        Args:
            gpus: VGroup of GPU boxes
            mode: "allgather" (bidirectional ring) or "reducescatter" (reverse ring)
        """
        arrows = VGroup()
        positions = [gpu.get_center() for gpu in gpus]
        
        if mode == "allgather":
            # All-to-all communication: simplified with curved arrows
            for i in range(self.NUM_GPUS - 1):
                # Arrow from left to right
                arrow = CurvedArrow(
                    positions[i] + UP * 0.2, 
                    positions[i + 1] + UP * 0.2,
                    color=BLUE_B, stroke_width=3, angle=TAU/8
                )
                arrows.add(arrow)
                # Arrow from right to left
                arrow_back = CurvedArrow(
                    positions[i + 1] + DOWN * 0.2,
                    positions[i] + DOWN * 0.2,
                    color=BLUE_B, stroke_width=3, angle=TAU/8
                )
                arrows.add(arrow_back)
        elif mode == "reducescatter":
            # ReduceScatter: arrows converging to center then scattering
            for i in range(self.NUM_GPUS - 1):
                arrow = Arrow(
                    positions[i] + RIGHT * 0.3,
                    positions[i + 1] + LEFT * 0.3,
                    color=RED_B, stroke_width=3, buff=0.1
                )
                arrows.add(arrow)
        
        return arrows
    
    def show_step(self, step_number: int, step_text: str, old_text: Text = None) -> Text:
        """Display or update step text."""
        new_text = Text(f"Step {step_number}: {step_text}", 
                       font_size=self.STEP_FONT_SIZE, color=WHITE)
        new_text.to_edge(DOWN, buff=0.5)  # Position at bottom to avoid overlap
        
        if old_text is None:
            self.play(Write(new_text), run_time=0.6)
        else:
            self.play(ReplacementTransform(old_text, new_text), run_time=0.4)
        
        return new_text

    def construct(self):
        """Main animation sequence."""
        # Title
        title = Text("PyTorch FSDP2 (fully_shard) Execution Flow", 
                    font_size=42, color=BLUE)
        self.play(FadeIn(title), run_time=0.8)
        self.wait(0.5)
        self.play(title.animate.to_edge(UP, buff=0.4))
        self.wait(0.5)

        # Create GPU grid - position it in the center/slightly above center
        gpus = self.create_gpu_grid()
        gpus.shift(UP * 0.3)  # Shift up slightly to make room for step text at bottom
        self.play(FadeIn(gpus), run_time=0.8)
        self.wait(0.5)

        # === STEP 1: Pre-forward AllGather ===
        step_text = self.show_step(1, "Pre-forward: AllGather → Full Params on Each GPU")
        self.wait(0.3)
        
        # Show communication arrows
        comm_arrows = self.create_communication_arrows(gpus, mode="allgather")
        self.play(Create(comm_arrows), run_time=0.8)
        self.wait(0.5)
        
        # Update labels to show full parameters
        full_param_labels = [f"GPU{i}: FULL param" for i in range(self.NUM_GPUS)]
        self.update_gpu_labels(gpus, full_param_labels)
        self.wait(0.5)
        
        # Fade out communication arrows
        self.play(FadeOut(comm_arrows), run_time=0.4)
        self.wait(0.3)

        # === STEP 2: Forward Compute ===
        step_text = self.show_step(2, "Forward Compute with Full Parameters", step_text)
        highlight = self.highlight_gpus(gpus, color=YELLOW)
        self.wait(1)
        self.play(FadeOut(highlight), run_time=0.4)
        self.wait(0.3)

        # === STEP 3: Post-forward Reshard ===
        step_text = self.show_step(3, "Post-forward: Reshard → Keep Only Local Shard", step_text)
        self.wait(0.3)
        
        shard_labels = [f"GPU{i}: shard{i}" for i in range(self.NUM_GPUS)]
        self.update_gpu_labels(gpus, shard_labels, color=BLUE)
        self.wait(0.5)

        # === STEP 4: Pre-backward AllGather ===
        step_text = self.show_step(4, "Pre-backward: AllGather Parameters Again", step_text)
        self.wait(0.3)
        
        comm_arrows = self.create_communication_arrows(gpus, mode="allgather")
        self.play(Create(comm_arrows), run_time=0.8)
        self.wait(0.5)
        
        # Update labels back to full parameters
        self.update_gpu_labels(gpus, full_param_labels)
        self.wait(0.5)
        
        self.play(FadeOut(comm_arrows), run_time=0.4)
        self.wait(0.3)

        # === STEP 5: Backward Compute ===
        step_text = self.show_step(5, "Backward Compute → Generate Full Gradients", step_text)
        highlight = self.highlight_gpus(gpus, color=GREEN)
        self.wait(1)
        self.play(FadeOut(highlight), run_time=0.4)
        self.wait(0.3)

        # === STEP 6: Post-backward ReduceScatter ===
        step_text = self.show_step(6, "Post-backward: ReduceScatter → Gradient Shards", step_text)
        self.wait(0.3)
        
        comm_arrows = self.create_communication_arrows(gpus, mode="reducescatter")
        self.play(Create(comm_arrows), run_time=0.8)
        self.wait(0.5)
        
        grad_labels = [f"GPU{i}: grad{i}" for i in range(self.NUM_GPUS)]
        self.update_gpu_labels(gpus, grad_labels, color=RED)
        self.wait(0.5)
        
        self.play(FadeOut(comm_arrows), run_time=0.4)
        self.wait(0.3)

        # === Final Summary ===
        self.play(FadeOut(step_text), run_time=0.4)
        final = Text(
            "FSDP2 Complete:\nEach GPU holds only its shard of params & gradients",
            font_size=28, color=GREEN, t2c={"FSDP2": BLUE}
        )
        final.to_edge(DOWN, buff=0.5)
        self.play(Write(final), run_time=1.0)
        self.wait(2)

