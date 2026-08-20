import math
import random
import svgwrite


# ============================================================
# generate_cover_bg_v5_svgwrite.py
#
# Distributed AI / systems background
# SVG version:
# - more polished than matplotlib
# - stronger glass / glow feeling
# - fewer messy arrows
# - cleaner orbit / field-line structure
# ============================================================


# ----------------------------
# Canvas
# ----------------------------
W, H = 1800, 2400
OUTPUT = "distributed_cover_bg_v5.svg"

random.seed(8)

dwg = svgwrite.Drawing(
    OUTPUT,
    size=(W, H),
    profile="full"
)

dwg.viewbox(0, 0, W, H)

# white background
dwg.add(
    dwg.rect(
        insert=(0, 0),
        size=(W, H),
        fill="white"
    )
)


# ----------------------------
# Defs: gradients + filters
# ----------------------------
defs = dwg.defs


def add_linear_gradient(name, stops, start=("0%", "0%"), end=("100%", "100%")):
    g = dwg.linearGradient(
        id=name,
        start=start,
        end=end
    )
    for offset, color, opacity in stops:
        g.add_stop_color(offset, color, opacity)
    defs.add(g)
    return f"url(#{name})"


def add_radial_gradient(name, stops):
    g = dwg.radialGradient(id=name)
    for offset, color, opacity in stops:
        g.add_stop_color(offset, color, opacity)
    defs.add(g)
    return f"url(#{name})"


def add_blur_filter(name, std_dev):
    f = dwg.filter(id=name)
    f.feGaussianBlur(stdDeviation=std_dev)
    defs.add(f)
    return f"url(#{name})"


glass_purple = add_linear_gradient(
    "glass_purple",
    [
        ("0%", "#ffffff", 0.55),
        ("35%", "#b58cff", 0.34),
        ("100%", "#5c35e8", 0.26),
    ],
    start=("0%", "0%"),
    end=("100%", "100%")
)

glass_blue = add_linear_gradient(
    "glass_blue",
    [
        ("0%", "#ffffff", 0.55),
        ("45%", "#68c7ff", 0.32),
        ("100%", "#2f6dff", 0.22),
    ],
    start=("0%", "0%"),
    end=("100%", "100%")
)

glass_pink = add_linear_gradient(
    "glass_pink",
    [
        ("0%", "#ffffff", 0.52),
        ("42%", "#ff77cf", 0.34),
        ("100%", "#8c55ff", 0.24),
    ],
    start=("0%", "0%"),
    end=("100%", "100%")
)

cube_top_grad = add_linear_gradient(
    "cube_top_grad",
    [
        ("0%", "#ffffff", 0.70),
        ("50%", "#bde9ff", 0.48),
        ("100%", "#ee8bff", 0.38),
    ],
    start=("0%", "0%"),
    end=("100%", "100%")
)

cube_left_grad = add_linear_gradient(
    "cube_left_grad",
    [
        ("0%", "#8fd9ff", 0.48),
        ("50%", "#4c91ff", 0.40),
        ("100%", "#725eff", 0.34),
    ],
    start=("0%", "0%"),
    end=("100%", "100%")
)

cube_right_grad = add_linear_gradient(
    "cube_right_grad",
    [
        ("0%", "#ffa2df", 0.46),
        ("45%", "#b96cff", 0.40),
        ("100%", "#5845e8", 0.34),
    ],
    start=("0%", "0%"),
    end=("100%", "100%")
)

node_purple_grad = add_radial_gradient(
    "node_purple_grad",
    [
        ("0%", "#ffffff", 0.95),
        ("30%", "#c7a4ff", 0.96),
        ("100%", "#6736e8", 0.98),
    ]
)

node_pink_grad = add_radial_gradient(
    "node_pink_grad",
    [
        ("0%", "#ffffff", 0.95),
        ("30%", "#ff9ee0", 0.96),
        ("100%", "#e51c8b", 0.98),
    ]
)

node_blue_grad = add_radial_gradient(
    "node_blue_grad",
    [
        ("0%", "#ffffff", 0.95),
        ("32%", "#7cc8ff", 0.96),
        ("100%", "#006ef0", 0.98),
    ]
)

node_cyan_grad = add_radial_gradient(
    "node_cyan_grad",
    [
        ("0%", "#ffffff", 0.95),
        ("32%", "#8eeaff", 0.96),
        ("100%", "#00a7cf", 0.98),
    ]
)

glow_filter = add_blur_filter("soft_glow", 18)
strong_glow_filter = add_blur_filter("strong_glow", 34)
tiny_glow_filter = add_blur_filter("tiny_glow", 8)


# ----------------------------
# Coordinate helpers
# ----------------------------
CX, CY = W / 2, H / 2 + 60


def P(x, y):
    """
    Logical coordinate to SVG coordinate.
    Logical space roughly:
    x: -4 to 4
    y: -5 to 5
    """
    scale = 205
    return CX + x * scale, CY - y * scale


def poly_points(points):
    return [P(x, y) for x, y in points]


def color_with_opacity(color, opacity):
    return color, opacity


# ----------------------------
# SVG helpers
# ----------------------------
def add_glow_circle(x, y, r, color, opacity=0.18, blur=True):
    px, py = P(x, y)

    dwg.add(
        dwg.circle(
            center=(px, py),
            r=r,
            fill=color,
            opacity=opacity,
            filter=strong_glow_filter if blur else None
        )
    )


def add_node(x, y, r, grad_url, glow_color):
    px, py = P(x, y)

    dwg.add(
        dwg.circle(
            center=(px, py),
            r=r * 2.8,
            fill=glow_color,
            opacity=0.16,
            filter=glow_filter
        )
    )

    dwg.add(
        dwg.circle(
            center=(px, py),
            r=r,
            fill=grad_url,
            opacity=0.98
        )
    )

    dwg.add(
        dwg.circle(
            center=(px - r * 0.25, py - r * 0.28),
            r=r * 0.26,
            fill="white",
            opacity=0.60
        )
    )


def add_platform(cx, cy, w, h, fill_url, stroke_color, opacity=0.55):
    pts = [
        (cx, cy + h / 2),
        (cx + w / 2, cy),
        (cx, cy - h / 2),
        (cx - w / 2, cy),
    ]

    dwg.add(
        dwg.polygon(
            points=poly_points(pts),
            fill=fill_url,
            stroke=stroke_color,
            stroke_width=1.5,
            opacity=opacity
        )
    )


def add_network(cx, cy, scale, stroke_color, node_grad, glow_color):
    pts = [
        (0.00, 0.60),
        (-0.55, 0.25),
        (0.55, 0.25),
        (-0.55, -0.34),
        (0.55, -0.34),
        (0.00, -0.68),
        (0.00, 0.00),
    ]

    pts = [(cx + x * scale, cy + y * scale) for x, y in pts]

    edges = [
        (0, 1), (0, 2), (0, 6),
        (1, 2), (1, 3), (1, 6),
        (2, 4), (2, 6),
        (3, 4), (3, 5), (3, 6),
        (4, 5), (4, 6),
        (5, 6),
    ]

    for i, j in edges:
        x1, y1 = P(*pts[i])
        x2, y2 = P(*pts[j])

        dwg.add(
            dwg.line(
                start=(x1, y1),
                end=(x2, y2),
                stroke=stroke_color,
                stroke_width=2.0,
                opacity=0.34,
                stroke_linecap="round"
            )
        )

    for x, y in pts:
        add_node(x, y, r=22, grad_url=node_grad, glow_color=glow_color)


def add_soft_link(p0, p1, p2, color, width=2.0, opacity=0.12):
    """
    Soft quadratic curve without arrow.
    """
    x0, y0 = P(*p0)
    x1, y1 = P(*p1)
    x2, y2 = P(*p2)

    d = f"M {x0},{y0} Q {x1},{y1} {x2},{y2}"

    dwg.add(
        dwg.path(
            d=d,
            fill="none",
            stroke=color,
            stroke_width=width,
            opacity=opacity,
            stroke_linecap="round"
        )
    )


def add_small_cube(cx, cy, s, color1, color2):
    """
    Tiny isometric cube.
    """
    top = [
        (cx, cy + s),
        (cx + s, cy + s * 0.50),
        (cx, cy),
        (cx - s, cy + s * 0.50),
    ]
    left = [
        (cx - s, cy + s * 0.50),
        (cx, cy),
        (cx, cy - s),
        (cx - s, cy - s * 0.50),
    ]
    right = [
        (cx + s, cy + s * 0.50),
        (cx, cy),
        (cx, cy - s),
        (cx + s, cy - s * 0.50),
    ]

    for pts, col, op in [
        (top, color1, 0.34),
        (left, color2, 0.24),
        (right, color1, 0.20),
    ]:
        dwg.add(
            dwg.polygon(
                points=poly_points(pts),
                fill=col,
                stroke="white",
                stroke_width=1.0,
                opacity=op
            )
        )


def add_center_cube(cx, cy, s):
    """
    Larger polished translucent central cube.
    """
    # glow behind cube
    px, py = P(cx, cy - 0.18)

    dwg.add(
        dwg.ellipse(
            center=(px, py + 48),
            r=(210, 150),
            fill="#7b4dff",
            opacity=0.16,
            filter=strong_glow_filter
        )
    )

    top = [
        (cx, cy + s * 0.68),
        (cx + s * 0.68, cy + s * 0.32),
        (cx, cy - s * 0.04),
        (cx - s * 0.68, cy + s * 0.32),
    ]

    left = [
        (cx - s * 0.68, cy + s * 0.32),
        (cx, cy - s * 0.04),
        (cx, cy - s * 0.94),
        (cx - s * 0.68, cy - s * 0.58),
    ]

    right = [
        (cx + s * 0.68, cy + s * 0.32),
        (cx, cy - s * 0.04),
        (cx, cy - s * 0.94),
        (cx + s * 0.68, cy - s * 0.58),
    ]

    faces = [
        (left, cube_left_grad),
        (right, cube_right_grad),
        (top, cube_top_grad),
    ]

    for pts, fill in faces:
        dwg.add(
            dwg.polygon(
                points=poly_points(pts),
                fill=fill,
                stroke="white",
                stroke_width=2.0,
                opacity=0.90
            )
        )

    # internal white grid
    grid_lines = [
        # top grid
        ((cx - s * 0.34, cy + s * 0.50), (cx + s * 0.34, cy + s * 0.14)),
        ((cx - s * 0.34, cy + s * 0.14), (cx + s * 0.34, cy + s * 0.50)),

        # vertical separator
        ((cx, cy + s * 0.66), (cx, cy - s * 0.94)),

        # face hint lines
        ((cx - s * 0.34, cy + s * 0.14), (cx - s * 0.34, cy - s * 0.76)),
        ((cx + s * 0.34, cy + s * 0.14), (cx + s * 0.34, cy - s * 0.76)),

        ((cx - s * 0.54, cy - s * 0.34), (cx, cy - s * 0.64)),
        ((cx + s * 0.54, cy - s * 0.34), (cx, cy - s * 0.64)),
    ]

    for a, b in grid_lines:
        x1, y1 = P(*a)
        x2, y2 = P(*b)
        dwg.add(
            dwg.line(
                start=(x1, y1),
                end=(x2, y2),
                stroke="white",
                stroke_width=1.3,
                opacity=0.42,
                stroke_linecap="round"
            )
        )


# ----------------------------
# Layout
# ----------------------------
nodes = {
    "top": {
        "pos": (0.0, 2.15),
        "color": "#7a45f5",
        "platform": glass_purple,
        "node": node_purple_grad,
        "glow": "#7a45f5",
    },
    "left": {
        "pos": (-2.45, 0.62),
        "color": "#ef2f95",
        "platform": glass_pink,
        "node": node_pink_grad,
        "glow": "#ef2f95",
    },
    "right": {
        "pos": (2.45, 0.62),
        "color": "#1479f5",
        "platform": glass_blue,
        "node": node_blue_grad,
        "glow": "#1479f5",
    },
    "bottom": {
        "pos": (0.0, -2.18),
        "color": "#10b6d9",
        "platform": glass_blue,
        "node": node_cyan_grad,
        "glow": "#10b6d9",
    },
}

cube_center = (0.0, -0.12)


# ----------------------------
# Soft local links
# ----------------------------
# No arrows, no spaghetti.
add_soft_link(
    p0=(0.0, 1.44),
    p1=(0.22, 0.78),
    p2=(0.0, 0.44),
    color="#7a45f5",
    width=2.0,
    opacity=0.15
)

add_soft_link(
    p0=(-1.45, 0.38),
    p1=(-0.98, -0.02),
    p2=(-0.46, -0.02),
    color="#ef2f95",
    width=1.9,
    opacity=0.13
)

add_soft_link(
    p0=(1.45, 0.38),
    p1=(0.98, -0.02),
    p2=(0.46, -0.02),
    color="#1479f5",
    width=1.9,
    opacity=0.13
)

add_soft_link(
    p0=(0.0, -1.42),
    p1=(-0.22, -0.88),
    p2=(0.0, -0.62),
    color="#10b6d9",
    width=2.0,
    opacity=0.15
)


# ----------------------------
# Platforms and subnetworks
# ----------------------------
for item in nodes.values():
    x, y = item["pos"]

    add_platform(
        cx=x,
        cy=y,
        w=1.72,
        h=0.84,
        fill_url=item["platform"],
        stroke_color=item["color"],
        opacity=0.42
    )

    add_network(
        cx=x,
        cy=y,
        scale=0.74,
        stroke_color=item["color"],
        node_grad=item["node"],
        glow_color=item["glow"]
    )


# ----------------------------
# Central cube
# ----------------------------
add_center_cube(
    cx=cube_center[0],
    cy=cube_center[1],
    s=0.86
)


# ----------------------------
# Floating small cubes
# ----------------------------
floating = [
    (-3.20, 2.08, 0.14, "#ff68c9", "#a868ff"),
    (3.15, 2.08, 0.16, "#b48cff", "#557eff"),
    (-2.90, -2.78, 0.12, "#8a6cff", "#69b7ff"),
    (2.75, -2.72, 0.11, "#a987ff", "#67c8ff"),
    (-0.90, 0.70, 0.12, "#72caff", "#466cff"),
    (1.05, 0.90, 0.12, "#cf78ff", "#745cff"),
]

for x, y, s, c1, c2 in floating:
    add_small_cube(x, y, s, c1, c2)


# ----------------------------
# Sparse particles
# ----------------------------
particle_colors = [
    "#7a45f5",
    "#ef2f95",
    "#1479f5",
    "#10b6d9",
    "#b75cff",
]

for _ in range(18):
    x = random.uniform(-3.2, 3.2)
    y = random.uniform(-2.8, 2.8)

    if abs(x) < 0.8 and abs(y) < 0.8:
        continue

    px, py = P(x, y)
    r = random.uniform(4.0, 9.0)
    col = random.choice(particle_colors)

    dwg.add(
        dwg.circle(
            center=(px, py),
            r=r * 1.9,
            fill=col,
            opacity=0.10,
            filter=tiny_glow_filter
        )
    )

    dwg.add(
        dwg.circle(
            center=(px, py),
            r=r,
            fill=col,
            opacity=random.uniform(0.35, 0.58)
        )
    )


# ----------------------------
# Save
# ----------------------------
dwg.save()
print(f"Saved to {OUTPUT}")