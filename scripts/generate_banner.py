"""Generate ultra-high-definition banner for Resonance repository."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance


def generate_banner(output_path: str = "assets/banner.png"):
    width = 2400
    height = 700

    # 1. Base dark background (#0B0F19 to #0D1527)
    bg = Image.new("RGBA", (width, height), (11, 15, 25, 255))
    draw = ImageDraw.Draw(bg)

    # Subtle gradient across width
    for x in range(width):
        alpha = x / width
        r = int(10 + 6 * alpha)
        g = int(14 + 10 * alpha)
        b = int(24 + 20 * alpha)
        draw.line([(x, 0), (x, height)], fill=(r, g, b, 255))

    # 2. Blend Albrecht Dürer's Melencolia I on left
    try:
        durer = Image.open("assets/banner_0.png").convert("RGBA")
        # Crop and resize to fit height
        dw, dh = durer.size
        target_h = height
        target_w = int(dw * (target_h / dh))
        durer_resized = durer.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Convert to dark slate monochrome
        durer_gray = durer_resized.convert("L")
        durer_tinted = Image.new("RGBA", (target_w, target_h), (14, 22, 38, 255))
        durer_colored = Image.composite(
            Image.new("RGBA", (target_w, target_h), (80, 110, 160, 255)),
            durer_tinted,
            durer_gray,
        )

        # Create smooth horizontal fade-out gradient mask
        mask = Image.new("L", (target_w, target_h), 0)
        mask_draw = ImageDraw.Draw(mask)
        fade_start = int(target_w * 0.25)
        fade_end = target_w
        for x in range(target_w):
            if x < fade_start:
                val = 140  # semi-subtle opacity
            else:
                progress = (x - fade_start) / max(1, (fade_end - fade_start))
                val = int(140 * (1.0 - progress)**1.5)
            mask_draw.line([(x, 0), (x, target_h)], fill=val)

        bg.paste(durer_colored, (0, 0), mask)
    except Exception as e:
        print(f"Note on Dürer blend: {e}")

    # 3. Draw high-tech subtle grid and circuit track accents
    grid_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    grid_draw = ImageDraw.Draw(grid_overlay)

    # Grid lines
    for gy in range(0, height, 70):
        grid_draw.line([(0, gy), (width, gy)], fill=(30, 45, 75, 40), width=1)
    for gx in range(0, width, 100):
        grid_draw.line([(gx, 0), (gx, height)], fill=(30, 45, 75, 40), width=1)

    bg = Image.alpha_composite(bg, grid_overlay)

    # 4. Generate Matplotlib Resonance & Phase Curves on the right side
    fig = plt.figure(figsize=(9.5, 5.0), dpi=150, facecolor="none")
    ax1 = fig.add_axes([0.1, 0.15, 0.85, 0.75])
    ax1.set_facecolor("none")

    # Physics parameters
    f = np.logspace(1.5, 4.5, 600)  # ~30 Hz to 30 kHz
    w = 2 * np.pi * f
    R = 40.0
    L = 0.015  # 15 mH
    C = 1.5e-6  # 1.5 uF
    w0 = 1.0 / np.sqrt(L * C)
    f0 = w0 / (2 * np.pi)

    # Series RLC impedance & current amplitude
    Z_mag = np.sqrt(R**2 + (w * L - 1.0 / (w * C))**2)
    I_amp = (0.1 / Z_mag) * 1000.0  # mA

    # Series RC and RL comparison curves
    Z_rc = np.sqrt(R**2 + (1.0 / (w * C))**2)
    I_rc = (0.1 / Z_rc) * 1000.0
    Z_rl = np.sqrt(R**2 + (w * L)**2)
    I_rl = (0.1 / Z_rl) * 1000.0

    # Plot glowing curves
    # Glow effect
    for lw, al in [(8, 0.12), (5, 0.25), (2.5, 0.95)]:
        ax1.semilogx(f, I_amp, color="#38BDF8", linewidth=lw, alpha=al)
    ax1.semilogx(f, I_rc, color="#F43F5E", linewidth=1.8, linestyle="--", alpha=0.75, label="Series RC")
    ax1.semilogx(f, I_rl, color="#A855F7", linewidth=1.8, linestyle=":", alpha=0.75, label="Series RL")

    # Resonant frequency peak indicator
    I_peak = (0.1 / R) * 1000.0
    ax1.scatter([f0], [I_peak], color="#38BDF8", s=140, zorder=5, edgecolors="#FFFFFF", linewidths=2.5)
    ax1.axvline(f0, color="#38BDF8", linestyle=":", alpha=0.6, linewidth=1.5)

    # Annotate resonance peak
    ax1.annotate(
        r"$\mathbf{f_0 = 1061\ Hz}$" + "\n" + r"$\mathbf{|I|_{max} = 2.50\ mA}$",
        xy=(f0, I_peak),
        xytext=(f0 * 1.5, I_peak * 0.88),
        color="#F8FAFC",
        fontsize=10.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0F172A", edgecolor="#38BDF8", alpha=0.9, lw=1.2),
    )

    # Style axes
    ax1.tick_params(colors="#94A3B8", labelsize=9)
    for spine in ax1.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.2)
    ax1.set_xlabel(r"Probe Frequency $f$ (Hz)", color="#CBD5E1", fontsize=10.5, labelpad=4)
    ax1.set_ylabel(r"Current $|I|$ (mA RMS)", color="#CBD5E1", fontsize=10.5, labelpad=4)
    ax1.grid(True, which="both", color="#1E293B", linestyle="--", linewidth=0.7, alpha=0.7)
    leg = ax1.legend(loc="upper left", facecolor="#0F172A", edgecolor="#334155", fontsize=9.5)
    for text in leg.get_texts():
        text.set_color("#E2E8F0")

    # Render figure to image
    fig.canvas.draw()
    plot_rgba = np.asarray(fig.canvas.buffer_rgba())
    plot_im = Image.fromarray(plot_rgba)
    plt.close(fig)

    # Paste plot onto the right side of banner
    bg.paste(plot_im, (width - plot_im.width - 40, (height - plot_im.height) // 2 + 35), plot_im)

    # 5. Render Crisp Typography (Title, Subtitle, Badges)
    draw = ImageDraw.Draw(bg)

    try:
        font_eyebrow = ImageFont.truetype("/System/Library/Fonts/Supplemental/HelveticaNeue.ttc", 30)
        font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/HelveticaNeue.ttc", 84)
        font_subtitle = ImageFont.truetype("/System/Library/Fonts/Supplemental/HelveticaNeue.ttc", 36)
        font_desc = ImageFont.truetype("/System/Library/Fonts/Supplemental/HelveticaNeue.ttc", 26)
        font_badge = ImageFont.truetype("/System/Library/Fonts/Supplemental/HelveticaNeue.ttc", 22)
    except Exception:
        font_eyebrow = ImageFont.load_default()
        font_title = font_eyebrow
        font_subtitle = font_eyebrow
        font_desc = font_eyebrow
        font_badge = font_eyebrow

    text_x = 480
    text_y = 120

    # Eyebrow Tag
    draw.text((text_x, text_y), "HARRIOT AI  •  ACTIVE REINFORCEMENT LEARNING", font=font_eyebrow, fill=(56, 189, 248, 255))

    # Glow under title
    glow_box = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_box)
    glow_draw.text((text_x - 2, text_y + 43), "RESONANCE", font=font_title, fill=(56, 189, 248, 60))
    glow_draw.text((text_x + 2, text_y + 47), "RESONANCE", font=font_title, fill=(56, 189, 248, 60))
    glow_box = glow_box.filter(ImageFilter.GaussianBlur(8))
    bg = Image.alpha_composite(bg, glow_box)
    draw = ImageDraw.Draw(bg)

    # Main Title
    draw.text((text_x, text_y + 45), "RESONANCE", font=font_title, fill=(255, 255, 255, 255))

    # Subtitle
    draw.text(
        (text_x, text_y + 150),
        "Active Physics Experimentation & RL Environment",
        font=font_subtitle,
        fill=(226, 232, 240, 255),
    )

    # Descriptive detail
    draw.text(
        (text_x, text_y + 205),
        "Autonomous model discrimination, sequential EIG, and complex impedance spectroscopy.",
        font=font_desc,
        fill=(148, 163, 184, 255),
    )

    # Feature badges at the bottom
    badges = [
        ("Gymnasium 1.0+", (16, 185, 129)),       # Emerald
        ("SFT + Multi-Turn GRPO", (56, 189, 248)), # Cyan
        ("MemoryGuard Safe", (168, 85, 247)),      # Violet
        ("UGPhysics & PhysReason", (245, 158, 11)),# Amber
    ]

    badge_x = text_x
    badge_y = text_y + 270
    for label, color in badges:
        # Measure text
        bbox = draw.textbbox((0, 0), label, font=font_badge)
        bw = (bbox[2] - bbox[0]) + 28
        bh = (bbox[3] - bbox[1]) + 18

        # Draw rounded rectangle badge
        draw.rounded_rectangle(
            [(badge_x, badge_y), (badge_x + bw, badge_y + bh)],
            radius=8,
            fill=(color[0], color[1], color[2], 35),
            outline=(color[0], color[1], color[2], 180),
            width=1,
        )
        # Dot
        draw.ellipse([(badge_x + 10, badge_y + bh // 2 - 4), (badge_x + 18, badge_y + bh // 2 + 4)], fill=color)
        draw.text((badge_x + 24, badge_y + 7), label, font=font_badge, fill=(241, 245, 249, 255))
        badge_x += bw + 16

    # 6. Save image
    bg.save(output_path, "PNG", optimize=True)
    print(f"Successfully generated high-definition banner at {output_path} ({width}x{height})!")


if __name__ == "__main__":
    generate_banner()
