#!/usr/bin/env python3
"""
Synthetic Dataset Generator for CUDA at Scale Capstone Project.
Generates 120+ standard Netpbm PPM (P6 binary RGB) and PGM (P5 binary Grayscale)
images with various frequencies, geometric patterns, and textures.
Pure Python standard library implementation - zero external dependencies required.
"""

import argparse
import math
import os
import struct
import sys

def write_pgm(filepath: str, width: int, height: int, pixels: bytes):
    """Writes a P5 (binary grayscale) Netpbm image file."""
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(pixels)

def write_ppm(filepath: str, width: int, height: int, rgb_bytes: bytes):
    """Writes a P6 (binary RGB) Netpbm image file."""
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(rgb_bytes)

def generate_pattern(pattern_id: int, width: int, height: int) -> bytes:
    """Generates procedural RGB pixel byte buffer based on pattern type."""
    buf = bytearray(width * height * 3)
    cx = width / 2.0
    cy = height / 2.0
    mode = pattern_id % 6

    idx = 0
    for y in range(height):
        dy = y - cy
        for x in range(width):
            dx = x - cx
            dist = math.sqrt(dx * dx + dy * dy)

            if mode == 0:
                # Concentric circles / ripples
                freq = 0.05 + (pattern_id % 5) * 0.02
                val = int(127.5 * (1.0 + math.sin(dist * freq)))
                r = val
                g = int(127.5 * (1.0 + math.cos(dist * freq * 0.7)))
                b = (x * 255 // width)
            elif mode == 1:
                # High frequency checkerboard / grid
                block_size = max(4, 8 + (pattern_id % 8) * 4)
                check = ((x // block_size) + (y // block_size)) % 2
                base = 230 if check else 25
                r = base
                g = (base + y * 50 // height) % 256
                b = (255 - base)
            elif mode == 2:
                # Radial gradient & angular rays
                angle = math.atan2(dy, dx)
                rays = int(127.5 * (1.0 + math.sin(angle * (4 + pattern_id % 6))))
                fade = max(0.0, 1.0 - dist / (width * 0.6))
                r = int(rays * fade)
                g = int(255 * (x / width) * fade)
                b = int(255 * (y / height) * fade)
            elif mode == 3:
                # Dual diagonal wave interference
                w1 = math.sin((x + y) * 0.08)
                w2 = math.cos((x - y) * 0.08)
                inter = int(127.5 * (1.0 + 0.5 * (w1 + w2)))
                r = inter
                g = int(255 * (1.0 - abs(dx) / cx))
                b = int(255 * (1.0 - abs(dy) / cy))
            elif mode == 4:
                # Edge bars and step gradients
                stripe = (x + y * 2) % 40
                bar = 255 if stripe < 20 else 0
                r = bar
                g = int(x * 255 / width)
                b = bar ^ int(y * 255 / height)
            else:
                # Pseudo-random textured noise pattern
                seed = (x * 374761393 + y * 668265263 + pattern_id * 314159) & 0xFFFFFFFF
                seed = (seed ^ (seed >> 13)) * 1274126177 & 0xFFFFFFFF
                val = (seed >> 16) & 0xFF
                r = val
                g = (val * 3) & 0xFF
                b = (val * 7) & 0xFF

            buf[idx] = max(0, min(255, r))
            buf[idx + 1] = max(0, min(255, g))
            buf[idx + 2] = max(0, min(255, b))
            idx += 3

    return bytes(buf)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset of images")
    parser.add_argument("--num_images", type=int, default=120, help="Number of images to generate (default: 120)")
    parser.add_argument("--width", type=int, default=512, help="Image width in pixels (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Image height in pixels (default: 512)")
    parser.add_argument("--output_dir", type=str, default="data/inputs", help="Output directory path")
    parser.add_argument("--format", type=str, default="ppm", choices=["ppm", "pgm", "both"], help="Format: ppm (RGB) or pgm (Grayscale)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating {args.num_images} images ({args.width}x{args.height}) in '{args.output_dir}'...")

    for i in range(1, args.num_images + 1):
        pixels_rgb = generate_pattern(i, args.width, args.height)

        if args.format in ["ppm", "both"]:
            filename = f"img_{i:04d}.ppm"
            filepath = os.path.join(args.output_dir, filename)
            write_ppm(filepath, args.width, args.height, pixels_rgb)

        if args.format in ["pgm", "both"]:
            # Convert RGB to grayscale
            gray_buf = bytearray(args.width * args.height)
            for p in range(args.width * args.height):
                r = pixels_rgb[p * 3]
                g = pixels_rgb[p * 3 + 1]
                b = pixels_rgb[p * 3 + 2]
                gray_buf[p] = int(0.299 * r + 0.587 * g + 0.114 * b)
            filename = f"img_{i:04d}.pgm"
            filepath = os.path.join(args.output_dir, filename)
            write_pgm(filepath, args.width, args.height, bytes(gray_buf))

        if i % 20 == 0 or i == args.num_images:
            print(f"  Generated [{i:03d}/{args.num_images:03d}] images...")

    print(f"Dataset generation complete. {args.num_images} test files ready.")

if __name__ == "__main__":
    main()
