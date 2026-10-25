"""產生程式圖示 src/omni2/data/omni2.ico（多種尺寸，PNG 格式的 ICO）。

    uv run python tools/make_icon.py
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QColor, QFont, QFontDatabase, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath,
)

SIZES = (16, 24, 32, 48, 64, 128, 256)
OUT = Path(__file__).resolve().parents[1] / "src" / "omni2" / "data" / "omni2.ico"


def font_family() -> str:
    # offscreen 平台不會自己讀 .ttc，明確載入微軟正黑體粗體
    fid = QFontDatabase.addApplicationFont("C:/Windows/Fonts/msjhbd.ttc")
    families = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
    return families[0] if families else "Noto Sans TC"


def render(size: int, family: str) -> bytes:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    rect = QRectF(0, 0, size, size).adjusted(size * 0.04, size * 0.04, -size * 0.04, -size * 0.04)
    gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
    gradient.setColorAt(0, QColor("#2563eb"))
    gradient.setColorAt(1, QColor("#7c3aed"))
    path = QPainterPath()
    path.addRoundedRect(rect, size * 0.22, size * 0.22)
    p.fillPath(path, gradient)
    font = QFont(family)
    font.setBold(True)
    font.setPixelSize(int(size * 0.62))
    p.setFont(font)
    p.setPen(QColor("white"))
    p.drawText(rect.adjusted(0, -size * 0.03, 0, 0), Qt.AlignmentFlag.AlignCenter, "譯")
    p.end()
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buf, "PNG")
    return bytes(data)


def main() -> int:
    QGuiApplication(sys.argv)
    family = font_family()
    images = [render(s, family) for s in SIZES]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for size, png in zip(SIZES, images):
        dim = 0 if size >= 256 else size  # ICO 裡 0 代表 256
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(png), offset + len(blobs))
        blobs += png
    OUT.write_bytes(header + entries + blobs)
    print(OUT, OUT.stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
