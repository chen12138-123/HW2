from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RunInfo:
    name: str
    run_dir: Path
    best_metrics_path: Path
    history_path: Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_run(run_dir: Path, name: str) -> RunInfo:
    best = run_dir / "best_metrics.json"
    hist = run_dir / "history.json"
    if not best.exists():
        raise FileNotFoundError(f"Missing best_metrics.json: {best}")
    if not hist.exists():
        raise FileNotFoundError(f"Missing history.json: {hist}")
    return RunInfo(name=name, run_dir=run_dir, best_metrics_path=best, history_path=hist)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_pdf", type=str, default=r"c:\Desktop\HW2\HW2_实验报告.pdf")
    p.add_argument("--task1_baseline", type=str, required=True)
    p.add_argument("--task1_scratch", type=str, required=True)
    p.add_argument("--task1_attention", type=str, required=True)
    p.add_argument("--task3_ce", type=str, required=True)
    p.add_argument("--task3_dice", type=str, required=True)
    p.add_argument("--task3_ce_dice", type=str, required=True)
    p.add_argument("--github_url", type=str, default="(请填写你的GitHub仓库链接)")
    p.add_argument("--weights_url", type=str, default="(请填写你的模型权重网盘链接)")
    p.add_argument("--team", type=str, default="(请填写组员姓名/学号/分工)")
    return p.parse_args()


def _ensure_curve_pngs(run_dir: Path, kind: str) -> None:
    if kind == "task1":
        loss_png = run_dir / "loss_curve.png"
        acc_png = run_dir / "acc_curve.png"
        if loss_png.exists() and acc_png.exists():
            return
        import subprocess, sys

        subprocess.check_call([sys.executable, str(run_dir.parent.parent / "plot_curves.py"), "--run_dir", str(run_dir)])
        return

    if kind == "task3":
        loss_png = run_dir / "loss_curve.png"
        miou_png = run_dir / "miou_curve.png"
        if loss_png.exists() and miou_png.exists():
            return
        import subprocess, sys

        subprocess.check_call([sys.executable, str(run_dir.parent.parent / "plot_curves.py"), "--run_dir", str(run_dir)])
        return


def _best_line_task1(best_obj: dict) -> str:
    return f"best_val_acc={best_obj.get('best_val_acc', 'NA')}, epoch={best_obj.get('epoch', 'NA')}"


def _best_line_task3(best_obj: dict) -> str:
    v = best_obj.get("val", {})
    return f"best_score={best_obj.get('best_score', 'NA')}, epoch={best_obj.get('epoch', 'NA')}, val_miou={v.get('miou','NA')}, val_acc={v.get('acc','NA')}"


def main() -> None:
    args = parse_args()

    t1_base = _find_run(Path(args.task1_baseline), "Task1 Baseline (pretrained)")
    t1_scratch = _find_run(Path(args.task1_scratch), "Task1 Ablation (scratch)")
    t1_att = _find_run(Path(args.task1_attention), "Task1 Attention (SE/CBAM)")

    t3_ce = _find_run(Path(args.task3_ce), "Task3 CE")
    t3_dice = _find_run(Path(args.task3_dice), "Task3 Dice")
    t3_ced = _find_run(Path(args.task3_ce_dice), "Task3 CE+Dice")

    for r in [t1_base, t1_scratch, t1_att]:
        _ensure_curve_pngs(r.run_dir, "task1")
    for r in [t3_ce, t3_dice, t3_ced]:
        _ensure_curve_pngs(r.run_dir, "task3")

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    font_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ]
    font_path = next((p for p in font_candidates if Path(p).exists()), None)
    if font_path:
        pdfmetrics.registerFont(TTFont("CNFont", font_path))
        font_name = "CNFont"
    else:
        font_name = "Helvetica"

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    w, h = A4

    def title(text: str) -> None:
        c.setFont(font_name, 16)
        c.drawString(2 * cm, h - 2 * cm, text)

    def paragraph(y: float, text: str, size: int = 11) -> float:
        c.setFont(font_name, size)
        max_w = w - 4 * cm
        x = 2 * cm
        line = ""
        for ch in text:
            if c.stringWidth(line + ch, font_name, size) > max_w:
                c.drawString(x, y, line)
                y -= 0.6 * cm
                line = ch
            else:
                line += ch
        if line:
            c.drawString(x, y, line)
            y -= 0.6 * cm
        return y

    def image(y: float, path: Path, height_cm: float = 6.0) -> float:
        img = ImageReader(str(path))
        iw, ih = img.getSize()
        target_h = height_cm * cm
        target_w = target_h * (iw / ih)
        c.drawImage(img, 2 * cm, y - target_h, width=target_w, height=target_h)
        return y - target_h - 0.4 * cm

    title("HW2 实验报告（任务1 & 任务3）")
    y = h - 3 * cm
    y = paragraph(y, f"小组信息：{args.team}")
    y = paragraph(y, f"GitHub Repo：{args.github_url}")
    y = paragraph(y, f"模型权重下载：{args.weights_url}")
    y = paragraph(y, "说明：本报告由脚本自动汇总训练日志与曲线图生成。若课程要求必须使用 wandb/swanlab 截图，可用本仓库 README 中命令开启记录后截图替换。")
    c.showPage()

    title("任务1：Flowers102 微调分类")
    y = h - 3 * cm
    y = paragraph(y, "模型：ResNet-18/34（ImageNet 预训练微调 / 随机初始化对比 / 注意力模块SE或CBAM）")
    y = paragraph(y, "数据集：torchvision Flowers102，官方 train/val/test 划分。评价指标：Top-1 Accuracy。")
    y = paragraph(y, "实验设置：AdamW；backbone 使用较小学习率，fc 使用较大学习率；OneCycleLR；label smoothing（可调）。")

    base_best = _load_json(t1_base.best_metrics_path)
    scratch_best = _load_json(t1_scratch.best_metrics_path)
    att_best = _load_json(t1_att.best_metrics_path)

    y = paragraph(y, f"Baseline：{t1_base.run_dir.name}，{_best_line_task1(base_best)}")
    y = paragraph(y, f"Scratch：{t1_scratch.run_dir.name}，{_best_line_task1(scratch_best)}")
    y = paragraph(y, f"Attention：{t1_att.run_dir.name}，{_best_line_task1(att_best)}")

    y = paragraph(y, "曲线：train/val loss 与 val accuracy（用于报告截图）")
    y = image(y, t1_base.run_dir / "loss_curve.png", height_cm=5.2)
    y = image(y, t1_base.run_dir / "acc_curve.png", height_cm=5.2)
    c.showPage()

    title("任务3：U-Net 语义分割（从零训练 + Dice Loss 工程）")
    y = h - 3 * cm
    y = paragraph(y, "模型：从零手写 U-Net（编码器/解码器/Skip Connection），不使用预训练权重。")
    y = paragraph(y, "数据集：Stanford Background Dataset（iccv09Data），labels/*.regions.txt 负数像素视为 unknown 并 ignore。")
    y = paragraph(y, "损失函数对比：CE / Dice / CE+Dice。评价指标：验证集 mIoU（主）与像素 Accuracy（辅）。")

    ce_best = _load_json(t3_ce.best_metrics_path)
    dice_best = _load_json(t3_dice.best_metrics_path)
    ced_best = _load_json(t3_ced.best_metrics_path)

    y = paragraph(y, f"CE：{t3_ce.run_dir.name}，{_best_line_task3(ce_best)}")
    y = paragraph(y, f"Dice：{t3_dice.run_dir.name}，{_best_line_task3(dice_best)}")
    y = paragraph(y, f"CE+Dice：{t3_ced.run_dir.name}，{_best_line_task3(ced_best)}")

    y = paragraph(y, "曲线：train/val loss、val mIoU、val pixel accuracy")
    y = image(y, t3_ced.run_dir / "loss_curve.png", height_cm=4.8)
    y = image(y, t3_ced.run_dir / "miou_curve.png", height_cm=4.8)
    y = image(y, t3_ced.run_dir / "acc_curve.png", height_cm=4.8)
    c.showPage()

    title("复现实验（运行方式）")
    y = h - 3 * cm
    y = paragraph(y, "任务1训练：")
    y = paragraph(y, "python task1/train.py --model resnet34 --pretrained --epochs 30 --lr_head 3e-4 --lr_backbone 3e-5 --label_smoothing 0.1 --device cuda", size=10)
    y = paragraph(y, "任务3三组对比：")
    y = paragraph(y, "python task3/train.py --loss ce --epochs 50 --device cuda", size=10)
    y = paragraph(y, "python task3/train.py --loss dice --epochs 50 --device cuda", size=10)
    y = paragraph(y, "python task3/train.py --loss ce_dice --epochs 50 --device cuda", size=10)
    y = paragraph(y, "注意：最终提交时，请在报告中补充你的 GitHub 链接与权重网盘链接。")

    c.save()


if __name__ == "__main__":
    main()
