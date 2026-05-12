from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2


def get_next_run_dir(base_root: str | Path, prefix: str) -> Path:
    """自动扫描 base_root 下已有的 prefix_run_XXX 目录，返回下一个运行目录。"""
    base_root = Path(base_root)
    base_root.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(prefix)}_run_(\d+)$")
    max_id = 0
    for p in base_root.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if m:
            max_id = max(max_id, int(m.group(1)))

    return base_root / f"{prefix}_run_{max_id + 1:03d}"


def create_run_paths(base_root: str | Path, prefix: str) -> Dict[str, Path]:
    """创建本次运行的总目录和常用子目录。"""
    run_dir = get_next_run_dir(base_root, prefix)

    paths = {
        "run_dir": run_dir,
        "video_dir": run_dir / "videos",
        "cluster_plot_dir": run_dir / "cluster_plots",
        "detect_frame_dir": run_dir / "frame_detections",
        "quiver_debug_dir": run_dir / "quiver_debug",
        "compare_pre_dir": run_dir / "compare_pre",
        "compare_post_dir": run_dir / "compare_post",
        "excel_dir": run_dir / "excel",
        "stats_dir": run_dir / "stats",
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    paths["stats_csv"] = paths["stats_dir"] / "flow_stats_two_stage.csv"
    paths["raw_speed_hist_csv"] = paths["stats_dir"] / "flow_speed_hist_raw.csv"
    paths["raw_speed_hist_png"] = paths["stats_dir"] / "flow_speed_hist_raw.png"
    paths["run_log"] = run_dir / "run.log"
    paths["run_config_json"] = run_dir / "run_config.json"
    return paths


def _normalize_log_level(level_name: str) -> int:
    return getattr(logging, str(level_name).upper(), logging.INFO)


def setup_run_logger(paths: Dict[str, Path], log_cfg: Dict[str, Any], logger_name: Optional[str] = None) -> logging.Logger:
    """创建同时写控制台和 run.log 的 logger。"""
    name = logger_name or f"flow_run_{paths['run_dir'].name}"
    logger = logging.getLogger(name)
    logger.setLevel(_normalize_log_level(log_cfg.get("LOG_LEVEL", "INFO")))
    logger.propagate = False

    # 避免重复添加 handler
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_cfg.get("LOG_TO_CONSOLE", True):
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(_normalize_log_level(log_cfg.get("LOG_LEVEL", "INFO")))
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if log_cfg.get("LOG_TO_FILE", True):
        fh = logging.FileHandler(paths["run_log"], encoding="utf-8")
        fh.setLevel(_normalize_log_level(log_cfg.get("LOG_LEVEL", "INFO")))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_run_config(paths: Dict[str, Path], config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Path:
    out_path = paths["run_config_json"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    if logger is not None:
        logger.info("运行配置已保存: %s", out_path)
    return out_path


def open_video_writer(
    output_path_no_suffix: Path,
    fps: int,
    size_wh: Tuple[int, int],
    logger: Optional[logging.Logger] = None,
):
    """优先 mp4，失败则退回 avi。"""
    w, h = size_wh

    out_mp4 = output_path_no_suffix.with_suffix(".mp4")
    out_avi = output_path_no_suffix.with_suffix(".avi")

    fourcc_mp4 = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc_mp4, fps, (w, h))
    if writer.isOpened():
        if logger is not None:
            logger.info("VideoWriter 初始化成功: %s", out_mp4)
        return writer, out_mp4

    if logger is not None:
        logger.warning("mp4 VideoWriter 初始化失败，尝试 avi: %s", out_mp4)

    fourcc_avi = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(out_avi), fourcc_avi, fps, (w, h))
    if writer.isOpened():
        if logger is not None:
            logger.info("VideoWriter 回退 avi 成功: %s", out_avi)
        return writer, out_avi

    if logger is not None:
        logger.error("VideoWriter 初始化失败（mp4/avi 都失败）: %s", output_path_no_suffix)
    return None, None


def open_requested_writers(
    paths: Dict[str, Path],
    video_flags: Dict[str, bool],
    fps: int,
    size_wh: Tuple[int, int],
    logger: Optional[logging.Logger] = None,
):
    """根据开关创建需要的视频 writer。"""
    name_map = {
        "hsv": "flow_hsv_two_stage",
        "quiver": "flow_quiver_two_stage",
        "region": "flow_regions_two_stage",
        "box": "flow_boxes_two_stage",
        "combined": "flow_regions_boxes_two_stage",
    }

    writers = {}
    writer_paths = {}
    for key, enabled in video_flags.items():
        if not enabled:
            continue
        base_name = name_map[key]
        writer, out_path = open_video_writer(paths["video_dir"] / base_name, fps, size_wh, logger=logger)
        if writer is None:
            raise RuntimeError(f"VideoWriter 初始化失败：{base_name}")
        writers[key] = writer
        writer_paths[key] = out_path
    return writers, writer_paths


def release_writers(writers: Dict[str, cv2.VideoWriter], logger: Optional[logging.Logger] = None):
    for name, writer in writers.items():
        if writer is not None:
            writer.release()
            if logger is not None:
                logger.debug("VideoWriter 已释放: %s", name)


def safe_imwrite(path: Path, image, logger: Optional[logging.Logger] = None) -> bool:
    if image is None:
        if logger is not None:
            logger.warning("跳过保存空图像: %s", path)
        return False
    ok = cv2.imwrite(str(path), image)
    if logger is not None:
        if ok:
            logger.debug("图像已保存: %s", path)
        else:
            logger.error("图像保存失败: %s", path)
    return bool(ok)
