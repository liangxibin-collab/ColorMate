"""
Android Bridge - 连接Kotlin UI和Python核心的桥接模块

提供简单的函数接口供Kotlin/Java通过Chaquopy调用。
所有文件路径使用Android文件系统的绝对路径。
"""

from .style_trainer import ColorStyleTrainer, ColorStyleModel
from .color_matcher import ColorMatcher
import os
import pickle
import json
import traceback


def train_model(reference_dir, model_path, progress_callback=None):
    """
    从参考照片目录训练模型
    
    Args:
        reference_dir: 包含参考照片的目录路径
        model_path: 模型保存路径（.pkl文件）
        progress_callback: 进度回调函数
    
    Returns:
        dict: {success: bool, message: str, num_images: int}
    """
    try:
        # 收集目录下所有jpg/jpeg文件
        valid_exts = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
        image_paths = []
        
        for fname in os.listdir(reference_dir):
            ext = os.path.splitext(fname)[1]
            if ext in valid_exts:
                image_paths.append(os.path.join(reference_dir, fname))
        
        if len(image_paths) == 0:
            return {
                "success": False,
                "message": "目录中没有找到JPG/JPEG图片",
                "num_images": 0
            }
        
        image_paths.sort()  # 固定顺序
        
        # 训练
        trainer = ColorStyleTrainer(sample_size=500)
        model = trainer.train(image_paths)
        
        # 保存模型
        model.save(model_path)
        
        return {
            "success": True,
            "message": f"训练完成，使用了 {len(image_paths)} 张参考照片",
            "num_images": len(image_paths)
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "message": f"训练失败: {str(e)}",
            "num_images": 0
        }


def match_image(model_path, target_path, output_path, strength=1.0):
    """
    对单张图片应用追色
    
    Args:
        model_path: 模型文件路径
        target_path: 目标图片路径
        output_path: 输出图片路径
        strength: 迁移强度 (0.0~1.0)
    
    Returns:
        dict: {success: bool, message: str}
    """
    try:
        model = ColorStyleModel.load(model_path)
        matcher = ColorMatcher(strength=strength, preserve_natural=True)
        matcher.match(model, target_path, output_path)
        
        return {
            "success": True,
            "message": "追色完成"
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "message": f"追色失败: {str(e)}"
        }


def match_batch(model_path, target_dir, output_dir, strength=1.0):
    """
    批量追色
    
    Args:
        model_path: 模型文件路径
        target_dir: 目标图片目录
        output_dir: 输出目录
        strength: 迁移强度
    
    Returns:
        dict: {success: bool, message: str, processed: int, failed: int}
    """
    try:
        model = ColorStyleModel.load(model_path)
        matcher = ColorMatcher(strength=strength, preserve_natural=True)
        
        os.makedirs(output_dir, exist_ok=True)
        
        valid_exts = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
        processed = 0
        failed = 0
        
        for fname in os.listdir(target_dir):
            ext = os.path.splitext(fname)[1]
            if ext not in valid_exts:
                continue
            
            src = os.path.join(target_dir, fname)
            dst = os.path.join(output_dir, f"color_matched_{fname}")
            
            try:
                matcher.match(model, src, dst)
                processed += 1
            except Exception as e:
                failed += 1
        
        return {
            "success": True,
            "message": f"批量追色完成: {processed}成功, {failed}失败",
            "processed": processed,
            "failed": failed
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "message": f"批量追色失败: {str(e)}",
            "processed": 0,
            "failed": 0
        }


def get_model_info(model_path):
    """
    获取模型信息
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        dict: 模型基本信息
    """
    try:
        model = ColorStyleModel.load(model_path)
        return {
            "success": True,
            "name": model.name,
            "num_training_images": model.num_training_images,
            "version": model.version,
            "hue_peaks": model.hue_hist_peaks if hasattr(model, 'hue_hist_peaks') else [],
            "lab_means": model.lab_means,
            "lab_stds": model.lab_stds,
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


def list_training_images(reference_dir):
    """
    列出训练目录中的图片
    
    Returns:
        dict: {success: bool, images: list[str], count: int}
    """
    try:
        valid_exts = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
        images = sorted([
            f for f in os.listdir(reference_dir)
            if os.path.splitext(f)[1] in valid_exts
        ])
        return {
            "success": True,
            "images": images,
            "count": len(images),
            "directory": reference_dir
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }
