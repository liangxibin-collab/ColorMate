"""
ColorStyleTrainer - 色彩风格训练器

从数百张参考照片中学习色彩风格特征，生成风格模型。
使用多种统计方法捕捉影调、色彩分布和色调曲线特征。

核心方法：
1. Lab色彩空间统计匹配（均值、标准差、相关矩阵）
2. 色调直方图（Hue histogram）分布
3. 饱和度特征映射
4. 亮度/对比度曲线特征
5. 颜色相关矩阵（捕捉颜色间的关系）
6. 基于分位数的色调映射曲线
"""

import json
import pickle
import numpy as np
from pathlib import Path
from collections import OrderedDict

# 尝试导入OpenCV和PIL，优先OpenCV
try:
    import cv2
    USE_CV2 = True
except ImportError:
    USE_CV2 = False
    from PIL import Image

# 色彩特征提取的bin数量
HUE_BINS = 360        # 色调分360个bin
SAT_BINS = 100        # 饱和度分100个bin
LIGHT_BINS = 100      # 明度分100个bin
L_HIST_BINS = 100     # L通道直方图bin数

# 分位点数量
QUANTILE_N = 100      # 色调映射曲线采样点

# 特征版本号
MODEL_VERSION = 2


class ColorStyleModel:
    """色彩风格模型，保存所有学习到的特征"""
    
    def __init__(self, name="ColorStyle"):
        self.name = name
        self.version = MODEL_VERSION
        
        # --- Lab统计特征 ---
        self.lab_means = None       # [L_mean, a_mean, b_mean]
        self.lab_stds = None        # [L_std, a_std, b_std]
        self.lab_cov = None         # 3x3协方差矩阵
        
        # --- 色调分布特征 ---
        self.hue_hist = None        # 归一化色调直方图 (360,)
        self.hue_hist_peaks = None  # 主色调峰值
        
        # --- 饱和度特征 ---
        self.saturation_hist = None  # 饱和度分布 (100,)
        self.saturation_stats = None # [mean_sat, std_sat]
        
        # --- 亮度特征 ---
        self.lightness_hist = None   # 亮度分布 (100,)
        self.lightness_stats = None  # [mean_L, std_L]
        
        # --- 色调映射曲线 (分位数匹配) ---
        self.l_quantiles = None      # L通道分位映射 (QUANTILE_N,)
        self.a_quantiles = None      # a通道分位映射
        self.b_quantiles = None      # b通道分位映射
        
        # --- L通道直方图分布 ---
        self.l_hist = None           # L通道直方图
        
        # --- 色域信息 ---
        self.gamut_convex_hull = None  # Lab凸包简化
        
        # --- 模型元数据 ---
        self.num_training_images = 0
        self.image_shape_stats = None  # 训练集尺寸统计
        
    def to_dict(self):
        """序列化为字典"""
        result = OrderedDict()
        result["name"] = self.name
        result["version"] = self.version
        result["num_training_images"] = self.num_training_images
        
        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                result[key] = val.tolist()
            elif val is not None:
                result[key] = val
                
        return result
    
    def save(self, path):
        """保存模型到文件"""
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(path):
        """从文件加载模型"""
        with open(path, 'rb') as f:
            return pickle.load(f)


class ColorStyleTrainer:
    """色彩风格训练器：从一组参考照片中学习色彩风格"""
    
    def __init__(self, sample_size=500):
        """
        Args:
            sample_size: 每张照片的像素采样数（用于统计计算）
        """
        self.sample_size = sample_size
        
    def train(self, image_paths, model=None, progress_callback=None):
        """
        从一组图片中训练色彩风格模型
        
        Args:
            image_paths: 图片路径列表
            model: 可选的已有模型（用于增量训练）
            progress_callback: 进度回调 func(current, total)
            
        Returns:
            ColorStyleModel
        """
        if model is None:
            model = ColorStyleModel()
        
        n = len(image_paths)
        if n == 0:
            raise ValueError("No training images provided")
        
        model.num_training_images = n
        
        # 累积所有采样像素用于统计
        all_lab_pixels = []       # Lab空间像素
        all_hue_values = []       # 色调值
        all_sat_values = []       # 饱和度值
        all_light_values = []     # 亮度值
        all_l_values = []         # L通道值
        
        # 每张图片的分位数
        all_l_quantiles = []
        all_a_quantiles = []
        all_b_quantiles = []
        
        # 每张图片的直方图
        all_hue_hists = []
        all_sat_hists = []
        all_light_hists = []
        
        for i, img_path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i + 1, n)
            
            # 读取图片
            img_bgr, img_rgb = self._read_image(img_path)
            if img_bgr is None:
                continue
            
            # 转换到Lab空间
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            lab_float = lab.astype(np.float32)
            
            # 转换到HSV空间
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            hsv_float = hsv.astype(np.float32)
            
            # 采样像素
            h, w = lab.shape[:2]
            total_pixels = h * w
            sample_size = min(self.sample_size, total_pixels)
            
            # 随机采样
            indices = np.random.choice(total_pixels, sample_size, replace=False)
            sampled_lab = lab_float.reshape(-1, 3)[indices]
            sampled_hsv = hsv_float.reshape(-1, 3)[indices]
            
            all_lab_pixels.append(sampled_lab)
            all_hue_values.extend(sampled_hsv[:, 0])  # H: 0-180 (OpenCV)
            all_sat_values.extend(sampled_hsv[:, 1])  # S: 0-255
            all_light_values.extend(sampled_hsv[:, 2]) # V: 0-255
            
            # L通道直方图
            l_channel = lab[:, :, 0].flatten()
            all_l_values.extend(l_channel)
            
            # 每张图片的分位数
            l_flat = lab[:, :, 0].flatten().astype(np.float32)
            a_flat = lab[:, :, 1].flatten().astype(np.float32)
            b_flat = lab[:, :, 2].flatten().astype(np.float32)
            
            # 采样分位数 (用全部像素计算分位数，或者采样)
            if total_pixels > 100000:
                q_indices = np.random.choice(total_pixels, 100000, replace=False)
                l_samp = l_flat[q_indices]
                a_samp = a_flat[q_indices]
                b_samp = b_flat[q_indices]
            else:
                l_samp = l_flat
                a_samp = a_flat
                b_samp = b_flat
            
            quantiles = np.linspace(0, 1, QUANTILE_N)
            all_l_quantiles.append(np.quantile(l_samp, quantiles))
            all_a_quantiles.append(np.quantile(a_samp, quantiles))
            all_b_quantiles.append(np.quantile(b_samp, quantiles))
            
            # 每张图片的色调直方图
            h_bins = np.arange(HUE_BINS + 1)
            hue_hist, _ = np.histogram(sampled_hsv[:, 0], bins=h_bins, range=(0, 180))
            # 归一化
            hue_hist = hue_hist.astype(np.float32) / max(hue_hist.sum(), 1)
            
            # 饱和度直方图
            s_bins = np.arange(SAT_BINS + 1)
            sat_hist, _ = np.histogram(sampled_hsv[:, 1], bins=s_bins, range=(0, 256))
            sat_hist = sat_hist.astype(np.float32) / max(sat_hist.sum(), 1)
            
            # 亮度直方图
            v_bins = np.arange(LIGHT_BINS + 1)
            light_hist, _ = np.histogram(sampled_hsv[:, 2], bins=v_bins, range=(0, 256))
            light_hist = light_hist.astype(np.float32) / max(light_hist.sum(), 1)
            
            # 对直方图应用小幅高斯平滑，减少噪声
            from scipy.ndimage import gaussian_filter1d
            all_hue_hists.append(gaussian_filter1d(hue_hist, sigma=1.0))
            all_sat_hists.append(gaussian_filter1d(sat_hist, sigma=1.0))
            all_light_hists.append(gaussian_filter1d(light_hist, sigma=1.0))
        
        if len(all_lab_pixels) == 0:
            raise ValueError("No valid images could be read for training")
        
        # ============================================
        # 1. Lab统计特征
        # ============================================
        all_pixels = np.vstack(all_lab_pixels)
        model.lab_means = np.mean(all_pixels, axis=0).tolist()
        model.lab_stds = np.std(all_pixels, axis=0).tolist()
        model.lab_cov = np.cov(all_pixels, rowvar=False).tolist()
        
        # ============================================
        # 2. 色调分布特征
        # ============================================
        # 平均色调直方图
        mean_hue_hist = np.mean(all_hue_hists, axis=0)
        model.hue_hist = mean_hue_hist.tolist()
        
        # 找到色调峰值（主要色调）
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(mean_hue_hist, height=np.max(mean_hue_hist) * 0.15, distance=15)
        peak_heights = properties['peak_heights']
        # 按高度排序
        sorted_idx = np.argsort(peak_heights)[::-1]
        model.hue_hist_peaks = [
            {"hue": float(peaks[j] * 180.0 / HUE_BINS), "height": float(peak_heights[j])}
            for j in sorted_idx[:10]  # 最多10个峰值
        ]
        
        # ============================================
        # 3. 饱和度特征
        # ============================================
        mean_sat_hist = np.mean(all_sat_hists, axis=0)
        model.saturation_hist = mean_sat_hist.tolist()
        all_sat_vals = np.array(all_sat_values)
        model.saturation_stats = [float(np.mean(all_sat_vals)), float(np.std(all_sat_vals))]
        
        # ============================================
        # 4. 亮度特征
        # ============================================
        mean_light_hist = np.mean(all_light_hists, axis=0)
        model.lightness_hist = mean_light_hist.tolist()
        all_light_vals = np.array(all_light_values)
        model.lightness_stats = [float(np.mean(all_light_vals)), float(np.std(all_light_vals))]
        
        # ============================================
        # 5. 色调映射曲线 (平均分位点)
        # ============================================
        model.l_quantiles = np.mean(all_l_quantiles, axis=0).tolist()
        model.a_quantiles = np.mean(all_a_quantiles, axis=0).tolist()
        model.b_quantiles = np.mean(all_b_quantiles, axis=0).tolist()
        
        # ============================================
        # 6. L通道直方图
        # ============================================
        all_l = np.array(all_l_values)
        l_hist, _ = np.histogram(all_l, bins=L_HIST_BINS, range=(0, 256))
        l_hist = l_hist.astype(np.float32) / max(l_hist.sum(), 1)
        from scipy.ndimage import gaussian_filter1d
        model.l_hist = gaussian_filter1d(l_hist, sigma=2.0).tolist()
        
        return model
    
    def _read_image(self, img_path):
        """
        读取图片，支持路径字符串或Path对象
        Returns: (bgr_image, rgb_image) 或 (None, None)
        """
        path = str(img_path)
        
        if USE_CV2:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                # 尝试用PIL读取
                try:
                    from PIL import Image
                    pil_img = Image.open(path).convert('RGB')
                    img_rgb = np.array(pil_img)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    return img_bgr, img_rgb
                except:
                    return None, None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img, img_rgb
        else:
            try:
                pil_img = Image.open(path).convert('RGB')
                img_rgb = np.array(pil_img)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                return img_bgr, img_rgb
            except:
                return None, None
