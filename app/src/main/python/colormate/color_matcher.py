"""
ColorMatcher - 色彩风格匹配器

将训练好的色彩风格模型应用到目标照片上。
使用多层色彩迁移策略，从粗到细地调整目标照片的颜色。

迁移策略（按应用顺序）：
1. Lab色彩空间全局统计匹配（均值、标准差对齐）
2. 色调直方图匹配（Reinhard色彩迁移增强版）
3. 饱和度自适应映射
4. 色调映射曲线匹配（分位数匹配）
5. 局部对比度增强（可选）
6. 颜色自然度保护（防止色偏过重）
"""

import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter

try:
    import cv2
    USE_CV2 = True
except ImportError:
    USE_CV2 = False
    from PIL import Image


class ColorMatcher:
    """色彩风格匹配器"""
    
    def __init__(self, strength=1.0, preserve_natural=True):
        """
        Args:
            strength: 迁移强度 (0.0 ~ 1.0)，1.0为完全迁移
            preserve_natural: 是否保护自然度（防止色偏）
        """
        self.strength = max(0.0, min(1.0, strength))
        self.preserve_natural = preserve_natural
    
    def match(self, model, target_path, output_path=None, strength=None):
        """
        对目标图片应用色彩风格迁移
        
        Args:
            model: ColorStyleModel 实例
            target_path: 目标图片路径
            output_path: 输出路径（可选，为None时返回数组）
            strength: 覆盖初始化时的强度设置
            
        Returns:
            如果 output_path 为 None，返回 RGB numpy array
            否则保存到文件
        """
        if strength is not None:
            self.strength = max(0.0, min(1.0, strength))
        
        s = self.strength
        
        # 读取目标图片
        img_bgr, img_rgb = self._read_image(target_path)
        if img_bgr is None:
            raise IOError(f"Cannot read image: {target_path}")
        
        original = img_rgb.copy().astype(np.float32)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # =============================================
        # 第1层：Lab统计匹配（基础色调调整）
        # =============================================
        lab_matched = self._statistical_match(
            lab, 
            np.array(model.lab_means),
            np.array(model.lab_stds),
            s
        )
        
        # =============================================
        # 第2层：色调映射曲线匹配（分位数匹配）
        # =============================================
        lab_curve = self._quantile_matching(
            lab_matched,
            np.array(model.l_quantiles),
            np.array(model.a_quantiles),
            np.array(model.b_quantiles),
            s * 0.8  # 曲线匹配强度稍低
        )
        
        # =============================================
        # 第3层：色调直方图匹配
        # =============================================
        # 先转回RGB进行处理
        rgb_mid = cv2.cvtColor(lab_curve.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        
        # 色调匹配在HSV空间做
        hsv = cv2.cvtColor(rgb_mid.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        target_hue_hist = np.array(model.hue_hist)
        hsv_matched = self._hue_matching(hsv, target_hue_hist, s * 0.5)
        
        # 饱和度匹配
        sat_stats = model.saturation_stats
        hsv_matched = self._saturation_matching(hsv_matched, sat_stats, s * 0.6)
        
        # 转换回RGB
        rgb_mid2 = cv2.cvtColor(hsv_matched.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        
        # =============================================
        # 第4层：自然度保护
        # =============================================
        if self.preserve_natural:
            result = self._blend_with_original(original, rgb_mid2, s)
        else:
            result = rgb_mid2
        
        # 剪裁到有效范围
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        if output_path:
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), result_bgr)
            return str(output_path)
        
        return result
    
    def _statistical_match(self, lab, target_means, target_stds, strength):
        """
        Lab空间统计匹配：均值对齐 + 标准差缩放
        
        这是Reinhard色彩迁移的核心算法，带保护机制
        """
        lab_out = lab.copy()
        
        # 分离通道
        l = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]
        
        # 均值对齐
        l_mean, l_std = np.mean(l), np.std(l)
        a_mean, a_std = np.mean(a), np.std(a)
        b_mean, b_std = np.mean(b), np.std(b)
        
        # Lab L范围[0,255], a/b范围[-128,127]
        # 保护边缘值不被过冲
        
        # L通道调整
        if l_std > 0:
            l_new = (l - l_mean) * (target_stds[0] / l_std) + target_means[0]
            l_new = l * (1 - strength) + l_new * strength
            # L通道有效范围 [0, 255]
            l_new = np.clip(l_new, 0, 255)
            lab_out[:, :, 0] = l_new
        
        # a通道调整
        if a_std > 0:
            a_new = (a - a_mean) * (target_stds[1] / a_std) + target_means[1]
            a_new = a * (1 - strength) + a_new * strength
            lab_out[:, :, 1] = a_new
        
        # b通道调整
        if b_std > 0:
            b_new = (b - b_mean) * (target_stds[2] / b_std) + target_means[2]
            b_new = b * (1 - strength) + b_new * strength
            lab_out[:, :, 2] = b_new
        
        return lab_out
    
    def _quantile_matching(self, lab, target_l_q, target_a_q, target_b_q, strength):
        """
        分位数匹配：将目标图片的每个Lab通道的亮度分布映射到参考风格的分位数曲线
        
        这样可以精确控制影调分布
        """
        lab_out = lab.copy().astype(np.float32)
        n_quantiles = len(target_l_q)
        quantiles = np.linspace(0, 1, n_quantiles)
        
        # 为每个通道创建映射函数
        for ch_idx, target_q in enumerate([target_l_q, target_a_q, target_b_q]):
            channel = lab[:, :, ch_idx].flatten().astype(np.float32)
            
            # 计算当前图片的分位数
            src_q = np.quantile(channel, quantiles)
            
            # 创建差值映射
            # 对每个像素: 找到其在源分位数中的位置，映射到目标分位数
            # 使用线性插值
            f = interpolate.interp1d(
                src_q, target_q, 
                bounds_error=False, 
                fill_value=(target_q[0], target_q[-1])
            )
            
            mapped = f(channel)
            
            # 混合
            blended = channel * (1 - strength) + mapped * strength
            lab_out[:, :, ch_idx] = blended.reshape(lab.shape[:2])
        
        return lab_out
    
    def _hue_matching(self, hsv, target_hue_hist, strength):
        """
        色调直方图匹配：调整HSV空间的H通道，使色调分布接近参考风格
        
        使用直方图规定化（Histogram Specification）
        """
        if strength <= 0:
            return hsv
        
        hsv_out = hsv.copy()
        h_channel = hsv[:, :, 0]  # H: 0-180 (OpenCV)
        
        # 目标色调直方图（从360 bin重采样到180 bin）
        target_h_180 = np.array(target_hue_hist)
        if len(target_h_180) == 360:
            target_h_180 = target_h_180.reshape(2, 180).mean(axis=0)
        
        # 计算当前图片的H通道直方图
        h_flat = h_channel.flatten()
        hist_curr, _ = np.histogram(h_flat, bins=180, range=(0, 180))
        hist_curr = hist_curr.astype(np.float32)
        hist_curr = hist_curr / max(hist_curr.sum(), 1)
        
        # 直方图规定化
        # 计算累积分布
        cdf_curr = np.cumsum(hist_curr)
        cdf_curr = cdf_curr / cdf_curr[-1]
        
        cdf_target = np.cumsum(target_h_180)
        cdf_target = cdf_target / cdf_target[-1]
        
        # 建立映射: 源bins -> 目标bins
        mapping = np.zeros(180, dtype=np.float32)
        for i in range(180):
            diff = np.abs(cdf_curr[i] - cdf_target)
            mapping[i] = np.argmin(diff)
        
        # 应用映射（带羽化避免Blocky artifact）
        mapped_h = mapping[h_flat.astype(np.int32)]
        
        # 混合
        h_new = h_flat * (1 - strength) + mapped_h * strength
        h_new = np.clip(h_new, 0, 179)
        
        hsv_out[:, :, 0] = h_new.reshape(h_channel.shape)
        
        return hsv_out
    
    def _saturation_matching(self, hsv, target_sat_stats, strength):
        """
        饱和度匹配：调整S通道的均值和标准差
        """
        if strength <= 0:
            return hsv
        
        hsv_out = hsv.copy()
        s_channel = hsv[:, :, 1].astype(np.float32)
        
        s_mean = np.mean(s_channel)
        s_std = np.std(s_channel)
        
        t_mean, t_std = target_sat_stats
        
        # 保护低饱和度区域（如天空、皮肤）
        if s_std > 0:
            # 计算每个像素的调整量，与当前饱和度值成正比
            s_new = (s_channel - s_mean) * (t_std / max(s_std, 1.0)) + t_mean
            
            # 对原饱和度很低的区域施加更少的调整
            low_sat_mask = s_channel < 30
            blend_factor = np.where(low_sat_mask, 0.3, 1.0)
            
            s_blended = s_channel * (1 - strength * blend_factor) + s_new * strength * blend_factor
            s_blended = np.clip(s_blended, 0, 255)
            
            hsv_out[:, :, 1] = s_blended
        
        return hsv_out
    
    def _blend_with_original(self, original, matched, strength):
        """
        与原图混合，保留部分原始细节和自然感
        
        同时使用边缘保护蒙版：在边缘区域保持更多原始信息
        """
        original_f = original.astype(np.float32)
        matched_f = matched.astype(np.float32)
        
        # 计算Lab空间的颜色偏移幅度
        orig_bgr = cv2.cvtColor(original_f.astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)
        match_bgr = cv2.cvtColor(matched_f.astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)
        
        orig_lab = cv2.cvtColor(orig_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        match_lab = cv2.cvtColor(match_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # ab平面上的色偏幅度
        ab_diff = np.sqrt(
            (match_lab[:, :, 1] - orig_lab[:, :, 1]) ** 2 + 
            (match_lab[:, :, 2] - orig_lab[:, :, 2]) ** 2
        )
        
        # 色偏保护：色偏大的区域，用更低的强度混合
        max_diff = np.percentile(ab_diff, 95)
        if max_diff > 0:
            protect_mask = np.clip(1.0 - ab_diff / max_diff, 0.2, 1.0)
        else:
            protect_mask = np.ones_like(ab_diff)
        
        # 边缘检测：边缘区域保留更多原始细节
        gray = cv2.cvtColor(matched_f.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        edges = gaussian_filter(edges, sigma=2.0)
        
        # 在边缘区域，保留更多原始信息
        edge_protect = 1.0 - edges * 0.3  # 边缘区域最多保留30%原始
        
        # 合并保护蒙版
        combined_mask = protect_mask * edge_protect
        
        # 应用混合
        # 低强度时更保守，高强度时更大胆
        alpha = strength * combined_mask
        alpha = np.clip(alpha, 0, 1)
        
        # 3通道蒙版
        mask_3ch = np.stack([alpha, alpha, alpha], axis=2)
        
        result = original_f * (1 - mask_3ch) + matched_f * mask_3ch
        
        # 额外保护：如果ab偏移过大，退回到更自然的版本
        extreme_mask = (match_lab[:, :, 1] - orig_lab[:, :, 1]) > 50
        extreme_mask |= (match_lab[:, :, 2] - orig_lab[:, :, 2]) > 50
        if np.any(extreme_mask):
            extreme_mask_3ch = np.stack([extreme_mask] * 3, axis=2)
            result = np.where(
                extreme_mask_3ch,
                original_f * 0.5 + matched_f * 0.5,
                result
            )
        
        return result
    
    def _read_image(self, img_path):
        """读取图片"""
        path = str(img_path)
        if USE_CV2:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                from PIL import Image
                pil_img = Image.open(path).convert('RGB')
                img_rgb = np.array(pil_img)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                return img_bgr, img_rgb
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img, img_rgb
        else:
            from PIL import Image
            pil_img = Image.open(path).convert('RGB')
            img_rgb = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            return img_bgr, img_rgb
