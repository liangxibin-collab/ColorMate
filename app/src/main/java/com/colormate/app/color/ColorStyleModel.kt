package com.colormate.app.color

import java.io.Serializable

/**
 * 色彩风格模型 — 保存所有学习到的特征
 * 对应原Python版 ColorStyleModel
 */
class ColorStyleModel(
    val name: String = "ColorStyle",
    var numTrainingImages: Int = 0
) : Serializable {
    // Lab统计特征
    var labMeans: FloatArray? = null     // [L_mean, a_mean, b_mean]
    var labStds: FloatArray? = null      // [L_std, a_std, b_std]

    // 色调分布
    var hueHist: FloatArray? = null      // 360 bins
    var hueHistPeaks: List<HuePeak>? = null

    // 饱和度
    var saturationStats: FloatArray? = null  // [mean, std]

    // 色调映射曲线（分位数）
    var lQuantiles: FloatArray? = null
    var aQuantiles: FloatArray? = null
    var bQuantiles: FloatArray? = null

    data class HuePeak(val hue: Float, val height: Float) : Serializable
}
