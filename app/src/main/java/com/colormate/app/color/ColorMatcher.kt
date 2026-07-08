package com.colormate.app.color

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.sqrt
import kotlin.math.min
import kotlin.math.max
import kotlin.math.ceil
import java.io.File

/**
 * 色彩风格匹配器 — 将训练好的色彩风格模型应用到目标照片上
 *
 * 对应原Python版 ColorMatcher
 *
 * 迁移策略（与原Python版一致）：
 * 1. Lab色彩空间全局统计匹配（均值、标准差对齐）
 * 2. 色调直方图匹配（Reinhard色彩迁移增强版）
 * 3. 饱和度自适应映射
 * 4. 色调映射曲线匹配（分位数匹配）
 * 5. 颜色自然度保护（防止色偏过重）
 */
class ColorMatcher(
    private var strength: Float = 1.0f,
    private val preserveNatural: Boolean = true
) {

    companion object {
        private const val HUE_BINS = 360
        private const val QUANTILE_N = 100
    }

    /**
     * 对目标图片应用色彩风格迁移
     * @param model 训练好的 ColorStyleModel
     * @param targetPath 目标图片路径
     * @param outputPath 输出路径
     * @param matchStrength 可选的强度覆盖
     */
    fun match(
        model: ColorStyleModel,
        targetPath: String,
        outputPath: String,
        matchStrength: Float? = null
    ) {
        if (matchStrength != null) strength = matchStrength
        strength = strength.coerceIn(0f, 1f)
        val s = strength

        // 读取目标图片为 Bitmap → Mat
        val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val bitmap = BitmapFactory.decodeFile(targetPath, options)
            ?: throw java.io.IOException("Cannot read image: $targetPath")

        val originalRgb = Mat()
        Utils.bitmapToMat(bitmap, originalRgb)

        // 保存原始RGB用于后续混合
        val original = Mat()
        originalRgb.copyTo(original)

        // 转Lab
        val lab = Mat()
        Imgproc.cvtColor(originalRgb, lab, Imgproc.COLOR_RGB2Lab)
        val labFloat = Mat()
        lab.convertTo(labFloat, CvType.CV_32FC3)

        // ========== 第1层：Lab统计匹配 ==========
        val labStats = statisticalMatch(labFloat, model.labMeans!!, model.labStds!!, s)

        // ========== 第2层：色调映射曲线匹配 ==========
        val labCurve = quantileMatching(labStats, model.lQuantiles!!, model.aQuantiles!!, model.bQuantiles!!, s * 0.8f)

        // ========== 第3层：色调+饱和度匹配 ==========
        val labCurveU8 = Mat()
        labCurve.convertTo(labCurveU8, CvType.CV_8UC3)
        val rgbMid = Mat()
        Imgproc.cvtColor(labCurveU8, rgbMid, Imgproc.COLOR_Lab2RGB)

        val hsv = Mat()
        Imgproc.cvtColor(rgbMid, hsv, Imgproc.COLOR_RGB2HSV)
        val hsvFloat = Mat()
        hsv.convertTo(hsvFloat, CvType.CV_32FC3)

        // 色调匹配
        val hsvHue = hueMatching(hsvFloat, model.hueHist!!, s * 0.5f)

        // 饱和度匹配
        val hsvSat = saturationMatching(hsvHue, model.saturationStats!!, s * 0.6f)

        val hsvSatU8 = Mat()
        hsvSat.convertTo(hsvSatU8, CvType.CV_8UC3)
        val rgbMid2 = Mat()
        Imgproc.cvtColor(hsvSatU8, rgbMid2, Imgproc.COLOR_HSV2RGB)

        // ========== 第4层：自然度保护 ==========
        val result: Mat
        if (preserveNatural) {
            result = blendWithOriginal(original, rgbMid2, s)
        } else {
            result = rgbMid2
        }

        // 剪裁并保存
        val resultClipped = Mat()
        Core.max(result, Scalar.all(0.0), resultClipped)
        Core.min(resultClipped, Scalar.all(255.0), resultClipped)
        resultClipped.convertTo(resultClipped, CvType.CV_8UC3)

        // 转BGR保存（OpenCV imwrite需要BGR）
        val resultBgr = Mat()
        Imgproc.cvtColor(resultClipped, resultBgr, Imgproc.COLOR_RGB2BGR)

        val outBitmap = Bitmap.createBitmap(resultBgr.cols(), resultBgr.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(resultBgr, outBitmap)

        File(outputPath).parentFile?.mkdirs()
        File(outputPath).outputStream().use { fos ->
            outBitmap.compress(Bitmap.CompressFormat.JPEG, 95, fos)
        }
    }

    // ========== 各层匹配算法 ==========

    /**
     * Lab空间统计匹配：均值对齐 + 标准差缩放
     * 对应Python版 _statistical_match
     */
    private fun statisticalMatch(
        lab: Mat, targetMeans: FloatArray, targetStds: FloatArray, s: Float
    ): Mat {
        val result = Mat()
        lab.copyTo(result)

        val channels = mutableListOf<Mat>()
        Core.split(result, channels)

        for (ch in 0..2) {
            val channel = channels[ch]
            val mean = Core.mean(channel).`val`[0]
            val std = computeStdDev(channel, mean)

            if (std > 0) {
                val newChannel = Mat()
                Core.subtract(channel, Scalar.all(mean.toDouble()), newChannel)
                Core.multiply(newChannel, Scalar.all((targetStds[ch] / std).toDouble()), newChannel)
                Core.add(newChannel, Scalar.all(targetMeans[ch].toDouble()), newChannel)

                // 混合
                Core.addWeighted(channel, (1 - s).toDouble(), newChannel, s.toDouble(), 0.0, channel)
            }

            // L通道 [0, 255], a/b范围约[-128, 127]
            when (ch) {
                0 -> Core.max(Scalar.all(0.0), channel, channel).also {
                    Core.min(channel, Scalar.all(255.0), channel)
                }
                1, 2 -> {
                    Core.max(Scalar.all(-128.0), channel, channel).also {
                        Core.min(channel, Scalar.all(127.0), channel)
                    }
                }
            }
        }

        Core.merge(channels, result)
        return result
    }

    /**
     * 分位数匹配：将目标图片的Lab通道分布映射到参考风格的分位数曲线
     * 对应Python版 _quantile_matching
     */
    private fun quantileMatching(
        lab: Mat, targetLQ: FloatArray, targetAQ: FloatArray, targetBQ: FloatArray, s: Float
    ): Mat {
        val result = Mat()
        lab.copyTo(result)

        val channels = mutableListOf<Mat>()
        Core.split(result, channels)

        val targetQs = listOf(targetLQ, targetAQ, targetBQ)

        for (ch in 0..2) {
            val channel = channels[ch]
            val totalPixels = channel.rows() * channel.cols()
            val flat = FloatArray(totalPixels)
            channel.get(0, 0, flat)

            // 计算当前图片分位数
            val sorted = flat.sorted().toFloatArray()
            val srcQ = computeQuantiles(sorted, QUANTILE_N)

            // 线性插值映射
            val mapped = FloatArray(flat.size)
            for (i in flat.indices) {
                mapped[i] = interpolate1D(srcQ, targetQs[ch], flat[i])
            }

            // 混合
            for (i in flat.indices) {
                flat[i] = flat[i] * (1 - s) + mapped[i] * s
            }
            channel.put(0, 0, flat)
        }

        Core.merge(channels, result)
        return result
    }

    /**
     * 色调直方图匹配：调整HSV空间的H通道
     * 对应Python版 _hue_matching
     */
    private fun hueMatching(hsv: Mat, targetHueHist: FloatArray, s: Float): Mat {
        if (s <= 0) return hsv

        val result = Mat()
        hsv.copyTo(result)

        val channels = mutableListOf<Mat>()
        Core.split(result, channels)
        val hChannel = channels[0]

        val totalPixels = hChannel.rows() * hChannel.cols()
        val flat = FloatArray(totalPixels)
        hChannel.get(0, 0, flat)

        // 目标直方图（360 bin → 180 bin for OpenCV）
        val targetH180 = if (targetHueHist.size == 360) {
            FloatArray(180) { i -> (targetHueHist[i * 2] + targetHueHist[i * 2 + 1]) / 2f }
        } else {
            targetHueHist
        }

        // 当前直方图 (180 bins, H: 0-180)
        val histCurr = FloatArray(180)
        for (v in flat) {
            val bin = (v / 180f * 180).toInt().coerceIn(0, 179)
            histCurr[bin]++
        }
        val histSum = histCurr.sum()
        if (histSum > 0) for (i in histCurr.indices) histCurr[i] /= histSum

        // CDF计算
        val cdfCurr = computeCdf(histCurr)
        val cdfTarget = computeCdf(targetH180)

        // 建立映射: 源bin → 目标bin
        val mapping = IntArray(180)
        for (i in 0 until 180) {
            var bestJ = 0
            var bestDiff = Float.MAX_VALUE
            for (j in 0 until 180) {
                val diff = kotlin.math.abs(cdfCurr[i] - cdfTarget[j])
                if (diff < bestDiff) {
                    bestDiff = diff
                    bestJ = j
                }
            }
            mapping[i] = bestJ
        }

        // 应用映射并混合
        for (i in flat.indices) {
            val srcBin = (flat[i] / 180f * 180).toInt().coerceIn(0, 179)
            val mappedValue = mapping[srcBin].toFloat() * 180f / 180f
            flat[i] = flat[i] * (1 - s) + mappedValue * s
            flat[i] = flat[i].coerceIn(0f, 179f)
        }
        hChannel.put(0, 0, flat)

        Core.merge(channels, result)
        return result
    }

    /**
     * 饱和度匹配：调整S通道的均值和标准差
     * 对应Python版 _saturation_matching
     */
    private fun saturationMatching(hsv: Mat, targetSatStats: FloatArray, s: Float): Mat {
        if (s <= 0) return hsv

        val result = Mat()
        hsv.copyTo(result)

        val channels = mutableListOf<Mat>()
        Core.split(result, channels)
        val sChannel = channels[1]

        val totalPixels = sChannel.rows() * sChannel.cols()
        val flat = FloatArray(totalPixels)
        sChannel.get(0, 0, flat)

        val sMean = flat.average().toFloat()
        val sStd = computeStdDev(flat, sMean)

        val tMean = targetSatStats[0]
        val tStd = targetSatStats[1]

        if (sStd > 0) {
            for (i in flat.indices) {
                val newVal = (flat[i] - sMean) * (tStd / sStd) + tMean

                // 低饱和度区域保护
                val blendFactor = if (flat[i] < 30f) 0.3f else 1.0f
                val blended = flat[i] * (1 - s * blendFactor) + newVal * s * blendFactor
                flat[i] = blended.coerceIn(0f, 255f)
            }
        }
        sChannel.put(0, 0, flat)

        Core.merge(channels, result)
        return result
    }

    /**
     * 与原图混合，保留原始细节和自然感
     * 对应Python版 _blend_with_original
     */
    private fun blendWithOriginal(original: Mat, matched: Mat, s: Float): Mat {
        if (original.size() != matched.size()) {
            val resized = Mat()
            Imgproc.resize(matched, resized, original.size())
            return blendWithOriginal(original, resized, s)
        }

        val origF = Mat()
        original.convertTo(origF, CvType.CV_32FC3)

        val matchF = Mat()
        matched.convertTo(matchF, CvType.CV_32FC3)

        // 转Lab计算色偏
        val origLab = Mat()
        val matchLab = Mat()
        Imgproc.cvtColor(original, origLab, Imgproc.COLOR_RGB2Lab)
        Imgproc.cvtColor(matched, matchLab, Imgproc.COLOR_RGB2Lab)

        val origLabF = Mat()
        val matchLabF = Mat()
        origLab.convertTo(origLabF, CvType.CV_32FC3)
        matchLab.convertTo(matchLabF, CvType.CV_32FC3)

        val totalPixels = origLabF.rows() * origLabF.cols()
        val origAB = FloatArray(totalPixels * 2)  // [a1, b1, a2, b2, ...]
        val matchAB = FloatArray(totalPixels * 2)

        // Extract a, b channels
        val origChannels = mutableListOf<Mat>()
        val matchChannels = mutableListOf<Mat>()
        Core.split(origLabF, origChannels)
        Core.split(matchLabF, matchChannels)

        val origA = FloatArray(totalPixels)
        val origB = FloatArray(totalPixels)
        val matchA = FloatArray(totalPixels)
        val matchB = FloatArray(totalPixels)
        origChannels[1].get(0, 0, origA)
        origChannels[2].get(0, 0, origB)
        matchChannels[1].get(0, 0, matchA)
        matchChannels[2].get(0, 0, matchB)

        // 计算ab色偏幅度
        val abDiff = FloatArray(totalPixels)
        var maxDiff = 0f
        for (i in 0 until totalPixels) {
            val d = sqrt(
                (matchA[i] - origA[i]) * (matchA[i] - origA[i]) +
                (matchB[i] - origB[i]) * (matchB[i] - origB[i])
            )
            abDiff[i] = d
            if (d > maxDiff) maxDiff = d
        }

        // 保护蒙版
        val protectMask = if (maxDiff > 0) {
            FloatArray(totalPixels) { (1f - abDiff[it] / maxDiff).coerceIn(0.2f, 1f) }
        } else {
            FloatArray(totalPixels) { 1f }
        }

        // 边缘检测保护
        val gray = Mat()
        Imgproc.cvtColor(matched, gray, Imgproc.COLOR_RGB2GRAY)
        val edges = Mat()
        Imgproc.Canny(gray, edges, 50.0, 150.0)

        val edgeFlat = ByteArray(totalPixels)
        edges.get(0, 0, edgeFlat)

        // 应用混合
        val result = Mat(origF.size(), CvType.CV_32FC3)
        val origFlat = FloatArray(totalPixels * 3)
        val matchFlat = FloatArray(totalPixels * 3)
        val resultFlat = FloatArray(totalPixels * 3)
        origF.get(0, 0, origFlat)
        matchF.get(0, 0, matchFlat)

        for (i in 0 until totalPixels) {
            val edgeFactor = 1.0f - (edgeFlat[i].toInt() and 0xFF) / 255f * 0.3f
            val alpha = (s * protectMask[i] * edgeFactor).coerceIn(0f, 1f)

            for (ch in 0..2) {
                resultFlat[i * 3 + ch] = origFlat[i * 3 + ch] * (1 - alpha) +
                        matchFlat[i * 3 + ch] * alpha
            }
        }
        result.put(0, 0, resultFlat)

        return result
    }

    // ========== 工具方法 ==========

    private fun computeStdDev(mat: Mat, mean: Double): Float {
        val flat = FloatArray(mat.rows() * mat.cols())
        mat.get(0, 0, flat)
        return computeStdDev(flat, mean.toFloat())
    }

    private fun computeStdDev(values: FloatArray, mean: Float): Float {
        if (values.isEmpty()) return 0f
        val variance = values.map { (it - mean) * (it - mean) }.average()
        return sqrt(variance).toFloat()
    }

    private fun computeQuantiles(sorted: FloatArray, n: Int): FloatArray {
        val result = FloatArray(n)
        for (i in 0 until n) {
            val q = i.toFloat() / (n - 1)
            val idx = q * (sorted.size - 1)
            val idxLow = idx.toInt().coerceIn(0, sorted.size - 1)
            val idxHigh = (idxLow + 1).coerceAtMost(sorted.size - 1)
            val frac = idx - idxLow
            result[i] = sorted[idxLow] * (1 - frac) + sorted[idxHigh] * frac
        }
        return result
    }

    /**
     * 一维线性插值
     */
    private fun interpolate1D(x: FloatArray, y: FloatArray, query: Float): Float {
        if (query <= x.first()) return y.first()
        if (query >= x.last()) return y.last()

        var low = 0
        var high = x.size - 1
        while (high - low > 1) {
            val mid = (low + high) / 2
            if (x[mid] <= query) low = mid
            else high = mid
        }

        val frac = (query - x[low]) / (x[high] - x[low])
        return y[low] * (1 - frac) + y[high] * frac
    }

    private fun computeCdf(hist: FloatArray): FloatArray {
        val cdf = FloatArray(hist.size)
        var sum = 0f
        for (i in hist.indices) {
            sum += hist[i]
            cdf[i] = sum
        }
        if (cdf.last() > 0) {
            for (i in cdf.indices) cdf[i] /= cdf.last()
        }
        return cdf
    }
}
