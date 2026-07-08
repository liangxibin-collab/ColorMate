package com.colormate.app.color

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.*

/**
 * 色彩风格训练器 — 从一组参考照片中学习色彩风格特征
 *
 * 对应原Python版 ColorStyleTrainer
 *
 * 核心方法（与Python版一致）：
 * 1. Lab色彩空间统计匹配（均值、标准差）
 * 2. 色调直方图 (Hue histogram) 分布
 * 3. 饱和度特征映射
 * 4. 基于分位数的色调映射曲线
 */
class ColorStyleTrainer(private val sampleSize: Int = 500) {

    companion object {
        private const val HUE_BINS = 360
        private const val SAT_BINS = 100
        private const val QUANTILE_N = 100
        private const val L_HIST_BINS = 100
    }

    /**
     * 从一组图片训练色彩风格模型
     * @param imagePaths 图片文件路径列表
     * @return ColorStyleModel
     */
    fun train(imagePaths: List<String>): ColorStyleModel {
        val n = imagePaths.size
        require(n > 0) { "No training images provided" }

        val model = ColorStyleModel(numTrainingImages = n)

        // 累积统计
        val allLabPixels = mutableListOf<FloatArray>()  // 每张图片的采样像素（Lab）
        val allHueHists = mutableListOf<FloatArray>()
        val allSatVals = mutableListOf<Float>()
        val allLQuantiles = mutableListOf<FloatArray>()
        val allAQuantiles = mutableListOf<FloatArray>()
        val allBQuantiles = mutableListOf<FloatArray>()

        val validExts = setOf("jpg", "jpeg")

        for (imgPath in imagePaths) {
            if (File(imgPath).extension.lowercase() !in validExts) continue

            val lab = loadLabImage(imgPath) ?: continue
            val hsv = convertToHsv(lab)

            val h = lab.rows()
            val w = lab.cols()
            val totalPixels = h * w
            val sampleCount = minOf(sampleSize, totalPixels)

            // 随机采样
            val rng = Random(42)
            val indices = (0 until totalPixels).toMutableList().also { it.shuffle(rng) }
                .take(sampleCount)

            // 收集Lab采样
            val labFlat = FloatArray(sampleCount * 3)
            for (i in 0 until sampleCount) {
                val idx = indices[i]
                val row = idx / w
                val col = idx % w
                labFlat[i * 3] = lab.get(row, col)[0]
                labFlat[i * 3 + 1] = lab.get(row, col)[1]
                labFlat[i * 3 + 2] = lab.get(row, col)[2]
            }
            allLabPixels.add(labFlat)

            // 收集HSV采样
            for (i in 0 until sampleCount) {
                val idx = indices[i]
                val row = idx / w
                val col = idx % w
                allSatVals.add(hsv.get(row, col)[1])
            }

            // 色调直方图
            val hueHist = FloatArray(HUE_BINS)
            for (i in 0 until sampleCount) {
                val idx = indices[i]
                val row = idx / w
                val col = idx % w
                val hue = hsv.get(row, col)[0].toInt() // H: 0-180 (OpenCV)
                val bin = ((hue.toFloat() / 180f) * HUE_BINS).toInt().coerceIn(0, HUE_BINS - 1)
                hueHist[bin]++
            }
            // 归一化
            val hueSum = hueHist.sum()
            if (hueSum > 0) {
                for (j in hueHist.indices) hueHist[j] /= hueSum
            }
            allHueHists.add(hueHist)

            // 分位数（从全部像素或大采样计算）
            val quantileSample = if (totalPixels > 100000) {
                val qIndices = (0 until totalPixels).toMutableList().also { it.shuffle(rng) }
                    .take(100000)
                val lSamp = FloatArray(qIndices.size)
                val aSamp = FloatArray(qIndices.size)
                val bSamp = FloatArray(qIndices.size)
                for (i in qIndices.indices) {
                    val row = qIndices[i] / w
                    val col = qIndices[i] % w
                    lSamp[i] = lab.get(row, col)[0]
                    aSamp[i] = lab.get(row, col)[1]
                    bSamp[i] = lab.get(row, col)[2]
                }
                Triple(lSamp, aSamp, bSamp)
            } else {
                val lSamp = FloatArray(totalPixels)
                val aSamp = FloatArray(totalPixels)
                val bSamp = FloatArray(totalPixels)
                for (i in 0 until totalPixels) {
                    val row = i / w
                    val col = i % w
                    lSamp[i] = lab.get(row, col)[0]
                    aSamp[i] = lab.get(row, col)[1]
                    bSamp[i] = lab.get(row, col)[2]
                }
                Triple(lSamp, aSamp, bSamp)
            }

            allLQuantiles.add(computeQuantiles(quantileSample.first, QUANTILE_N))
            allAQuantiles.add(computeQuantiles(quantileSample.second, QUANTILE_N))
            allBQuantiles.add(computeQuantiles(quantileSample.third, QUANTILE_N))
        }

        require(allLabPixels.isNotEmpty()) { "No valid images could be read for training" }

        // ========== 1. Lab统计特征 ==========
        val allPixels = allLabPixels.flatMap { it.toList() }.toFloatArray()
        val nPixels = allPixels.size / 3
        val flattened = Array(3) { FloatArray(nPixels) }
        for (i in 0 until nPixels) {
            flattened[0][i] = allPixels[i * 3]
            flattened[1][i] = allPixels[i * 3 + 1]
            flattened[2][i] = allPixels[i * 3 + 2]
        }
        model.labMeans = FloatArray(3) { ch -> flattened[ch].average().toFloat() }
        model.labStds = FloatArray(3) { ch ->
            val mean = model.labMeans!![ch]
            val variance = flattened[ch].map { (it - mean) * (it - mean) }.average()
            kotlin.math.sqrt(variance).toFloat()
        }

        // ========== 2. 色调分布 ==========
        val meanHueHist = FloatArray(HUE_BINS) { j ->
            allHueHists.map { it[j] }.average().toFloat()
        }
        model.hueHist = meanHueHist

        // 找峰值（简化版：局部最大值）
        model.hueHistPeaks = findHuePeaks(meanHueHist)

        // ========== 3. 饱和度特征 ==========
        val satMean = allSatVals.average().toFloat()
        val satStd = kotlin.math.sqrt(
            allSatVals.map { (it - satMean) * (it - satMean) }.average()
        ).toFloat()
        model.saturationStats = floatArrayOf(satMean, satStd)

        // ========== 4. 色调映射曲线 ==========
        model.lQuantiles = FloatArray(QUANTILE_N) { j ->
            allLQuantiles.map { it[j] }.average().toFloat()
        }
        model.aQuantiles = FloatArray(QUANTILE_N) { j ->
            allAQuantiles.map { it[j] }.average().toFloat()
        }
        model.bQuantiles = FloatArray(QUANTILE_N) { j ->
            allBQuantiles.map { it[j] }.average().toFloat()
        }

        return model
    }

    // ========== 工具方法 ==========

    private fun loadLabImage(path: String): Mat? {
        return try {
            // Try loading via Bitmap first (more reliable on Android)
            val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
            val bitmap = BitmapFactory.decodeFile(path, options) ?: return null
            val rgb = Mat()
            Utils.bitmapToMat(bitmap, rgb)
            val lab = Mat()
            Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
            lab
        } catch (e: Exception) {
            null
        }
    }

    private fun convertToHsv(lab: Mat): Mat {
        // Convert Lab back to RGB then to HSV
        val rgb = Mat()
        Imgproc.cvtColor(lab, rgb, Imgproc.COLOR_Lab2RGB)
        val hsv = Mat()
        Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
        return hsv
    }

    private fun computeQuantiles(values: FloatArray, n: Int): FloatArray {
        val sorted = values.sorted().toFloatArray()
        val result = FloatArray(n)
        for (i in 0 until n) {
            val q = i.toFloat() / (n - 1)
            val idx = (q * (sorted.size - 1))
            val idxLow = idx.toInt().coerceIn(0, sorted.size - 1)
            val idxHigh = (idxLow + 1).coerceAtMost(sorted.size - 1)
            val frac = idx - idxLow
            result[i] = sorted[idxLow] * (1 - frac) + sorted[idxHigh] * frac
        }
        return result
    }

    private fun findHuePeaks(hist: FloatArray, minHeightFraction: Float = 0.15f, minDistance: Int = 15): List<ColorStyleModel.HuePeak> {
        val maxVal = hist.maxOrNull() ?: 0f
        val threshold = maxVal * minHeightFraction

        val peaks = mutableListOf<Pair<Int, Float>>()
        var i = 1
        while (i < hist.size - 1) {
            if (hist[i] > hist[i - 1] && hist[i] > hist[i + 1] && hist[i] >= threshold) {
                peaks.add(i to hist[i])
                i += minDistance  // skip ahead
            } else {
                i++
            }
        }

        peaks.sortByDescending { it.second }
        return peaks.take(10).map { (idx, height) ->
            ColorStyleModel.HuePeak(hue = idx.toFloat() * 180f / HUE_BINS, height = height)
        }
    }
}
