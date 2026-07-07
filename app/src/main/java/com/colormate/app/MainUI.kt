package com.colormate.app

import android.graphics.Color
import android.net.Uri
import android.view.Gravity
import android.view.View
import android.widget.*
import androidx.core.content.res.ResourcesCompat
import androidx.core.graphics.toColor
import androidx.core.view.setPadding

/**
 * ColorMate 主界面 - 纯代码布局（无需XML）
 */
class MainUI(private val activity: MainActivity, private val onAction: (Action) -> Unit) {

    sealed class Action {
        data object SELECT_TRAINING_DIR : Action()
        data object START_TRAINING : Action()
        data object SELECT_TARGET : Action()
        data object START_MATCHING : Action()
        data object SELECT_MODEL : Action()
        data object SET_STRENGTH : Action()
        data object CLEAR_MODEL : Action()
        data class UPDATE_STRENGTH(val value: Float) : Action()
    }

    // UI组件
    val root: ScrollView
    private val container: LinearLayout

    // 状态展示
    val modelStatus: TextView
    private val trainingCountText: TextView
    private val logView: TextView

    // 按钮
    private val btnSelectDir: Button
    private val btnTrain: Button
    private val btnSelectTarget: Button
    private val btnMatch: Button
    private val btnLoadModel: Button
    private val btnClearModel: Button

    // 图片预览
    private val targetPreview: ImageView
    private val resultPreview: ImageView

    // 进度指示
    private val progressContainer: LinearLayout
    private val progressBar: ProgressBar
    private val progressText: TextView

    // 强度滑块
    private val strengthSlider: SeekBar
    private val strengthText: TextView

    // 状态跟踪
    var currentTargetUri: Uri? = null
    private var isModelLoaded = false
    private var trainingCount = 0

    init {
        // ============================================
        // 根布局
        // ============================================
        root = ScrollView(activity).apply {
            isFillViewport = true
        }

        container = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24)
            setBackgroundColor(Color.parseColor("#1A1A2E"))
        }

        root.addView(container)

        // ============================================
        // 标题
        // ============================================
        container.addView(TextView(activity).apply {
            text = "🎨 ColorMate"
            textSize = 28f
            typeface = ResourcesCompat.getFont(activity, R.font.sans_serif)
            setTextColor(Color.parseColor("#E94560"))
            gravity = Gravity.CENTER
            setPadding(0, 16, 0, 8)
        })

        container.addView(TextView(activity).apply {
            text = "AI 照片追色工具"
            textSize = 14f
            setTextColor(Color.parseColor("#888888"))
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 24)
        })

        // ============================================
        // 模型状态栏
        // ============================================
        modelStatus = createInfoCard("📊 模型状态", "未训练")
        container.addView(modelStatus)

        trainingCountText = createSmallText("")
        container.addView(trainingCountText)

        container.addView(createDivider())

        // ============================================
        // 第1部分：训练
        // ============================================
        container.addView(createSectionHeader("📚 第1步：训练色彩风格"))

        val trainLayout = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 8, 0, 8)
        }

        btnSelectDir = createButton("📁 选择训练目录（含数百张参考照片）")
        btnSelectDir.setOnClickListener { onAction(Action.SELECT_TRAINING_DIR) }
        trainLayout.addView(btnSelectDir)

        progressContainer = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            visibility = View.GONE
            setPadding(0, 8, 0, 8)
        }

        progressBar = ProgressBar(activity, null, android.R.attr.progressBarStyleHorizontal)
        progressBar.layoutParams = LinearLayout.LayoutParams(0, 48, 1f)

        progressText = TextView(activity).apply {
            setPadding(12, 0, 0, 0)
            setTextColor(Color.parseColor("#AAAAAA"))
            textSize = 12f
        }
        progressContainer.addView(progressBar)
        progressContainer.addView(progressText)
        trainLayout.addView(progressContainer)

        btnTrain = createButton("▶️ 开始训练", accent = true)
        btnTrain.isEnabled = false
        btnTrain.setOnClickListener { onAction(Action.START_TRAINING) }
        trainLayout.addView(btnTrain)

        container.addView(trainLayout)
        container.addView(createDivider())

        // ============================================
        // 第2部分：追色
        // ============================================
        container.addView(createSectionHeader("🖌 第2步：追色匹配"))

        val matchLayout = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 8, 0, 8)
        }

        btnSelectTarget = createButton("🖼 选择目标照片")
        btnSelectTarget.setOnClickListener { onAction(Action.SELECT_TARGET) }
        matchLayout.addView(btnSelectTarget)

        // 预览区域
        val previewLayout = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }

        targetPreview = ImageView(activity).apply {
            layoutParams = LinearLayout.LayoutParams(0, 360, 1f).apply {
                setMargins(0, 8, 8, 8)
            }
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            background = ResourcesCompat.getDrawable(
                activity.resources, android.R.drawable.dark_header, null
            )
            setImageDrawable(ResourcesCompat.getDrawable(
                activity.resources, android.R.drawable.ic_menu_gallery, null
            ))
        }

        val arrowText = TextView(activity).apply {
            text = "→"
            textSize = 32f
            setTextColor(Color.parseColor("#E94560"))
            gravity = Gravity.CENTER
            setPadding(4, 0, 4, 0)
        }

        resultPreview = ImageView(activity).apply {
            layoutParams = LinearLayout.LayoutParams(0, 360, 1f).apply {
                setMargins(8, 8, 0, 8)
            }
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            background = ResourcesCompat.getDrawable(
                activity.resources, android.R.drawable.dark_header, null
            )
            setImageDrawable(ResourcesCompat.getDrawable(
                activity.resources, android.R.drawable.ic_menu_gallery, null
            ))
        }

        previewLayout.addView(targetPreview)
        previewLayout.addView(arrowText)
        previewLayout.addView(resultPreview)
        matchLayout.addView(previewLayout)

        // 强度控制
        val strengthLayout = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 8, 0, 8)
        }

        strengthLayout.addView(TextView(activity).apply {
            text = "强度:"
            setTextColor(Color.parseColor("#CCCCCC"))
            textSize = 14f
            setPadding(0, 0, 8, 0)
        })

        strengthText = TextView(activity).apply {
            text = "1.0"
            setTextColor(Color.parseColor("#E94560"))
            textSize = 14f
            setPadding(0, 0, 8, 0)
        }

        strengthSlider = SeekBar(activity).apply {
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            max = 100
            progress = 100
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar, p: Int, fromUser: Boolean) {
                    val v = p / 100f
                    strengthText.text = String.format("%.1f", v)
                    onAction(Action.UPDATE_STRENGTH(v))
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }

        strengthLayout.addView(strengthSlider)
        matchLayout.addView(strengthLayout)

        btnMatch = createButton("✨ 开始追色", accent = true)
        btnMatch.isEnabled = false
        btnMatch.setOnClickListener { onAction(Action.START_MATCHING) }
        matchLayout.addView(btnMatch)

        container.addView(matchLayout)
        container.addView(createDivider())

        // ============================================
        // 第3部分：模型管理
        // ============================================
        container.addView(createSectionHeader("💾 模型管理"))

        val modelLayout = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(0, 8, 0, 8)
        }

        btnLoadModel = createButton("📂 导入已有模型")
        btnLoadModel.setOnClickListener { onAction(Action.SELECT_MODEL) }
        modelLayout.addView(btnLoadModel)

        btnClearModel = createButton("🗑 清除模型", danger = true)
        btnClearModel.isEnabled = false
        btnClearModel.setOnClickListener { onAction(Action.CLEAR_MODEL) }
        modelLayout.addView(btnClearModel)

        container.addView(modelLayout)
        container.addView(createDivider())

        // ============================================
        // 日志
        // ============================================
        container.addView(createSectionHeader("📋 运行日志"))

        logView = TextView(activity).apply {
            setPadding(16)
            setTextColor(Color.parseColor("#BBBBBB"))
            textSize = 12f
            text = "就绪。选择训练目录开始吧！"
            setLineSpacing(4f, 1.0f)
            setBackgroundColor(Color.parseColor("#16213E"))
            minHeight = 200
        }
        container.addView(logView)

        // 底部留白
        container.addView(TextView(activity).apply {
            text = ""
            minimumHeight = 48
        })
    }

    // ============================================
    // UI 更新方法
    // ============================================

    fun log(message: String) {
        activity.runOnUiThread {
            val current = logView.text.toString()
            logView.text = if (current == "就绪。选择训练目录开始吧！") {
                message
            } else {
                "$current\n$message"
            }
            // 自动滚动到底部
            root.post { root.fullScroll(View.FOCUS_DOWN) }
        }
    }

    fun setTrainingProgress(active: Boolean) {
        activity.runOnUiThread {
            progressContainer.visibility = if (active) View.VISIBLE else View.GONE
            progressBar.isIndeterminate = active
            progressText.text = if (active) "训练中..." else ""
            btnTrain.isEnabled = !active && trainingCount > 0
            btnSelectDir.isEnabled = !active
        }
    }

    fun setMatchingProgress(active: Boolean) {
        activity.runOnUiThread {
            btnMatch.text = if (active) "⏳ 追色中..." else "✨ 开始追色"
            btnMatch.isEnabled = !active && isModelLoaded && currentTargetUri != null
        }
    }

    fun updateTrainingCount(count: Int) {
        trainingCount = count
        trainingCountText.text = if (count > 0) "📸 找到 $count 张照片" else ""
    }

    fun setModelLoaded(loaded: Boolean) {
        isModelLoaded = loaded
        btnClearModel.isEnabled = loaded
        activity.runOnUiThread {
            updateUIState()
        }
    }

    fun setTargetImage(uri: Uri) {
        currentTargetUri = uri
        activity.runOnUiThread {
            targetPreview.setImageURI(uri)
            updateUIState()
        }
    }

    fun setResultImage(uri: Uri) {
        activity.runOnUiThread {
            resultPreview.setImageURI(uri)
        }
    }

    private fun updateUIState() {
        btnMatch.isEnabled = isModelLoaded && currentTargetUri != null
        btnTrain.isEnabled = trainingCount > 0
    }

    fun setCanTrain(can: Boolean) {
        btnTrain.isEnabled = can && trainingCount > 0
    }

    fun setCanMatch(can: Boolean) {
        btnMatch.isEnabled = can
    }

    // ============================================
    // 辅助UI创建方法
    // ============================================

    private fun createButton(text: String, accent: Boolean = false, danger: Boolean = false): Button {
        return Button(activity).apply {
            this.text = text
            setPadding(16, 12, 16, 12)
            textSize = 14f
            setTextColor(Color.WHITE)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 4, 0, 4) }

            background = ResourcesCompat.getDrawable(
                activity.resources,
                android.R.drawable.btn_default,
                null
            )

            setBackgroundColor(
                when {
                    danger -> Color.parseColor("#B83232")
                    accent -> Color.parseColor("#E94560")
                    else -> Color.parseColor("#0F3460")
                }
            )
        }
    }

    private fun createSectionHeader(text: String): TextView {
        return TextView(activity).apply {
            this.text = text
            textSize = 16f
            setTextColor(Color.parseColor("#E94560"))
            typeface = ResourcesCompat.getFont(activity, R.font.sans_serif)
            setPadding(0, 16, 0, 4)
        }
    }

    private fun createInfoCard(label: String, value: String): TextView {
        return TextView(activity).apply {
            text = "$label: $value"
            textSize = 15f
            setTextColor(Color.parseColor("#CCCCCC"))
            setPadding(16, 12, 16, 12)
            setBackgroundColor(Color.parseColor("#16213E"))
        }
    }

    private fun createSmallText(text: String): TextView {
        return TextView(activity).apply {
            this.text = text
            textSize = 12f
            setTextColor(Color.parseColor("#888888"))
            setPadding(8, 4, 8, 4)
        }
    }

    private fun createDivider(): View {
        return View(activity).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 1
            ).apply { setMargins(0, 16, 0, 8) }
            setBackgroundColor(Color.parseColor("#333355"))
        }
    }
}
