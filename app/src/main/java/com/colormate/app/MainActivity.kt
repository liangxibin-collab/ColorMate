package com.colormate.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.colormate.app.color.ColorMatcher
import com.colormate.app.color.ColorStyleModel
import com.colormate.app.color.ColorStyleTrainer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.io.*
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {

    // UI状态
    private lateinit var ui: MainUI
    private var trainedModelPath: String? = null
    private var trainingDirPath: String? = null
    private var currentStrength: Float = 1.0f

    private val PERMISSION_REQUEST_CODE = 1001

    // 文件选择器
    private val selectTrainingDirLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocumentTree()
    ) { uri: Uri? ->
        uri?.let { handleTrainingDirSelected(it) }
    }

    private val selectTargetImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { handleTargetImageSelected(it) }
    }

    private val selectModelFileLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { loadModelFromUri(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 初始化OpenCV
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV初始化失败", Toast.LENGTH_LONG).show()
        }

        ui = MainUI(this) { action ->
            handleUIAction(action)
        }
        setContentView(ui.root)

        checkPermissions()
        updateUI()
    }

    private fun checkPermissions() {
        val neededPermissions = mutableListOf<String>()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED
        ) {
            neededPermissions.add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }

        if (android.os.Build.VERSION.SDK_INT >= 33) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES)
                != PackageManager.PERMISSION_GRANTED
            ) {
                neededPermissions.add(Manifest.permission.READ_MEDIA_IMAGES)
            }
        }

        if (neededPermissions.isNotEmpty()) {
            requestPermissions(
                neededPermissions.toTypedArray(),
                PERMISSION_REQUEST_CODE
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            val denied = grantResults.filter { it != PackageManager.PERMISSION_GRANTED }
            if (denied.isNotEmpty()) {
                Toast.makeText(this, "需要存储权限才能读取图片", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun handleUIAction(action: MainUI.Action) {
        when (action) {
            MainUI.Action.SELECT_TRAINING_DIR -> selectTrainingDir()
            MainUI.Action.START_TRAINING -> startTraining()
            MainUI.Action.SELECT_TARGET -> selectTargetImage()
            MainUI.Action.START_MATCHING -> startMatching()
            MainUI.Action.SELECT_MODEL -> selectModelFile()
            MainUI.Action.SET_STRENGTH -> { }
            MainUI.Action.CLEAR_MODEL -> clearModel()
            is MainUI.Action.UPDATE_STRENGTH -> currentStrength = action.value
        }
    }

    // ==============================
    // 训练流程
    // ==============================

    private fun selectTrainingDir() {
        selectTrainingDirLauncher.launch(null)
    }

    private fun handleTrainingDirSelected(uri: Uri) {
        contentResolver.takePersistableUriPermission(
            uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
        )
        trainingDirPath = uri.toString()
        ui.modelStatus.text = "已选择训练目录"
        ui.updateTrainingCount(0)
        updateUI()

        lifecycleScope.launch(Dispatchers.IO) {
            val count = countImagesInDir(uri)
            withContext(Dispatchers.Main) {
                ui.updateTrainingCount(count)
                ui.modelStatus.text = if (count > 0) "训练目录：$count 张图片"
                else "目录中没有找到JPG图片"
            }
        }
    }

    private fun countImagesInDir(uri: Uri): Int {
        val docId = DocumentsContract.getTreeDocumentId(uri)
        val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(uri, docId)
        var count = 0
        try {
            contentResolver.query(childrenUri, null, null, null, null)?.use { cursor ->
                while (cursor.moveToNext()) {
                    val name = cursor.getString(
                        cursor.getColumnIndexOrThrow(OpenableColumns.DISPLAY_NAME)
                    ) ?: continue
                    val ext = name.substringAfterLast('.', "").lowercase()
                    if (ext in listOf("jpg", "jpeg")) count++
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return count
    }

    private fun startTraining() {
        val dirPath = trainingDirPath ?: run {
            Toast.makeText(this, "请先选择训练目录", Toast.LENGTH_SHORT).show()
            return
        }

        ui.setTrainingProgress(true)
        ui.log("开始训练...")

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val tempTrainDir = File(cacheDir, "training_images")
                tempTrainDir.mkdirs()
                copyTrainingImages(dirPath, tempTrainDir)

                val imagePaths = tempTrainDir.listFiles()
                    ?.filter { it.extension.lowercase() in listOf("jpg", "jpeg") }
                    ?.map { it.absolutePath }
                    ?.sorted()

                if (imagePaths.isNullOrEmpty()) {
                    withContext(Dispatchers.Main) {
                        ui.log("❌ 训练目录中没有找到JPG图片")
                        ui.setTrainingProgress(false)
                    }
                    return@launch
                }

                ui.log("  加载了 ${imagePaths.size} 张参考照片...")

                // 使用Kotlin版训练器
                val trainer = ColorStyleTrainer(sampleSize = 500)
                val model = trainer.train(imagePaths)

                // 保存模型
                val modelFile = File(cacheDir, "color_style_model.pkl")
                // 使用简化的对象序列化
                saveModel(model, modelFile.absolutePath)

                withContext(Dispatchers.Main) {
                    trainedModelPath = modelFile.absolutePath
                    ui.log("✅ 训练成功！使用了 ${imagePaths.size} 张参考照片")
                    ui.modelStatus.text = "已训练：${imagePaths.size} 张照片"
                    ui.setModelLoaded(true)
                    tempTrainDir.deleteRecursively()
                }
                ui.setTrainingProgress(false)
                updateUI()
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    ui.log("❌ 训练异常: ${e.message}")
                    ui.setTrainingProgress(false)
                }
            }
        }
    }

    // 简化的模型序列化/反序列化
    private fun saveModel(model: ColorStyleModel, path: String) {
        ObjectOutputStream(FileOutputStream(path)).use { oos ->
            oos.writeObject(model)
        }
    }

    private fun loadModel(path: String): ColorStyleModel? {
        return try {
            ObjectInputStream(FileInputStream(path)).use { ois ->
                ois.readObject() as ColorStyleModel
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun copyTrainingImages(uri: String, targetDir: File) {
        val treeUri = Uri.parse(uri)
        val docId = DocumentsContract.getTreeDocumentId(treeUri)
        val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, docId)

        contentResolver.query(childrenUri, null, null, null, null)?.use { cursor ->
            while (cursor.moveToNext()) {
                val name = cursor.getString(
                    cursor.getColumnIndexOrThrow(OpenableColumns.DISPLAY_NAME)
                ) ?: continue
                val ext = name.substringAfterLast('.', "").lowercase()
                if (ext !in listOf("jpg", "jpeg")) continue

                val docIdChild = cursor.getString(
                    cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DOCUMENT_ID)
                )
                val fileUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, docIdChild)

                try {
                    contentResolver.openInputStream(fileUri)?.use { input ->
                        File(targetDir, name).outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                } catch (e: Exception) {
                    // skip
                }
            }
        }
    }

    // ==============================
    // 追色流程
    // ==============================

    private fun selectTargetImage() {
        selectTargetImageLauncher.launch("image/*")
    }

    private fun handleTargetImageSelected(uri: Uri) {
        ui.setTargetImage(uri)
        updateUI()
    }

    private fun startMatching() {
        val modelPath = trainedModelPath ?: run {
            Toast.makeText(this, "请先训练或加载模型", Toast.LENGTH_SHORT).show()
            return
        }

        val targetUri = ui.currentTargetUri ?: run {
            Toast.makeText(this, "请先选择目标图片", Toast.LENGTH_SHORT).show()
            return
        }

        ui.setMatchingProgress(true)
        ui.log("开始追色...强度: ${String.format("%.1f", currentStrength)}")

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // 复制目标图片到临时文件
                val tempInput = File(cacheDir, "target_input.jpg")
                contentResolver.openInputStream(targetUri)?.use { input ->
                    tempInput.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }

                // 准备输出路径
                val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
                    .format(Date())
                val outputFile = File(
                    Environment.getExternalStoragePublicDirectory(
                        Environment.DIRECTORY_PICTURES
                    ),
                    "ColorMate/color_matched_$timeStamp.jpg"
                )
                outputFile.parentFile?.mkdirs()

                // 加载模型
                val model = loadModel(modelPath)
                    ?: throw Exception("模型文件损坏或格式不正确")

                // 使用Kotlin版追色器
                val matcher = ColorMatcher(strength = currentStrength, preserveNatural = true)
                matcher.match(model, tempInput.absolutePath, outputFile.absolutePath)

                withContext(Dispatchers.Main) {
                    ui.log("✅ 追色完成！已保存到: ${outputFile.absolutePath}")
                    ui.setResultImage(Uri.fromFile(outputFile))

                    // 刷新图库
                    val mediaScanIntent = Intent(
                        Intent.ACTION_MEDIA_SCANNER_SCAN_FILE
                    )
                    mediaScanIntent.data = Uri.fromFile(outputFile)
                    sendBroadcast(mediaScanIntent)

                    Toast.makeText(this@MainActivity, "追色完成！", Toast.LENGTH_SHORT).show()
                    ui.setMatchingProgress(false)
                    tempInput.delete()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    ui.log("❌ 追色异常: ${e.message}")
                    ui.setMatchingProgress(false)
                }
            }
        }
    }

    // ==============================
    // 模型管理
    // ==============================

    private fun selectModelFile() {
        selectModelFileLauncher.launch("application/octet-stream")
    }

    private fun loadModelFromUri(uri: Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val modelFile = File(cacheDir, "loaded_model.dat")
                contentResolver.openInputStream(uri)?.use { input ->
                    modelFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }

                val model = loadModel(modelFile.absolutePath)

                withContext(Dispatchers.Main) {
                    if (model != null) {
                        trainedModelPath = modelFile.absolutePath
                        ui.log("✅ 模型加载成功！基于 ${model.numTrainingImages} 张照片训练")
                        ui.modelStatus.text = "已加载模型：${model.numTrainingImages} 张照片"
                        ui.setModelLoaded(true)
                    } else {
                        ui.log("❌ 模型加载失败: 文件格式不兼容")
                    }
                    updateUI()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    ui.log("❌ 模型导入异常: ${e.message}")
                }
            }
        }
    }

    private fun clearModel() {
        trainedModelPath = null
        trainedModelPath?.let { File(it).delete() }
        ui.setModelLoaded(false)
        ui.modelStatus.text = "未加载模型"
        updateUI()
    }

    private fun updateUI() {
        ui.setCanTrain(trainingDirPath != null)
        ui.setCanMatch(trainedModelPath != null && ui.currentTargetUri != null)
    }
}
