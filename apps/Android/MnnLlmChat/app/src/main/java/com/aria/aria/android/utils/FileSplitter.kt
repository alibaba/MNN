// Created by ruoyi.sjd on 2025/1/14.
// Copyright (c) 2024 Alibaba Group Holding Limited. All rights reserved.

package com.alibaba.mnnllm.android.utils

import android.util.Log
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException

/**
 * Utility class for splitting large files into smaller chunks and merging them back
 */
object FileSplitter {
    private const val TAG = "FileSplitter"
    const val MAX_CHUNK_SIZE = 1024 * 1024 * 1024L // 1GB
    
    /**
     * Information about file splits
     */
    data class SplitInfo(
        val originalFileName: String,
        val originalFileSize: Long,
        val chunkSize: Long,
        val totalChunks: Int,
        val chunks: List<ChunkInfo>
    )
    
    data class ChunkInfo(
        val chunkIndex: Int,
        val chunkFileName: String,
        val chunkSize: Long,
        val checksum: String? = null
    )
    
    /**
     * Split a large file into smaller chunks
     * @param sourceFile The file to split
     * @param outputDir Directory to store the chunks
     * @param chunkSize Maximum size of each chunk (default: 1GB)
     * @return SplitInfo containing information about the splits
     */
    fun splitFile(
        sourceFile: File, 
        outputDir: File, 
        chunkSize: Long = MAX_CHUNK_SIZE
    ): SplitInfo? {
        if (!sourceFile.exists()) {
            Log.e(TAG, "Source file does not exist: ${sourceFile.absolutePath}")
            return null
        }
        
        if (!outputDir.exists()) {
            outputDir.mkdirs()
        }
        
        val fileSize = sourceFile.length()
        if (fileSize <= chunkSize) {
            Log.d(TAG, "File ${sourceFile.name} is already small enough (${fileSize} bytes), no splitting needed")
            return null
        }
        
        val totalChunks = ((fileSize + chunkSize - 1) / chunkSize).toInt()
        val chunks = mutableListOf<ChunkInfo>()
        
        Log.d(TAG, "Splitting file ${sourceFile.name} (${fileSize} bytes) into $totalChunks chunks")
        
        try {
            FileInputStream(sourceFile).use { inputStream ->
                val buffer = ByteArray(chunkSize.toInt())
                
                for (chunkIndex in 0 until totalChunks) {
                    val chunkFileName = "${sourceFile.name}.part${chunkIndex + 1}"
                    val chunkFile = File(outputDir, chunkFileName)
                    
                    val bytesRead = inputStream.read(buffer)
                    if (bytesRead == -1) break
                    
                    FileOutputStream(chunkFile).use { outputStream ->
                        outputStream.write(buffer, 0, bytesRead)
                    }
                    
                    val actualChunkSize = bytesRead.toLong()
                    val chunkInfo = ChunkInfo(
                        chunkIndex = chunkIndex + 1,
                        chunkFileName = chunkFileName,
                        chunkSize = actualChunkSize
                    )
                    chunks.add(chunkInfo)
                    
                    Log.d(TAG, "Created chunk $chunkFileName (${actualChunkSize} bytes)")
                }
            }
            
            val splitInfo = SplitInfo(
                originalFileName = sourceFile.name,
                originalFileSize = fileSize,
                chunkSize = chunkSize,
                totalChunks = totalChunks,
                chunks = chunks
            )
            
            // Save split info to JSON file
            val splitInfoFile = File(outputDir, "splits_info.json")
            val gson = Gson()
            val json = gson.toJson(splitInfo)
            splitInfoFile.writeText(json)
            
            Log.d(TAG, "Split info saved to ${splitInfoFile.absolutePath}")
            return splitInfo
            
        } catch (e: IOException) {
            Log.e(TAG, "Failed to split file ${sourceFile.name}", e)
            return null
        }
    }
    
    /**
     * Merge split files back into the original file
     * @param splitInfo Information about the splits
     * @param chunksDir Directory containing the chunk files
     * @param outputFile The output file to merge into
     * @return true if merge was successful
     */
    fun mergeFiles(splitInfo: SplitInfo, chunksDir: File, outputFile: File): Boolean {
        if (!chunksDir.exists()) {
            Log.e(TAG, "Chunks directory does not exist: ${chunksDir.absolutePath}")
            return false
        }
        
        Log.d(TAG, "Merging ${splitInfo.totalChunks} chunks into ${outputFile.name}")
        Log.d(TAG, "Expected final size: ${splitInfo.originalFileSize} bytes")
        Log.d(TAG, "Output file path: ${outputFile.absolutePath}")
        
        // Pre-validate all chunk files before starting merge
        val sortedChunks = splitInfo.chunks.sortedBy { it.chunkIndex }
        Log.d(TAG, "Sorted chunks: ${sortedChunks.map { "${it.chunkFileName}(${it.chunkSize})" }.joinToString(", ")}")
        var totalExpectedSize = 0L
        
        for (chunkInfo in sortedChunks) {
            val chunkFile = File(chunksDir, chunkInfo.chunkFileName)
            if (!chunkFile.exists()) {
                Log.e(TAG, "Chunk file does not exist: ${chunkFile.absolutePath}")
                return false
            }
            
            val actualSize = chunkFile.length()
            if (actualSize != chunkInfo.chunkSize) {
                Log.e(TAG, "Chunk file ${chunkInfo.chunkFileName} size mismatch: expected ${chunkInfo.chunkSize}, actual $actualSize")
                return false
            }
            
            totalExpectedSize += chunkInfo.chunkSize
            Log.d(TAG, "Validated chunk ${chunkInfo.chunkFileName}: ${chunkInfo.chunkSize} bytes")
        }
        
        if (totalExpectedSize != splitInfo.originalFileSize) {
            Log.e(TAG, "Total chunk sizes ($totalExpectedSize) do not match expected original size (${splitInfo.originalFileSize})")
            return false
        }
        
        try {
            // Ensure output directory exists
            outputFile.parentFile?.mkdirs()
            
            // Delete output file if it exists to ensure clean merge
            if (outputFile.exists()) {
                outputFile.delete()
            }
            
            var totalBytesWritten = 0L
            
            FileOutputStream(outputFile).use { outputStream ->
                for (chunkInfo in sortedChunks) {
                    val chunkFile = File(chunksDir, chunkInfo.chunkFileName)
                    Log.d(TAG, "Merging chunk ${chunkInfo.chunkFileName} (${chunkInfo.chunkSize} bytes)")
                    
                    FileInputStream(chunkFile).use { inputStream ->
                        val buffer = ByteArray(8192)
                        var bytesRead: Int
                        var chunkBytesWritten = 0L
                        
                        while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                            outputStream.write(buffer, 0, bytesRead)
                            chunkBytesWritten += bytesRead
                            totalBytesWritten += bytesRead
                        }
                        
                        if (chunkBytesWritten != chunkInfo.chunkSize) {
                            Log.e(TAG, "Chunk ${chunkInfo.chunkFileName} bytes written ($chunkBytesWritten) != expected (${chunkInfo.chunkSize})")
                            return false
                        }
                    }
                    
                    Log.d(TAG, "Successfully merged chunk ${chunkInfo.chunkFileName}")
                }
            }
            
            // Verify final file size
            val mergedFileSize = outputFile.length()
            if (mergedFileSize != splitInfo.originalFileSize) {
                Log.e(TAG, "CRITICAL: Merged file size ($mergedFileSize) does not match original size (${splitInfo.originalFileSize})")
                Log.e(TAG, "Total bytes written: $totalBytesWritten")
                
                // Delete the corrupted file
                outputFile.delete()
                return false
            }
            
            Log.d(TAG, "Successfully merged file ${outputFile.name} (${mergedFileSize} bytes)")
            return true
            
        } catch (e: IOException) {
            Log.e(TAG, "Failed to merge files", e)
            // Clean up partial file
            if (outputFile.exists()) {
                outputFile.delete()
            }
            return false
        }
    }
    
    /**
     * Load split information from JSON file
     * @param splitInfoFile The JSON file containing split information
     * @return SplitInfo or null if loading failed
     */
    fun loadSplitInfo(splitInfoFile: File): SplitInfo? {
        if (!splitInfoFile.exists()) {
            Log.d(TAG, "Split info file does not exist: ${splitInfoFile.absolutePath}")
            return null
        }
        
        try {
            val json = splitInfoFile.readText()
            val gson = Gson()
            return gson.fromJson(json, SplitInfo::class.java)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load split info from ${splitInfoFile.absolutePath}", e)
            return null
        }
    }
    
    /**
     * Check if a file needs to be merged (has split info)
     * @param modelDir Directory containing the model files
     * @return true if the directory contains split files that need merging
     */
    fun needsMerging(modelDir: File): Boolean {
        val splitInfoFile = File(modelDir, "splits_info.json")
        return splitInfoFile.exists()
    }
    
    /**
     * Merge all split files in a model directory
     * @param modelDir Directory containing the model files
     * @return true if all files were successfully merged
     */
    fun mergeAllSplitFiles(modelDir: File): Boolean {
        val splitInfoFile = File(modelDir, "splits_info.json")
        val splitInfo = loadSplitInfo(splitInfoFile) ?: return true // No split info, nothing to merge
        
        Log.d(TAG, "Merging split files in ${modelDir.absolutePath}")
        Log.d(TAG, "Split info: ${splitInfo.totalChunks} chunks, original size: ${splitInfo.originalFileSize} bytes")
        
        // Find all files that need merging
        val filesToMerge = mutableListOf<String>()
        for (chunkInfo in splitInfo.chunks) {
            // Extract original filename by removing .partX suffix
            val originalFileName = chunkInfo.chunkFileName.replace(Regex("\\.part\\d+$"), "")
            Log.d(TAG, "Chunk: ${chunkInfo.chunkFileName} -> Original: $originalFileName")
            if (!filesToMerge.contains(originalFileName)) {
                filesToMerge.add(originalFileName)
            }
        }
        
        Log.d(TAG, "Files to merge: ${filesToMerge.joinToString(", ")}")
        
        var allMerged = true
        val mergedFiles = mutableListOf<String>()
        val failedFiles = mutableListOf<String>()
        
        for (originalFileName in filesToMerge) {
            val outputFile = File(modelDir, originalFileName)
            if (outputFile.exists()) {
                Log.d(TAG, "File $originalFileName already exists, verifying size...")
                val existingSize = outputFile.length()
                if (existingSize == splitInfo.originalFileSize) {
                    Log.d(TAG, "Existing file $originalFileName has correct size ($existingSize bytes), skipping merge")
                    mergedFiles.add(originalFileName)
                    continue
                } else {
                    Log.w(TAG, "Existing file $originalFileName has wrong size ($existingSize bytes), expected ${splitInfo.originalFileSize}, will re-merge")
                    outputFile.delete()
                }
            }
            
            Log.d(TAG, "Starting merge for file: $originalFileName")
            val merged = mergeFiles(splitInfo, modelDir, outputFile)
            
            if (!merged) {
                Log.e(TAG, "CRITICAL: Failed to merge file $originalFileName")
                failedFiles.add(originalFileName)
                allMerged = false
            } else {
                // Verify the merged file size
                val mergedSize = outputFile.length()
                if (mergedSize != splitInfo.originalFileSize) {
                    Log.e(TAG, "CRITICAL: Merged file $originalFileName has wrong size ($mergedSize bytes), expected ${splitInfo.originalFileSize}")
                    outputFile.delete()
                    failedFiles.add(originalFileName)
                    allMerged = false
                } else {
                    Log.d(TAG, "Successfully merged file $originalFileName (${mergedSize} bytes)")
                    mergedFiles.add(originalFileName)
                    
                    // Successfully merged, now delete the part files
                    deletePartFiles(modelDir, originalFileName, splitInfo)
                }
            }
        }
        
        // If all files were successfully merged, delete the split info file
        if (allMerged && mergedFiles.isNotEmpty()) {
            try {
                splitInfoFile.delete()
                Log.d(TAG, "Deleted split info file: ${splitInfoFile.absolutePath}")
            } catch (e: Exception) {
                Log.w(TAG, "Failed to delete split info file: ${splitInfoFile.absolutePath}", e)
            }
        }
        
        if (failedFiles.isNotEmpty()) {
            Log.e(TAG, "Failed to merge files: ${failedFiles.joinToString(", ")}")
            Log.e(TAG, "Successfully merged files: ${mergedFiles.joinToString(", ")}")
        }
        
        return allMerged
    }
    
    /**
     * Delete part files after successful merge
     * @param modelDir Directory containing the model files
     * @param originalFileName The original file name (without .part extension)
     * @param splitInfo Information about the splits
     */
    private fun deletePartFiles(modelDir: File, originalFileName: String, splitInfo: SplitInfo) {
        try {
            // Double-check that the merged file exists and has correct size before deleting parts
            val mergedFile = File(modelDir, originalFileName)
            if (!mergedFile.exists()) {
                Log.e(TAG, "CRITICAL: Merged file $originalFileName does not exist, cannot delete part files")
                return
            }
            
            val mergedSize = mergedFile.length()
            if (mergedSize != splitInfo.originalFileSize) {
                Log.e(TAG, "CRITICAL: Merged file $originalFileName has wrong size ($mergedSize bytes), expected ${splitInfo.originalFileSize}, cannot delete part files")
                return
            }
            
            Log.d(TAG, "Verified merged file $originalFileName has correct size ($mergedSize bytes), proceeding to delete part files")
            
            // Find all part files for this original file
            val partFiles = splitInfo.chunks.filter { 
                it.chunkFileName.startsWith(originalFileName) && it.chunkFileName.matches(Regex(".*\\.part\\d+$"))
            }
            
            var allPartsDeleted = true
            for (chunkInfo in partFiles) {
                val partFile = File(modelDir, chunkInfo.chunkFileName)
                if (partFile.exists()) {
                    val deleted = partFile.delete()
                    if (deleted) {
                        Log.d(TAG, "Deleted part file: ${chunkInfo.chunkFileName}")
                    } else {
                        Log.w(TAG, "Failed to delete part file: ${chunkInfo.chunkFileName}")
                        allPartsDeleted = false
                    }
                } else {
                    Log.d(TAG, "Part file already deleted: ${chunkInfo.chunkFileName}")
                }
            }
            
            if (allPartsDeleted) {
                Log.d(TAG, "All part files for $originalFileName have been successfully deleted")
            } else {
                Log.w(TAG, "Some part files for $originalFileName could not be deleted")
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error deleting part files for $originalFileName", e)
        }
    }
}
