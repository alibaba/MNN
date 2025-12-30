package com.alibaba.mls.api.download

import org.junit.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class DownloadInfoTest {

    @Test
    fun testIsDownloading() {
        val info = DownloadInfo(downloadState = DownloadState.DOWNLOADING)
        assertTrue(info.isDownloading())
        
        val info2 = DownloadInfo(downloadState = DownloadState.NOT_START)
        assertFalse(info2.isDownloading())
    }

    @Test
    fun testIsComplete() {
        val info = DownloadInfo(downloadState = DownloadState.DOWNLOAD_SUCCESS)
        assertTrue(info.isComplete())
        
        val info2 = DownloadInfo(downloadState = DownloadState.DOWNLOADING)
        assertFalse(info2.isComplete())
    }

    @Test
    fun testCanDownload() {
        assertTrue(DownloadInfo(downloadState = DownloadState.NOT_START).canDownload())
        assertTrue(DownloadInfo(downloadState = DownloadState.DOWNLOAD_FAILED).canDownload())
        assertTrue(DownloadInfo(downloadState = DownloadState.DOWNLOAD_PAUSED).canDownload())
        
        assertFalse(DownloadInfo(downloadState = DownloadState.DOWNLOADING).canDownload())
        assertFalse(DownloadInfo(downloadState = DownloadState.DOWNLOAD_SUCCESS).canDownload())
    }
}
