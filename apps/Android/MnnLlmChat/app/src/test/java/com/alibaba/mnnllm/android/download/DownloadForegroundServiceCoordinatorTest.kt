package com.alibaba.mnnllm.android.download

import org.junit.Assert.assertEquals
import org.junit.Test

class DownloadForegroundServiceCoordinatorTest {

    private val gateway = FakeGateway()
    private val coordinator = DownloadForegroundServiceCoordinator(gateway)

    @Test
    fun downloadingStateStartsForegroundServiceOnce() {
        coordinator.onDownloadStateChanged(
            modelId = "model-a",
            modelName = "Model A",
            isDownloading = true
        )

        assertEquals(listOf(FakeGateway.Event.Show(1, "Model A")), gateway.events)
    }

    @Test
    fun repeatedDownloadingStateDoesNotRestartForegroundService() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", true)

        assertEquals(listOf(FakeGateway.Event.Show(1, "Model A")), gateway.events)
    }

    @Test
    fun additionalDownloadUpdatesForegroundNotificationCount() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-b", "Model B", true)

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"),
                FakeGateway.Event.Show(2, "Model A")
            ),
            gateway.events
        )
    }

    @Test
    fun endingLastDownloadStopsForegroundService() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"),
                FakeGateway.Event.Stop
            ),
            gateway.events
        )
    }

    @Test
    fun endingPrimaryDownloadPromotesNextActiveModelName() {
        coordinator.onDownloadStateChanged("model-a", "Model A", true)
        coordinator.onDownloadStateChanged("model-b", "Model B", true)
        coordinator.onDownloadStateChanged("model-a", "Model A", false)

        assertEquals(
            listOf(
                FakeGateway.Event.Show(1, "Model A"),
                FakeGateway.Event.Show(2, "Model A"),
                FakeGateway.Event.Show(1, "Model B")
            ),
            gateway.events
        )
    }

    private class FakeGateway : DownloadForegroundServiceCoordinator.Gateway {
        val events = mutableListOf<Event>()

        override fun show(downloadCount: Int, modelName: String?): Boolean {
            events += Event.Show(downloadCount, modelName)
            return true
        }

        override fun stop() {
            events += Event.Stop
        }

        sealed interface Event {
            data class Show(val downloadCount: Int, val modelName: String?) : Event
            data object Stop : Event
        }
    }
}
