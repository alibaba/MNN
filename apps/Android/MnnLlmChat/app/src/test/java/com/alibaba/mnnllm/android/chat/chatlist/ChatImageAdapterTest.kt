package com.alibaba.mnnllm.android.chat.chatlist

import android.net.Uri
import android.widget.ImageView
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.widgets.FullScreenImageViewer
import io.mockk.*
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

/**
 * Unit tests for ChatImageAdapter
 * Tests chat message image display and full-screen viewer integration
 */
@RunWith(RobolectricTestRunner::class)
class ChatImageAdapterTest {

    private val testUri1 = Uri.parse("content://test/image1.jpg")
    private val testUri2 = Uri.parse("content://test/image2.jpg")
    private val testUri3 = Uri.parse("content://test/image3.jpg")

    @Before
    fun setup() {
        // Mock the static method showImagePopup
        mockkObject(FullScreenImageViewer)
        every { FullScreenImageViewer.showImagePopup(any(), any<List<Uri>>(), any()) } just Runs
        every { FullScreenImageViewer.showImagePopup(any(), any<Uri>()) } just Runs
    }

    @Test
    fun `test adapter creates with empty list`() {
        // Given
        val emptyList = emptyList<Uri>()

        // When
        val adapter = ChatImageAdapter(emptyList)

        // Then
        assertEquals(0, adapter.itemCount)
    }

    @Test
    fun `test adapter creates with single image`() {
        // Given
        val images = listOf(testUri1)

        // When
        val adapter = ChatImageAdapter(images)

        // Then
        assertEquals(1, adapter.itemCount)
    }

    @Test
    fun `test adapter creates with multiple images`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)

        // When
        val adapter = ChatImageAdapter(images)

        // Then
        assertEquals(3, adapter.itemCount)
    }

    @Test
    fun `test getItemCount returns correct count`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3, testUri1, testUri2)
        val adapter = ChatImageAdapter(images)

        // When
        val count = adapter.itemCount

        // Then
        assertEquals(5, count)
    }

    @Test
    fun `test onCreateViewHolder creates ViewHolder with correct layout`() {
        // Given
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        val adapter = ChatImageAdapter(listOf(testUri1))

        // When
        val viewHolder = adapter.onCreateViewHolder(parent, 0)

        // Then
        assertNotNull(viewHolder)
        assertNotNull(viewHolder.itemView)
        // Verify the layout contains the expected ImageView
        val imageView = viewHolder.itemView.findViewById<ImageView>(R.id.chat_image_item)
        assertNotNull("Layout should contain chat_image_item ImageView", imageView)
    }

    @Test
    fun `test onBindViewHolder binds image at position 0`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        val adapter = ChatImageAdapter(images)
        val viewHolder = adapter.onCreateViewHolder(parent, 0)

        // When
        adapter.onBindViewHolder(viewHolder, 0)

        // Then
        val imageView = viewHolder.itemView.findViewById<ImageView>(R.id.chat_image_item)
        assertNotNull(imageView)
        // Note: Robolectric doesn't fully support ImageView.setImageURI verification
        // but we verify no crash occurs during binding
    }

    @Test
    fun `test onBindViewHolder binds image at different positions`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        val adapter = ChatImageAdapter(images)

        // When & Then - test each position
        for (position in images.indices) {
            val viewHolder = adapter.onCreateViewHolder(parent, 0)
            // Should not throw exception
            adapter.onBindViewHolder(viewHolder, position)
            
            val imageView = viewHolder.itemView.findViewById<ImageView>(R.id.chat_image_item)
            assertNotNull("ImageView should exist at position $position", imageView)
        }
    }

    @Test
    fun `test clicking image at position 0 opens full screen viewer`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        val adapter = ChatImageAdapter(images)
        val viewHolder = adapter.onCreateViewHolder(parent, 0)
        adapter.onBindViewHolder(viewHolder, 0)

        // When
        val imageView = viewHolder.itemView.findViewById<ImageView>(R.id.chat_image_item)
        imageView.performClick()

        // Then
        verify {
            FullScreenImageViewer.showImagePopup(
                context,
                images,
                0
            )
        }
    }

    @Test
    fun `test clicking image at position 1 opens full screen viewer with correct index`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        val adapter = ChatImageAdapter(images)
        val viewHolder = adapter.onCreateViewHolder(parent, 0)
        adapter.onBindViewHolder(viewHolder, 1)

        // When
        val imageView = viewHolder.itemView.findViewById<ImageView>(R.id.chat_image_item)
        imageView.performClick()

        // Then
        verify {
            FullScreenImageViewer.showImagePopup(
                context,
                images,
                1
            )
        }
    }

    @Test
    fun `test clicking image at last position opens full screen viewer with correct index`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        val adapter = ChatImageAdapter(images)
        val viewHolder = adapter.onCreateViewHolder(parent, 0)
        adapter.onBindViewHolder(viewHolder, 2)

        // When
        val imageView = viewHolder.itemView.findViewById<ImageView>(R.id.chat_image_item)
        imageView.performClick()

        // Then
        verify {
            FullScreenImageViewer.showImagePopup(
                context,
                images,
                2
            )
        }
    }

    @Test
    fun `test full screen viewer receives all images in list`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        val adapter = ChatImageAdapter(images)
        val viewHolder = adapter.onCreateViewHolder(parent, 0)
        adapter.onBindViewHolder(viewHolder, 1)

        // When
        val imageView = viewHolder.itemView.findViewById<ImageView>(R.id.chat_image_item)
        imageView.performClick()

        // Then - verify all images are passed, not just the clicked one
        verify {
            FullScreenImageViewer.showImagePopup(
                context,
                match { it.size == 3 && it.containsAll(images) },
                1
            )
        }
    }

    @Test
    fun `test multiple image clicks open viewer with different positions`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        val adapter = ChatImageAdapter(images)

        // When - click images at different positions
        val viewHolder0 = adapter.onCreateViewHolder(parent, 0)
        adapter.onBindViewHolder(viewHolder0, 0)
        viewHolder0.itemView.findViewById<ImageView>(R.id.chat_image_item).performClick()

        val viewHolder2 = adapter.onCreateViewHolder(parent, 0)
        adapter.onBindViewHolder(viewHolder2, 2)
        viewHolder2.itemView.findViewById<ImageView>(R.id.chat_image_item).performClick()

        // Then
        verify {
            FullScreenImageViewer.showImagePopup(context, images, 0)
            FullScreenImageViewer.showImagePopup(context, images, 2)
        }
    }

    @Test
    fun `test adapter with single image still passes list to viewer`() {
        // Given
        val images = listOf(testUri1)
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        val adapter = ChatImageAdapter(images)
        val viewHolder = adapter.onCreateViewHolder(parent, 0)
        adapter.onBindViewHolder(viewHolder, 0)

        // When
        val imageView = viewHolder.itemView.findViewById<ImageView>(R.id.chat_image_item)
        imageView.performClick()

        // Then - should pass list with single item, not just the URI
        verify {
            FullScreenImageViewer.showImagePopup(
                context,
                match { it.size == 1 && it[0] == testUri1 },
                0
            )
        }
    }

    @Test
    fun `test adapter with large image list`() {
        // Given
        val images = (1..20).map { Uri.parse("content://test/image$it.jpg") }
        val adapter = ChatImageAdapter(images)

        // When
        val count = adapter.itemCount

        // Then
        assertEquals(20, count)
    }

    @Test
    fun `test ViewHolder reuse with different positions updates correctly`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        val adapter = ChatImageAdapter(images)
        val viewHolder = adapter.onCreateViewHolder(parent, 0)

        // When - simulate ViewHolder recycling
        adapter.onBindViewHolder(viewHolder, 0)
        val imageView = viewHolder.itemView.findViewById<ImageView>(R.id.chat_image_item)
        imageView.performClick()

        // Rebind to different position
        adapter.onBindViewHolder(viewHolder, 2)
        imageView.performClick()

        // Then - should have clicked with correct positions
        verify {
            FullScreenImageViewer.showImagePopup(context, images, 0)
            FullScreenImageViewer.showImagePopup(context, images, 2)
        }
    }

    @Test
    fun `test viewer is not invoked until image is clicked`() {
        // Given
        val images = listOf(testUri1, testUri2)
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        val adapter = ChatImageAdapter(images)
        val viewHolder = adapter.onCreateViewHolder(parent, 0)

        // When - only bind, don't click
        adapter.onBindViewHolder(viewHolder, 0)

        // Then - viewer should not be invoked
        verify(exactly = 0) {
            FullScreenImageViewer.showImagePopup(any(), any<List<Uri>>(), any())
        }
    }
}
