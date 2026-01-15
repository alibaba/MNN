package com.alibaba.mnnllm.android.widgets

import android.net.Uri
import android.widget.ImageView
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mockito
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

/**
 * Unit tests for FullScreenImageAdapter
 * Tests ViewPager2 adapter functionality for full-screen image viewing
 */
@RunWith(RobolectricTestRunner::class)
class FullScreenImageAdapterTest {

    private lateinit var mockOnClick: () -> Unit
    private val testUri1 = Uri.parse("content://test/image1.jpg")
    private val testUri2 = Uri.parse("content://test/image2.jpg")
    private val testUri3 = Uri.parse("content://test/image3.jpg")

    @Before
    fun setup() {
        @Suppress("UNCHECKED_CAST")
        mockOnClick = Mockito.mock(Function0::class.java) as () -> Unit
    }

    @Test
    fun `test adapter creates with empty list`() {
        // Given
        val emptyList = emptyList<Uri>()

        // When
        val adapter = FullScreenImageAdapter(emptyList, mockOnClick)

        // Then
        assertEquals(0, adapter.itemCount)
    }

    @Test
    fun `test adapter creates with single image`() {
        // Given
        val images = listOf(testUri1)

        // When
        val adapter = FullScreenImageAdapter(images, mockOnClick)

        // Then
        assertEquals(1, adapter.itemCount)
    }

    @Test
    fun `test adapter creates with multiple images`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)

        // When
        val adapter = FullScreenImageAdapter(images, mockOnClick)

        // Then
        assertEquals(3, adapter.itemCount)
    }

    @Test
    fun `test getItemCount returns correct count`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3, testUri1, testUri2)
        val adapter = FullScreenImageAdapter(images, mockOnClick)

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
        val adapter = FullScreenImageAdapter(listOf(testUri1), mockOnClick)

        // When
        val viewHolder = adapter.onCreateViewHolder(parent, 0)

        // Then
        assertNotNull(viewHolder)
        assertNotNull(viewHolder.itemView)
        // Verify the layout contains the expected ImageView
        val imageView = viewHolder.itemView.findViewById<ImageView>(R.id.preview_image)
        assertNotNull("Layout should contain preview_image ImageView", imageView)
    }

    @Test
    fun `test onBindViewHolder binds image at position 0`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        val adapter = FullScreenImageAdapter(images, mockOnClick)
        val viewHolder = adapter.onCreateViewHolder(parent, 0)

        // When
        adapter.onBindViewHolder(viewHolder, 0)

        // Then
        val imageView = viewHolder.itemView.findViewById<ImageView>(R.id.preview_image)
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
        val adapter = FullScreenImageAdapter(images, mockOnClick)

        // When & Then - test each position
        for (position in images.indices) {
            val viewHolder = adapter.onCreateViewHolder(parent, 0)
            // Should not throw exception
            adapter.onBindViewHolder(viewHolder, position)
            
            val imageView = viewHolder.itemView.findViewById<ImageView>(R.id.preview_image)
            assertNotNull("ImageView should exist at position $position", imageView)
        }
    }

    @Test
    fun `test ViewHolder image click triggers callback`() {
        // Given
        val images = listOf(testUri1)
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        
        var clickCount = 0
        val countingCallback: () -> Unit = { clickCount++ }
        val adapter = FullScreenImageAdapter(images, countingCallback)
        
        val viewHolder = adapter.onCreateViewHolder(parent, 0)
        adapter.onBindViewHolder(viewHolder, 0)

        // When
        val imageView = viewHolder.itemView.findViewById<ImageView>(R.id.preview_image)
        imageView.performClick()

        // Then
        assertEquals("Click callback should be invoked once", 1, clickCount)
    }

    @Test
    fun `test multiple ViewHolders each trigger callback independently`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        
        var clickCount = 0
        val countingCallback: () -> Unit = { clickCount++ }
        val adapter = FullScreenImageAdapter(images, countingCallback)

        // When - create and click multiple ViewHolders
        val viewHolder1 = adapter.onCreateViewHolder(parent, 0)
        adapter.onBindViewHolder(viewHolder1, 0)
        
        val viewHolder2 = adapter.onCreateViewHolder(parent, 0)
        adapter.onBindViewHolder(viewHolder2, 1)

        val imageView1 = viewHolder1.itemView.findViewById<ImageView>(R.id.preview_image)
        val imageView2 = viewHolder2.itemView.findViewById<ImageView>(R.id.preview_image)

        imageView1.performClick()
        imageView2.performClick()

        // Then - callback should be invoked twice
        assertEquals("Click callback should be invoked twice", 2, clickCount)
    }

    @Test
    fun `test adapter with large number of images`() {
        // Given
        val images = (1..100).map { Uri.parse("content://test/image$it.jpg") }
        val adapter = FullScreenImageAdapter(images, mockOnClick)

        // When
        val count = adapter.itemCount

        // Then
        assertEquals(100, count)
    }

    @Test
    fun `test ViewHolder reuse with different positions`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        val adapter = FullScreenImageAdapter(images, mockOnClick)
        val viewHolder = adapter.onCreateViewHolder(parent, 0)

        // When - bind different positions to same ViewHolder (simulating recycling)
        adapter.onBindViewHolder(viewHolder, 0)
        adapter.onBindViewHolder(viewHolder, 1)
        adapter.onBindViewHolder(viewHolder, 2)

        // Then - should not crash, ImageView should exist
        val imageView = viewHolder.itemView.findViewById<ImageView>(R.id.preview_image)
        assertNotNull(imageView)
    }

    @Test
    fun `test callback is not invoked until image is clicked`() {
        // Given
        val images = listOf(testUri1)
        val context = RuntimeEnvironment.getApplication()
        val parent = RecyclerView(context)
        
        var clickCount = 0
        val countingCallback: () -> Unit = { clickCount++ }
        val adapter = FullScreenImageAdapter(images, countingCallback)
        val viewHolder = adapter.onCreateViewHolder(parent, 0)

        // When
        adapter.onBindViewHolder(viewHolder, 0)

        // Then - callback should not be invoked yet
        assertEquals("Callback should not be invoked before click", 0, clickCount)
    }
}
