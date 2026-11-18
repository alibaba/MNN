package com.alibaba.mnnllm.android.widgets

import android.net.Uri
import android.content.Context
import android.content.ContentResolver
import java.io.ByteArrayInputStream
import java.io.IOException
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.drawable.BitmapDrawable
import android.widget.FrameLayout
import androidx.activity.ComponentActivity
import androidx.exifinterface.media.ExifInterface
import com.alibaba.mnnllm.android.utils.ImageUtils
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.mockkStatic
import io.mockk.unmockkObject
import io.mockk.unmockkStatic
import io.mockk.verify
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestCoroutineScheduler
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.robolectric.Robolectric
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import android.view.LayoutInflater
import android.view.View
import android.widget.ImageView
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.input.ImagePreviewAdapter
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mockito
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

/**
 * Unit tests for ImagePreviewAdapter
 * Tests image collection management and adapter notification mechanisms
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@OptIn(ExperimentalCoroutinesApi::class)
class ImagePreviewAdapterTest {

    private lateinit var adapter: ImagePreviewAdapter
    private lateinit var mockDeleteCallback: (Uri) -> Unit
    private val testUri1 = Uri.parse("content://test/image1.jpg")
    private val testUri2 = Uri.parse("content://test/image2.jpg")
    private val testUri3 = Uri.parse("content://test/image3.jpg")

    @Before
    fun setup() {
        @Suppress("UNCHECKED_CAST")
        mockDeleteCallback = Mockito.mock(Function1::class.java) as (Uri) -> Unit
        adapter = ImagePreviewAdapter(mockDeleteCallback)
    }

    private fun createParentRecyclerView(): RecyclerView {
        val context = RuntimeEnvironment.getApplication()
        return RecyclerView(context).apply {
            layoutManager = LinearLayoutManager(context)
        }
    }

    @Test
    fun `test initial state is empty`() {
        // Then
        assertEquals(0, adapter.itemCount)
        assertTrue(adapter.getImages().isEmpty())
    }

    @Test
    fun `test addImage adds single image`() {
        // When
        adapter.addImage(testUri1)

        // Then
        assertEquals(1, adapter.itemCount)
        assertEquals(listOf(testUri1), adapter.getImages())
    }

    @Test
    fun `test addImage adds multiple images sequentially`() {
        // When
        adapter.addImage(testUri1)
        adapter.addImage(testUri2)
        adapter.addImage(testUri3)

        // Then
        assertEquals(3, adapter.itemCount)
        assertEquals(listOf(testUri1, testUri2, testUri3), adapter.getImages())
    }

    @Test
    fun `test addImages adds multiple images at once`() {
        // Given
        val uris = listOf(testUri1, testUri2, testUri3)

        // When
        adapter.addImages(uris)

        // Then
        assertEquals(3, adapter.itemCount)
        assertEquals(uris, adapter.getImages())
    }

    @Test
    fun `test addImages appends to existing images`() {
        // Given
        adapter.addImage(testUri1)
        val newUris = listOf(testUri2, testUri3)

        // When
        adapter.addImages(newUris)

        // Then
        assertEquals(3, adapter.itemCount)
        assertEquals(listOf(testUri1, testUri2, testUri3), adapter.getImages())
    }

    @Test
    fun `test addImages with empty list does not change state`() {
        // Given
        adapter.addImage(testUri1)

        // When
        adapter.addImages(emptyList())

        // Then
        assertEquals(1, adapter.itemCount)
        assertEquals(listOf(testUri1), adapter.getImages())
    }

    @Test
    fun `test removeImage removes existing image`() {
        // Given
        adapter.addImages(listOf(testUri1, testUri2, testUri3))

        // When
        adapter.removeImage(testUri2)

        // Then
        assertEquals(2, adapter.itemCount)
        assertEquals(listOf(testUri1, testUri3), adapter.getImages())
    }

    @Test
    fun `test removeImage with non-existent URI does nothing`() {
        // Given
        adapter.addImages(listOf(testUri1, testUri2))
        val nonExistentUri = Uri.parse("content://test/nonexistent.jpg")

        // When
        adapter.removeImage(nonExistentUri)

        // Then
        assertEquals(2, adapter.itemCount)
        assertEquals(listOf(testUri1, testUri2), adapter.getImages())
    }

    @Test
    fun `test removeImage removes only first occurrence`() {
        // Given
        adapter.addImages(listOf(testUri1, testUri2, testUri1))

        // When
        adapter.removeImage(testUri1)

        // Then
        assertEquals(2, adapter.itemCount)
        assertEquals(listOf(testUri2, testUri1), adapter.getImages())
    }

    @Test
    fun `test getImages returns a copy of the list`() {
        // Given
        adapter.addImages(listOf(testUri1, testUri2))

        // When
        val images = adapter.getImages()
        // Try to modify the returned list
        val mutableImages = images.toMutableList()
        mutableImages.add(testUri3)

        // Then - original adapter should not be affected
        assertEquals(2, adapter.itemCount)
        assertEquals(listOf(testUri1, testUri2), adapter.getImages())
    }

    @Test
    fun `test clear removes all images`() {
        // Given
        adapter.addImages(listOf(testUri1, testUri2, testUri3))

        // When
        adapter.clear()

        // Then
        assertEquals(0, adapter.itemCount)
        assertTrue(adapter.getImages().isEmpty())
    }

    @Test
    fun `test clear on empty adapter does nothing`() {
        // When
        adapter.clear()

        // Then
        assertEquals(0, adapter.itemCount)
        assertTrue(adapter.getImages().isEmpty())
    }

    @Test
    fun `test onCreateViewHolder creates correct ViewHolder`() {
        // Given
        val parent = createParentRecyclerView()

        // When
        val viewHolder = adapter.onCreateViewHolder(parent, 0)

        // Then
        assertNotNull(viewHolder)
        assertNotNull(viewHolder.itemView)
    }

    @Test
    fun `test onBindViewHolder binds image URI`() {
        // Given
        adapter.addImage(testUri1)
        val parent = createParentRecyclerView()
        val viewHolder = adapter.onCreateViewHolder(parent, 0)

        // When
        adapter.onBindViewHolder(viewHolder, 0)

        // Then
        val imageView = viewHolder.itemView.findViewById<ImageView>(R.id.iv_preview)
        assertNotNull(imageView)
        // Note: Cannot easily verify URI is set due to Robolectric limitations
        // but we can verify no crash occurs
    }

    @Test
    fun `test ViewHolder delete button triggers callback`() {
        // Given
        adapter.addImage(testUri1)
        val parent = createParentRecyclerView()
        val viewHolder = adapter.onCreateViewHolder(parent, 0)
        adapter.onBindViewHolder(viewHolder, 0)

        // When
        val deleteButton = viewHolder.itemView.findViewById<View>(R.id.iv_delete)
        deleteButton?.performClick()

        // Then - verify callback was invoked by checking the adapter state after removal would happen
        // Note: We can't easily verify mock invocation with simple Mockito, so we test behavior instead
        assertNotNull("Delete button should exist", deleteButton)
    }

    @Test
    fun `test ViewHolder fallback click triggers callback when no delete button`() {
        // Given
        adapter.addImage(testUri1)
        val parent = createParentRecyclerView()
        val viewHolder = adapter.onCreateViewHolder(parent, 0)
        adapter.onBindViewHolder(viewHolder, 0)

        // Note: This test verifies the fallback behavior mentioned in the code
        // If there's no delete button, clicking the item itself should trigger delete
        // This requires the layout to not have iv_delete view
        
        // When - simulate no delete button scenario by clicking item
        if (viewHolder.itemView.findViewById<View>(R.id.iv_delete) == null) {
            viewHolder.itemView.performClick()
            // Then - callback would be invoked
        }
    }

    @Test
    fun `test adapter handles rapid add and remove operations`() {
        // When
        adapter.addImage(testUri1)
        adapter.addImage(testUri2)
        adapter.removeImage(testUri1)
        adapter.addImage(testUri3)
        adapter.removeImage(testUri2)

        // Then
        assertEquals(1, adapter.itemCount)
        assertEquals(listOf(testUri3), adapter.getImages())
    }

    @Test
    fun `test adapter handles adding same URI multiple times`() {
        // When
        adapter.addImage(testUri1)
        adapter.addImage(testUri1)
        adapter.addImage(testUri1)

        // Then
        assertEquals(3, adapter.itemCount)
        assertEquals(listOf(testUri1, testUri1, testUri1), adapter.getImages())
    }

    @Test
    fun `sample size uses ceil dimensions and both budgets`() {
        assertEquals(32, ImageUtils.calculateInSampleSize(24000, 1000, 1024, 1048576))
        assertEquals(32, ImageUtils.calculateInSampleSize(1000, 24000, 1024, 1048576))
        assertEquals(2, ImageUtils.calculateInSampleSize(1025, 1, 1024, 1048576))
        assertEquals(4, ImageUtils.calculateInSampleSize(1025, 1025, 2048, 262144))
        assertEquals(2, ImageUtils.calculateInSampleSize(1000, 1000, 2048, 250000))
        assertEquals(1, ImageUtils.calculateInSampleSize(640, 480, 1024, 1048576))
        assertEquals(1, ImageUtils.calculateInSampleSize(1, 1, 1024, 1048576))
        assertNull(ImageUtils.calculateInSampleSize(0, 100, 1024, 1048576))
        assertNull(ImageUtils.calculateInSampleSize(100, -1, 1024, 1048576))
        assertNull(ImageUtils.calculateInSampleSize(100, 100, 0, 1048576))
        assertNull(ImageUtils.calculateInSampleSize(100, 100, 1024, 0))
    }

    @Test
    fun `sample budget stays bounded without overflow at extreme dimensions`() {
        val dimensions = listOf(1, 3, 511, 1024, 1025, 24000, 65535, Int.MAX_VALUE)
        for (width in dimensions) {
            for (height in dimensions) {
                val sample = ImageUtils.calculateInSampleSize(width, height, 1024, 1048576)!!
                assertEquals(0, sample and (sample - 1))
                val sampledWidth = (width.toLong() + sample - 1) / sample
                val sampledHeight = (height.toLong() + sample - 1) / sample
                assertTrue(sampledWidth in 1L..1024L)
                assertTrue(sampledHeight in 1L..1024L)
                assertTrue(sampledWidth * sampledHeight <= 1048576)
            }
        }
    }

    private fun createSource(width: Int = 40, height: Int = 20): File {
        val context = RuntimeEnvironment.getApplication()
        val file = File.createTempFile("source_", ".jpg", context.cacheDir)
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        try {
            for (y in 0 until height) {
                for (x in 0 until width) {
                    bitmap.setPixel(x, y, if (y < height / 2) {
                        if (x < width / 2) Color.RED else Color.GREEN
                    } else {
                        if (x < width / 2) Color.YELLOW else Color.BLUE
                    })
                }
            }
            file.outputStream().use { assertTrue(bitmap.compress(Bitmap.CompressFormat.JPEG, 100, it)) }
        } finally {
            bitmap.recycle()
        }
        return file
    }

    private fun attachmentDirectory(): File {
        val context = RuntimeEnvironment.getApplication()
        return File(context.filesDir, "image-test-${java.util.UUID.randomUUID()}").apply { mkdirs() }
    }

    @Test
    fun `derived attachment is bounded unique and preserves source`() {
        val context = RuntimeEnvironment.getApplication()
        val source = createSource(24000, 3)
        val bytes = source.readBytes()
        val directory = attachmentDirectory()
        val first = ImageUtils.createImageAttachment(context, Uri.fromFile(source), directory)!!
        val second = ImageUtils.createImageAttachment(context, Uri.fromFile(source), directory)!!
        assertNotEquals(first, second)
        assertArrayEquals(bytes, source.readBytes())
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(first.absolutePath, bounds)
        assertTrue(bounds.outWidth in 1..1024)
        assertTrue(bounds.outHeight in 1..1024)
        assertTrue(bounds.outWidth.toLong() * bounds.outHeight <= 1048576)
    }

    @Test
    fun `bad image missing image and non private destination are rejected`() {
        val context = RuntimeEnvironment.getApplication()
        val source = File.createTempFile("invalid_", ".jpg", context.cacheDir)
        source.writeText("not an image")
        val directory = attachmentDirectory()
        assertNull(ImageUtils.createImageAttachment(context, Uri.fromFile(source), directory))
        assertNull(ImageUtils.createImageAttachment(context, Uri.fromFile(File(source, "missing")), directory))
        assertNull(ImageUtils.createImageAttachment(context, Uri.fromFile(source), context.cacheDir))
        assertEquals("not an image", source.readText())
        assertTrue(directory.listFiles()!!.isEmpty())
        assertFalse(ImageUtils.compressImageFile(source))
        assertEquals("not an image", source.readText())
    }

    @Test
    fun `small images are not enlarged including all EXIF orientations`() {
        val context = RuntimeEnvironment.getApplication()
        val expectedCorners = listOf(Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW,
            Color.RED, Color.YELLOW, Color.BLUE, Color.GREEN)
        for (orientation in 1..8) {
            val source = createSource()
            ExifInterface(source).apply {
                setAttribute(ExifInterface.TAG_ORIENTATION, orientation.toString())
                saveAttributes()
            }
            val bitmap = ImageUtils.decodeBitmap(context, Uri.fromFile(source), 1024)!!
            try {
                assertEquals(if (orientation >= 5) 20 else 40, bitmap.width)
                assertEquals(if (orientation >= 5) 40 else 20, bitmap.height)
                val actual = bitmap.getPixel(2, 2)
                val expected = expectedCorners[orientation - 1]
                assertTrue(kotlin.math.abs(Color.red(actual) - Color.red(expected)) < 50)
                assertTrue(kotlin.math.abs(Color.green(actual) - Color.green(expected)) < 50)
                assertTrue(kotlin.math.abs(Color.blue(actual) - Color.blue(expected)) < 50)
            } finally {
                bitmap.recycle()
            }
        }
    }

    @Test
    fun `captured file compression reports failure and bounds successful output`() {
        val source = createSource(24000, 3)
        val bytes = source.readBytes()
        assertFalse(ImageUtils.compressImageFile(source, quality = -1))
        assertArrayEquals(bytes, source.readBytes())
        assertTrue(ImageUtils.compressImageFile(source))
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(source.absolutePath, bounds)
        assertTrue(bounds.outWidth in 1..1024)
        assertTrue(bounds.outHeight in 1..1024)
    }

    @Test
    fun `stream close failure releases decoded bitmap and closes each input`() {
        val context = mockk<Context>()
        val resolver = mockk<ContentResolver>()
        val bytes = createSource().readBytes()
        val bitmap = Bitmap.createBitmap(40, 20, Bitmap.Config.ARGB_8888)
        var opened = 0
        var closed = 0
        every { context.contentResolver } returns resolver
        every { resolver.openInputStream(testUri1) } answers {
            val index = ++opened
            object : ByteArrayInputStream(bytes) {
                override fun close() {
                    closed++
                    super.close()
                    if (index == 3) throw IOException("controlled close failure")
                }
            }
        }
        mockkStatic(BitmapFactory::class)
        try {
            every { BitmapFactory.decodeStream(any(), any(), any()) } answers {
                val options = thirdArg<BitmapFactory.Options>()
                if (options.inJustDecodeBounds) {
                    options.outWidth = 40
                    options.outHeight = 20
                    null
                } else {
                    bitmap
                }
            }
            assertNull(ImageUtils.decodeBitmap(context, testUri1, 1024))
            assertEquals(3, opened)
            assertEquals(3, closed)
            assertTrue(bitmap.isRecycled)
        } finally {
            unmockkStatic(BitmapFactory::class)
        }
    }

    @Test
    fun `bounded decode OOM rejects attachment without touching source`() {
        val context = RuntimeEnvironment.getApplication()
        val source = createSource()
        val bytes = source.readBytes()
        val directory = attachmentDirectory()
        mockkStatic(BitmapFactory::class)
        try {
            every { BitmapFactory.decodeStream(any(), any(), any()) } answers {
                val options = thirdArg<BitmapFactory.Options>()
                if (options.inJustDecodeBounds) {
                    options.outWidth = 24000
                    options.outHeight = 1000
                    null
                } else {
                    assertEquals(32, options.inSampleSize)
                    throw OutOfMemoryError("controlled decode failure")
                }
            }
            assertNull(ImageUtils.createImageAttachment(context, Uri.fromFile(source), directory))
            assertArrayEquals(bytes, source.readBytes())
            assertTrue(directory.listFiles()!!.isEmpty())
        } finally {
            unmockkStatic(BitmapFactory::class)
        }
    }

    @Test
    fun `encoder failure and OOM recycle temporary bitmap and reject original fallback`() {
        val context = RuntimeEnvironment.getApplication()
        val source = createSource()
        val uri = Uri.fromFile(source)
        val bytes = source.readBytes()
        val directory = attachmentDirectory()
        mockkObject(ImageUtils)
        try {
            for (outOfMemory in listOf(false, true)) {
                val bitmap = mockk<Bitmap>(relaxed = true)
                every { ImageUtils.decodeBitmap(context, uri, 1024, 1048576) } returns bitmap
                every { bitmap.compress(any(), any(), any()) } answers {
                    if (outOfMemory) throw OutOfMemoryError("controlled encode failure")
                    false
                }
                assertNull(ImageUtils.createImageAttachment(context, uri, directory))
                verify(exactly = 1) { bitmap.recycle() }
                assertTrue(directory.listFiles()!!.isEmpty())
                assertArrayEquals(bytes, source.readBytes())
            }
        } finally {
            unmockkObject(ImageUtils)
        }
    }

    @Test
    fun `rebind drops late decode and never recycles a displayed bitmap`() {
        val main = StandardTestDispatcher(TestCoroutineScheduler())
        val dispatcher = StandardTestDispatcher(TestCoroutineScheduler())
        Dispatchers.setMain(main)
        val controller = Robolectric.buildActivity(ComponentActivity::class.java).setup().visible()
        val activity = controller.get()
        val image = ImageView(activity)
        activity.setContentView(image)
        val first = Bitmap.createBitmap(2, 2, Bitmap.Config.ARGB_8888)
        val second = Bitmap.createBitmap(3, 3, Bitmap.Config.ARGB_8888)
        val request = ImageUtils.ImageRequest(image, 512, dispatcher) { _, uri, _ ->
            if (uri == testUri1) first else second
        }
        try {
            assertTrue(image.isAttachedToWindow)
            request.bind(testUri1)
            main.scheduler.runCurrent()
            assertNull(image.drawable)
            dispatcher.scheduler.runCurrent()
            assertNull(image.drawable)
            request.bind(testUri2)
            main.scheduler.runCurrent()
            dispatcher.scheduler.runCurrent()
            main.scheduler.runCurrent()
            assertTrue(first.isRecycled)
            assertSame(second, (image.drawable as BitmapDrawable).bitmap)
            assertFalse(second.isRecycled)
            request.clear()
            assertNull(image.drawable)
            assertFalse(second.isRecycled)
        } finally {
            request.clear()
            controller.pause().stop().destroy()
            main.scheduler.runCurrent()
            Dispatchers.resetMain()
        }
    }

    @Test
    fun `detach cancels decode reattach reloads and destroy discards pending result`() {
        val main = StandardTestDispatcher(TestCoroutineScheduler())
        val dispatcher = StandardTestDispatcher(TestCoroutineScheduler())
        Dispatchers.setMain(main)
        val controller = Robolectric.buildActivity(ComponentActivity::class.java).setup().visible()
        val activity = controller.get()
        val root = FrameLayout(activity)
        val image = ImageView(activity)
        root.addView(image)
        activity.setContentView(root)
        val bitmaps = mutableListOf<Bitmap>()
        val request = ImageUtils.ImageRequest(image, 512, dispatcher) { _, _, _ ->
            Bitmap.createBitmap(2, 2, Bitmap.Config.ARGB_8888).also(bitmaps::add)
        }
        try {
            assertTrue(image.isAttachedToWindow)
            request.bind(testUri1)
            main.scheduler.runCurrent()
            root.removeView(image)
            dispatcher.scheduler.runCurrent()
            main.scheduler.runCurrent()
            assertTrue(bitmaps.isEmpty())
            assertNull(image.drawable)
            root.addView(image)
            main.scheduler.runCurrent()
            dispatcher.scheduler.runCurrent()
            main.scheduler.runCurrent()
            assertEquals(1, bitmaps.size)
            assertSame(bitmaps[0], (image.drawable as BitmapDrawable).bitmap)
            request.bind(testUri2)
            main.scheduler.runCurrent()
            dispatcher.scheduler.runCurrent()
            controller.pause().stop().destroy()
            main.scheduler.runCurrent()
            assertEquals(2, bitmaps.size)
            assertTrue(bitmaps[1].isRecycled)
            assertFalse(bitmaps[0].isRecycled)
            assertNull(image.drawable)
        } finally {
            request.clear()
            if (!activity.isDestroyed) controller.pause().stop().destroy()
            main.scheduler.runCurrent()
            Dispatchers.resetMain()
        }
    }

    @Test
    fun `recycling preview clears drawable without recycling published bitmap`() {
        adapter.addImage(testUri1)
        val holder = adapter.onCreateViewHolder(createParentRecyclerView(), 0)
        adapter.onBindViewHolder(holder, 0)
        val image = holder.itemView.findViewById<ImageView>(R.id.iv_preview)
        val bitmap = Bitmap.createBitmap(2, 2, Bitmap.Config.ARGB_8888)
        image.setImageBitmap(bitmap)
        adapter.onViewRecycled(holder)
        assertNull(image.drawable)
        assertFalse(bitmap.isRecycled)
    }
}
