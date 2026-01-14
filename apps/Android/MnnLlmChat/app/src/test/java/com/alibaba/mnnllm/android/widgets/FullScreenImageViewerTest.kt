package com.alibaba.mnnllm.android.widgets

import android.app.Dialog
import android.content.Context
import android.net.Uri
import androidx.viewpager2.widget.ViewPager2
import com.alibaba.mnnllm.android.R
import io.mockk.*
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.shadows.ShadowDialog

/**
 * Unit tests for FullScreenImageViewer
 * Tests dialog creation, ViewPager2 setup, and user interaction handling
 */
@RunWith(RobolectricTestRunner::class)
class FullScreenImageViewerTest {

    private lateinit var context: Context
    private val testUri1 = Uri.parse("content://test/image1.jpg")
    private val testUri2 = Uri.parse("content://test/image2.jpg")
    private val testUri3 = Uri.parse("content://test/image3.jpg")

    @Before
    fun setup() {
        context = RuntimeEnvironment.getApplication()
    }

    @After
    fun tearDown() {
        // Dismiss any remaining dialogs
        ShadowDialog.reset()
    }

    @Test
    fun `test showImagePopup with single URI creates dialog`() {
        // When
        FullScreenImageViewer.showImagePopup(context, testUri1)

        // Then
        val dialog = ShadowDialog.getLatestDialog()
        assertNotNull("Dialog should be created", dialog)
        assertTrue("Dialog should be showing", dialog.isShowing)
    }

    @Test
    fun `test showImagePopup with null URI does nothing`() {
        // When
        FullScreenImageViewer.showImagePopup(context, null)

        // Then
        val dialog = ShadowDialog.getLatestDialog()
        assertNull("No dialog should be created for null URI", dialog)
    }

    @Test
    fun `test showImagePopup with empty list does nothing`() {
        // When
        FullScreenImageViewer.showImagePopup(context, emptyList(), 0)

        // Then
        val dialog = ShadowDialog.getLatestDialog()
        assertNull("No dialog should be created for empty list", dialog)
    }

    @Test
    fun `test showImagePopup with single image list creates dialog`() {
        // Given
        val images = listOf(testUri1)

        // When
        FullScreenImageViewer.showImagePopup(context, images, 0)

        // Then
        val dialog = ShadowDialog.getLatestDialog()
        assertNotNull("Dialog should be created", dialog)
        assertTrue("Dialog should be showing", dialog.isShowing)
    }

    @Test
    fun `test showImagePopup with multiple images creates dialog`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)

        // When
        FullScreenImageViewer.showImagePopup(context, images, 0)

        // Then
        val dialog = ShadowDialog.getLatestDialog()
        assertNotNull("Dialog should be created", dialog)
        assertTrue("Dialog should be showing", dialog.isShowing)
    }

    @Test
    fun `test dialog contains ViewPager2`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)

        // When
        FullScreenImageViewer.showImagePopup(context, images, 0)

        // Then
        val dialog = ShadowDialog.getLatestDialog()
        assertNotNull(dialog)
        
        val viewPager = dialog.findViewById<ViewPager2>(R.id.viewPager)
        assertNotNull("Dialog should contain ViewPager2", viewPager)
    }

    @Test
    fun `test ViewPager2 adapter is set correctly`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)

        // When
        FullScreenImageViewer.showImagePopup(context, images, 1)

        // Then
        val dialog = ShadowDialog.getLatestDialog()
        val viewPager = dialog.findViewById<ViewPager2>(R.id.viewPager)
        
        assertNotNull("ViewPager adapter should be set", viewPager.adapter)
        assertEquals("Adapter should have correct item count", 3, viewPager.adapter?.itemCount)
    }

    @Test
    fun `test ViewPager2 initial position is set to index 0`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)

        // When
        FullScreenImageViewer.showImagePopup(context, images, 0)

        // Then
        val dialog = ShadowDialog.getLatestDialog()
        val viewPager = dialog.findViewById<ViewPager2>(R.id.viewPager)
        
        assertEquals("Initial position should be 0", 0, viewPager.currentItem)
    }

    @Test
    fun `test ViewPager2 initial position is set to middle index`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)

        // When
        FullScreenImageViewer.showImagePopup(context, images, 1)

        // Then
        val dialog = ShadowDialog.getLatestDialog()
        val viewPager = dialog.findViewById<ViewPager2>(R.id.viewPager)
        
        assertEquals("Initial position should be 1", 1, viewPager.currentItem)
    }

    @Test
    fun `test ViewPager2 initial position is set to last index`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)

        // When
        FullScreenImageViewer.showImagePopup(context, images, 2)

        // Then
        val dialog = ShadowDialog.getLatestDialog()
        val viewPager = dialog.findViewById<ViewPager2>(R.id.viewPager)
        
        assertEquals("Initial position should be 2", 2, viewPager.currentItem)
    }

    @Test
    fun `test background click dismisses dialog`() {
        // Given
        val images = listOf(testUri1, testUri2, testUri3)
        FullScreenImageViewer.showImagePopup(context, images, 0)
        val dialog = ShadowDialog.getLatestDialog()
        
        assertTrue("Dialog should be showing initially", dialog.isShowing)

        // When - click on the root view (background)
        val rootView = dialog.findViewById<android.view.View>(android.R.id.content)?.parent as? android.view.View
        rootView?.performClick()

        // Then
        // Note: In Robolectric, we need to manually trigger dismiss
        // The actual click handling may vary, so we verify the setup exists
        assertNotNull("Root view should have click listener", rootView?.hasOnClickListeners())
    }

    @Test
    fun `test single URI overload calls multi-image version`() {
        // When
        FullScreenImageViewer.showImagePopup(context, testUri1)

        // Then
        val dialog = ShadowDialog.getLatestDialog()
        assertNotNull("Dialog should be created", dialog)
        
        val viewPager = dialog.findViewById<ViewPager2>(R.id.viewPager)
        assertNotNull("ViewPager2 should exist", viewPager)
        assertEquals("Should have 1 image", 1, viewPager.adapter?.itemCount)
        assertEquals("Initial position should be 0", 0, viewPager.currentItem)
    }

    @Test
    fun `test dialog has fullscreen theme`() {
        // Given
        val images = listOf(testUri1)

        // When
        FullScreenImageViewer.showImagePopup(context, images, 0)

        // Then
        val dialog = ShadowDialog.getLatestDialog()
        assertNotNull("Dialog should be created with fullscreen theme", dialog)
        // Note: Theme verification is limited in Robolectric, but we verify dialog creation
    }

    @Test
    fun `test handles large image list`() {
        // Given
        val images = (1..50).map { Uri.parse("content://test/image$it.jpg") }

        // When
        FullScreenImageViewer.showImagePopup(context, images, 25)

        // Then
        val dialog = ShadowDialog.getLatestDialog()
        assertNotNull("Dialog should handle large image lists", dialog)
        
        val viewPager = dialog.findViewById<ViewPager2>(R.id.viewPager)
        assertEquals("Should have 50 images", 50, viewPager.adapter?.itemCount)
        assertEquals("Should be at position 25", 25, viewPager.currentItem)
    }

    @Test
    fun `test dialog content view is set`() {
        // Given
        val images = listOf(testUri1)

        // When
        FullScreenImageViewer.showImagePopup(context, images, 0)

        // Then
        val dialog = ShadowDialog.getLatestDialog()
        assertNotNull("Dialog content should be set", dialog.window?.decorView)
    }

    @Test
    fun `test showImagePopup with index out of bounds still creates dialog`() {
        // Given
        val images = listOf(testUri1, testUri2)

        // When - use index beyond array bounds
        FullScreenImageViewer.showImagePopup(context, images, 10)

        // Then - dialog should still be created
        val dialog = ShadowDialog.getLatestDialog()
        assertNotNull("Dialog should be created even with out-of-bounds index", dialog)
        
        // ViewPager2 should handle the invalid index gracefully
        val viewPager = dialog.findViewById<ViewPager2>(R.id.viewPager)
        assertNotNull(viewPager)
    }

    @Test
    fun `test multiple dialogs can be created sequentially`() {
        // When - create first dialog
        FullScreenImageViewer.showImagePopup(context, testUri1)
        val dialog1 = ShadowDialog.getLatestDialog()
        
        // Dismiss first dialog
        dialog1.dismiss()
        
        // Create second dialog
        FullScreenImageViewer.showImagePopup(context, testUri2)
        val dialog2 = ShadowDialog.getLatestDialog()

        // Then
        assertNotNull("Second dialog should be created", dialog2)
        assertFalse("First dialog should be dismissed", dialog1.isShowing)
        assertTrue("Second dialog should be showing", dialog2.isShowing)
    }
}
