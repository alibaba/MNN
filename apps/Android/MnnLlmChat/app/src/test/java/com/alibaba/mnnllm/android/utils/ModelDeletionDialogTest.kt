package com.alibaba.mnnllm.android.utils

import android.app.Application
import android.content.DialogInterface
import android.view.View
import android.view.ViewGroup
import android.widget.CheckBox
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.alibaba.mnnllm.android.R
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.android.controller.ActivityController
import org.robolectric.annotation.Config
import org.robolectric.shadows.ShadowDialog

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28], application = Application::class)
class ModelDeletionDialogTest {
    private lateinit var controller: ActivityController<AppCompatActivity>
    private lateinit var activity: AppCompatActivity

    @Before
    fun setUp() {
        controller = Robolectric.buildActivity(AppCompatActivity::class.java)
        activity = controller.get()
        activity.setTheme(R.style.AppTheme)
        controller.setup()
    }

    @After
    fun tearDown() {
        ShadowDialog.getLatestDialog()?.dismiss()
        controller.pause().stop().destroy()
    }

    @Test
    fun confirmationDefaultsToRetainingHistory() {
        val choices = mutableListOf<Boolean>()
        DialogUtils.showModelDeletionDialog(activity) { choices.add(it) }
        val dialog = ShadowDialog.getLatestDialog() as AlertDialog
        assertTrue(findCheckbox(dialog.window!!.decorView)!!.isChecked)
        assertTrue(choices.isEmpty())

        dialog.getButton(DialogInterface.BUTTON_POSITIVE).performClick()

        assertEquals(listOf(true), choices)
    }

    @Test
    fun uncheckingExplicitlyRequestsFullDeletion() {
        val choices = mutableListOf<Boolean>()
        DialogUtils.showModelDeletionDialog(activity) { choices.add(it) }
        val dialog = ShadowDialog.getLatestDialog() as AlertDialog
        findCheckbox(dialog.window!!.decorView)!!.isChecked = false
        assertTrue(choices.isEmpty())

        dialog.getButton(DialogInterface.BUTTON_POSITIVE).performClick()

        assertEquals(listOf(false), choices)
    }

    @Test
    fun cancelAndDismissNeverConfirmDeletion() {
        val choices = mutableListOf<Boolean>()
        DialogUtils.showModelDeletionDialog(activity) { choices.add(it) }
        (ShadowDialog.getLatestDialog() as AlertDialog)
            .getButton(DialogInterface.BUTTON_NEGATIVE).performClick()
        DialogUtils.showModelDeletionDialog(activity) { choices.add(it) }
        ShadowDialog.getLatestDialog().cancel()
        DialogUtils.showModelDeletionDialog(activity) { choices.add(it) }
        ShadowDialog.getLatestDialog().dismiss()

        assertTrue(choices.isEmpty())
    }

    private fun findCheckbox(view: View): CheckBox? {
        if (view is CheckBox) return view
        if (view is ViewGroup) {
            for (index in 0 until view.childCount) {
                findCheckbox(view.getChildAt(index))?.let { return it }
            }
        }
        return null
    }
}
