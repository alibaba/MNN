// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelsettings

import android.content.Context
import android.view.Menu
import android.view.View
import androidx.appcompat.widget.PopupMenu

/**
 * Helper class for handling dropdown menu operations
 */
class DropDownMenuHelper {
    
    companion object {
        /**
         * Create and display a dropdown menu
         *
         * @param context The context
         * @param anchorView The view to anchor the popup menu to
         * @param items List of items to display in the dropdown
         * @param itemToString Function to convert an item to its string representation
         * @param currentIndex The current selected value as a string
         * @param onItemSelected Callback when an item is selected, provides (index, item)
         */
        fun showDropDownMenu(
            context: Context, 
            anchorView: View,
            items: List<Any>,
            itemToString: (Any) -> String = { it.toString() },
            currentIndex: Int,
            onItemSelected: (Int, Any) -> Unit
        ) {
            if (items.isEmpty()) return

            val popupMenu = PopupMenu(context, anchorView)
            
            items.forEachIndexed { index, item ->
                val itemText = itemToString(item)
                if (index == currentIndex) {
                    popupMenu.menu.add(Menu.NONE, index, index, "$itemText ✓")
                } else {
                    popupMenu.menu.add(Menu.NONE, index, index, itemText)
                }
            }
            
            // Try to show icons in popup menu (if available)
            try {
                val fieldMPopup = PopupMenu::class.java.getDeclaredField("mPopup")
                fieldMPopup.isAccessible = true
                val mPopup = fieldMPopup.get(popupMenu)
                mPopup?.javaClass?.getDeclaredMethod("setForceShowIcon", Boolean::class.java)?.invoke(mPopup, true)
            } catch (e: Exception) {
                try {
                    val menuBuilderClass = Class.forName("com.android.internal.view.menu.MenuBuilder")
                    val setOptionalIconsVisibleMethod = menuBuilderClass.getDeclaredMethod("setOptionalIconsVisible", Boolean::class.javaPrimitiveType)
                    setOptionalIconsVisibleMethod.isAccessible = true
                    setOptionalIconsVisibleMethod.invoke(popupMenu.menu, true)
                } catch (ex: Exception) {
                    // Ignore exception if we can't show icons
                }
            }
            
            popupMenu.setOnMenuItemClickListener { menuItem ->
                val selectedIndex = menuItem.itemId
                val selectedValue = items[selectedIndex]
                onItemSelected.invoke(selectedIndex, selectedValue)
                true
            }
            
            popupMenu.show()
        }
    }
}