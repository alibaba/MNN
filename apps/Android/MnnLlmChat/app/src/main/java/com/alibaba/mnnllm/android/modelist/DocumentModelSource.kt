package com.alibaba.mnnllm.android.modelist

import android.content.ContentResolver
import android.net.Uri
import android.provider.DocumentsContract
import java.io.IOException
import java.io.InputStream

/** SAF document IDs are opaque: resolve every child via DocumentsContract, never as file paths. */
class DocumentModelSource private constructor(
    private val resolver: ContentResolver,
    private val tree: Uri,
    private val id: String,
    override val name: String,
    override val isDirectory: Boolean
) : LocalModelImporter.Source {
    override fun children(): List<LocalModelImporter.Source> {
        val uri = DocumentsContract.buildChildDocumentsUriUsingTree(tree, id)
        return resolver.query(uri, COLUMNS, null, null, null)?.use { cursor ->
            buildList {
                while (cursor.moveToNext()) {
                    add(DocumentModelSource(resolver, tree, cursor.getString(0), cursor.getString(1),
                        cursor.getString(2) == DocumentsContract.Document.MIME_TYPE_DIR))
                }
            }
        } ?: throw IOException("Cannot read the selected folder")
    }

    override fun open(): InputStream = resolver.openInputStream(
        DocumentsContract.buildDocumentUriUsingTree(tree, id)
    ) ?: throw IOException("Cannot read $name")

    companion object {
        private val COLUMNS = arrayOf(DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME, DocumentsContract.Document.COLUMN_MIME_TYPE)

        fun fromTree(resolver: ContentResolver, tree: Uri): DocumentModelSource {
            val id = DocumentsContract.getTreeDocumentId(tree)
            val uri = DocumentsContract.buildDocumentUriUsingTree(tree, id)
            return resolver.query(uri, COLUMNS, null, null, null)?.use { cursor ->
                if (!cursor.moveToFirst()) throw IOException("Cannot read the selected folder")
                DocumentModelSource(resolver, tree, id, cursor.getString(1),
                    cursor.getString(2) == DocumentsContract.Document.MIME_TYPE_DIR)
            } ?: throw IOException("Cannot read the selected folder")
        }
    }
}
