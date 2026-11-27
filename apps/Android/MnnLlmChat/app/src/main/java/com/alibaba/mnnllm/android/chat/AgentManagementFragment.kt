package com.alibaba.mnnllm.android.chat

import android.app.Dialog
import android.os.Bundle
import android.text.InputFilter
import android.text.InputType
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.DialogFragment
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.agent.AgentConsentController
import com.alibaba.mnnllm.android.chat.agent.AgentMemory
import com.alibaba.mnnllm.android.chat.agent.AgentSkill
import com.alibaba.mnnllm.android.chat.agent.AgentStoreLimits
import com.alibaba.mnnllm.android.chat.agent.SqliteAgentStore
import com.google.android.material.button.MaterialButton
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class AgentManagementFragment : DialogFragment() {
    private val consent = AgentConsentController()
    private lateinit var store: SqliteAgentStore
    private lateinit var root: View
    private lateinit var memoriesColumn: LinearLayout
    private lateinit var skillsColumn: LinearLayout
    private lateinit var toolHistorySection: LinearLayout
    private lateinit var toolHistoryColumn: LinearLayout
    private lateinit var clearToolHistoryButton: MaterialButton
    private var busy = false
    private var childDialog: androidx.appcompat.app.AlertDialog? = null

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        store = SqliteAgentStore(requireContext(), consent)
        root = layoutInflater.inflate(R.layout.dialog_agent_manage, null)
        memoriesColumn = root.findViewById(R.id.ll_agent_memories)
        skillsColumn = root.findViewById(R.id.ll_agent_skills)
        toolHistorySection = root.findViewById(R.id.section_agent_tool_history)
        toolHistoryColumn = root.findViewById(R.id.ll_agent_tool_history)
        clearToolHistoryButton = root.findViewById(R.id.btn_agent_clear_tool_history)
        root.findViewById<MaterialButton>(R.id.btn_agent_add_skill).setOnClickListener {
            if (!busy) editSkill(null)
        }
        root.findViewById<MaterialButton>(R.id.btn_agent_clear_memories).setOnClickListener {
            if (!busy) confirmDelete(getString(R.string.agent_clear_memories)) { store.clearMemories() }
        }
        root.findViewById<MaterialButton>(R.id.btn_agent_clear_skills).setOnClickListener {
            if (!busy) confirmDelete(getString(R.string.agent_clear_skills)) { store.clearSkills() }
        }
        root.findViewById<MaterialButton>(R.id.btn_agent_clear_all).setOnClickListener {
            if (!busy) confirmDelete(getString(R.string.agent_clear_all)) { store.clearAll() }
        }
        clearToolHistoryButton.setOnClickListener {
            val sessionId = arguments?.getString(ARG_SESSION_ID) ?: return@setOnClickListener
            if (!busy) confirmDelete(getString(R.string.agent_clear_tool_history)) { store.clearToolSteps(sessionId) }
        }
        return MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.agent_manage)
            .setView(root)
            .setPositiveButton(android.R.string.ok, null)
            .create()
    }

    override fun onStart() {
        super.onStart()
        reload()
    }

    private fun item(text: String, onClick: () -> Unit) = TextView(requireContext()).apply {
        this.text = text
        setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodyMedium)
        setTextColor(com.google.android.material.color.MaterialColors.getColor(
            this, com.google.android.material.R.attr.colorOnSurface))
        val vertical = (10 * resources.displayMetrics.density).toInt()
        setPadding(0, vertical, 0, vertical)
        val outValue = android.util.TypedValue()
        context.theme.resolveAttribute(android.R.attr.selectableItemBackground, outValue, true)
        setBackgroundResource(outValue.resourceId)
        isClickable = true
        isFocusable = true
        setOnClickListener { if (!busy) onClick() }
    }

    private fun emptyItem(text: String) = TextView(requireContext()).apply {
        this.text = text
        setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodySmall)
        setTextColor(com.google.android.material.color.MaterialColors.getColor(
            this, com.google.android.material.R.attr.colorOnSurfaceVariant))
        val vertical = (10 * resources.displayMetrics.density).toInt()
        setPadding(0, vertical, 0, vertical)
    }

    private fun reload() = operation {
        val memories = store.allMemories()
        val skills = store.allSkills()
        val sessionId = arguments?.getString(ARG_SESSION_ID)
        val toolHistory = sessionId?.let { store.toolStepsForSession(it) }.orEmpty()
        memoriesColumn.removeAllViews()
        if (memories.isEmpty()) {
            memoriesColumn.addView(emptyItem(getString(R.string.agent_memory_empty)))
        } else {
            memories.forEach { memory ->
                memoriesColumn.addView(item(memory.content) { showMemory(memory) })
            }
        }
        skillsColumn.removeAllViews()
        if (skills.isEmpty()) {
            skillsColumn.addView(emptyItem(getString(R.string.agent_skills_empty)))
        } else {
            skills.forEach { skill ->
                skillsColumn.addView(item(skill.name + "\n" + skill.actionTemplate) { showSkill(skill) })
            }
        }
        if (sessionId != null) {
            toolHistorySection.visibility = View.VISIBLE
            toolHistoryColumn.removeAllViews()
            if (toolHistory.isEmpty()) {
                toolHistoryColumn.addView(emptyItem(getString(R.string.agent_tool_history_empty)))
            } else {
                toolHistory.asReversed().forEachIndexed { index, record ->
                    val title = getString(R.string.agent_tool_turn_label, toolHistory.size - index, record.time)
                    toolHistoryColumn.addView(item(title) { showToolRecord(title, record) })
                }
            }
            clearToolHistoryButton.isEnabled = toolHistory.isNotEmpty()
        } else {
            toolHistorySection.visibility = View.GONE
        }
        root.findViewById<MaterialButton>(R.id.btn_agent_clear_memories).isEnabled = memories.isNotEmpty()
        root.findViewById<MaterialButton>(R.id.btn_agent_clear_skills).isEnabled = skills.isNotEmpty()
        root.findViewById<MaterialButton>(R.id.btn_agent_clear_all).isEnabled =
            memories.isNotEmpty() || skills.isNotEmpty()
    }

    private fun showToolRecord(title: String, record: com.alibaba.mnnllm.android.chat.agent.AgentToolStepRecord) {
        val text = record.steps.joinToString("\n\n") {
            getString(AgentStepUi.title(it.kind)) + " · " + getString(AgentStepUi.status(it.status)) + "\n" + it.tool
        }
        childDialog = MaterialAlertDialogBuilder(requireContext()).setTitle(title).setMessage(text)
            .setPositiveButton(android.R.string.ok, null).show()
    }

    private fun showMemory(memory: AgentMemory) {
        if (busy) return
        childDialog = MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.agent_memories)
            .setMessage(memory.content)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.agent_delete_item) { _, _ ->
                confirmDelete(memory.content) { store.deleteMemory(memory.content) }
            }.show()
    }

    private fun showSkill(skill: AgentSkill) {
        if (busy) return
        childDialog = MaterialAlertDialogBuilder(requireContext())
            .setTitle(skill.name)
            .setMessage(skill.triggers.joinToString(", ") + "\n\n" + skill.actionTemplate)
            .setNegativeButton(android.R.string.cancel, null)
            .setNeutralButton(R.string.agent_edit_skill) { _, _ -> editSkill(skill) }
            .setPositiveButton(R.string.agent_delete_item) { _, _ ->
                confirmDelete(skill.name) { store.deleteSkill(skill.name) }
            }.show()
    }

    private fun confirmDelete(title: String, delete: suspend () -> Unit) {
        childDialog = MaterialAlertDialogBuilder(requireContext())
            .setTitle(title)
            .setMessage(R.string.agent_delete_notice)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.agent_delete_item) { _, _ ->
                operation { delete(); reloadAfterOperation() }
            }.show()
    }

    private fun column() = LinearLayout(requireContext()).apply {
        orientation = LinearLayout.VERTICAL
        val padding = (16 * resources.displayMetrics.density).toInt()
        setPadding(padding, padding, padding, padding)
    }

    private fun label(text: String) = TextView(requireContext()).apply {
        this.text = text
        setTextIsSelectable(true)
        setPadding(0, 12, 0, 12)
    }

    private fun input(hint: Int, maximum: Int, text: String = "") = EditText(requireContext()).apply {
        setHint(hint)
        inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
        filters = arrayOf(InputFilter.LengthFilter(maximum))
        setText(text)
    }

    private fun editSkill(existing: AgentSkill?) {
        val form = column()
        val name = input(R.string.agent_skill_name, AgentStoreLimits.MAX_NAME_LENGTH, existing?.name.orEmpty())
        name.isEnabled = existing == null
        val triggers = input(R.string.agent_skill_triggers, 520, existing?.triggers?.joinToString(", ").orEmpty())
        val procedure = input(R.string.agent_skill_procedure, AgentStoreLimits.MAX_SKILL_LENGTH, existing?.actionTemplate.orEmpty())
        form.addView(label(getString(R.string.agent_skill_text_notice)))
        form.addView(name)
        form.addView(triggers)
        form.addView(procedure)
        val editor = MaterialAlertDialogBuilder(requireContext())
            .setTitle(if (existing == null) R.string.agent_add_skill else R.string.agent_edit_skill)
            .setView(android.widget.ScrollView(requireContext()).apply { addView(form) })
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.agent_save_skill, null)
            .create()
        editor.setOnShowListener {
            editor.getButton(androidx.appcompat.app.AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val skill = try {
                    val keywords = triggers.text.toString().split(',', '，', '\n').map { it.trim() }.filter { it.isNotBlank() }
                    require(keywords.size in 1..AgentStoreLimits.MAX_TRIGGERS &&
                        keywords.all { it.length <= AgentStoreLimits.MAX_NAME_LENGTH })
                    AgentStoreLimits.skill(AgentSkill(name.text.toString(), keywords, procedure.text.toString()))
                } catch (_: IllegalArgumentException) {
                    procedure.error = getString(R.string.agent_skill_invalid)
                    return@setOnClickListener
                }
                operation {
                    if (existing == null && store.allSkills().any { it.name == skill.name }) {
                        name.error = getString(R.string.agent_skill_exists)
                        return@operation
                    }
                    val grant = consent.confirmEnable(networkAllowed = false, persistenceAllowed = true)
                    try { check(store.saveSkill(skill, grant.revision)) }
                    finally { consent.disable() }
                    editor.dismiss()
                    reloadAfterOperation()
                }
            }
        }
        childDialog = editor
        editor.show()
    }

    private fun reloadAfterOperation() { root.post { if (isAdded) reload() } }

    private fun operation(block: suspend () -> Unit) {
        if (busy || !isAdded) return
        busy = true
        lifecycleScope.launch {
            try { block() }
            catch (e: CancellationException) { throw e }
            catch (_: Exception) {
                context?.let { Toast.makeText(it, R.string.agent_operation_failed, Toast.LENGTH_LONG).show() }
            } finally { busy = false }
        }
    }

    override fun onDestroy() {
        childDialog?.dismiss()
        childDialog = null
        super.onDestroy()
        if (::store.isInitialized) CoroutineScope(Dispatchers.IO + SupervisorJob()).launch {
            consent.disable()
            store.close()
        }
    }

    companion object {
        const val TAG = "AgentManagement"
        const val ARG_SESSION_ID = "agent_session_id"
    }
}

internal object AgentStepUi {
    fun title(kind: com.alibaba.mnnllm.android.chat.agent.AgentStepKind): Int = when (kind) {
        com.alibaba.mnnllm.android.chat.agent.AgentStepKind.PREPARING -> R.string.agent_tool_preparing
        com.alibaba.mnnllm.android.chat.agent.AgentStepKind.TIME -> R.string.agent_tool_time
        com.alibaba.mnnllm.android.chat.agent.AgentStepKind.CALCULATOR -> R.string.agent_tool_calculator
        com.alibaba.mnnllm.android.chat.agent.AgentStepKind.SEARCH -> R.string.agent_tool_search
        com.alibaba.mnnllm.android.chat.agent.AgentStepKind.READ_PAGE -> R.string.agent_tool_page
        com.alibaba.mnnllm.android.chat.agent.AgentStepKind.MEMORY -> R.string.agent_tool_memory
        com.alibaba.mnnllm.android.chat.agent.AgentStepKind.SKILL -> R.string.agent_tool_skill
        com.alibaba.mnnllm.android.chat.agent.AgentStepKind.OTHER -> R.string.agent_tool_other
    }

    fun status(status: com.alibaba.mnnllm.android.chat.agent.AgentStepStatus): Int = when (status) {
        com.alibaba.mnnllm.android.chat.agent.AgentStepStatus.RUNNING -> R.string.agent_step_running
        com.alibaba.mnnllm.android.chat.agent.AgentStepStatus.DONE -> R.string.agent_step_done
        com.alibaba.mnnllm.android.chat.agent.AgentStepStatus.FAILED -> R.string.agent_step_failed
        com.alibaba.mnnllm.android.chat.agent.AgentStepStatus.SKIPPED -> R.string.agent_step_skipped
        com.alibaba.mnnllm.android.chat.agent.AgentStepStatus.CANCELLED -> R.string.agent_step_cancelled
    }
}
