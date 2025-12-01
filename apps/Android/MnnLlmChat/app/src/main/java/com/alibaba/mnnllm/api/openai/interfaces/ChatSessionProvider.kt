package com.alibaba.mnnllm.api.openai.interfaces

import com.alibaba.mnnllm.android.llm.LlmSession

/** * chatsessionproviderinterface * fordecoupleapi.openaimodule andandroid.chatmoduledirectlydependency * * thisinterfacedefineapi.openaimoduleneededcorefunctionality, * avoiddirectlyaccessingChatPresenterprivatemembeenr*/
interface ChatSessionProvider {
    
    /** * getcurrentLLMsessioninstance * @return LlmSessioninstance，if not availableactivesessionthenreturnnull*/
    fun getLlmSession(): LlmSession?
    
    /** * check whether validactivechatsession * @return trueif there isactivesession,falseotherwise*/
    fun hasActiveSession(): Boolean
    
    /** * getcurrentsessionID * @return sessionID，if not availableactivesessionthenreturnnull*/
    fun getCurrentSessionId(): String?
}