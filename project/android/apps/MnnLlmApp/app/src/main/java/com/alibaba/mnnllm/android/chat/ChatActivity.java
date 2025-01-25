// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat;

import static com.alibaba.mnnllm.android.chat.VoiceRecordingModule.REQUEST_RECORD_AUDIO_PERMISSION;
import static com.alibaba.mnnllm.android.utils.KeyboardUtils.hideKeyboard;
import static com.alibaba.mnnllm.android.utils.KeyboardUtils.showKeyboard;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.util.Log;
import android.view.KeyEvent;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.alibaba.mnnllm.android.ChatService;
import com.alibaba.mnnllm.android.ChatSession;
import com.alibaba.mnnllm.android.R;
import com.alibaba.mnnllm.android.utils.FileUtils;
import com.alibaba.mnnllm.android.utils.ModelUtils;
import com.alibaba.mnnllm.android.utils.AudioPlayService;
import com.alibaba.mnnllm.android.utils.PreferenceUtils;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;

public class ChatActivity extends AppCompatActivity {

    private RecyclerView recyclerView;
    private ChatRecyclerViewAdapter adapter;
    private EditText editUserMessage;
    private ImageView buttonSend;

    private ImageView imageMore;
    private View layoutModelLoading;

    private DateFormat dateFormat;
    private ChatSession chatSession;
    private String chatSessionId;

    private String modelName;
    private ScheduledExecutorService chatExecutor;

    private LinearLayoutManager linearLayoutManager;

    private ChatDataManager chatDataManager;

    private boolean isUserScrolling = false;

    public static final String TAG = "ChatActivity";
    private VoiceRecordingModule voiceRecordingModule;

    private boolean isAudioModel = false;
    private AttachmentPickerModule attachmentPickerModule;
    private View buttonSwitchVoice;

    private ChatDataItem currentUserMessage;

    private boolean isGenerating = false;
    private boolean isLoading = false;
    private String sessionName;
    private boolean stopGenerating = false;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_chat);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        modelName = getIntent().getStringExtra("modelName");
        layoutModelLoading = findViewById(R.id.layout_model_loading);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        }
        chatExecutor = Executors.newScheduledThreadPool(1);
        chatDataManager = ChatDataManager.getInstance(this);
        this.setupSession();
        dateFormat = new SimpleDateFormat("hh:mm aa", Locale.getDefault());
        this.setupRecyclerView();
        setupEditText();
        buttonSend = findViewById(R.id.bt_send);
        buttonSend.setEnabled(false);
        buttonSend.setOnClickListener(view -> handleSendClick());
        isAudioModel = ModelUtils.isAudioModel(modelName);
        setupVoiceRecordingModule();
        setupAttachmentPickerModule();
        smoothScrollToBottom();
    }

    private void handleSendClick() {
        Log.d(TAG, "handleSendClick isGenerating : " + isGenerating);
        if (isGenerating) {
            stopGenerating = true;
        } else {
            sendUserMessage();
        }
    }

    private void setupSession() {
        ChatService chatService = ChatService.provide();
        chatSessionId = getIntent().getStringExtra("chatSessionId");
        List<ChatDataItem> chatDataItemList;
        if (!TextUtils.isEmpty(chatSessionId)) {
            chatDataItemList = chatDataManager.getChatDataBySession(chatSessionId);
            if (chatDataItemList != null && !chatDataItemList.isEmpty()) {
                sessionName = chatDataItemList.get(0).getText();
            }
        } else {
            chatDataItemList = null;
        }
        if (ModelUtils.isDiffusionModel(modelName)) {
            String diffusionDir = getIntent().getStringExtra("diffusionDir");
            chatSession =  chatService.createDiffusionSession(diffusionDir, chatSessionId, chatDataItemList);
        } else {
            String configFilePath = getIntent().getStringExtra("configFilePath");
            chatSession = chatService.createSession(configFilePath, true, chatSessionId, chatDataItemList);
        }
        chatSessionId = chatSession.getSessionId();
        chatSession.setKeepHistory(!ModelUtils.isVisualModel(modelName) && !ModelUtils.isAudioModel(modelName));
        Log.d(TAG, "current SessionId: " + chatSessionId);
        chatExecutor.submit(() -> {
            Log.d(TAG, "chatSession loading");
            setIsLoading(true);
            chatSession.load();
            setIsLoading(false);
            Log.d(TAG, "chatSession loaded");
        });
    }

    private void setIsLoading(boolean loading) {
        isLoading = loading;
        runOnUiThread(() -> {
            if (!loading && voiceRecordingModule != null) {
                voiceRecordingModule.onEnabled();
            }
            updateSenderButton();
            layoutModelLoading.setVisibility(loading ? View.VISIBLE : View.GONE);
            if (getSupportActionBar() != null) {
                getSupportActionBar().setDisplayHomeAsUpEnabled(true);
                getSupportActionBar().setTitle(loading ? getString(R.string.model_loading) : modelName);
            }
        });
    }

    private void setupRecyclerView() {
        recyclerView = findViewById(R.id.recyclerView);
        recyclerView.setItemAnimator(null);
        linearLayoutManager = new LinearLayoutManager(this);
        recyclerView.setLayoutManager(linearLayoutManager);
        adapter = new ChatRecyclerViewAdapter(this, initData(), this.modelName);
        recyclerView.setAdapter(adapter);
        recyclerView.addOnScrollListener(new RecyclerView.OnScrollListener() {
            @Override
            public void onScrollStateChanged(@NonNull RecyclerView recyclerView, int newState) {
                super.onScrollStateChanged(recyclerView, newState);
            }

            @Override
            public void onScrolled(@NonNull RecyclerView recyclerView, int dx, int dy) {
                super.onScrolled(recyclerView, dx, dy);
                if (Math.abs(dy) > 0) {
                    isUserScrolling = true;
                }
            }
            public boolean isUserScrolling() {
                return isUserScrolling;
            }
        });
    }
    private void setupEditText() {
        editUserMessage = findViewById(R.id.et_message);
        editUserMessage.setOnEditorActionListener((v, actionId, event) -> {
            if ((event != null && event.getKeyCode() == KeyEvent.KEYCODE_ENTER && event.getAction() == KeyEvent.ACTION_DOWN)) {
                Log.d(TAG, "onEditorAction" + actionId + "  getAction: " + event.getAction() + "code: " + event.getKeyCode());
                sendUserMessage();
                return true;
            }
            return false;
        });
        editUserMessage.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {

            }

            @Override
            public void afterTextChanged(Editable s) {
                updateSenderButton();
                updateVoiceButtonVisibility();
            }
        });
    }

    public List<ChatDataItem> initData() {
        List<ChatDataItem> data = new ArrayList<>();
        data.add(new ChatDataItem(dateFormat.format(new Date()), ChatViewHolders.HEADER, ""));
        data.add(new ChatDataItem(dateFormat.format(new Date()), ChatViewHolders.ASSISTANT,
                getString(ModelUtils.isDiffusionModel(modelName) ?
                        R.string.model_hello_prompt_diffusion : R.string.model_hello_prompt, modelName)));
        List<ChatDataItem> savedHistory = chatSession.getSavedHistory();
        if (savedHistory != null && !savedHistory.isEmpty()) {
            data.addAll(savedHistory);
        }
        return data;
    }

    private void setupAttachmentPickerModule() {
        imageMore = findViewById(R.id.bt_plus);
        buttonSwitchVoice = findViewById(R.id.bt_switch_audio);
        if (!ModelUtils.isVisualModel(this.modelName) && !ModelUtils.isAudioModel(this.modelName)) {
            imageMore.setVisibility(View.GONE);
            return;
        }
        attachmentPickerModule = new AttachmentPickerModule(this);
        attachmentPickerModule.setOnImagePickCallback(new AttachmentPickerModule.ImagePickCallback() {
            @Override
            public void onAttachmentPicked(Uri attachmentUri, AttachmentPickerModule.AttachmentType type) {
                imageMore.setVisibility(View.GONE);
                updateVoiceButtonVisibility();
                currentUserMessage = new ChatDataItem(ChatViewHolders.USER);
                if (type == AttachmentPickerModule.AttachmentType.Audio) {
                    currentUserMessage.setAudioUri(attachmentUri);
                } else {
                    currentUserMessage.setImageUri(attachmentUri);
                }
                updateSenderButton();
            }
            @Override
            public void onAttachmentRemoved() {
                currentUserMessage = null;
                imageMore.setVisibility(View.VISIBLE);
                updateSenderButton();
                updateVoiceButtonVisibility();
            }

            @Override
            public void onAttachmentLayoutShow() {
                imageMore.setImageResource(R.drawable.ic_bottom);
            }
            @Override
            public void onAttachmentLayoutHide() {
                imageMore.setImageResource(R.drawable.ic_plus);
            }
        });
        imageMore.setOnClickListener(v -> {
            if (voiceRecordingModule  != null) {
                voiceRecordingModule.exitRecordingMode();
            }
            attachmentPickerModule.toggleAttachmentVisibility();
        });
    }

    private void updateVoiceButtonVisibility() {
        boolean visible = true;
        if (!ModelUtils.isAudioModel(modelName)) {
            visible = false;
        } else if (isGenerating) {
            visible = false;
        } else if (currentUserMessage != null) {
            visible = false;
        } else if (!TextUtils.isEmpty(editUserMessage.getText().toString())) {
            visible = false;
        }
        buttonSwitchVoice.setVisibility(visible ? View.VISIBLE : View.GONE);
    }

    private void updateSenderButton() {
        boolean enabled = true;
        if (isLoading) {
            enabled = false;
        } else if (currentUserMessage == null && TextUtils.isEmpty(editUserMessage.getText().toString())) {
            enabled = false;
        }
        if (isGenerating) {
            enabled = true;
        }
        buttonSend.setEnabled(enabled);
        buttonSend.setImageResource(!isGenerating ? R.drawable.button_send  : R.drawable.ic_stop);
    }

    private void setupVoiceRecordingModule() {
        voiceRecordingModule = new VoiceRecordingModule(this);
        voiceRecordingModule.setOnVoiceRecordingListener(new VoiceRecordingModule.VoiceRecordingListener() {
            @Override
            public void onEnterRecordingMode() {
                editUserMessage.setVisibility(View.GONE);
                hideKeyboard(editUserMessage);
                if (attachmentPickerModule != null) {
                    attachmentPickerModule.hideAttachmentLayout();
                }
            }

            @Override
            public void onLeaveRecordingMode() {
                editUserMessage.setVisibility(View.VISIBLE);
                editUserMessage.requestFocus();
                showKeyboard(editUserMessage);
            }

            @Override
            public void onRecordSuccess(float duration, String recordingFilePath) {
                ChatDataItem chatDataItem = ChatDataItem.createAudioInputData(dateFormat.format(new Date()), "", recordingFilePath, duration);
                handleSendMessage(chatDataItem);
            }

            @Override
            public void onRecordCanceled() {

            }
        });
        voiceRecordingModule.setup(isAudioModel);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_chat, menu);
        menu.findItem(R.id.show_performance_metrics)
                .setChecked(PreferenceUtils.getBoolean(this, PreferenceUtils.KEY_SHOW_PERFORMACE_METRICS, false));
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if (item.getItemId() == R.id.start_new_chat) {
            handleNewSession();
        } else if (item.getItemId() == R.id.show_performance_metrics) {
            item.setChecked(!item.isChecked());
            PreferenceUtils.setBoolean(this, PreferenceUtils.KEY_SHOW_PERFORMACE_METRICS, item.isChecked());
            adapter.notifyItemRangeChanged(0, adapter.getItemCount());
        } else if (item.getItemId() == android.R.id.home) {
            finish();
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                voiceRecordingModule.handlePermissionAllowed();
            } else {
                voiceRecordingModule.handlePermissionDenied();
            }
        }
    }

    private void handleNewSession() {
        if (!isGenerating) {
            currentUserMessage = null;
            chatSessionId = chatSession.generateNewSession();
            this.sessionName = null;
            chatExecutor.execute(() -> chatSession.reset());
            chatDataManager.deleteAllChatData(chatSessionId);
            if (adapter.reset()) {
                Toast.makeText(this, R.string.new_conversation_started, Toast.LENGTH_LONG).show();
            }
        } else {
            Toast.makeText(this, "Cannot Reset when generating", Toast.LENGTH_LONG).show();
        }
    }

    private void setIsGenerating(boolean isGenerating) {
        this.isGenerating = isGenerating;
        updateSenderButton();
        updateVoiceButtonVisibility();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (attachmentPickerModule != null && attachmentPickerModule.canHandleResult(requestCode)) {
            attachmentPickerModule.onActivityResult(requestCode, resultCode, data);
        }
    }

    private void smoothScrollToBottom() {
        Log.d(TAG, "smoothScrollToBottom");
        recyclerView.post(() -> {
            int position = adapter.getItemCount() - 1;
            recyclerView.scrollToPosition(position);
            recyclerView.post(() -> recyclerView.scrollToPosition(position));
        });
    }

    private void scrollToEnd() {
        recyclerView.postDelayed(() -> {
            int position = adapter.getItemCount() - 1;
            linearLayoutManager.scrollToPositionWithOffset(position, -9999);
        }, 100);
    }

    private void sendUserMessage() {
        if (!buttonSend.isEnabled()) {
            return;
        }
        String inputString = editUserMessage.getText().toString().trim();
        if (currentUserMessage == null) {
            currentUserMessage = new ChatDataItem(ChatViewHolders.USER);
        }
        currentUserMessage.setText(inputString);
        currentUserMessage.setTime(dateFormat.format(new Date()));
        handleSendMessage(currentUserMessage);
        currentUserMessage = null;
    }

    private void handleSendMessage(ChatDataItem userData) {
        setIsGenerating(true);
        editUserMessage.setText("");
        adapter.addItem(userData);
        addResponsePlaceholder();
        String input;
        boolean hasSessionName = !TextUtils.isEmpty(this.sessionName);
        String sessionName = null;
        if (userData.getAudioUri() != null) {
            String audioPath = attachmentPickerModule.getPathForUri(userData.getAudioUri());
            if (audioPath == null) {
                Toast.makeText(this, "Audio file not found", Toast.LENGTH_LONG).show();
                return;
            }
            if (userData.getAudioDuration() <= 0.1) {
                userData.setAudioDuration(FileUtils.getAudioDuration(audioPath));
            }
            input = String.format("<audio>%s</audio>%s", audioPath, userData.getText());
            if (!hasSessionName) {
                sessionName = "[Audio]" + userData.getText();
            }
        } else if (userData.getImageUri() != null) {
            String imagePath = attachmentPickerModule.getPathForUri(userData.getImageUri());
            if (imagePath == null) {
                Toast.makeText(this, "image file not found", Toast.LENGTH_LONG).show();
                return;
            }
            input = String.format("<img>%s</img>%s", imagePath, userData.getText());
            if (!hasSessionName) {
                sessionName = "[Image]" + userData.getText();
            }
        } else {
            input = userData.getText();
            if (!hasSessionName) {
                sessionName = userData.getText();
            }
        }
        if (!hasSessionName) {
            chatDataManager.addOrUpdateSession(chatSessionId, modelName);
            this.sessionName = sessionName.length() > 100 ? sessionName.substring(0, 100) : sessionName;
            chatDataManager.updateSessionName(this.chatSessionId, this.sessionName);
        }
        if (ModelUtils.isDiffusionModel(this.modelName)) {
            chatExecutor.execute(() -> submitRequest(input));
        } else {
            chatExecutor.execute(() -> submitRequest(input));
        }
        chatDataManager.addChatData(chatSessionId, userData);
        if (attachmentPickerModule != null) {
            attachmentPickerModule.clearInput();
        }
        smoothScrollToBottom();
        hideKeyboard(editUserMessage);
    }

    private void addResponsePlaceholder() {
        adapter.addItem(new ChatDataItem(dateFormat.format(new Date()), ChatViewHolders.ASSISTANT, ""));
        smoothScrollToBottom();
    }

    private void submitRequest(String input) {
        isUserScrolling = false;
        stopGenerating = false;
        StringBuilder stringBuilder = new StringBuilder();
        ChatDataItem chatDataItem = adapter.getRecentItem();
        HashMap<String, Object> benchMarkResult;
        if (ModelUtils.isDiffusionModel(this.modelName)) {
            String diffusionDestPath = FileUtils.generateDestDiffusionFilePath(this, chatSessionId);
            benchMarkResult = chatSession.generateDiffusion(input,  diffusionDestPath, progress-> {
                if ("100".equals(progress)) {
                    chatDataItem.setText(getString(R.string.diffusion_generated_message));
                    chatDataItem.setImageUri(Uri.parse(diffusionDestPath));
                } else {
                    chatDataItem.setText(getString(R.string.diffusion_generate_progress, progress));
                }
                runOnUiThread(() -> updateAssistantResponse(chatDataItem));
                return false;
            });
        } else {
            benchMarkResult = chatSession.generate(input, progress -> {
                if (progress != null) {
                    stringBuilder.append(progress);
                    chatDataItem.setText(stringBuilder.toString());
                    runOnUiThread(() -> updateAssistantResponse(chatDataItem));
                }
                if (stopGenerating) {
                    Log.d(TAG, "stopGenerating requeted");
                }
                return stopGenerating;
            });
        }
        Log.d(TAG, "submitRequest benchMark: " + benchMarkResult);
        HashMap<String, Object> finalBenchMarkResult = benchMarkResult;
        runOnUiThread(() -> {
            chatDataItem.setBenchmarkInfo(ModelUtils.generateBenchMarkString(finalBenchMarkResult));
            updateAssistantResponse(chatDataItem);
        });
        chatDataManager.addChatData(chatSessionId, chatDataItem);
        this.getWindow().getDecorView().getHandler().post(() -> setIsGenerating(false));
    }

    private void updateAssistantResponse(ChatDataItem chatDataItem) {
        adapter.updateRecentItem(chatDataItem);
        if (!isUserScrolling) {
            scrollToEnd();
        }
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        chatExecutor.submit(() -> {
            chatSession.reset();
            chatSession.release();
            chatExecutor.shutdownNow();
        });
    }

    @Override
    protected void onStop() {
        super.onStop();
        AudioPlayService.getInstance().destroy();
    }

    public String getSessionId() {
        return chatSessionId;
    }

    public String getModelName() {
        return modelName;
    }
}