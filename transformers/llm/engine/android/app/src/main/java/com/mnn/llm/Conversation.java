package com.mnn.llm;

import android.Manifest;
import android.app.Notification;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.provider.MediaStore;
import android.provider.Settings;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.LayoutInflater;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;
import android.content.Intent;
import android.content.Context;
import android.widget.TextView;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.io.File;
import java.io.IOException;

public class Conversation extends BaseActivity {

    private static final int REQUEST_CODE = 100;
    private RecyclerView mRecyclerView;
    private ConversationRecyclerView mAdapter;
    private ImageView imagePreview;
    private Uri imageUri;
    private EditText text;
    private Button send;
    private DateFormat mDateFormat;
    private Chat mChat;
    private String selectedImagePath;
    private final ScheduledExecutorService executor = Executors.newScheduledThreadPool(1);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_conversation);

        mChat = (Chat) getIntent().getSerializableExtra("chat");
        mDateFormat = new SimpleDateFormat("hh:mm aa");

        setupToolbarWithUpNav(R.id.toolbar, "mnn-llm", R.drawable.ic_action_back);

        mRecyclerView = findViewById(R.id.recyclerView);
        mRecyclerView.setLayoutManager(new LinearLayoutManager(this));
        mAdapter = new ConversationRecyclerView(this, initData());
        mRecyclerView.setAdapter(mAdapter);

        text = findViewById(R.id.et_message);
        text.setOnClickListener(view -> smoothScrollToBottom());

        findViewById(R.id.bt_select_image).setOnClickListener(view -> selectImage());
        send = findViewById(R.id.bt_send);
        send.setOnClickListener(view -> handleSendClick());
        imagePreview = findViewById(R.id.image_preview);
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_CODE);
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                intent.setData(Uri.parse("package:" + getPackageName()));
                startActivityForResult(intent, REQUEST_CODE);
            }
        }
        TextView promptBubble1 = findViewById(R.id.promptBubble1);
        promptBubble1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                promptBubble1.setBackgroundResource(R.drawable.bubble_background_onclick);
                new Handler().postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        promptBubble1.setBackgroundResource(R.drawable.bubble_background);
                    }
                }, 200);
                EditText inputField = findViewById(R.id.et_message);
                inputField.setText(promptBubble1.getText().toString());
            }
        });
        TextView promptBubble2 = findViewById(R.id.promptBubble2);
        promptBubble2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                promptBubble2.setBackgroundResource(R.drawable.bubble_background_onclick);
                new Handler().postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        promptBubble2.setBackgroundResource(R.drawable.bubble_background);
                    }
                }, 200);
                EditText inputField = findViewById(R.id.et_message);
                inputField.setText(promptBubble2.getText().toString());
            }
        });

        String imgDir = new File(getFilesDir(), "image").getAbsolutePath();
        ImageView img1 = findViewById(R.id.examplePicture1);
        ImageView img2 = findViewById(R.id.examplePicture2);
        ImageView img3 = findViewById(R.id.examplePicture3);
        File imgFile1 = new File(imgDir,"1.png");
        if (imgFile1.exists()) {
            Bitmap myBitmap = BitmapFactory.decodeFile(imgFile1.getAbsolutePath());
            img1.setImageBitmap(myBitmap);
            img1.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    imageUri = Uri.fromFile(imgFile1);
                    selectedImagePath = imgFile1.getAbsolutePath();
                    imagePreview.setImageURI(imageUri);
                    imagePreview.setVisibility(View.VISIBLE);
                }
            });
        }
        File imgFile2 = new File(imgDir,"2.png");
        if (imgFile2.exists()) {
            Bitmap myBitmap = BitmapFactory.decodeFile(imgFile2.getAbsolutePath());
            img2.setImageBitmap(myBitmap);
            img2.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    imageUri = Uri.fromFile(imgFile2);
                    selectedImagePath = imgFile2.getAbsolutePath();
                    imagePreview.setImageURI(imageUri);
                    imagePreview.setVisibility(View.VISIBLE);
                }
            });
        }
        File imgFile3 = new File(imgDir,"3.png");
        if (imgFile3.exists()) {
            Bitmap myBitmap = BitmapFactory.decodeFile(imgFile3.getAbsolutePath());
            img3.setImageBitmap(myBitmap);
            img3.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    imageUri = Uri.fromFile(imgFile3);
                    selectedImagePath = imgFile3.getAbsolutePath();
                    imagePreview.setImageURI(imageUri);
                    imagePreview.setVisibility(View.VISIBLE);
                }
            });
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == RESULT_OK && data != null) {
            imageUri = data.getData();
            selectedImagePath = getPathFromUri(imageUri);
            imagePreview.setImageURI(imageUri);
            imagePreview.setVisibility(View.VISIBLE);
        }
    }

    private String getPathFromUri(Uri uri) {
        String[] projection = {MediaStore.Images.Media.DATA};
        Cursor cursor = getContentResolver().query(uri, projection, null, null, null);
        if (cursor != null) {
            int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
            cursor.moveToFirst();
            String path = cursor.getString(column_index);
            cursor.close();
            return path;
        }
        return null;
    }

    private void selectImage() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), 1);
    }

    private void smoothScrollToBottom() {
        mRecyclerView.post(() -> {
            int position = mAdapter.getItemCount() - 1;
            mRecyclerView.scrollToPosition(position);
            // 通过延迟来确保最后一个 Item 被完全展示
            mRecyclerView.post(() -> mRecyclerView.scrollToPosition(position));
        });
    }

    private void handleSendClick() {
        if (imagePreview.getVisibility() == View.VISIBLE) {
            imagePreview.setVisibility(View.GONE);
        } else {
            imageUri = null;
        }
        String inputString = text.getText().toString().trim();
        if (!inputString.isEmpty() || selectedImagePath != null) {
            String combinedInput = inputString;
            if (selectedImagePath != null) {
                combinedInput = String.format("<img>%s</img>%s", selectedImagePath, combinedInput);
            }
            addUserMessage(inputString, imageUri);
            text.setText("");
            selectedImagePath = null;

            if (inputString.equals("/reset")) {
                mChat.Reset();
            } else {
                addBotResponsePlaceholder();
                String finalCombinedInput = combinedInput;
                executor.execute(() -> handleBotResponse(finalCombinedInput));
            }
        }
    }

    private void addUserMessage(String message, Uri image) {
        ChatData userData = new ChatData(mDateFormat.format(new Date()), "2", message);
        userData.setImageUri(image);
        mAdapter.addItem(userData);
        smoothScrollToBottom();
    }

    private void addBotResponsePlaceholder() {
        mAdapter.addItem(new ChatData(mDateFormat.format(new Date()), "1", ""));
        smoothScrollToBottom();
    }

    private void handleBotResponse(String input) {
        mChat.Submit(input);
        String lastResponse = "";
        while (!lastResponse.contains("<eop>")) {
            try {
                Thread.sleep(50);
                String response = new String(mChat.Response());
                if (!response.equals(lastResponse)) {
                    lastResponse = response;
                    runOnUiThread(() -> updateBotResponse(response.replaceFirst("<eop>", "")));
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        mChat.Done();
    }

    private void updateBotResponse(String responseText) {
        mAdapter.updateRecentItem(new ChatData(mDateFormat.format(new Date()), "1", responseText));
    }

    public List<ChatData> initData() {
        List<ChatData> data = new ArrayList<>();
        data.add(new ChatData(new SimpleDateFormat("yyyy-MM-dd").format(new Date()), "0", ""));
        data.add(new ChatData(mDateFormat.format(new Date()), "1", "Hello，I'm mnn-llm，you can ask me anything。"));
        return data;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_userphoto, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        Toast.makeText(getBaseContext(), "Memory Clear", Toast.LENGTH_SHORT).show();
        mChat.Reset();
        return true;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdownNow();
    }

    private static class ChatData {
        private String time, type, text;
        private Uri imageUri;

        public ChatData(String time, String type, String text) {
            this.time = time;
            this.type = type;
            this.text = text;
        }

        public String getTime() { return time; }
        public String getType() { return type; }
        public String getText() { return text; }
        public Uri getImageUri() { return imageUri; }
        public void setImageUri(Uri image) { imageUri = image; }
    }

    private static class ConversationRecyclerView extends RecyclerView.Adapter<RecyclerView.ViewHolder> {
        private final List<ChatData> items;
        private final Context mContext;

        private static final int DATE = 0, YOU = 1, ME = 2;

        public ConversationRecyclerView(Context context, List<ChatData> items) {
            this.mContext = context;
            this.items = items;
        }

        @Override
        public int getItemCount() { return items.size(); }

        @Override
        public int getItemViewType(int position) {
            switch (items.get(position).getType()) {
                case "0": return DATE;
                case "1": return YOU;
                case "2": return ME;
                default: return -1;
            }
        }

        @Override
        public RecyclerView.ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
            LayoutInflater inflater = LayoutInflater.from(parent.getContext());
            View view;
            switch (viewType) {
                case DATE: view = inflater.inflate(R.layout.layout_holder_date, parent, false); return new ViewHolder(view, true);
                case YOU: view = inflater.inflate(R.layout.layout_holder_you, parent, false); return new ViewHolder(view, false);
                case ME: default: view = inflater.inflate(R.layout.layout_holder_me, parent, false); return new ViewHolder(view, false);
            }
        }

        @Override
        public void onBindViewHolder(RecyclerView.ViewHolder holder, int position) {
            ViewHolder viewHolder = (ViewHolder) holder;
            ChatData chatData = items.get(position);
            viewHolder.bind(chatData);
            if (viewHolder.imageView == null) {
                return;
            }
            if (chatData.getImageUri() != null) {
                viewHolder.imageView.setVisibility(View.VISIBLE);
                viewHolder.imageView.setImageURI(chatData.getImageUri());
            } else {
                viewHolder.imageView.setVisibility(View.GONE);
            }
        }

        public void addItem(ChatData item) {
            items.add(item);
            notifyItemInserted(items.size() - 1);
        }

        public void updateRecentItem(ChatData item) {
            items.set(items.size() - 1, item);
            notifyItemChanged(items.size() - 1);
        }

        private static class ViewHolder extends RecyclerView.ViewHolder {
            public ImageView imageView;
            private TextView time, chatText;

            ViewHolder(View view, boolean isDate) {
                super(view);
                if (isDate) {
                    time = view.findViewById(R.id.tv_date);
                } else {
                    time = view.findViewById(R.id.tv_time);
                    chatText = view.findViewById(R.id.tv_chat_text);
                    imageView = view.findViewById(R.id.tv_chat_image);
                }
            }

            void bind(ChatData data) {
                if (chatText != null) {
                    chatText.setText(data.getText());
                }
                if (time != null) {
                    time.setText(data.getTime());
                }
            }
        }
    }
}