package com.mnn.llm;

import android.Manifest;
import android.app.Notification;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.provider.Settings;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.LayoutInflater;
import android.view.ViewTreeObserver;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.Toast;
import android.content.Intent;
import android.content.Context;
import android.widget.TextView;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.appbar.AppBarLayout;

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
    private LinearLayoutManager mLinearLayoutManager;
    private ConversationRecyclerView mAdapter;
    private ImageButton imageButton;
    private boolean isPreview;
    private Uri imageUri;
    private EditText text;
    private ImageView send;
    private DateFormat mDateFormat;
    private Chat mChat;
    private String selectedImagePath;
    private ImageButton toggleButton;
    private ConstraintLayout targetLayout;
    private boolean isLayoutVisible = true;
    private ImageButton clear;
    private ImageButton closeButton;
    private ImageButton menuButton;
    private TextView speed;
    private boolean isGenerating = false;
    private final ScheduledExecutorService executor = Executors.newScheduledThreadPool(1);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_conversation);

        mChat = (Chat) getIntent().getSerializableExtra("chat");
        mDateFormat = new SimpleDateFormat("hh:mm aa");

        mRecyclerView = findViewById(R.id.recyclerView);
        mLinearLayoutManager = new LinearLayoutManager(this);
        mRecyclerView.setLayoutManager(mLinearLayoutManager);
        mAdapter = new ConversationRecyclerView(this, initData());
        mRecyclerView.setAdapter(mAdapter);

        closeButton = findViewById(R.id.close);
        closeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                AppBarLayout layout = findViewById(R.id.toolbar);
                layout.setVisibility(View.GONE);
            }
        });

        speed = findViewById(R.id.speed);
        speed.setText("0.0");

        clear = findViewById(R.id.clear);
        clear.setVisibility(View.VISIBLE);

        clear.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                clear.setBackgroundResource(R.drawable.refresh_square_onclick);
                new Handler().postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        clear.setBackgroundResource(R.drawable.refresh_square);
                    }
                }, 200);
                runOnUiThread(()->clearMessage());
            }
        });

        closeButton = findViewById(R.id.close);
        closeButton.setVisibility(View.VISIBLE);
        closeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                closeButton.setBackgroundResource(R.drawable.close_square_com_onclick);
                new Handler().postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        closeButton.setBackgroundResource(R.drawable.close_square_com);
                    }
                }, 200);
                runOnUiThread(()->closeBar());
            }
        });

        menuButton = findViewById(R.id.displayBar);
        menuButton.setVisibility(View.GONE);
        menuButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                menuButton.setBackgroundResource(R.drawable.menu_alt_2_svgrepo_com_onclick);
                new Handler().postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        menuButton.setBackgroundResource(R.drawable.menu_alt_2_svgrepo_com);
                    }
                }, 200);
                runOnUiThread(()->openBar());
            }
        });


        text = findViewById(R.id.et_message);
        text.setOnClickListener(view -> smoothScrollToBottom());
        text.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {

            }

            @Override
            public void afterTextChanged(Editable s) {
                updateSenderButton();
            }
        });

        imageButton = findViewById(R.id.bt_select_image);
        imageButton.setOnClickListener(view -> selectImage());
        send = findViewById(R.id.bt_send);
        send.setEnabled(false);
        send.setOnClickListener(view -> handleSendClick());
        isPreview = false;

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

        ImageView img1 = findViewById(R.id.examplePicture1);
        ImageView img2 = findViewById(R.id.examplePicture2);
        ImageView img3 = findViewById(R.id.examplePicture3);
        ImageView img4 = findViewById(R.id.examplePicture4);
        ImageView img5 = findViewById(R.id.examplePicture5);
        ConstraintLayout promptBubble1 = findViewById(R.id.promptBubble1);
        TextView text1 = findViewById(R.id.textView1);
        initPredefinePrompt(promptBubble1, img1, text1, 1);
        ConstraintLayout promptBubble2 = findViewById(R.id.promptBubble2);
        TextView text2 = findViewById(R.id.textView2);
        initPredefinePrompt(promptBubble2, img2, text2, 2);
        ConstraintLayout promptBubble3 = findViewById(R.id.promptBubble3);
        TextView text3 = findViewById(R.id.textView3);
        initPredefinePrompt(promptBubble3, img3, text3, 3);
        ConstraintLayout promptBubble4 = findViewById(R.id.promptBubble4);
        TextView text4 = findViewById(R.id.textView4);
        initPredefinePrompt(promptBubble4, img4,text4, 4);
        ConstraintLayout promptBubble5 = findViewById(R.id.promptBubble5);
        TextView text5 = findViewById(R.id.textView5);
        initPredefinePrompt(promptBubble5, img5, text5, 5);

        toggleButton = findViewById(R.id.toggleButton);
        targetLayout = findViewById(R.id.predefineView);
        targetLayout.setVisibility(View.GONE);

        toggleButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isLayoutVisible) {
                    targetLayout.setVisibility(View.GONE);
                    toggleButton.setVisibility(View.VISIBLE);
                    isLayoutVisible = false;
                } else {
                    targetLayout.setVisibility(View.VISIBLE);
                    toggleButton.setVisibility(View.GONE);
                    isLayoutVisible = true;
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == RESULT_OK && data != null) {
            imageUri = data.getData();
            selectedImagePath = getPathFromUri(imageUri);
            if(selectedImagePath == null){
                imageUri = null;
                isPreview = false;
            }
            else{
                imageButton.setImageURI(imageUri);
                isPreview = true;
            }
        }
    }


    private void initPredefinePrompt(ConstraintLayout promptView, ImageView imgView, TextView textView, int index) {
        String imgDir = new File(getFilesDir(), "image").getAbsolutePath();
        File imgFile = new File(imgDir,String.format("%d.png", index));
        if (imgFile.exists()) {
            Bitmap myBitmap = BitmapFactory.decodeFile(imgFile.getAbsolutePath());
            imgView.setImageBitmap(myBitmap);
        }
        promptView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                promptView.setBackgroundResource(R.drawable.bubble_background_onclick);
                new Handler().postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        promptView.setBackgroundResource(R.drawable.bubble_background);
                    }
                }, 200);
                imageUri = Uri.fromFile(imgFile);
                selectedImagePath = imgFile.getAbsolutePath();
                imageButton.setImageURI(imageUri);
                isPreview = true;
                EditText inputField = findViewById(R.id.et_message);

                inputField.setText(textView.getText().toString());
                targetLayout.setVisibility(View.GONE);
                toggleButton.setVisibility(View.VISIBLE);
                isLayoutVisible = false;
            }
        });
    }
    private void updateSenderButton() {
        boolean enabled = true;
        if (TextUtils.isEmpty(text.getText().toString())) {
            enabled = false;
        }
        send.setEnabled(enabled);
    }

    private String getPathFromUri(Uri uri) {
        if (DocumentsContract.isDocumentUri(this, uri)) {
            String docId = DocumentsContract.getDocumentId(uri);
            String[] split = docId.split(":");
            String type = split[0];

            if ("image".equalsIgnoreCase(type)) {
                Uri contentUri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI;
                String selection = "_id=?";
                String[] selectionArgs = new String[]{split[1]};

                return getDataColumn(this, contentUri, selection, selectionArgs);
            }
        } else if ("content".equalsIgnoreCase(uri.getScheme())) {
            return getDataColumn(this, uri, null, null);
        } else if ("file".equalsIgnoreCase(uri.getScheme())) {
            return uri.getPath();
        }
        return null;
    }
    private String getDataColumn(Context context, Uri uri, String selection, String[] selectionArgs) {
        Cursor cursor = null;
        final String column = "_data";
        final String[] projection = {column};

        try {
            cursor = context.getContentResolver().query(uri, projection, selection, selectionArgs, null);
            if (cursor != null && cursor.moveToFirst()) {
                final int column_index = cursor.getColumnIndexOrThrow(column);
                return cursor.getString(column_index);
            }
        } finally {
            if (cursor != null) {
                cursor.close();
            }
        }
        return null;
    }
    private void selectImage() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), 1);
    }

    private void smoothScrollToBottom() {
        mRecyclerView.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
            @Override
            public void onGlobalLayout() {
                mRecyclerView.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                int position = mAdapter.getItemCount() - 1;
                mLinearLayoutManager.scrollToPositionWithOffset(position, -9999);
            }
        });
    }

    private void closeBar(){
        RelativeLayout layout = findViewById(R.id.brandInfo);
        layout.setVisibility(View.GONE);
        closeButton.setVisibility(View.GONE);
        clear.setVisibility(View.GONE);
        menuButton.setVisibility(View.VISIBLE);
    }
    private void openBar(){
        RelativeLayout layout = findViewById(R.id.brandInfo);
        layout.setVisibility(View.VISIBLE);
        closeButton.setVisibility(View.VISIBLE);
        clear.setVisibility(View.VISIBLE);
        menuButton.setVisibility(View.GONE);
    }
    /*private void scrollToBottom() {
        mRecyclerView.post(() -> mRecyclerView.smoothScrollToPosition(mAdapter.getItemCount() - 1));
    }*/

    private void toggleKeyboard(View view, boolean show) {
        InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
        if (show) {
            imm.showSoftInput(view, InputMethodManager.SHOW_IMPLICIT);
        } else {
            imm.hideSoftInputFromWindow(view.getWindowToken(), 0);
        }
    }

    private void handleSendClick() {
        if (!send.isEnabled()) {
            return;
        }
        toggleKeyboard(send,false);
        if (isPreview) {
            Drawable drawable = getResources().getDrawable(R.drawable.ic_image);
            imageButton.setImageDrawable(drawable);
            isPreview = false;
        }
        String inputString = text.getText().toString().trim();
        if (!inputString.isEmpty() || selectedImagePath != null) {
            String combinedInput = inputString;
            if (selectedImagePath != null) {
                mChat.Reset();
                combinedInput = String.format("<img>%s</img>%s", selectedImagePath, combinedInput);
            }
            addUserMessage(inputString  , imageUri);
            text.setText("");
            selectedImagePath = null;
            imageUri = null;

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

    private void clearMessage() {
        if(isGenerating){
            return;
        }
        mChat.Reset();
        mAdapter.clearItem();
        smoothScrollToBottom();
    }

    private void handleBotResponse(String input) {
        isGenerating = true;
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
        float total_speed = mChat.Done();
        isGenerating = false;
        String formattedValue = Integer.toString((int)total_speed);
        runOnUiThread(() -> speed.setText(formattedValue));
    }

    private void updateBotResponse(String responseText) {
        mAdapter.updateRecentItem(new ChatData(mDateFormat.format(new Date()), "1", responseText));
        smoothScrollToBottom();
    }

    public List<ChatData> initData() {
        List<ChatData> data = new ArrayList<>();
        data.add(new ChatData(mDateFormat.format(new Date()), "1", "Hello, I'm mnn vision assistant, you can ask me anything."));
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

        public void clearItem() {
            if (items.isEmpty()) {
                return;
            }
            ChatData firstItem = items.get(0);
            items.clear();
            items.add(firstItem);
            notifyDataSetChanged();

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
                    chatText = view.findViewById(R.id.tv_chat_text);
                    imageView = view.findViewById(R.id.tv_chat_image);
                }
            }

            void bind(ChatData data) {
                if (chatText != null) {
                    chatText.setText(data.getText());
                }
            }
        }
    }
}