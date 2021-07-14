package com.taobao.android.mnnapp;

import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentActivity;
import android.support.v4.app.FragmentManager;
import android.support.v4.app.FragmentStatePagerAdapter;
import android.support.v4.view.ViewPager;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.ImageView;
import android.widget.LinearLayout;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends FragmentActivity implements OnClickListener {
    //主要要继承自FragmentActivity,这样才能在初始适配器类是使用getSupportFragmentManager方法获取FragmentManager对象

    private final String TAG = "MainActivity";

    private ViewPager mViewPager;
    private List<Class> fragments;
    private ViewPagerFragmentAdapter viewPagerFragmentAdapter;
    private LinearLayout videoLayout;
    private LinearLayout imageLayout, meLayout;
    private ImageView mImageViewVideo, mImageViewImage, mImageViewMe;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.activity_main);
        initFragments();
        // 初始化控件
        initView();
        // 注册单击监听
        initEvent();
        // 初始化适配器类
        viewPagerFragmentAdapter = new ViewPagerFragmentAdapter(getSupportFragmentManager(), fragments);
        mViewPager.setAdapter(viewPagerFragmentAdapter);
        // mViewPager.setOffscreenPageLimit(1);
    }

    private void initFragments() {
        if (fragments == null) {
            fragments = new ArrayList<>();
        }
        fragments.clear();
        fragments.add(VideoFragment.class);
        fragments.add(ImageFragment.class);
        fragments.add(MeFragment.class);
    }

    private void initEvent() {
        videoLayout.setOnClickListener(this);
        imageLayout.setOnClickListener(this);
        meLayout.setOnClickListener(this);
        mImageViewVideo.setImageResource(R.drawable.video_yes);
        mViewPager.addOnPageChangeListener(new ViewPager.OnPageChangeListener() {
            // ViewPager滑动切换监听
            @Override
            public void onPageSelected(int arg0) {
                int currentItem = mViewPager.getCurrentItem();
                resetImages();
                switch (currentItem) {
                    case 0:
                        mImageViewVideo.setImageResource(R.drawable.video_yes);
                        break;
                    case 1:
                        mImageViewImage.setImageResource(R.drawable.image_yes);
                        break;
                    case 2:
                        mImageViewMe.setImageResource(R.drawable.me_yes);
                        break;
                    default:
                        break;
                }
            }

            @Override
            public void onPageScrolled(int arg0, float arg1, int arg2) {

            }

            @Override
            public void onPageScrollStateChanged(int arg0) {

            }
        });
    }

    private void initView() {
        mViewPager = findViewById(R.id.viewpager);
        videoLayout = findViewById(R.id.video);
        imageLayout = findViewById(R.id.image);
        meLayout = findViewById(R.id.me);
        mImageViewVideo = findViewById(R.id.img_video);
        mImageViewImage = findViewById(R.id.img_image);
        mImageViewMe = findViewById(R.id.img_me);
    }

    @Override
    public void onClick(View v) {
        resetImages();
        switch (v.getId()) {
            case R.id.video:
                mViewPager.setCurrentItem(0);
                mImageViewVideo.setImageResource(R.drawable.video_yes);
                break;
            case R.id.image:
                mViewPager.setCurrentItem(1);
                mImageViewImage.setImageResource(R.drawable.image_yes);

                break;
            case R.id.me:
                mViewPager.setCurrentItem(2);
                mImageViewMe.setImageResource(R.drawable.me_yes);
                break;
            default:
                break;
        }
    }

    private void resetImages() {
        //重置图片
        mImageViewVideo.setImageResource(R.drawable.video_no);
        mImageViewImage.setImageResource(R.drawable.image_no);
        mImageViewMe.setImageResource(R.drawable.me_no);
    }

    public class ViewPagerFragmentAdapter extends FragmentStatePagerAdapter {
        private List<Class> fragments;

        public ViewPagerFragmentAdapter(FragmentManager fm, List<Class> fragments) {
            super(fm);
            this.fragments = fragments;
        }

        @Override
        public Fragment getItem(int position) {
            // 返回子View对象
            try {
                // 反射加载Fragment
                // Ref: https://www.cnblogs.com/weimore/p/7466630.html
                return (Fragment) fragments.get(position).newInstance();
            }
            catch (InstantiationException e) {
                e.printStackTrace();
            }
            catch (IllegalAccessException e) {
                e.printStackTrace();
            }
            return null;
        }

        @Override
        public int getCount() {
            //返回子View的个数
            return fragments.size();
        }

        @Override
        public Object instantiateItem(ViewGroup container, int position) {
            //初始子View方法
            return super.instantiateItem(container, position);
        }

        @Override
        public void destroyItem(ViewGroup container, int position, Object object) {
            //销毁子View
            super.destroyItem(container, position, object);
        }
    }
}
