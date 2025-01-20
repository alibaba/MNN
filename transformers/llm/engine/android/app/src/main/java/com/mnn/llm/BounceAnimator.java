package com.mnn.llm;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.view.View;
import android.view.animation.BounceInterpolator;

public class BounceAnimator {
    public static void startAdvancedBounce(View view) {
        view.animate()
                .scaleX(1.2f)
                .scaleY(1.2f)
                .alpha(1f)
                .setDuration(600)
                .setInterpolator(new BounceInterpolator())
                .setListener(new AnimatorListenerAdapter() {
                    @Override
                    public void onAnimationEnd(Animator animation) {
                        super.onAnimationEnd(animation);
                        if (view.getTag(R.id.anim_running) != null) {
                            view.animate()
                                    .scaleX(0.8f)
                                    .scaleY(0.8f)
                                    .alpha(0.7f)
                                    .setDuration(600)
                                    .setInterpolator(new BounceInterpolator())
                                    .start();
                        }
                    }
                })
                .start();
        view.setTag(R.id.anim_running, true);
    }
    public static void stopBounce(View view) {
        view.setTag(R.id.anim_running, null);
        view.animate().cancel();
        view.setScaleX(1f);
        view.setScaleY(1f);
        view.setAlpha(1f);
    }
}
