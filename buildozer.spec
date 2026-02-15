[app]
title = 全麻将通用辅助
package.name = mahjongassist
package.domain = org.mahjong.assist
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json
version = 1.0.0

# 安卓核心配置
android.api = 33
android.ndk = 25b
android.sdk = 24
android.archs = arm64-v8a, armeabi-v7a
android.ndk_path = ~/Android/Sdk/ndk/25.2.9519653
android.sdk_path = ~/Android/Sdk

# 关键权限（悬浮窗+屏幕捕获）
android.permissions = SYSTEM_ALERT_WINDOW, RECORD_AUDIO, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, MEDIA_CONTENT_CONTROL, CAPTURE_VIDEO_OUTPUT, CAPTURE_AUDIO_OUTPUT, INTERNET
android.activity_always_on_top = True  # 悬浮窗置顶
android.enable_androidx = True
android.use_aapt2 = True

# 依赖（重新集成OpenCV，使用验证过的方法）
requirements = python3,kivy,plyer,numpy,opencv-python==4.8.0.74
android.add_ndk_modules = opencv

# 打包优化
p4a.local_recipes = ./recipes
android.add_assets = ./assets
android.icon = ./icon.png  # 可选：添加APP图标

# 使用国内镜像源
p4a.url = https://gitee.com/mirrors/python-for-android.git
p4a.branch = master

[buildozer]
log_level = 2
warn_on_root = 1