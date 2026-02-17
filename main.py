import cv2
import numpy as np
import json
import threading
import time
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.core.window import Window
from kivy.clock import Clock, mainthread
from kivy.graphics import Color, Rectangle
from plyer import android  # 安卓原生权限/屏幕捕获

# ==============================================
# 1. 全麻将规则库（覆盖所有主流玩法）
# ==============================================
ALL_MAHJONG_RULES = {
    # 基础通用
    "大众推倒胡": {"can_chow": True, "can_pong": True, "can_kong": True, "only_zimo": False, "must_drop_suit": False, "fan_limit": 0},
    "国标麻将": {"can_chow": True, "can_pong": True, "can_kong": True, "only_zimo": False, "must_drop_suit": False, "fan_limit": 8},
    # 地方特色
    "广东鸡平胡": {"can_chow": False, "can_pong": True, "can_kong": True, "only_zimo": False, "must_drop_suit": False, "mahjong_type": "guangdong"},
    "广东推倒胡": {"can_chow": True, "can_pong": True, "can_kong": True, "only_zimo": False, "must_drop_suit": False, "mahjong_type": "guangdong"},
    "四川血战到底": {"can_chow": False, "can_pong": True, "can_kong": True, "only_zimo": True, "must_drop_suit": True, "mahjong_type": "sichuan"},
    "四川血流成河": {"can_chow": False, "can_pong": True, "can_kong": True, "only_zimo": False, "must_drop_suit": True, "mahjong_type": "sichuan"},
    "长沙麻将": {"can_chow": False, "can_pong": True, "can_kong": True, "only_zimo": True, "must_drop_suit": True, "mahjong_type": "hunan"},
    "武汉红中赖子": {"can_chow": True, "can_pong": True, "can_kong": True, "only_zimo": False, "must_drop_suit": False, "mahjong_type": "hubei"},
    "杭州麻将": {"can_chow": True, "can_pong": True, "can_kong": True, "only_zimo": False, "must_drop_suit": False, "mahjong_type": "zhejiang"},
    "东北麻将": {"can_chow": True, "can_pong": True, "can_kong": True, "only_zimo": False, "must_drop_suit": False, "mahjong_type": "northeast"},
    # 特殊玩法
    "七对专用": {"can_chow": False, "can_pong": False, "can_kong": False, "only_zimo": False, "must_drop_suit": False, "only_7pairs": True},
    "十三幺专用": {"can_chow": False, "can_pong": False, "can_kong": False, "only_zimo": True, "must_drop_suit": False, "only_13orphans": True}
}

# ==============================================
# 2. 核心麻将引擎（全规则适配+对手推算）
# ==============================================
class MahjongEngine:
    def __init__(self):
        self.current_rule = "广东鸡平胡"
        self.rule_config = ALL_MAHJONG_RULES[self.current_rule]
        
        # 牌型数据
        self.self_tiles = []  # 自己手牌
        self.discarded_all = []  # 桌面所有弃牌
        self.opp_discarded = [[], [], []]  # 三家对手弃牌
        self.remaining_tiles = self._init_remaining_tiles()  # 剩余牌
        
        # 分析结果
        self.ting_list = []  # 听牌列表
        self.best_discard = "无"  # 最优出牌
        self.risk_tiles = []  # 点炮风险牌
        self.opp_ting_prob = [0.0, 0.0, 0.0]  # 对手听牌概率

    def _init_remaining_tiles(self):
        """初始化所有麻将牌（136张）"""
        tiles = []
        # 万/条/筒 1-9，各4张
        for suit in ["万", "条", "筒"]:
            for num in range(1, 10):
                tiles += [f"{num}{suit}"] * 4
        # 字牌 东/南/西/北/中/发/白，各4张
        for word in ["东", "南", "西", "北", "中", "发", "白"]:
            tiles += [word] * 4
        return tiles

    def switch_rule(self, rule_name):
        """切换麻将规则"""
        self.current_rule = rule_name
        self.rule_config = ALL_MAHJONG_RULES[rule_name]
        self.refresh_analysis()  # 切换规则后重新分析

    def update_tiles(self, self_tiles, discarded_all, opp_discarded):
        """更新牌型数据"""
        self.self_tiles = self._normalize_tiles(self_tiles)
        self.discarded_all = self._normalize_tiles(discarded_all)
        self.opp_discarded = [self._normalize_tiles(opp) for opp in opp_discarded]
        
        # 更新剩余牌
        used = self.self_tiles + self.discarded_all
        self.remaining_tiles = [t for t in self._init_remaining_tiles() if t not in used]
        
        # 实时分析
        self.refresh_analysis()

    def _normalize_tiles(self, tiles):
        """标准化牌型格式（去重、排序）"""
        if not tiles:
            return []
        return sorted(list(set([t.strip() for t in tiles if t.strip()])))

    def is_hu(self, tiles):
        """胡牌判断（适配所有规则）"""
        tile_count = len(tiles)
        # 七对专用规则
        if self.rule_config.get("only_7pairs"):
            return self._check_7_pairs(tiles)
        # 十三幺专用规则
        if self.rule_config.get("only_13orphans"):
            return self._check_13_orphans(tiles)
        # 通用3n+2规则
        if tile_count != 14:
            return False
        return self._check_3n_2(tiles.copy()) or (not self.rule_config.get("only_3n2") and self._check_7_pairs(tiles))

    def _check_3n_2(self, tiles):
        """检查3n+2结构（顺子/刻子+将牌）"""
        if not tiles:
            return True
        first_tile = tiles[0]
        # 先找将牌（对子）
        if tiles.count(first_tile) >= 2:
            temp = tiles.copy()
            temp.remove(first_tile)
            temp.remove(first_tile)
            if self._check_melds(temp):
                return True
        # 先找刻子
        if tiles.count(first_tile) >= 3:
            temp = tiles.copy()
            for _ in range(3):
                temp.remove(first_tile)
            if self._check_melds(temp):
                return True
        return False

    def _check_melds(self, tiles):
        """检查顺子/刻子组合"""
        if not tiles:
            return True
        first_tile = tiles[0]
        # 刻子
        if tiles.count(first_tile) >= 3:
            temp = tiles.copy()
            for _ in range(3):
                temp.remove(first_tile)
            if self._check_melds(temp):
                return True
        # 顺子（仅万条筒）
        if first_tile[-1] in ["万", "条", "筒"]:
            num = int(first_tile[:-1])
            suit = first_tile[-1]
            tile2 = f"{num+1}{suit}"
            tile3 = f"{num+2}{suit}"
            if tile2 in tiles and tile3 in tiles:
                temp = tiles.copy()
                temp.remove(first_tile)
                temp.remove(tile2)
                temp.remove(tile3)
                if self._check_melds(temp):
                    return True
        return False

    def _check_7_pairs(self, tiles):
        """检查七对"""
        if len(tiles) != 14:
            return False
        tile_counts = {t: tiles.count(t) for t in set(tiles)}
        return all(v == 2 for v in tile_counts.values()) and len(tile_counts) == 7

    def _check_13_orphans(self, tiles):
        """检查十三幺"""
        orphans = ["1万", "9万", "1条", "9条", "1筒", "9筒", "东", "南", "西", "北", "中", "发", "白"]
        if len(tiles) != 14:
            return False
        # 13种幺九牌各1张 + 任意1张将牌
        has_all_orphans = all(o in tiles for o in orphans)
        return has_all_orphans and any(tiles.count(o) == 2 for o in orphans)

    def calc_ting(self):
        """计算听牌列表（适配当前规则）"""
        self.ting_list = []
        if len(self.self_tiles) not in [13, 12]:  # 正常听牌前手牌数
            return []
        
        # 遍历所有剩余牌，判断摸牌后是否胡牌
        for tile in self.remaining_tiles:
            test_tiles = self.self_tiles + [tile]
            if self.is_hu(test_tiles):
                self.ting_list.append(tile)
        return self.ting_list

    def calc_best_discard(self):
        """计算最优出牌（避点炮+优先成型）"""
        if not self.self_tiles:
            return "无"
        
        # 优先级：孤张 > 边张 > 搭子 > 危险牌（反向排除）
        risk_scores = self._calc_risk_score()
        orphan_tiles = self._find_orphan_tiles()
        
        # 优先打孤张（无风险→低风险）
        if orphan_tiles:
            orphan_risk = [(t, risk_scores.get(t, 0)) for t in orphan_tiles]
            orphan_risk.sort(key=lambda x: x[1])
            return orphan_risk[0][0]
        
        # 无孤张则打低风险边张
        edge_tiles = self._find_edge_tiles()
        if edge_tiles:
            edge_risk = [(t, risk_scores.get(t, 0)) for t in edge_tiles]
            edge_risk.sort(key=lambda x: x[1])
            return edge_risk[0][0]
        
        # 最后打风险最低的牌
        all_risk = [(t, risk_scores.get(t, 0)) for t in self.self_tiles]
        all_risk.sort(key=lambda x: x[1])
        return all_risk[0][0]

    def _find_orphan_tiles(self):
        """找孤张（无搭子、无对子）"""
        orphans = []
        for tile in self.self_tiles:
            if self.self_tiles.count(tile) == 1:
                # 万条筒检查相邻牌
                if tile[-1] in ["万", "条", "筒"]:
                    num = int(tile[:-1])
                    suit = tile[-1]
                    has_near = f"{num-1}{suit}" in self.self_tiles or f"{num+1}{suit}" in self.self_tiles
                    if not has_near:
                        orphans.append(tile)
                # 字牌直接算孤张
                else:
                    orphans.append(tile)
        return orphans

    def _find_edge_tiles(self):
        """找边张（1/9的万条筒）"""
        edges = []
        for tile in self.self_tiles:
            if tile[-1] in ["万", "条", "筒"]:
                num = int(tile[:-1])
                if num in [1, 9]:
                    edges.append(tile)
        return edges

    def calc_opp_risk(self):
        """推算对手听牌概率+点炮风险"""
        self.opp_ting_prob = [0.0, 0.0, 0.0]
        self.risk_tiles = []
        
        # 分析每家对手弃牌规律
        for i, opp_d in enumerate(self.opp_discarded):
            # 弃牌少→听牌概率低；弃牌集中某花色→听牌概率高
            discard_count = len(opp_d)
            if discard_count < 5:
                self.opp_ting_prob[i] = 0.1
            elif discard_count < 10:
                self.opp_ting_prob[i] = 0.4
            else:
                # 统计弃牌花色分布
                suit_counts = {"万":0, "条":0, "筒":0, "字":0}
                for t in opp_d:
                    if t[-1] in suit_counts:
                        suit_counts[t[-1]] += 1
                    else:
                        suit_counts["字"] += 1
                # 单花色弃牌占比>70% → 听牌概率高
                max_suit = max(suit_counts.values())
                if max_suit / discard_count > 0.7:
                    self.opp_ting_prob[i] = 0.8
                else:
                    self.opp_ting_prob[i] = 0.5
        
        # 计算点炮风险（对手听牌概率高的花色→风险高）
        risk_scores = {}
        for tile in self.self_tiles:
            risk = 0.0
            for i, prob in enumerate(self.opp_ting_prob):
                # 对手弃牌少的花色→风险高
                opp_suit_discard = [t for t in self.opp_discarded[i] if t[-1] == tile[-1]]
                suit_discard_ratio = len(opp_suit_discard) / len(self.opp_discarded[i]) if self.opp_discarded[i] else 0
                risk += prob * (1 - suit_discard_ratio)
            risk_scores[tile] = risk
            if risk > 0.6:
                self.risk_tiles.append(tile)
        
        return risk_scores

    def refresh_analysis(self):
        """全量刷新分析（听牌+最优出牌+风险）"""
        self.calc_ting()
        risk_scores = self.calc_opp_risk()
        self.best_discard = self.calc_best_discard()

# ==============================================
# 3. 屏幕视觉识别模块（轻量化模板匹配）
# ==============================================
class ScreenRecognizer:
    def __init__(self, engine):
        self.engine = engine
        self.is_running = False
        self.recognize_thread = None
        self.screen_capture = None
        self.has_permission = False
        
        # 识别优化
        self.recognize_cache = {}  # 识别缓存
        self.last_recognize_time = 0  # 上次识别时间
        self.min_recognize_interval = 0.3  # 最小识别间隔（秒）
        
        # 麻将牌模板（用于识别）
        self.tile_templates = {
            f"{num}{suit}": f"{num}{suit}" for num in range(1,10) for suit in ["万", "条", "筒"]
        }
        self.tile_templates.update({word: word for word in ["东", "南", "西", "北", "中", "发", "白"]})
        
    def start_recognize(self):
        """启动屏幕识别线程"""
        self.is_running = True
        self.recognize_thread = threading.Thread(target=self._recognize_loop, daemon=True)
        self.recognize_thread.start()

    def stop_recognize(self):
        """停止识别"""
        self.is_running = False
        if self.recognize_thread:
            self.recognize_thread.join()
        if self.screen_capture:
            self.screen_capture.stop()

    def _recognize_loop(self):
        """屏幕识别循环（优化版）"""
        while self.is_running:
            try:
                # 检查识别间隔，避免过度识别
                current_time = time.time()
                if current_time - self.last_recognize_time < self.min_recognize_interval:
                    time.sleep(0.1)
                    continue
                
                # 基础版本：使用模拟数据 + 简单屏幕捕获尝试
                # 实际安卓端：调用MediaProjection捕获屏幕帧
                if android:
                    # 尝试初始化屏幕捕获
                    if not self.screen_capture:
                        self._init_screen_capture()
                
                # 获取屏幕数据（先使用模拟数据，后续替换为真实截图）
                frame = self._get_screen_data()
                
                # 生成缓存键
                cache_key = str(hash(str(frame)))
                
                # 检查缓存
                if cache_key in self.recognize_cache:
                    # 使用缓存数据
                    cached_data = self.recognize_cache[cache_key]
                    self_tiles = cached_data['self_tiles']
                    discarded_all = cached_data['discarded_all']
                    opp_discarded = cached_data['opp_discarded']
                else:
                    # 识别手牌、弃牌
                    self_tiles = self._recognize_self_tiles(frame)
                    discarded_all = self._recognize_discarded(frame)
                    opp_discarded = self._recognize_opp_discarded(frame)
                    
                    # 缓存结果
                    self.recognize_cache[cache_key] = {
                        'self_tiles': self_tiles,
                        'discarded_all': discarded_all,
                        'opp_discarded': opp_discarded,
                        'timestamp': current_time
                    }
                    
                    # 清理过期缓存
                    self._clean_cache()
                
                # 更新引擎数据
                self.engine.update_tiles(self_tiles, discarded_all, opp_discarded)
                
                # 更新最后识别时间
                self.last_recognize_time = current_time
                
                time.sleep(0.1)  # 优化刷新间隔
            except Exception as e:
                print(f"识别出错：{e}")
                time.sleep(0.5)

    def _init_screen_capture(self):
        """初始化屏幕捕获"""
        try:
            # 这里将在后续版本实现真实的屏幕捕获
            # 目前使用模拟数据
            print("尝试初始化屏幕捕获...")
            self.has_permission = True
        except Exception as e:
            print(f"屏幕捕获初始化失败：{e}")
            self.has_permission = False
    
    def _clean_cache(self):
        """清理过期缓存"""
        try:
            current_time = time.time()
            # 清理30秒前的缓存
            expired_keys = [key for key, data in self.recognize_cache.items() 
                          if current_time - data['timestamp'] > 30]
            for key in expired_keys:
                del self.recognize_cache[key]
            
            # 限制缓存大小，最多保存100个缓存项
            if len(self.recognize_cache) > 100:
                # 按时间排序，删除最早的缓存
                sorted_keys = sorted(self.recognize_cache.keys(), 
                                   key=lambda k: self.recognize_cache[k]['timestamp'])
                for key in sorted_keys[:-100]:
                    del self.recognize_cache[key]
        except Exception as e:
            print(f"缓存清理失败：{e}")

    def _get_screen_data(self):
        """获取屏幕数据（集成OpenCV）"""
        try:
            # 测试OpenCV是否正常工作
            if 'cv2' in globals():
                # 创建一个简单的测试图像
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.putText(test_image, 'OpenCV Test', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                print("OpenCV集成成功！")
        except Exception as e:
            print(f"OpenCV测试失败：{e}")
        
        # 模拟数据：真实版本将替换为屏幕截图分析
        return {"self_tiles": ["4万", "5万", "6万", "4条", "5条", "6条", "8条", "8条", "2万", "8筒", "3筒"],
                "discarded": ["1万", "3万", "7条", "9筒"],
                "opp1_discarded": ["2条", "5筒"],
                "opp2_discarded": ["7万", "8万"],
                "opp3_discarded": ["1条", "4筒"]}

    def _recognize_self_tiles(self, frame):
        """识别自己手牌"""
        return frame["self_tiles"]

    def _recognize_discarded(self, frame):
        """识别桌面所有弃牌"""
        return frame["discarded"]

    def _recognize_opp_discarded(self, frame):
        """识别三家对手弃牌"""
        return [frame["opp1_discarded"], frame["opp2_discarded"], frame["opp3_discarded"]]

# ==============================================
# 4. 悬浮窗界面（规则切换+实时提示）
# ==============================================
class MahjongFloatWindow(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 悬浮窗样式
        self.size_hint = (None, None)
        self.size = (400, 180)
        self.pos_hint = {"top": 0.95, "right": 0.98}
        with self.canvas.before:
            Color(0, 0, 0, 0.8)  # 半透明黑底
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(pos=self.update_rect, size=self.update_rect)

        # 1. 规则选择下拉框
        self.rule_spinner = Spinner(
            text="广东鸡平胡",
            values=list(ALL_MAHJONG_RULES.keys()),
            size_hint=(1, 0.2),
            pos_hint={"x":0, "top":1},
            font_size=14
        )
        self.rule_spinner.bind(text=self.on_rule_change)
        self.add_widget(self.rule_spinner)

        # 2. 听牌提示
        self.ting_label = Label(
            text="听牌：无",
            size_hint=(1, 0.2),
            pos_hint={"x":0, "top":0.8},
            font_size=14,
            color=(0, 1, 0, 1)  # 绿色
        )
        self.add_widget(self.ting_label)

        # 3. 最优出牌提示
        self.discard_label = Label(
            text="最优打：无",
            size_hint=(1, 0.2),
            pos_hint={"x":0, "top":0.6},
            font_size=14,
            color=(1, 0, 0, 1)  # 红色
        )
        self.add_widget(self.discard_label)

        # 4. 点炮风险提示
        self.risk_label = Label(
            text="风险牌：无",
            size_hint=(1, 0.2),
            pos_hint={"x":0, "top":0.4},
            font_size=14,
            color=(1, 1, 0, 1)  # 黄色
        )
        self.add_widget(self.risk_label)

        # 5. 对手听牌概率
        self.opp_label = Label(
            text="对手听牌概率：0% 0% 0%",
            size_hint=(1, 0.2),
            pos_hint={"x":0, "top":0.2},
            font_size=12,
            color=(1, 1, 1, 1)  # 白色
        )
        self.add_widget(self.opp_label)

        # 初始化引擎和识别器
        self.engine = MahjongEngine()
        self.recognizer = ScreenRecognizer(self.engine)
        
        # 启动识别和UI刷新
        self.recognizer.start_recognize()
        Clock.schedule_interval(self.update_ui, 0.5)

    def update_rect(self, *args):
        """更新悬浮窗背景"""
        self.rect.pos = self.pos
        self.rect.size = self.size

    def on_rule_change(self, spinner, text):
        """切换麻将规则"""
        self.engine.switch_rule(text)

    @mainthread
    def update_ui(self, dt):
        """实时更新UI（主线程）"""
        # 听牌提示
        ting_text = "听牌：" + (",".join(self.engine.ting_list) if self.engine.ting_list else "未听")
        self.ting_label.text = ting_text
        
        # 最优出牌
        self.discard_label.text = f"最优打：{self.engine.best_discard}"
        
        # 风险牌
        risk_text = "风险牌：" + (",".join(self.engine.risk_tiles) if self.engine.risk_tiles else "无")
        self.risk_label.text = risk_text
        
        # 对手听牌概率
        opp_text = "对手听牌概率：" + " ".join([f"{int(p*100)}%" for p in self.engine.opp_ting_prob])
        self.opp_label.text = opp_text

    def on_stop(self):
        """APP停止时清理"""
        self.recognizer.stop_recognize()

# ==============================================
# 5. APP入口
# ==============================================
class MahjongAssistantApp(App):
    def build(self):
        # 设备兼容性检查
        self.check_device_compatibility()
        
        # 安卓悬浮窗配置
        if android:
            # 请求必要权限
            self.request_android_permissions()
        self.title = "全麻将通用视觉辅助"
        return MahjongFloatWindow()
    
    def check_device_compatibility(self):
        """检查设备兼容性"""
        try:
            # 检查Python版本
            import sys
            print(f"Python版本: {sys.version}")
            
            # 检查Kivy版本
            import kivy
            print(f"Kivy版本: {kivy.__version__}")
            
            # 检查Numpy版本
            import numpy
            print(f"NumPy版本: {numpy.__version__}")
            
            # 检查OpenCV版本
            if 'cv2' in globals():
                print(f"OpenCV版本: {cv2.__version__}")
            
            print("设备兼容性检查完成")
        except Exception as e:
            print(f"兼容性检查失败：{e}")
    
    def request_android_permissions(self):
        """请求Android权限"""
        try:
            # 核心权限列表
            permissions = [
                "SYSTEM_ALERT_WINDOW",  # 悬浮窗权限
                "RECORD_AUDIO",  # 录音权限（用于屏幕捕获）
                "READ_EXTERNAL_STORAGE",  # 读取存储
                "WRITE_EXTERNAL_STORAGE",  # 写入存储
                "MEDIA_CONTENT_CONTROL",  # 媒体控制
                "CAPTURE_VIDEO_OUTPUT",  # 视频捕获
                "CAPTURE_AUDIO_OUTPUT"  # 音频捕获
            ]
            
            # 请求权限
            android.permissions.request_permissions(permissions)
            print("权限请求已发送")
            
            # 检查悬浮窗权限
            self.check_overlay_permission()
        except Exception as e:
            print(f"权限请求失败：{e}")
    
    def check_overlay_permission(self):
        """检查悬浮窗权限"""
        try:
            # 这里将在后续版本实现完整的悬浮窗权限检查
            # 目前打印提示信息
            print("请确保授予应用悬浮窗权限和屏幕捕获权限")
            print("在系统设置中找到该应用，开启'显示在其他应用上层'权限")
        except Exception as e:
            print(f"悬浮窗权限检查失败：{e}")

    def on_stop(self):
        """APP退出"""
        self.root.on_stop()

if __name__ == "__main__":
    MahjongAssistantApp().run()