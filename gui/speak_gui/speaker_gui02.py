import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QRadioButton, QCheckBox, QScrollArea, QButtonGroup
)
from PyQt6.QtGui import QPixmap, QFont, QIcon, QPainter, QPainterPath, QColor
from PyQt6.QtCore import Qt, QSize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICONS_DIR = os.path.join(BASE_DIR, "icons")

DEFAULT_AVATAR_CONTENT_RATIO = 0.875


def create_rounded_pixmap(source_pixmap_path, target_size_int, inner_content_ratio=DEFAULT_AVATAR_CONTENT_RATIO):
    target_qsize = QSize(target_size_int, target_size_int)
    source_pixmap = QPixmap(source_pixmap_path)

    if source_pixmap.isNull():
        placeholder = QPixmap(target_qsize)
        placeholder.fill(Qt.GlobalColor.transparent)  # PyQt6 style
        p = QPainter(placeholder)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)  # PyQt6 style
        p.setBrush(Qt.GlobalColor.lightGray)  # PyQt6 style
        p.setPen(Qt.PenStyle.NoPen)  # PyQt6 style
        p.drawEllipse(0, 0, target_qsize.width(), target_qsize.height())
        p.setPen(Qt.GlobalColor.darkGray)  # PyQt6 style
        font = QFont()
        font.setPointSize(target_size_int // 2)
        p.setFont(font)
        p.drawText(placeholder.rect(), Qt.AlignmentFlag.AlignCenter, "?")  # PyQt6 style
        p.end()
        print(f"警告 (create_rounded_pixmap): 图片 {source_pixmap_path} 未找到或无法加载。使用占位符。")
        return placeholder

    source_content_dim = min(source_pixmap.width(), source_pixmap.height())
    actual_content_diameter_in_source = source_content_dim * inner_content_ratio

    if actual_content_diameter_in_source <= 0:
        print(f"警告 (create_rounded_pixmap): 无效的 inner_content_ratio for {source_pixmap_path}. 使用默认裁剪。")
        rounded_pixmap_fallback = QPixmap(target_qsize)
        rounded_pixmap_fallback.fill(Qt.GlobalColor.transparent)  # PyQt6 style
        painter_fb = QPainter(rounded_pixmap_fallback)
        painter_fb.setRenderHint(QPainter.RenderHint.Antialiasing)  # PyQt6 style
        clip_path_fb = QPainterPath()
        clip_path_fb.addEllipse(0, 0, target_qsize.width(), target_qsize.height())
        painter_fb.setClipPath(clip_path_fb)
        scaled_source_fb = source_pixmap.scaled(target_qsize, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                                Qt.TransformationMode.SmoothTransformation)  # PyQt6 style
        x_offset_fb = (target_qsize.width() - scaled_source_fb.width()) / 2
        y_offset_fb = (target_qsize.height() - scaled_source_fb.height()) / 2
        painter_fb.drawPixmap(int(x_offset_fb), int(y_offset_fb), scaled_source_fb)
        painter_fb.end()
        return rounded_pixmap_fallback

    scale_factor = target_size_int / actual_content_diameter_in_source
    scaled_source_width = source_pixmap.width() * scale_factor
    scaled_source_height = source_pixmap.height() * scale_factor
    scaled_source = source_pixmap.scaled(
        QSize(int(round(scaled_source_width)), int(round(scaled_source_height))),
        Qt.AspectRatioMode.KeepAspectRatio,  # PyQt6 style
        Qt.TransformationMode.SmoothTransformation  # PyQt6 style
    )

    content_tl_x_in_scaled = (scaled_source.width() - target_size_int) / 2.0
    content_tl_y_in_scaled = (scaled_source.height() - target_size_int) / 2.0
    draw_x = -content_tl_x_in_scaled
    draw_y = -content_tl_y_in_scaled

    rounded_pixmap = QPixmap(target_qsize)
    rounded_pixmap.fill(Qt.GlobalColor.transparent)  # PyQt6 style
    painter = QPainter(rounded_pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)  # PyQt6 style
    clip_path = QPainterPath()
    clip_path.addEllipse(0, 0, target_qsize.width(), target_qsize.height())
    painter.setClipPath(clip_path)
    painter.drawPixmap(int(round(draw_x)), int(round(draw_y)), scaled_source)
    painter.end()

    return rounded_pixmap


class VoiceAssistantUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("VoiceAssistantWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)  # MODIFIED for PyQt6

        self.setWindowTitle("智能语音助手")
        self.setGeometry(100, 100, 1000, 600)

        self.is_listening = False
        self.mic_icon_loaded = False
        self.mic_icon_path = os.path.join(ICONS_DIR, "voice.png")
        self.mic_q_icon = QIcon(self.mic_icon_path)
        self.mic_icon_size = QSize(50, 50)

        self.main_background_pixmap = None
        background_image_path_for_paint = os.path.join(ICONS_DIR, "background.png")
        if os.path.exists(background_image_path_for_paint):
            self.main_background_pixmap = QPixmap(background_image_path_for_paint)
            if self.main_background_pixmap.isNull():
                print(
                    f"警告 (VoiceAssistantUI.__init__): QPixmap未能从 {background_image_path_for_paint} 加载背景图片。")
                self.main_background_pixmap = None
        else:
            print(f"警告 (VoiceAssistantUI.__init__): 背景图片文件 {background_image_path_for_paint} 未找到。")

        self.init_ui()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            if self.main_background_pixmap and not self.main_background_pixmap.isNull():
                scaled_pixmap = self.main_background_pixmap.scaled(
                    self.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation  # PyQt6 style
                )
                x_offset = (self.width() - scaled_pixmap.width()) / 2
                y_offset = (self.height() - scaled_pixmap.height()) / 2
                painter.drawPixmap(int(x_offset), int(y_offset), scaled_pixmap)
            else:
                painter.fillRect(self.rect(), QColor("#f0f2f5"))
        finally:
            painter.end()
        super().paintEvent(event)  # Call super's paintEvent

    def _setup_radio_group(self, parent_layout, group_label_text, options_list, button_group_attr_name,
                           default_option_text=None):
        layout = QVBoxLayout()
        label = QLabel(group_label_text)
        label.setFont(QFont("Arial", 11, QFont.Weight.Bold))  # PyQt6 style
        label.setStyleSheet("background-color: transparent; color: black;")
        layout.addWidget(label)
        layout.addSpacing(5)
        button_group = QButtonGroup(self)
        setattr(self, button_group_attr_name, button_group)
        radio_style = "margin-bottom: 8px; background-color: transparent; color: black;"
        first_button = None
        target_button_found = False
        for text in options_list:
            radio = QRadioButton(text)
            radio.setStyleSheet(radio_style)
            button_group.addButton(radio)
            layout.addWidget(radio)
            if first_button is None: first_button = radio
            if text == default_option_text:
                radio.setChecked(True)
                target_button_found = True
        if not target_button_found and first_button:
            first_button.setChecked(True)
        parent_layout.addLayout(layout)

    def init_ui(self):
        overall_layout = QVBoxLayout(self)
        overall_layout.setContentsMargins(0, 0, 0, 0)
        overall_layout.setSpacing(0)

        self.content_widget = QWidget()
        self.content_widget.setObjectName("ContentLayer")
        self.content_widget.setStyleSheet("background-color: transparent;")

        main_content_layout = QHBoxLayout(self.content_widget)
        main_content_layout.setContentsMargins(0, 0, 0, 0)
        main_content_layout.setSpacing(0)

        left_widget = QWidget()
        left_widget.setStyleSheet("background-color: rgba(255, 255, 255, 0.7); border-radius: 10px;")
        left_panel = QVBoxLayout(left_widget)
        left_panel.setContentsMargins(20, 20, 20, 20)
        left_panel.setSpacing(15)

        avatar_path = os.path.join(ICONS_DIR, "avatar.png")
        self.avatar_main_label = QLabel()
        self.avatar_main_label.setFixedSize(100, 100)
        self.avatar_main_label.setPixmap(create_rounded_pixmap(avatar_path, 100))
        self.avatar_main_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # PyQt6 style
        self.avatar_main_label.setStyleSheet("background-color: transparent;")
        left_panel.addWidget(self.avatar_main_label, 0, Qt.AlignmentFlag.AlignHCenter)  # PyQt6 style

        name_label = QLabel("智能语音助手")
        name_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))  # PyQt6 style
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # PyQt6 style
        name_label.setStyleSheet("background-color: transparent; color: black;")
        left_panel.addWidget(name_label)

        self.status_label = QLabel("已准备就绪")
        self.status_label.setStyleSheet(
            "color: #333; background-color: rgba(240, 240, 240, 0.7); border-radius: 12px; padding: 6px 12px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # PyQt6 style
        left_panel.addWidget(self.status_label)
        left_panel.addSpacing(25)

        self._setup_radio_group(left_panel, "交互模式",
                                ["语音输入语音输出", "文本输入语音输出", "语音输入文本输出", "纯文本交互"],
                                "interaction_mode_group", "纯文本交互")
        left_panel.addSpacing(25)

        subtitle_layout = QVBoxLayout()
        subtitle_label = QLabel("字幕显示")
        subtitle_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))  # PyQt6 style
        subtitle_label.setStyleSheet("background-color: transparent; color: black;")
        subtitle_layout.addWidget(subtitle_label)
        subtitle_layout.addSpacing(5)
        checkbox_style = "margin-bottom: 8px; background-color: transparent; color: black;"

        # Store checkboxes as instance attributes if they need to be accessed from outside
        self.cb_show_subtitles = QCheckBox("显示字幕")  # Instance attribute
        self.cb_show_subtitles.setChecked(True)
        self.cb_show_subtitles.setStyleSheet(checkbox_style)
        subtitle_layout.addWidget(self.cb_show_subtitles)

        self.cb_show_english = QCheckBox("显示英文")  # Instance attribute
        self.cb_show_english.setChecked(True)
        self.cb_show_english.setStyleSheet(checkbox_style)
        subtitle_layout.addWidget(self.cb_show_english)
        left_panel.addLayout(subtitle_layout)
        left_panel.addSpacing(25)

        self._setup_radio_group(left_panel, "模型选择",
                                ["本地模型", "在线模型 - Kimi", "在线模型 - DeepSeek"],
                                "model_button_group", "本地模型")
        left_panel.addStretch()

        self.mic_btn = QPushButton()
        self.mic_btn.setFixedSize(80, 80)
        self.mic_btn.setCursor(Qt.CursorShape.PointingHandCursor)  # PyQt6 style

        self.mic_icon_loaded = (not self.mic_q_icon.isNull()) and os.path.exists(self.mic_icon_path)

        if self.mic_icon_loaded:
            self.mic_btn.setIcon(self.mic_q_icon)
            self.mic_btn.setIconSize(self.mic_icon_size)
            self.mic_btn.setText("")
            self.ready_mic_style = """
                QPushButton { background-color: transparent; border: none; border-radius: 40px; }
                QPushButton:hover { background-color: rgba(0,0,0,0.1); }
            """
        else:
            self.mic_btn.setText("REC")
            self.mic_btn.setFont(QFont("Arial", 10, QFont.Weight.Bold))  # PyQt6 style
            self.ready_mic_style = """
                QPushButton { background-color: #E0E0E0; color: #333333; border: 1px solid #CCCCCC; border-radius: 40px; }
                QPushButton:hover { background-color: #D0D0D0; }
            """
        self.listening_mic_style = """
            QPushButton { background-color: #79A6FF; border: 3px solid #CDE0FF; border-radius: 40px; color: white; }
        """
        self.mic_btn.setStyleSheet(self.ready_mic_style)
        self.mic_btn.clicked.connect(self.toggle_listening_state)
        left_panel.addWidget(self.mic_btn, 0, Qt.AlignmentFlag.AlignHCenter)  # PyQt6 style
        left_panel.addSpacing(10)

        center_widget = QWidget()
        center_widget.setStyleSheet("background-color: rgba(255, 255, 255, 0.8); border-radius: 10px;")
        center_panel = QVBoxLayout(center_widget)
        center_panel.setContentsMargins(15, 15, 15, 15)
        center_panel.setSpacing(10)

        top_buttons_layout = QHBoxLayout()
        voice_btn = QPushButton("语音助手")
        knowledge_btn = QPushButton("知识助手")
        button_height = 38
        voice_btn.setFixedHeight(button_height)
        knowledge_btn.setFixedHeight(button_height)
        common_btn_font = "font-weight: bold;"
        voice_btn.setStyleSheet(
            f"QPushButton {{ background-color: #448AFF; color: white; border-radius: {button_height // 2}px; padding: 5px 20px; border: none; {common_btn_font} }}")
        knowledge_btn.setStyleSheet(
            f"QPushButton {{ border: 1.5px solid #448AFF; color: #448AFF; border-radius: {button_height // 2}px; background-color: white; padding: 5px 20px; {common_btn_font} }}")
        voice_btn.setCursor(Qt.CursorShape.PointingHandCursor)  # PyQt6 style
        knowledge_btn.setCursor(Qt.CursorShape.PointingHandCursor)  # PyQt6 style
        top_buttons_layout.addWidget(voice_btn)
        top_buttons_layout.addSpacing(10)
        top_buttons_layout.addWidget(knowledge_btn)
        top_buttons_layout.addStretch()

        self.chat_area_layout = QVBoxLayout()
        self.chat_area_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # PyQt6 style
        self.chat_area_layout.setSpacing(12)

        scroll_content_widget = QWidget()
        scroll_content_widget.setStyleSheet("background-color: transparent;")
        scroll_content_widget.setLayout(self.chat_area_layout)

        self.chat_scroll_area = QScrollArea()
        self.chat_scroll_area.setStyleSheet("background: transparent; border: none;")
        self.chat_scroll_area.setWidgetResizable(True)
        self.chat_scroll_area.setWidget(scroll_content_widget)

        input_area_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("输入消息...")
        input_field_height = 40
        self.input_field.setFixedHeight(input_field_height)
        self.input_field.setStyleSheet(
            f"QLineEdit {{ border: 1px solid #D0D0D0; border-radius: {input_field_height // 2 - 1}px; padding-left: 15px; padding-right: 15px; background-color: white; font-size: 10pt; }}")
        self.input_field.returnPressed.connect(self.send_message)

        self.send_button = QPushButton("发送")
        self.send_button.setFixedHeight(input_field_height)
        self.send_button.setFixedWidth(80)
        self.send_button.setStyleSheet(
            f"QPushButton {{ background-color: #448AFF; color: white; border-radius: {input_field_height // 2 - 1}px; padding: 5px 10px; border: none; {common_btn_font} }}")
        self.send_button.setCursor(Qt.CursorShape.PointingHandCursor)  # PyQt6 style
        self.send_button.clicked.connect(self.send_message)
        input_area_layout.addWidget(self.input_field)
        input_area_layout.addSpacing(8)
        input_area_layout.addWidget(self.send_button)

        center_panel.addLayout(top_buttons_layout)
        center_panel.addWidget(self.chat_scroll_area)
        center_panel.addLayout(input_area_layout)

        main_content_layout.addWidget(left_widget, 25)
        main_content_layout.addWidget(center_widget, 75)
        overall_layout.addWidget(self.content_widget)

        # Initial messages for testing UI
        # initial_messages = [
        #     ("你好，有什么可以帮你？", "bot"),
        #     ("你好", "user"),
        #     ("抱歉，我暂时还不明白你的意思。", "bot"),
        # ]
        # for msg, sender_type in initial_messages:
        #     self.add_message(msg, sender=sender_type)

    def toggle_listening_state(self):
        self.is_listening = not self.is_listening
        if self.is_listening:
            self.status_label.setText("正在聆听...")
            self.mic_btn.setStyleSheet(self.listening_mic_style)
            if self.mic_icon_loaded:
                self.mic_btn.setIcon(self.mic_q_icon);
                self.mic_btn.setText("")
            else:
                self.mic_btn.setIcon(QIcon());
                self.mic_btn.setText("...")
        else:
            self.status_label.setText("已准备就绪")
            self.mic_btn.setStyleSheet(self.ready_mic_style)
            if self.mic_icon_loaded:
                self.mic_btn.setIcon(self.mic_q_icon);
                self.mic_btn.setText("")
            else:
                self.mic_btn.setIcon(QIcon());
                self.mic_btn.setText("REC")

    def send_message(self):  # This is for direct UI testing if run standalone
        user_text = self.input_field.text().strip()
        if not user_text: return
        self.add_message(user_text, sender="user")
        self.input_field.clear()
        # Simple echo/canned reply for standalone testing
        bot_reply = f"收到: {user_text}"
        if "你好" in user_text or "在吗" in user_text:
            bot_reply = "你好，有什么可以帮你？"
        elif "天气" in user_text:
            bot_reply = "今天天气不错，适合出去走走。"
        elif "你是谁" in user_text:
            bot_reply = "我是你的智能语音助手。"
        self.add_message(bot_reply, sender="bot")

    def add_message(self, text, sender="user"):
        message_container_widget = QWidget()
        message_container_widget.setStyleSheet("background-color: transparent;")
        message_layout = QHBoxLayout(message_container_widget)
        message_layout.setContentsMargins(0, 0, 0, 0)
        message_layout.setSpacing(8)

        avatar_label = QLabel()
        avatar_label.setFixedSize(36, 36)
        bubble_label = QLabel(text)
        bubble_label.setWordWrap(True)
        bubble_label.setMaximumWidth(450)
        bubble_label.setFont(QFont("Arial", 10))
        bubble_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)  # PyQt6 style

        bubble_radius = "18px";
        small_radius = "6px";
        common_bubble_padding = "padding: 10px 15px;"
        user_avatar_path = os.path.join(ICONS_DIR, "user.png")
        bot_avatar_path = os.path.join(ICONS_DIR, "avatar.png")

        if sender == "user":
            avatar_label.setPixmap(create_rounded_pixmap(user_avatar_path, 36))
            bubble_label.setStyleSheet(
                f"QLabel {{ background-color: #A0E880; color: black; {common_bubble_padding} border-radius: {bubble_radius}; border-top-right-radius: {small_radius}; }}")
            message_layout.addStretch();
            message_layout.addWidget(bubble_label);
            message_layout.addWidget(avatar_label, alignment=Qt.AlignmentFlag.AlignTop)  # PyQt6 style
        else:
            avatar_label.setPixmap(create_rounded_pixmap(bot_avatar_path, 36))
            bubble_label.setStyleSheet(
                f"QLabel {{ background-color: #FFFFFF; color: black; {common_bubble_padding} border-radius: {bubble_radius}; border-top-left-radius: {small_radius}; border: 1px solid #E0E0E0; }}")
            message_layout.addWidget(avatar_label, alignment=Qt.AlignmentFlag.AlignTop);  # PyQt6 style
            message_layout.addWidget(bubble_label);
            message_layout.addStretch()

        self.chat_area_layout.addWidget(message_container_widget)
        QApplication.processEvents()  # Force UI update
        self.chat_scroll_area.verticalScrollBar().setValue(self.chat_scroll_area.verticalScrollBar().maximum())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    if not os.path.exists(ICONS_DIR):
        try:
            os.makedirs(ICONS_DIR)
        except OSError as e:
            print(f"错误 (main): 创建 '{ICONS_DIR}' 目录失败: {e}")
    critical_icons = ["background.png", "avatar.png", "user.png", "voice.png"]
    for icon_name in critical_icons:
        if not os.path.exists(os.path.join(ICONS_DIR, icon_name)):
            print(f"警告 (main): 关键图标 '{icon_name}' 在 '{ICONS_DIR}' 中未找到！")
    window = VoiceAssistantUI()
    # Add initial messages for standalone testing
    initial_messages_standalone = [
        ("你好，我是语音助手，可以开始测试了。", "bot"),
    ]
    for msg, sender_type in initial_messages_standalone:
        window.add_message(msg, sender=sender_type)
    window.show()
    sys.exit(app.exec())