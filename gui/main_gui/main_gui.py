import os
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame
)
from PyQt6.QtGui import QPixmap, QFont, QPainter, QLinearGradient, QColor, QBrush
from PyQt6.QtCore import Qt, QRect # QRect is fine as is

FONT_PATH = "fonts/NotoSansSC-Regular.ttf" # Assuming this path is valid relative to where script runs
ICON_PATH = "icons" # Assuming this path is valid

# 自定义渐变圆角卡片
class GradientCard(QFrame):
    def __init__(self, width, height, color_start, color_end, radius=16, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.color_start = QColor(*color_start)
        self.color_end = QColor(*color_end)
        self.radius = radius
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, self.color_start)
        gradient.setColorAt(1, self.color_end)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing) # PyQt6 style
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen) # PyQt6 style
        painter.drawRoundedRect(0, 0, self.width(), self.height(), self.radius, self.radius)

# 主界面
class LearningApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deepseek 学习机")
        self.setGeometry(100, 100, 1280, 720)

        self.central = QWidget()
        self.setCentralWidget(self.central)

        self.main_layout = QVBoxLayout()
        self.central.setLayout(self.main_layout)

        self.init_top_bar()
        self.init_welcome_card()
        self.init_function_cards()
        self.init_subject_grid()
        self.init_weekly_card()
        self.init_recent_study()
        # self.init_float_button()

    def init_top_bar(self):
        top_bar = QHBoxLayout()

        title = QLabel("Deepseek学习机")
        title.setFont(QFont("Noto Sans SC", 20)) # Ensure font is available or use a fallback
        title.setStyleSheet("color: #3C3C3C;")
        top_bar.addWidget(title)

        top_bar.addStretch()

        search = QLabel()
        # Ensure ICON_PATH is correctly set up for these Pixmaps
        search.setPixmap(QPixmap(f"{ICON_PATH}/search.png").scaled(28, 28, Qt.AspectRatioMode.KeepAspectRatio)) # PyQt6 style
        user = QLabel()
        user.setPixmap(QPixmap(f"{ICON_PATH}/user.png").scaled(28, 28, Qt.AspectRatioMode.KeepAspectRatio)) # PyQt6 style
        top_bar.addWidget(search)
        top_bar.addSpacing(20)
        top_bar.addWidget(user)

        self.main_layout.addLayout(top_bar)

    def init_welcome_card(self):
        card = QFrame()
        card.setStyleSheet("background: white; border-radius: 16px;")
        card.setFixedHeight(100)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 10, 10, 10)

        label1 = QLabel("下午好，同学")
        label1.setFont(QFont("Noto Sans SC", 16))
        label2 = QLabel("今天想学习什么内容？")
        label2.setFont(QFont("Noto Sans SC", 12))
        label2.setStyleSheet("color: #666;")

        layout.addWidget(label1)
        layout.addWidget(label2)
        self.main_layout.addWidget(card)
        self.main_layout.addSpacing(10)

    def init_function_cards(self):
        layout = QHBoxLayout()

        # 拍照问答卡片
        camera_card = GradientCard(480, 120, (102, 126, 234), (118, 75, 162))
        cam_layout = QVBoxLayout(camera_card)
        icon = QLabel()
        icon.setPixmap(QPixmap(f"{ICON_PATH}/camera.png").scaled(36, 36))
        cam_layout.addWidget(icon)

        title1 = QLabel("拍照问答")
        title1.setFont(QFont("Noto Sans SC", 14))
        cam_layout.addWidget(title1)

        subtitle1 = QLabel("图像识别+答案解析")
        subtitle1.setFont(QFont("Noto Sans SC", 10))
        cam_layout.addWidget(subtitle1)

        # 智能对话卡片
        chat_card = GradientCard(480, 120, (255, 154, 158), (250, 208, 196))
        chat_layout = QVBoxLayout(chat_card)
        icon2 = QLabel()
        icon2.setPixmap(QPixmap(f"{ICON_PATH}/chat.png").scaled(36, 36))
        chat_layout.addWidget(icon2)

        title2 = QLabel("智能对话")
        title2.setFont(QFont("Noto Sans SC", 14))
        chat_layout.addWidget(title2)

        subtitle2 = QLabel("AI语义知识问答")
        subtitle2.setFont(QFont("Noto Sans SC", 10))
        chat_layout.addWidget(subtitle2)

        layout.addWidget(camera_card)
        layout.addSpacing(20)
        layout.addWidget(chat_card)
        layout.addStretch()

        self.main_layout.addLayout(layout)
        self.main_layout.addSpacing(20)

    def init_subject_grid(self):
        label = QLabel("学科选择")
        label.setFont(QFont("Noto Sans SC", 12))
        label.setStyleSheet("color: #666;")
        self.main_layout.addWidget(label)

        subjects = ["数学", "英语", "物理", "化学", "生物", "地理", "语文", "政治", "历史"]
        grid = QGridLayout()

        for i, subject in enumerate(subjects):
            icon_label = QLabel()
            icon_label.setPixmap(QPixmap(f"{ICON_PATH}/{subject_map(subject)}.png").scaled(36, 36))
            name_label = QLabel(subject)
            name_label.setFont(QFont("Noto Sans SC", 10))
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter) # PyQt6 style

            box = QVBoxLayout()
            box.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignCenter) # PyQt6 style
            box.addWidget(name_label)
            cell = QWidget()
            cell.setLayout(box)
            cell.setStyleSheet("background: #f0f0ff; border-radius: 12px;")
            grid.addWidget(cell, i // 5, i % 5)

        self.main_layout.addLayout(grid)

    def init_weekly_card(self):
        card = QFrame()
        card.setFixedHeight(100)
        card.setStyleSheet("background: white; border-radius: 16px;")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(20, 10, 10, 10)

        title = QLabel("本周学习")
        title.setFont(QFont("Noto Sans SC", 12))
        layout.addWidget(title)

        for val, label_text in zip(["8.2", "24", "86%"], ["学习时长(h)", "提问次数", "正确率"]): # Renamed label for clarity
            v = QVBoxLayout()
            t = QLabel(val)
            t.setFont(QFont("Noto Sans SC", 20))
            l_widget = QLabel(label_text) # Renamed l to l_widget to avoid conflict with layout
            l_widget.setFont(QFont("Noto Sans SC", 10))
            l_widget.setStyleSheet("color: #888;")
            v.addWidget(t)
            v.addWidget(l_widget)
            layout.addSpacing(40)
            layout.addLayout(v)

        self.main_layout.addWidget(card)

    def init_recent_study(self):
        label = QLabel("最近学习")
        label.setFont(QFont("Noto Sans SC", 12))
        label.setStyleSheet("color: #666;")
        self.main_layout.addWidget(label)

        records = [
            ("二次函数应用问题", "数学 · 30分钟前"),
            ("三角函数应用问题", "数学 · 2小时前"),
            ("英译汉练习题", "英语 · 昨天"),
            ("听力练习题", "英语 · 昨天"),
            ("牛顿第二定律推导", "物理 · 3天前"),
            ("有机化学反应", "化学 · 上周"),
        ]

        grid = QGridLayout()
        for i, (title, subtitle) in enumerate(records):
            icon_label = QLabel()
            icon_label.setPixmap(QPixmap(f"{ICON_PATH}/book.png").scaled(28, 28))

            text_layout = QVBoxLayout() # Renamed text to text_layout for clarity
            t = QLabel(title)
            t.setFont(QFont("Noto Sans SC", 12))
            s = QLabel(subtitle)
            s.setFont(QFont("Noto Sans SC", 9))
            s.setStyleSheet("color: #888;")
            text_layout.addWidget(t)
            text_layout.addWidget(s)

            hbox = QHBoxLayout()
            hbox.addWidget(icon_label)
            hbox.addLayout(text_layout)

            card = QWidget()
            card.setLayout(hbox)
            card.setStyleSheet("background: white; border-radius: 12px; padding: 6px;")
            grid.addWidget(card, i // 2, i % 2)

        self.main_layout.addLayout(grid)

    # def init_float_button(self):
    #     btn = QPushButton("+", self)
    #     btn.setStyleSheet("""
    #         QPushButton {
    #             background-color: #4285f4;
    #             color: white;
    #             font-size: 28px;
    #             border-radius: 30px;
    #         }
    #     """)
    #     btn.setFixedSize(60, 60)
    #     # Consider window resizing:
    #     # btn.move(self.centralWidget().width() - 80, self.centralWidget().height() - 100)
    #     # Or use a layout with alignment for more robust positioning.
    #     btn.move(self.width() - 80, self.height() - 100) # Relative to main window
    #     btn.show()

def subject_map(name):
    return {
        "数学": "math",
        "英语": "english",
        "物理": "physics",
        "化学": "chemistry",
        "生物": "biology",
        "地理": "geography",
        "语文": "notes",
        "政治": "politics",
        "历史": "history",
    }.get(name, "default")


if __name__ == "__main__":
    # Ensure FONT_PATH and ICON_PATH are accessible or handle potential errors
    # For example, by checking os.path.exists() or using absolute paths if necessary
    if not os.path.exists(FONT_PATH):
        print(f"警告: 字体文件未找到于 '{FONT_PATH}'. 可能导致显示问题。")
    if not os.path.exists(ICON_PATH) or not os.path.isdir(ICON_PATH):
        print(f"警告: 图标目录未找到于 '{ICON_PATH}'. 可能导致图标无法加载。")
        # Fallback or create dummy icon directory if critical
        # os.makedirs(ICON_PATH, exist_ok=True)


    app = QApplication(sys.argv)
    window = LearningApp()
    window.show()
    sys.exit(app.exec())