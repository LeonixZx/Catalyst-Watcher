import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QToolButton, QDesktopWidget, QSizeGrip
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, QSize, QRectF, QUrl
from PyQt5.QtGui import QPalette, QColor, QFont, QPainter, QPainterPath, QLinearGradient, QIcon

class CustomTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)  # Increased height for the title

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("Fear and Greed Index")
        self.title_label.setStyleSheet("""
            color: #CDD6F4;
            font-family: 'Arial';
            font-size: 11px;
            font-weight: bold;
        """)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label, 1)

        self.pin_button = self.create_button("#4CAF50")
        layout.addWidget(self.pin_button)

        self.minimize_button = self.create_button("#FFBD2E")
        self.close_button = self.create_button("#FF5F56")

        layout.addWidget(self.minimize_button)
        layout.addWidget(self.close_button)
        
    def create_button(self, color):
        button = QToolButton(self)
        button.setFixedSize(QSize(12, 12))
        button.setStyleSheet(f"""
            QToolButton {{
                background-color: {color};
                border: none;
                border-radius: 6px;
            }}
            QToolButton:hover {{
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                  stop:0 {color}, stop:1 #FFFFFF);
            }}
        """)
        return button
        
class FearGreedWidget(QWidget):
    def __init__(self, market="MY"):
        super().__init__()
        self.pinned = False
        self.dragging = False
        self.market = market
        self.initUI()
        
    def initUI(self):
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.setWindowTitle("Fear and Greed Index")
       
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(0)

        self.title_bar = CustomTitleBar(self)
        if self.market == "MY":
            self.title_bar.title_label.setText("Malaysia Fear and Greed Index")
            url = "https://www.malaysiastock.biz/Market-Gauge-New.aspx"
            self.setFixedSize(320, 550)  # Fixed size for MY widget
        else:
            self.title_bar.title_label.setText("Fear and Greed Index")
            url = "https://www.cnn.com/markets/fear-and-greed"  # Default to CNN if not MY
            self.setFixedSize(800, 600)  # Adjust size for other markets if needed

        self.title_bar.close_button.clicked.connect(self.close)
        self.title_bar.minimize_button.clicked.connect(self.showMinimized)
        self.title_bar.pin_button.clicked.connect(self.toggle_pin)
        main_layout.addWidget(self.title_bar)
        
        self.web_view = QWebEngineView()
        self.web_view.setUrl(QUrl(url))
        main_layout.addWidget(self.web_view)
        
        self.center_top()

        self.web_view.loadFinished.connect(self.on_load_finished)

    def on_load_finished(self, ok):
        if ok:
            if self.market == "MY":
                self.inject_my_scripts()
            
    def inject_my_scripts(self):
        js = """
        (function() {
            var indexElement = document.getElementById('MainContent_IndiceView1_IndiceText1_lblIndiceValue');
            if (indexElement) {
                var title = document.querySelector('.title-bar-title');
                if (title) {
                    title.textContent = 'Malaysia Fear and Greed Index: ' + indexElement.textContent;
                }
            }
        })();
        """
        self.web_view.page().runJavaScript(js)

    def center_top(self):
        screen = QDesktopWidget().screenNumber(QDesktopWidget().cursor().pos())
        center = QDesktopWidget().screenGeometry(screen).center()
        geo = self.geometry()
        geo.moveCenter(center)
        geo.setY(0)
        self.setGeometry(geo)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 15, 15)

        painter.setPen(Qt.NoPen)
        for i in range(10):
            opacity = 10 - i
            painter.setBrush(QColor(0, 0, 0, opacity))
            painter.drawPath(path.translated(0, i * 0.5))

        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(36, 36, 54))
        gradient.setColorAt(1, QColor(30, 30, 46))
        painter.setBrush(gradient)
        painter.drawPath(path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.y() < self.title_bar.height():
            self.dragging = True
            self.dragPosition = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.dragging:
            self.move(event.globalPos() - self.dragPosition)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.dragging = False

    def showMinimized(self):
        self.setWindowFlags(self.windowFlags() & ~Qt.Tool)
        super().showMinimized()
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.show()

    def toggle_pin(self):
        self.pinned = not self.pinned
        if self.pinned:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            self.title_bar.pin_button.setStyleSheet("""
                QToolButton {
                    background-color: #FF5722;
                    border: none;
                    border-radius: 6px;
                }
                QToolButton:hover {
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                      stop:0 #FF5722, stop:1 #FFFFFF);
                }
            """)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
            self.title_bar.pin_button.setStyleSheet("""
                QToolButton {
                    background-color: #4CAF50;
                    border: none;
                    border-radius: 6px;
                }
                QToolButton:hover {
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                      stop:0 #4CAF50, stop:1 #FFFFFF);
                }
            """)
        self.show()

    def closeEvent(self, event):
        event.accept()

def run_fear_greed_widget(market="MY"):
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    widget = FearGreedWidget(market)
    widget.show()
    return widget

if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_widget = run_fear_greed_widget("MY")  # Explicitly pass "MY" for Malaysia
    sys.exit(app.exec_())