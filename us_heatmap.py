import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QToolButton, QDesktopWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage, QWebEngineSettings
from PyQt5.QtCore import Qt, QSize, QRectF, QUrl, QTimer
from PyQt5.QtGui import QPalette, QColor, QFont, QPainter, QPainterPath, QLinearGradient, QIcon

class CustomWebEnginePage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        pass

class CustomTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("US Heatmap")
        self.title_label.setStyleSheet("""
            color: #CDD6F4;
            font-family: 'Arial';
            font-weight: bold;
        """)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label, 1)

        self.pin_button = self.create_button("#4CAF50")
        layout.addWidget(self.pin_button)

        self.minimize_button = self.create_button("#FFBD2E")
        self.maximize_button = self.create_button("#27C93F")
        self.close_button = self.create_button("#FF5F56")

        layout.addWidget(self.minimize_button)
        layout.addWidget(self.maximize_button)
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

class USHeatmapWidget(QWidget):
    def __init__(self, icon=None):
        super().__init__()
        self.pinned = False
        self.dragging = False
        if icon:
            self.setWindowIcon(icon)
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(0)

        self.setWindowTitle("US Stock Market Heatmap")

        self.title_bar = CustomTitleBar(self)
        self.title_bar.close_button.clicked.connect(self.hide)
        self.title_bar.minimize_button.clicked.connect(self.showMinimized)
        self.title_bar.maximize_button.clicked.connect(self.toggle_maximize)
        self.title_bar.pin_button.clicked.connect(self.toggle_pin)
        main_layout.addWidget(self.title_bar)
        
        self.content_widget = QWidget()
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        self.web_view = QWebEngineView()
        self.web_view.setPage(CustomWebEnginePage(self.web_view))
        content_layout.addWidget(self.web_view)
        
        main_layout.addWidget(self.content_widget)
        
        self.setFixedSize(1000, 600)
        self.center()

        self.load_heatmap()

    def load_heatmap(self):
        html_content = """
        <html>
        <head>
            <style>
                body, html {
                    margin: 0;
                    padding: 0;
                    height: 100%;
                    overflow: hidden;
                }
                .tradingview-widget-container {
                    height: 100%;
                }
            </style>
        </head>
        <body>
            <div class="tradingview-widget-container">
                <div class="tradingview-widget-container__widget"></div>
                <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-stock-heatmap.js" async>
                {
                "exchanges": [],
                "dataSource": "SPX500",
                "grouping": "sector",
                "blockSize": "market_cap_basic",
                "blockColor": "change",
                "locale": "en",
                "symbolUrl": "",
                "colorTheme": "dark",
                "hasTopBar": false,
                "isDataSetEnabled": false,
                "isZoomEnabled": true,
                "hasSymbolTooltip": true,
                "isMonoSize": false,
                "width": "100%",
                "height": "100%"
                }
                </script>
            </div>
        </body>
        </html>
        """
        self.web_view.setHtml(html_content)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def showEvent(self, event):
        super().showEvent(event)
        self.center()

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

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

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

def run_us_heatmap_widget(icon=None):
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    app.setStyle("Fusion")
    
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(30, 30, 46))
    dark_palette.setColor(QPalette.WindowText, QColor(205, 214, 244))
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, QColor(205, 214, 244))
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    
    widget = USHeatmapWidget(icon)
    widget.show()
    return widget

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = run_us_heatmap_widget()
    sys.exit(app.exec_())