import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QToolButton, QDesktopWidget, QFrame, QProgressBar
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtCore import Qt, QSize, QRectF, QUrl, QPoint, pyqtSignal, QTimer
from PyQt5.QtGui import QPalette, QColor, QFont, QPainter, QPainterPath, QLinearGradient, QIcon, QPolygon, QPen, QBrush

class CustomTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("US Fear and Greed Index")
        self.title_label.setStyleSheet("""
            color: #CDD6F4;
            font-family: 'Arial';
            font-weight: bold;
        """)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label, 1)  # 1 is the stretch factor

        self.pin_button = self.create_button("#4CAF50")  # Green color for the pin button
        self.minimize_button = self.create_button("#FFBD2E")
        self.close_button = self.create_button("#FF5F56")

        layout.addWidget(self.pin_button)
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


class FearGreedGauge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 50
        self.setFixedSize(300, 80)

    def set_value(self, value):
        self.value = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        gauge_height = 20
        gauge_top = rect.height() - gauge_height - 20  # Leave space for labels below
        gauge_rect = QRectF(0, gauge_top, rect.width(), gauge_height)

        # Draw background
        background_color = QColor(30, 30, 46)
        painter.fillRect(rect, background_color)

        # Draw gauge background
        gauge_background = QColor(53, 53, 78)
        painter.fillRect(gauge_rect, gauge_background)

        # Draw bar-by-bar gradient
        bar_count = 100
        bar_width = gauge_rect.width() / bar_count
        for i in range(bar_count):
            bar_rect = QRectF(i * bar_width, gauge_top, bar_width, gauge_height)
            if i < 25:
                color = QColor(255, int(255 * (i / 25)), 0, 200)  # Red to Yellow (Extreme Fear)
            elif i < 45:
                color = QColor(255, 255, int(255 * ((i - 25) / 20)), 200)  # Yellow to Green (Fear)
            elif i < 55:
                color = QColor(0, 255, 0, 200)  # Green (Neutral)
            elif i < 75:
                color = QColor(int(255 * (1 - (i - 55) / 20)), 255, 255, 200)  # Green to Cyan (Greed)
            else:
                color = QColor(0, int(255 * (1 - (i - 75) / 25)), 255, 200)  # Cyan to Blue (Extreme Greed)
            painter.fillRect(bar_rect, color)

        # Draw gauge border
        painter.setPen(QPen(QColor(205, 214, 244), 2))
        painter.drawRoundedRect(gauge_rect, 10, 10)

        # Draw indicator
        indicator_pos = int(self.value / 100 * gauge_rect.width())
        indicator_path = QPainterPath()
        indicator_path.moveTo(indicator_pos, gauge_top + gauge_height)
        indicator_path.lineTo(indicator_pos - 10, gauge_top + gauge_height + 15)
        indicator_path.lineTo(indicator_pos + 10, gauge_top + gauge_height + 15)
        indicator_path.closeSubpath()
        painter.fillPath(indicator_path, QColor(205, 214, 244))  # Lighter color

        # Draw labels
        painter.setPen(QColor(205, 214, 244))
        font = QFont("Arial", 8)
        painter.setFont(font)
        
        label_rect = QRectF(0, rect.height() - 20, rect.width(), 20)
        painter.drawText(label_rect, Qt.AlignLeft | Qt.AlignBottom, "EXTREME FEAR")
        painter.drawText(label_rect, Qt.AlignCenter | Qt.AlignBottom, "NEUTRAL")
        painter.drawText(label_rect, Qt.AlignRight | Qt.AlignBottom, "EXTREME GREED")

        # Draw value
        value_rect = QRectF(0, 0, rect.width(), gauge_top)
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(205, 214, 244))
        painter.drawText(value_rect, Qt.AlignCenter, f"{self.value}")
        
class WebEnginePage(QWebEnginePage):
    fear_greed_data = pyqtSignal(str, str, str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loadFinished.connect(self.onLoadFinished)
        self.attempt_count = 0
        self.max_attempts = 20
        self.extract_timer = QTimer(self)
        self.extract_timer.timeout.connect(self.attemptExtractData)

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print(f"JS Console ({level}): {message} at line {lineNumber}")

    def onLoadFinished(self, ok):
        if ok:
            print("Page loaded successfully, waiting before attempting to extract data...")
            QTimer.singleShot(5000, self.startExtraction)
        else:
            print("Page failed to load")
            self.fear_greed_data.emit("N/A", "Failed to load", "0")

    def startExtraction(self):
        self.attempt_count = 0
        self.attemptExtractData()

    def attemptExtractData(self):
        if self.attempt_count >= self.max_attempts:
            print("Max attempts reached, giving up")
            self.fear_greed_data.emit("N/A", "Failed to load", "0")
            return

        self.runJavaScript(r"""
            function extractData() {
                var container = document.querySelector('.market-fng-gauge');
                if (!container) {
                    console.log('Fear and Greed container not found');
                    return null;
                }
                
                var valueElement = container.querySelector('.market-fng-gauge__dial-number-value');
                var statusElement = container.querySelector('.market-fng-gauge__label');
                var rotationElement = container.querySelector('.market-fng-gauge__hand-svg');
                
                if (valueElement && statusElement && rotationElement) {
                    console.log('Found Fear & Greed elements');
                    var rotation = rotationElement.style.transform;
                    var rotationValue = rotation.match(/rotate\(([-\d.]+)deg\)/);
                    return [valueElement.textContent.trim(), statusElement.textContent.trim(), rotationValue ? rotationValue[1] : "0"];
                } else {
                    console.log('Fear & Greed elements not found within container');
                    return null;
                }
            }
            extractData();
        """, self.handleExtractedData)


    def handleExtractedData(self, result):
        print(f"Received data from JavaScript: {result}")
        if result:
            self.fear_greed_data.emit(result[0], result[1], result[2])
            self.extract_timer.stop()
        else:
            self.attempt_count += 1
            print(f"Data not found, attempt {self.attempt_count}")
            self.extract_timer.start(1000)  # Try again after 1 second



class LoadingScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 80)
        layout = QVBoxLayout(self)
        self.label = QLabel("Loading Fear and Greed Index...", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #CDD6F4;")
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2C2C2C;
                border: 1px solid #555555;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3A3A3A;
                width: 10px;
                margin: 0.5px;
            }
        """)
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        
class USFearGreedWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.setWindowTitle("US Fear and Greed Index")
       
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(0)

        self.title_bar = CustomTitleBar(self)
        self.title_bar.close_button.clicked.connect(self.close)
        self.title_bar.minimize_button.clicked.connect(self.showMinimized)
        self.title_bar.pin_button.clicked.connect(self.toggle_pin)
        main_layout.addWidget(self.title_bar)
        
        self.content_widget = QFrame(self)
        self.content_widget.setStyleSheet("""
            QFrame {
                background-color: #1E1E2E;
                border: 1px solid #3D3D5C;
                border-bottom-left-radius: 10px;
                border-bottom-right-radius: 10px;
            }
        """)
        content_layout = QVBoxLayout(self.content_widget)
        main_layout.addWidget(self.content_widget)

        self.loading_screen = LoadingScreen(self.content_widget)
        content_layout.addWidget(self.loading_screen)

        self.gauge_widget = FearGreedGauge(self.content_widget)
        content_layout.addWidget(self.gauge_widget, alignment=Qt.AlignCenter)
        self.gauge_widget.hide()

        self.status_label = QLabel("NEUTRAL", self.content_widget)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            font-family: 'Arial', sans-serif;
            font-size: 16px;
            font-weight: bold;
            color: #CDD6F4;
        """)
        content_layout.addWidget(self.status_label)
        self.status_label.hide()

        self.description_label = QLabel("", self.content_widget)
        self.description_label.setAlignment(Qt.AlignCenter)
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("""
            font-family: 'Arial', sans-serif;
            font-size: 12px;
            color: #A6ADC8;
        """)
        content_layout.addWidget(self.description_label)
        self.description_label.hide()

        self.web_view = QWebEngineView()
        self.web_page = WebEnginePage(self.web_view)
        self.web_view.setPage(self.web_page)
        self.web_page.fear_greed_data.connect(self.update_fear_greed_data)
        self.web_view.setUrl(QUrl("https://www.cnn.com/markets/fear-and-greed"))
        self.web_view.hide()  # Hide the web view

#        self.setFixedSize(320, 200)
        self.setFixedSize(320, 207)  # Instead of 200
        self.center_top()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create a rounded rectangle path
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 15, 15)

        # Create a subtle shadow effect
        painter.setPen(Qt.NoPen)
        for i in range(10):
            opacity = 10 - i
            painter.setBrush(QColor(0, 0, 0, opacity))
            painter.drawPath(path.translated(0, i * 0.5))

        # Fill the main shape
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(36, 36, 54))  # Slightly lighter at the top
        gradient.setColorAt(1, QColor(30, 30, 46))  # Original color at the bottom
        painter.setBrush(gradient)
        painter.drawPath(path)

    def update_fear_greed_data(self, value, status, rotation):
        try:
            self.loading_screen.hide()
            self.gauge_widget.show()
            self.status_label.show()
            self.description_label.show()
            
            int_value = int(value)
            self.gauge_widget.set_value(int_value)
            self.status_label.setText(status.upper())
            
            if int_value <= 25:
                description = "Extreme Fear: Investors are in a state of panic, potentially overselling."
            elif int_value <= 45:
                description = "Fear: There's significant worry in the market, but not panic."
            elif int_value <= 55:
                description = "Neutral: The market sentiment is balanced between optimism and caution."
            elif int_value <= 75:
                description = "Greed: Investors are showing increased optimism in the market."
            else:
                description = "Extreme Greed: Investors are overly optimistic, potentially overbuying."
            
            self.description_label.setText(description)
            
        except ValueError:
            print(f"Invalid value: {value}")

    def center_top(self):
        screen = QDesktopWidget().screenNumber(QDesktopWidget().cursor().pos())
        center = QDesktopWidget().screenGeometry(screen).center()
        geo = self.geometry()
        geo.moveCenter(center)
        geo.setY(0)
        self.setGeometry(geo)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.y() < self.title_bar.height():
            self.dragPosition = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and hasattr(self, 'dragPosition'):
            self.move(event.globalPos() - self.dragPosition)
            event.accept()

    def mouseReleaseEvent(self, event):
        if hasattr(self, 'dragPosition'):
            del self.dragPosition

    def showMinimized(self):
        self.setWindowFlags(self.windowFlags() & ~Qt.Tool)
        super().showMinimized()
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.show()

    def toggle_pin(self):
        self.pinned = not getattr(self, 'pinned', False)
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
                                                      stop:0 #FF5722, stop:1 #FF8A65);
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
                                                      stop:0 #4CAF50, stop:1 #81C784);
                }
            """)
        self.show()

    def closeEvent(self, event):
        event.accept()

def run_us_fear_greed_widget():
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
    
    widget = USFearGreedWidget()
    widget.show()
    return widget

if __name__ == '__main__':
    app = QApplication(sys.argv)
    us_widget = run_us_fear_greed_widget()
    sys.exit(app.exec_())