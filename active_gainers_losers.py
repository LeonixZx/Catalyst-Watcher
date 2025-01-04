import sys
import csv
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QWidget, QComboBox, QTabWidget, 
                             QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar, QFrame)
from PyQt5.QtCore import QTimer, Qt, QSize, QPropertyAnimation, QEasingCurve, QRectF
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QLinearGradient
import requests
from bs4 import BeautifulSoup
import multiprocessing

class RoundedCornerFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("roundedFrame")
        self.setStyleSheet("""
            #roundedFrame {
                background-color: #252526;
                border-radius: 10px;
            }
        """)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 10, 10)
        painter.fillPath(path, QColor("#252526"))
        
class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super(LoadingOverlay, self).__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 150);")
        
        layout = QVBoxLayout(self)
        self.spinner = QProgressBar(self)
        self.spinner.setRange(0, 0)  # Makes it an "infinite" progress bar
        self.spinner.setTextVisible(False)
        self.spinner.setFixedSize(100, 10)
        self.spinner.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4a4a4a;
                border-radius: 5px;
                background-color: #2c2c2c;
            }
            QProgressBar::chunk {
                background-color: #0088cc;
            }
        """)
        
        layout.addWidget(self.spinner, alignment=Qt.AlignCenter)
        
        self.label = QLabel("Loading...", self)
        self.label.setStyleSheet("color: white; font-size: 16px;")
        layout.addWidget(self.label, alignment=Qt.AlignCenter)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self.rect(), 10, 10)
        painter.fillPath(path, QColor(0, 0, 0, 150))


class MacOSTitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setObjectName("titleBar")
        self.setFixedHeight(32)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("Advanced Multi-Market Stocks Data")
        self.title_label.setStyleSheet("color: #D4D4D4; font-weight: bold;")
        layout.addWidget(self.title_label)

        layout.addStretch()

        button_data = [
            ("#4A4A4A", "Pin", self.toggle_pin),
            ("#FDBC40", "Minimize", self.parent.showMinimized),
            ("#34C749", "Maximize", self.toggle_maximize),
            ("#FC615D", "Close", self.parent.close)
        ]
        
        for color, tip, callback in button_data:
            button = QPushButton(self)
            button.setFixedSize(12, 12)
            button.setToolTip(tip)
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    border-radius: 6px;
                    border: none;
                }}
                QPushButton:hover {{
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                      stop:0 {color}, stop:1 {self.darker_color(color)});
                }}
            """)
            button.clicked.connect(callback)
            layout.addWidget(button)

        self.is_pinned = False

    def darker_color(self, color):
        c = QColor(color)
        h, s, v, _ = c.getHsv()
        return QColor.fromHsv(h, s, int(v * 0.8)).name()

    def toggle_maximize(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
        else:
            self.parent.showMaximized()

    def toggle_pin(self):
        self.is_pinned = not self.is_pinned
        if self.is_pinned:
            self.parent.setWindowFlags(self.parent.windowFlags() | Qt.WindowStaysOnTopHint)
            self.parent.show()
        else:
            self.parent.setWindowFlags(self.parent.windowFlags() & ~Qt.WindowStaysOnTopHint)
            self.parent.show()

class StocksWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.loading_overlay = None
        self.tab_widget = None
        self.initUI()
        self.init_loading_overlay()
        self.init_tabs()
        QTimer.singleShot(100, self.initial_refresh)
        
    def initUI(self):
        self.setWindowTitle('Advanced Multi-Market Stocks Data')
        self.setGeometry(100, 100, 1500, 800)
        
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        main_frame = RoundedCornerFrame(self)
        self.setCentralWidget(main_frame)
        
        main_layout = QVBoxLayout(main_frame)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.title_bar = MacOSTitleBar(self)
        main_layout.addWidget(self.title_bar)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(content_widget)
        
        market_layout = QHBoxLayout()
        self.market_combo = QComboBox()
        self.market_combo.addItems(['Malaysia', 'US'])
        self.market_combo.currentIndexChanged.connect(self.refresh_current_tab)
        market_layout.addWidget(QLabel('Select Market:'))
        market_layout.addWidget(self.market_combo)
        market_layout.addStretch(1)
        
        refresh_label = QLabel('Data refreshes every 5 minutes automatically')
        market_layout.addWidget(refresh_label)
        
        content_layout.addLayout(market_layout)
        
        self.market_combo.setCurrentText('Malaysia')
        
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.refresh_current_tab)
        content_layout.addWidget(self.tab_widget)
        
        self.category_headers = {
            'All Stocks': ['Symbol', 'Price', 'Change %', 'Volume', 'Rel Volume', 'Market cap', 'P/E', 'EPS dil TTM', 'EPS dil growth TTM YoY', 'Div yield % TTM', 'Sector', 'Analyst Rating'],
            'Top Gainers': ['Symbol', 'Change %', 'Price', 'Volume', 'Rel Volume', 'Market cap', 'P/E', 'EPS dil TTM', 'EPS dil growth TTM YoY', 'Div yield % TTM', 'Sector', 'Analyst Rating'],
            'Top Losers': ['Symbol', 'Change %', 'Price', 'Volume', 'Rel Volume', 'Market cap', 'P/E', 'EPS dil TTM', 'EPS dil growth TTM YoY', 'Div yield % TTM', 'Sector', 'Analyst Rating'],
            'Most Active': ['Symbol', 'Vol * Price', 'Price', 'Change %', 'Volume', 'Rel Volume', 'Market cap', 'P/E', 'EPS dil TTM', 'EPS dil growth TTM YoY', 'Div yield % TTM', 'Sector', 'Analyst Rating'],
            'Oversold': ['Symbol', 'RSI (14)', 'Price', 'Change %', 'Volume', 'Rel Volume', 'Market cap', 'P/E', 'EPS dil TTM', 'EPS dil growth TTM YoY', 'Div yield % TTM', 'Sector', 'Analyst Rating'],
            'Overbought': ['Symbol', 'RSI (14)', 'Price', 'Change %', 'Volume', 'Rel Volume', 'Market cap', 'P/E', 'EPS dil TTM', 'EPS dil growth TTM YoY', 'Div yield % TTM', 'Sector', 'Analyst Rating']
        }
        
        self.export_button = QPushButton('Export Data')
        self.export_button.clicked.connect(self.export_data)
        content_layout.addWidget(self.export_button)
        
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                color: #D4D4D4;
                font-family: 'Helvetica Neue', Arial, sans-serif;
            }
            QTableWidget {
                background-color: #252526;
                border: none;
                gridline-color: #3E3E3E;
            }
            QHeaderView::section {
                background-color: #252526;
                color: #569CD6;
                border: none;
                padding: 4px;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QComboBox, QPushButton {
                background-color: #3E3E3E;
                border: 1px solid #569CD6;
                padding: 5px;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #3E3E3E;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #2D2D2D;
                color: #D4D4D4;
                padding: 8px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #3E3E3E;
            }
            #titleBar {
                background-color: #1E1E1E;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
            }
        """)
        
        self.market_urls = {
            'Malaysia': {
                'All Stocks': 'https://www.tradingview.com/markets/stocks-malaysia/market-movers-all-stocks/',
                'Most Active': 'https://www.tradingview.com/markets/stocks-malaysia/market-movers-active/',
                'Top Gainers': 'https://www.tradingview.com/markets/stocks-malaysia/market-movers-gainers/',
                'Top Losers': 'https://www.tradingview.com/markets/stocks-malaysia/market-movers-losers/',
                'Oversold': 'https://www.tradingview.com/markets/stocks-malaysia/market-movers-oversold/',
                'Overbought': 'https://www.tradingview.com/markets/stocks-malaysia/market-movers-overbought/'
            },
            'US': {
                'All Stocks': 'https://www.tradingview.com/markets/stocks-usa/market-movers-all-stocks/',
                'Most Active': 'https://www.tradingview.com/markets/stocks-usa/market-movers-active/',
                'Top Gainers': 'https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/',
                'Top Losers': 'https://www.tradingview.com/markets/stocks-usa/market-movers-losers/',
                'Oversold': 'https://www.tradingview.com/markets/stocks-usa/market-movers-oversold/',
                'Overbought': 'https://www.tradingview.com/markets/stocks-usa/market-movers-overbought/'
            }
        }
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_all_tabs)
        self.timer.start(300000)  # 5 minutes

    def init_loading_overlay(self):
        if self.loading_overlay is None:
            self.loading_overlay = LoadingOverlay(self)
            self.loading_overlay.hide()

    def show_loading_overlay(self, message):
        if self.loading_overlay is None:
            self.init_loading_overlay()
        self.loading_overlay.label.setText(message)
        self.loading_overlay.resize(self.size())
        self.loading_overlay.show()

    def hide_loading_overlay(self):
        if self.loading_overlay:
            self.loading_overlay.hide()

    def init_tabs(self):
        print("Initializing tabs...")  # Debugging print
        if self.tab_widget is None:
            print("Error: tab_widget is None")
            return
        
        self.tab_widget.clear()  # Clear existing tabs
        for category in ['All Stocks', 'Most Active', 'Top Gainers', 'Top Losers', 'Oversold', 'Overbought']:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            table = QTableWidget()
            table.setObjectName(f"{category}Table")
            table.setColumnCount(len(self.category_headers[category]))
            table.setHorizontalHeaderLabels(self.category_headers[category])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.setSortingEnabled(False)
            layout.addWidget(table)
            self.tab_widget.addTab(tab, category)
        print(f"Created {self.tab_widget.count()} tabs")  # Debugging print

    def refresh_current_tab(self):
        self.show_loading_overlay("Refreshing data...")
        market = self.market_combo.currentText()
        category = self.tab_widget.tabText(self.tab_widget.currentIndex())
        url = self.market_urls[market][category]
        QTimer.singleShot(0, lambda: self.update_tab(self.tab_widget.currentIndex(), url))

    def update_tab(self, tab_index, url):
        print(f"Updating tab {tab_index}")  # Debugging print
        try:
            if self.tab_widget is None:
                raise ValueError("tab_widget is None")
            
            tab = self.tab_widget.widget(tab_index)
            if tab is None:
                raise ValueError(f"No tab found at index {tab_index}")
            
            table = tab.findChild(QTableWidget)
            if table is None:
                raise ValueError(f"No QTableWidget found in tab at index {tab_index}")
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            self.populate_table(table, soup)
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            if 'table' in locals():
                table.setRowCount(1)
                table.setColumnCount(1)
                table.setItem(0, 0, QTableWidgetItem(f"Error: {str(e)}"))
            else:
                print(f"Could not display error message in table: {str(e)}")
        finally:
            self.hide_loading_overlay()


    def populate_table(self, table, soup):
        rows = soup.find_all('tr', class_='row-RdUXZpkv')
        table.setRowCount(len(rows))
        
        category = self.tab_widget.tabText(self.tab_widget.currentIndex())
        headers = self.category_headers[category]
        
        for index, row in enumerate(rows):
            cells = row.find_all('td', class_='cell-RLhfr_y4')
            
            if len(cells) >= len(headers):
                for col, header in enumerate(headers):
                    if header == 'Symbol':
                        symbol = cells[0].find('a', class_='apply-common-tooltip')
                        item = QTableWidgetItem(symbol.text.strip().split()[0] if symbol else '')
                    elif header == 'RSI (14)':
                        rsi_value = cells[1].text.strip() if len(cells) > 1 else ''
                        item = QTableWidgetItem(rsi_value)
                    elif header == 'Vol * Price' and category == 'Most Active':
                        volume = self.extract_numeric_value(cells[4].text.strip())
                        price = self.extract_numeric_value(cells[2].text.strip())
                        vol_price = volume * price
                        item = QTableWidgetItem(f"{vol_price:,.0f}")
                    else:
                        cell_index = headers.index(header)
                        item = QTableWidgetItem(cells[cell_index].text.strip())
                    
                    if header in ['Change %', 'EPS dil growth TTM YoY']:
                        color = self.get_color_for_change(item.text())
                        item.setForeground(color)
                    elif header == 'RSI (14)':
                        color = self.get_color_for_rsi(float(item.text()) if item.text() else 0)
                        item.setForeground(color)
                    elif header == 'Sector':
                        item.setForeground(QColor('yellow'))
                    elif header == 'Analyst Rating':
                        color = self.get_color_for_rating(item.text())
                        item.setForeground(color)
                    
                    table.setItem(index, col, item)

    def get_color_for_rsi(self, rsi_value):
        if rsi_value >= 70:
            return QColor('red')
        elif rsi_value <= 30:
            return QColor('green')
        else:
            return QColor('white')

    def get_color_for_change(self, value):
        negative_indicators = ['-', '−', '–', '—', '‒']
        is_negative = any(indicator in value for indicator in negative_indicators)
        
        try:
            cleaned_value = ''.join(c for c in value if c.isdigit() or c in ['.', ','])
            num_value = float(cleaned_value.replace(',', ''))
            
            if is_negative:
                num_value = -num_value
            
            if num_value > 0:
                return QColor('green')
            elif num_value < 0:
                return QColor('red')
        except ValueError:
            pass
        return QColor('white')

    def get_color_for_rating(self, rating):
        rating_lower = rating.lower()
        if any(buy in rating_lower for buy in ['buy', 'strong buy']):
            return QColor('green')
        elif any(sell in rating_lower for sell in ['sell', 'strong sell']):
            return QColor('red')
        return QColor('white')

    def extract_numeric_value(self, text):
        try:
            return float(text.replace(',', '').replace('%', '').replace('K', '000').replace('M', '000000').replace('B', '000000000'))
        except ValueError:
            return 0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 10, 10)
        painter.fillPath(path, QColor("#252526"))

    def export_data(self):
        market = self.market_combo.currentText()
        category = self.tab_widget.tabText(self.tab_widget.currentIndex())
        filename, _ = QFileDialog.getSaveFileName(self, f"Export {market} - {category}", "", "CSV Files (*.csv)")
        if filename:
            table = self.tab_widget.currentWidget().findChild(QTableWidget)
            headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
            
            with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                
                for row in range(table.rowCount()):
                    row_data = []
                    for col in range(table.columnCount()):
                        item = table.item(row, col)
                        if item is not None:
                            text = item.text()
                            header = headers[col]
                            
                            if header in ['Volume', 'Market cap', 'Rel Volume']:
                                text = text.replace('ᵏ', 'K').replace('ᵐ', 'M')
                            elif header == 'Change %' or header == 'EPS dil growth TTM YoY' or header == 'Div yield % TTM':
                                if any(indicator in text for indicator in ['−', '–', '—', '‒']):
                                    text = '-' + text.lstrip('−–—‒')
                            
                            row_data.append(text)
                        else:
                            row_data.append('')
                    writer.writerow(row_data)
            
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText("Data exported successfully.")
            msg_box.setInformativeText("Would you like to open the file now?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.Yes)
            if msg_box.exec_() == QMessageBox.Yes:
                os.startfile(filename)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.loading_overlay and self.loading_overlay.isVisible():
            self.loading_overlay.resize(self.size())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.y() < self.title_bar.height():
            self.dragPos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and event.y() < self.title_bar.height():
            self.move(event.globalPos() - self.dragPos)
            event.accept()

    def initial_refresh(self):
        print("Performing initial refresh...")  # Debugging print
        self.show_loading_overlay("Loading initial data...")
        self.update_all_tabs()
        self.hide_loading_overlay()
        self.show()

    def update_all_tabs(self):
        print("Updating all tabs...")  # Debugging print
        market = self.market_combo.currentText()
        for index, category in enumerate(['All Stocks', 'Most Active', 'Top Gainers', 'Top Losers', 'Oversold', 'Overbought']):
            if category in self.market_urls[market]:
                self.show_loading_overlay(f"Loading {market} {category}...")
                self.update_tab(index, self.market_urls[market][category])
            else:
                print(f"No URL found for {market} {category}")

#if __name__ == '__main__':
#    app = QApplication(sys.argv)
#    ex = StocksWidget()
#    ex.show()
#    sys.exit(app.exec_())

def run_stocks_widget():
    app = QApplication(sys.argv)
    ex = StocksWidget()
    ex.show()
    app.exec_()

if __name__ == '__main__':
    run_stocks_widget()