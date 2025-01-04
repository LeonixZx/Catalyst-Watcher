import customtkinter as ctk
import subprocess
import sys
import os
from CTkMessagebox import CTkMessagebox
from PIL import Image
import threading
import time

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(application_path, relative_path)

class AIAnalysisMenu(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Analysis")
        
        # Set the icon for the window
        icon_path = get_resource_path("LogoAI.ico")
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)
        else:
            print(f"Warning: Icon file not found at {icon_path}")
        
        # Set window size
        window_width = 300
        window_height = 400
        self.geometry(f"{window_width}x{window_height}")
        self.resizable(False, False)


        # Center the window on the screen
        self.center_window()

        # Rest of your __init__ method remains the same
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(6, weight=1)
        
        
        # Load and set background image
#        bg_image = ctk.CTkImage(Image.open(get_resource_path("background.jpg")), size=(400, 500))
#        bg_label = ctk.CTkLabel(self, image=bg_image, text="")
#        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(6, weight=1)

        self.logo_image = ctk.CTkImage(Image.open(get_resource_path("LogoAI.ico")), size=(100, 100))
        self.logo_label = ctk.CTkLabel(self, image=self.logo_image, text="")
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.title_label = ctk.CTkLabel(self, text="AI Analysis", font=("Roboto", 24, "bold"))
        self.title_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        self.progress_bar = ctk.CTkProgressBar(self, width=300)
        self.progress_bar.grid(row=2, column=0, padx=20, pady=(0, 20))
        self.progress_bar.set(0)

        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.button_frame.grid_columnconfigure((0, 1), weight=1)

        self.install_button = ctk.CTkButton(self.button_frame, text="Install & Run", command=self.install_and_run, fg_color="green", hover_color="darkgreen")
        self.install_button.grid(row=0, column=0, padx=(0, 10), pady=10, sticky="ew")

        self.uninstall_button = ctk.CTkButton(self.button_frame, text="Uninstall", command=self.uninstall_python, fg_color="red", hover_color="darkred")
        self.uninstall_button.grid(row=0, column=1, padx=(10, 0), pady=10, sticky="ew")

        self.exit_button = ctk.CTkButton(self, text="Exit", command=self.quit, fg_color="gray", hover_color="darkgray")
        self.exit_button.grid(row=4, column=0, padx=20, pady=(10, 20), sticky="s")

        self.status_label = ctk.CTkLabel(self, text="", font=("Roboto", 12))
        self.status_label.grid(row=5, column=0, padx=20, pady=(0, 10))
        
        # Copyright label
        copyright_label = ctk.CTkLabel(self.master, text="© Created by LeonCybr Lab | 2024", font=("Roboto", 10))
        copyright_label.grid(row=6, column=0, padx=10, pady=5, sticky="e")
        
        self.process = None
        self.is_running = False


    def center_window(self):
        # Get the screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate the x and y coordinates for the window
        x = (screen_width - self.winfo_width()) // 2
        y = (screen_height - self.winfo_height()) // 2

        # Set the position of the window to the center of the screen
        self.geometry(f"+{x}+{y}")


    def run_command(self, command):
        def run_in_thread():
            self.is_running = True
            self.disable_buttons()
            self.status_label.configure(text="Operation in progress...")
            self.progress_bar.set(0)
            
            try:
                self.process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                while self.is_running:
                    output = self.process.stdout.readline()
                    if self.process.poll() is not None:
                        break
                    if output:
                        print(output.strip())
                        self.status_label.configure(text=output.strip())
                    self.update_progress()
                    time.sleep(0.1)  # Small delay to prevent GUI freezing
                
                returncode = self.process.poll()
                if returncode == 0:
                    CTkMessagebox(title="Success", message="Operation completed successfully.", icon="check")
                else:
                    CTkMessagebox(title="Error", message="An error occurred. Check console for details.", icon="cancel")
            except Exception as e:
                CTkMessagebox(title="Error", message=f"An error occurred: {e}", icon="cancel")
            finally:
                self.is_running = False
                self.process = None
                self.progress_bar.set(0)
                self.status_label.configure(text="")
                self.enable_buttons()

        threading.Thread(target=run_in_thread, daemon=True).start()

    def update_progress(self):
        if not self.is_running:
            return
        current = self.progress_bar.get()
        if current < 1:
            self.progress_bar.set(current + 0.01)
        else:
            self.progress_bar.set(0)
        self.after(50, self.update_progress)

    def disable_buttons(self):
        self.install_button.configure(state="disabled")
        self.uninstall_button.configure(state="disabled")

    def enable_buttons(self):
        self.install_button.configure(state="normal")
        self.uninstall_button.configure(state="normal")

    def install_and_run(self):
        batch_file = get_resource_path("run_ai_analysis.bat")
        self.run_command(f'"{batch_file}" install')

    def uninstall_python(self):
        if self.is_running:
            CTkMessagebox(title="Warning", message="An operation is already in progress. Please wait for it to complete.", icon="warning")
            return

        confirm = CTkMessagebox(title="Confirm Uninstall", 
                                message="Are you sure you want to uninstall AI Analysis?",
                                icon="warning",
                                option_1="Yes",
                                option_2="No")
        if confirm.get() == "Yes":
            batch_file = get_resource_path("run_ai_analysis.bat")
            self.run_command(f'"{batch_file}" uninstall')

    def quit(self):
        if self.is_running:
            confirm = CTkMessagebox(title="Confirm Exit", 
                                    message="An operation is in progress. Are you sure you want to exit?",
                                    icon="warning",
                                    option_1="Yes",
                                    option_2="No")
            if confirm.get() == "Yes":
                if self.process:
                    self.process.terminate()
                self.is_running = False
                super().quit()
        else:
            super().quit()

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    print("Executable path:", sys.executable)
    print("Resource path:", get_resource_path(""))
    
    model_files = [
        "Stock_Predictions_Model_lstm_30.keras",
        "Stock_Predictions_Model_cnn_lstm_30.keras",
        "Stock_Predictions_Model_rf_30.joblib",
        "Stock_Predictions_Model_xgb_30.joblib",
        "Stock_Predictions_Model_lstm_60.keras",
        "Stock_Predictions_Model_cnn_lstm_60.keras",
        "Stock_Predictions_Model_rf_60.joblib",
        "Stock_Predictions_Model_xgb_60.joblib",
        "Stock_Predictions_Model_lstm_90.keras",
        "Stock_Predictions_Model_cnn_lstm_90.keras",
        "Stock_Predictions_Model_rf_90.joblib",
        "Stock_Predictions_Model_xgb_90.joblib"
    ]
    
    for model_file in model_files:
        model_path = get_resource_path(model_file)
        if os.path.exists(model_path):
            print(f"Model file found: {model_file}")
        else:
            print(f"Model file not found: {model_file}")
    
    app = AIAnalysisMenu()
    app.mainloop()