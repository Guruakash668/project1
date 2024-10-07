import os
import wave
import time
import json
import joblib
import numpy as np
import sounddevice as sd
from kivymd.app import MDApp
from kivy.utils import platform
from kivymd.uix.button import MDRaisedButton, MDIconButton,MDFlatButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.textfield import MDTextField
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.uix.dialog import MDDialog
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
import librosa
from scipy.fft import fft, fftfreq


USER_DATA_FILE = 'user_data.json'
MODEL_FILE = 'random_forest_model.pkl'


class BackgroundScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layout = FloatLayout()

        background = Image(source='assets/1.jpg', allow_stretch=True, keep_ratio=False)
        layout.add_widget(background)

        self.add_widget(layout)
class LoginScreen(BackgroundScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app

        layout = MDBoxLayout(orientation='vertical', padding=20, spacing=10)

        self.username_field = MDTextField(
            hint_text="Username",
            size_hint_y=None,
            height=40,
            pos_hint={"center_x": 0.5},
            foreground_color = (1, 0, 0, 1)
        )
        layout.add_widget(self.username_field)

        self.password_field = MDTextField(
            hint_text="Password",
            password=True,
            size_hint_y=None,
            height=40,
            pos_hint={"center_x": 0.5},
            foreground_color = (1, 0, 0, 1)
        )
        layout.add_widget(self.password_field)

        self.error_label = MDLabel(text='', halign='center', color=(1, 0, 0, 1))
        layout.add_widget(self.error_label)

        login_button = MDRaisedButton(
            text='Login',
            size_hint_y=None,
            height=50,
            pos_hint={"center_x": 0.5}
        )
        login_button.bind(on_release=self.login)
        layout.add_widget(login_button)

        create_account_button = MDRaisedButton(
            text='Create Account',
            size_hint_y=None,
            height=50,
            pos_hint={"center_x": 0.5}
        )
        create_account_button.bind(on_release=self.app.show_create_account_screen)
        layout.add_widget(create_account_button)

        self.add_widget(layout)

    def login(self, instance):
        username = self.username_field.text
        password = self.password_field.text

        if username in self.app.user_data and self.app.user_data[username] == password:
            self.app.change_to_chassis_screen()
        else:
            self.error_label.text = 'Invalid username or password'


class CreateAccountScreen(BackgroundScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app

        layout = MDBoxLayout(orientation='vertical', padding=20, spacing=10)

        self.new_username_field = MDTextField(
            hint_text="New Username",
            size_hint_y=None,
            height=40,
            pos_hint={"center_x": 0.5}
        )
        layout.add_widget(self.new_username_field)

        self.new_password_field = MDTextField(
            hint_text="New Password",
            password=True,
            size_hint_y=None,
            height=40,
            pos_hint={"center_x": 0.5}
        )
        layout.add_widget(self.new_password_field)

        self.error_label = MDLabel(text='', halign='center', color=(1, 0, 0, 1))
        layout.add_widget(self.error_label)

        create_account_button = MDRaisedButton(
            text='Create Account',
            size_hint_y=None,
            height=50,
            pos_hint={"center_x": 0.5}
        )
        create_account_button.bind(on_release=self.create_account)
        layout.add_widget(create_account_button)

        back_button = MDRaisedButton(
            text='Back to Login',
            size_hint_y=None,
            height=50,
            pos_hint={"center_x": 0.5}
        )
        back_button.bind(on_release=self.app.show_login_screen)
        layout.add_widget(back_button)

        self.add_widget(layout)

    def create_account(self, instance):
        username = self.new_username_field.text
        password = self.new_password_field.text

        if username in self.app.user_data:
            self.error_label.text = 'Username already exists!'
        elif username == '' or password == '':
            self.error_label.text = 'Username and password cannot be empty!'
        else:
            self.app.user_data[username] = password
            self.app.save_user_data()
            self.error_label.text = 'Account created successfully! You can log in now.'


class ChassisNumberScreen(BackgroundScreen):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app

        layout = MDBoxLayout(orientation='vertical', padding=20, spacing=10)
        self.chassis_number_field = MDTextField(
            hint_text="Chassis Number",
            size_hint_y=None,
            height=40,
            pos_hint={"center_y": 0.61, "center_x": 0.5}
        )
        layout.add_widget(self.chassis_number_field)

        self.vehicle_name_field = MDTextField(
            hint_text="Vehicle Name",
            size_hint_y=None,
            height=40,
            pos_hint={"center_y": 0.61, "center_x": 0.5}
        )
        layout.add_widget(self.vehicle_name_field)

        self.error_label = MDLabel(text='', halign='center', color=(1, 0, 0, 1))
        layout.add_widget(self.error_label)

        submit_button = MDRaisedButton(
            text='Submit',
            size_hint_y=None,
            height=50,
            pos_hint={"center_x": 0.5}
        )
        submit_button.bind(on_release=self.submit_chassis_number)
        layout.add_widget(submit_button)

        self.add_widget(layout)

    def submit_chassis_number(self, instance):
        chassis_number = self.chassis_number_field.text
        vehicle_name = self.vehicle_name_field.text
        if chassis_number:
            self.app.chassis_number = chassis_number
            self.app.vehicle_name = vehicle_name
            self.app.change_to_engine_screen()
        else:
            self.chassis_number_field.hint_text = 'Please enter a valid chassis number!'
            self.chassis_number_field.text = ''


class EngineSoundAnalyzerApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recording = False
        self.audio_data = []
        self.screen_manager = ScreenManager()
        self.user_data = self.load_user_data()
        self.chassis_number = None  # Store chassis number here
        self.recorded_files = []
        self.results = []
        self.login_screen = LoginScreen(self, name='login')
        self.create_account_screen = CreateAccountScreen(self, name='create_account')
        self.chassis_number_screen = ChassisNumberScreen(self, name='chassis_number')

        self.screen_manager.add_widget(self.login_screen)
        self.screen_manager.add_widget(self.create_account_screen)
        self.screen_manager.add_widget(self.chassis_number_screen)
        self.recording_screen = self.create_recording_screen()
        self.screen_manager.add_widget(self.recording_screen)

        self.model = joblib.load(MODEL_FILE)

    def build(self):
        self.title = "Engine Sound Analyzer"
        return self.screen_manager

    def load_user_data(self):
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
        return {}

    def save_user_data(self):
        with open(USER_DATA_FILE, 'w') as f:
            json.dump(self.user_data, f)

    def create_recording_screen(self):
        screen = Screen(name='recording')
        layout = MDBoxLayout(orientation='vertical', padding=20, spacing=10)

        self.label = MDLabel(text='Press The Microphone Button To Capture Engine Sound.', halign='center')
        layout.add_widget(self.label)

        # Record button
        record_button = MDIconButton(
            icon='microphone',
            on_release=self.toggle_record,
            size_hint=(0.25, 0.25),
            size=(100, 100),
            pos_hint={"center_x": .5}
        )
        layout.add_widget(record_button)

        # Stop button
        stop_button = MDIconButton(
            icon='stop',
            on_release=self.stop_recording,
            size_hint=(0.25, 0.25),
            size=(100, 100),
            pos_hint={"center_x": .5}
        )
        layout.add_widget(stop_button)
        # Play button
        play_button = MDIconButton(
            icon='play',
            on_release=self.play_recording,
            size_hint=(0.25, 0.25),
            size=(100, 100),
            pos_hint={"center_x": .5}
        )
        layout.add_widget(play_button)

        self.prediction_label = MDLabel(text='', halign='center')
        layout.add_widget(self.prediction_label)

        screen.add_widget(layout)
        return screen

    def change_to_engine_screen(self):
        self.screen_manager.current = 'recording'

    def change_to_chassis_screen(self):
        self.screen_manager.current = 'chassis_number'

    def show_create_account_screen(self, instance):
        self.screen_manager.current = 'create_account'

    def show_login_screen(self, instance):
        self.screen_manager.current = 'login'

    def toggle_record(self, instance):
        if self.recording:
            self.stop_recording(instance)
        else:
            self.start_recording()

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.label.text = 'Recording...'
        self.stream = sd.InputStream(samplerate=44100, channels=1, callback=self.audio_callback)
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_data.append(indata.copy().astype(np.float32))  # Ensure it's a float32 NumPy array

    def stop_recording(self, instance):
        if self.recording:
            self.stream.stop()
            self.stream.close()
            self.recording = False
            self.save_recording()
            self.label.text = 'Recording stopped. Analyzing sound...'
            self.analyze_audio()

    def play_recording(self, instance):
        if not self.recorded_files:
            self.label.text = "No recordings available to play."
            return

        # Load the last recorded file
        last_recorded_file = self.recorded_files[-1]  # Get the most recent recording
        self.label.text = "Playing recording..."

        # Read the audio data from the file
        fs, data = self.load_audio_file(last_recorded_file)
        sd.play(data, fs)

        # Wait until the sound has finished playing
        sd.wait()
        self.label.text = "Playback finished."

    def load_audio_file(self, filepath):
        import scipy.io.wavfile as wav
        fs, data = wav.read(filepath)
        return fs, data

    def save_recording(self):
        audio_data_array = np.concatenate(self.audio_data, axis=0)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        chassis_number_safe = self.chassis_number.replace(" ", "_") if self.chassis_number else "unknown"
        filename = f'engine_sound_{chassis_number_safe}_{timestamp}.wav'
        if platform == 'android':
            save_path = f'/storage/emulated/0/Download/{filename}'
        else:
            save_path = filename

        with wave.open(save_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(audio_data_array.astype(np.int16).tobytes())

        self.label.text = f'Recording saved as: {filename}'
        self.recorded_files.append(save_path)

    def extract_features_from_file(self, signal, sample_rate=44100, target_length=25):
        # The signal should be a 1D NumPy array
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sample_rate))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate))

        # FFT to get frequency domain features
        N = len(signal)
        yf = fft(signal)
        xf = fftfreq(N, 1 / sample_rate)[:N // 2]
        amplitude_spectrum = 2.0 / N * np.abs(yf[:N // 2])

        # Pad or truncate FFT amplitude spectrum
        if len(amplitude_spectrum) > target_length:
            amplitude_spectrum = amplitude_spectrum[:target_length]
        else:
            amplitude_spectrum = np.pad(amplitude_spectrum, (0, target_length - len(amplitude_spectrum)), 'constant')

        # Split the frequency spectrum into bands
        low_band = amplitude_spectrum[:target_length // 3]
        mid_band = amplitude_spectrum[target_length // 3:2 * target_length // 3]
        high_band = amplitude_spectrum[2 * target_length // 3:]

        # MFCC
        mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13), axis=1)

        # Extract statistics for each band and MFCCs
        features = [
            zcr, spectral_centroid, spectral_bandwidth,
            np.mean(low_band), np.max(low_band), np.std(low_band),
            np.mean(mid_band), np.max(mid_band), np.std(mid_band),
            np.mean(high_band), np.max(high_band), np.std(high_band)
        ]

        # Combine features with MFCCs
        features.extend(mfccs)

        return np.array(features)

    def analyze_audio(self):
        if not self.audio_data:
            self.label.text = "No audio data recorded."
            return

        # Concatenate the recorded audio data into a single 2D array
        audio_data_array = np.concatenate(self.audio_data, axis=0)  # This should give you a 2D array
        audio_data_array = audio_data_array.flatten()  # Flatten the array to 1D

        # Extract features using the refined method
        features = self.extract_features_from_file(audio_data_array)

        # Make predictions with the model
        prediction = self.model.predict(features.reshape(1, -1))

        # Map numerical predictions to labels
        if prediction[0] == 1:
            predicted_label = 'NOK'
        else:
            predicted_label = 'OK'

        self.prediction_label.text = f'Prediction: {predicted_label}'

        # Show dialog for user confirmation
        self.show_prediction_dialog(predicted_label)

    def show_prediction_dialog(self, predicted_label):
        self.dialog = MDDialog(
            title="Prediction Confirmation",
            text=f"The model predicts: {predicted_label}. Is this correct?",
            buttons=[
                MDFlatButton(
                    text="OK",
                    on_release=lambda x: self.confirm_prediction(predicted_label, "OK")
                ),
                MDFlatButton(
                    text="NOK",
                    on_release=lambda x: self.confirm_prediction(predicted_label, "NOK")
                ),
                MDFlatButton(
                    text="Cancel",
                    on_release=self.close_dialog
                ),
            ],
        )
        self.dialog.open()

    def confirm_prediction(self, predicted_label, user_input):
        self.label.text = f"User confirmed: {user_input}"
        self.save_analysis_results(user_input,predicted_label)  # Save user input along with prediction
        self.close_dialog()

    def close_dialog(self, *args):
        self.dialog.dismiss()

    def save_analysis_results(self, user_input, predicted_label):
        result_data = {
            'chassis_number': self.chassis_number,
            'vehicle_name': self.vehicle_name,
            'user_prediction': user_input,
            'model_prediction': predicted_label
        }
        self.results.append(result_data)

        # Determine the file path based on the platform
        if platform == 'android':
            results_file_path = '/storage/emulated/0/Download/analysis_results.json'  # Path for Android
        else:
            results_file_path = os.path.join(os.getcwd(), 'analysis_results.json')  # Path for Laptop

        print("Saving results to:", results_file_path)  # Debugging line

        try:
            if os.path.exists(results_file_path):
                with open(results_file_path, 'r') as f:
                    existing_results = json.load(f)
                existing_results.append(result_data)
            else:
                existing_results = [result_data]

            with open(results_file_path, 'w') as f:
                json.dump(existing_results, f, indent=4)
        except Exception as e:
            print("Error saving results:", e)


Window.size = (300, 600)
if __name__ == '__main__':
    EngineSoundAnalyzerApp().run()
