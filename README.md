# 🤟 SignVision: AI-Powered ASL Translator

**A modern, full-stack web application designed to bridge the gap between silence and speech.**

SignVision provides real-time American Sign Language (ASL) gesture recognition, constructs autocorrected sentences, and delivers contextual, AI-powered responses using the DeepSeek/OpenRouter API.

> **Developed by Faizan Rasool and Anoosha Alina**

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Folder & File Structure](#-folder--file-structure)
- [Main Components](#-main-components)
- [Technologies Used](#-technologies-used)
- [Dataset](#-dataset)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Getting Your Own API Key](#-getting-your-own-api-key)
- [How to Use the App](#-how-to-use-the-app)
- [Customization & Tips](#-customization--tips)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 📖 Project Overview

**SignVision** is an accessibility tool that translates ASL gestures into text and engages users with AI-generated responses. Key features include:

- **Real-time Gesture Recognition:** Instant translation of hand signs via webcam.
- **Intelligent Sentence Construction:** Automatic correction and sentence assembly.
- **AI-Powered Interaction:** Context-aware answers to signed queries.
- **Modern UI:** A responsive and aesthetically pleasing web interface.

---

## 📁 Folder & File Structure

```text
ASL Complete GITHUB Ready/
│
├── ASl Signvision Fullapp .py         # 🚀 Main Full-Stack Application (Flask)
├── knn_asl_model.pkl                  # 🤖 Pre-trained KNN Model for Recognition
├── ASL Avatar.gif                     # 🧑‍🎤 Animated UI Avatar
├── Readme.txt                         # 📄 Project Documentation
│
├── Front End/
│   └── Front End.html                 # 🌐 Standalone Frontend (Reference)
│
├── Back End/
│   └── Back End.py                    # 🛠️ Standalone Backend (Reference)
│
├── Models/
│   └── (Reserved for additional models)
│
└── Training Code/
    └── ASL Training Code.ipynb        # 📓 Model Training Notebook
```

---

## 🧩 Main Components

| Component | Description |
| :--- | :--- |
| **`ASl Signvision Fullapp .py`** | The core Flask application managing the backend, API endpoints, video processing, and frontend serving. |
| **`knn_asl_model.pkl`** | The pre-trained K-Nearest Neighbors (KNN) model used to identify ASL gestures. |
| **`ASL Avatar.gif`** | Visual avatar displayed on the application homepage. |
| **`Training Code/`** | Contains the Jupyter Notebook (`.ipynb`) for retraining or fine-tuning the gesture recognition model. |

---

## 🛠 Technologies Used

*   **Core:** Python 3.7+
*   **Web Framework:** Flask & Flask-CORS
*   **Computer Vision:** OpenCV & MediaPipe
*   **Data Processing:** NumPy, Joblib
*   **NLP:** TextBlob (Autocorrect)
*   **AI Integration:** DeepSeek/OpenRouter API
*   **Frontend:** HTML5, CSS3, JavaScript, Tailwind CSS

---

## 📊 Dataset

The model was trained using the **ASL Alphabet Dataset**:
🔗 [Kaggle: khansatehreem/asl-alphabet-dataset](https://www.kaggle.com/datasets/khansatehreem/asl-alphabet-dataset)

*   **Retraining:** You can use the provided notebook in `Training Code/ASL Training Code.ipynb` to improve the model or train on new data.

---

## ⚡️ Setup & Installation

Follow these steps to get the project running locally:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Faizanras00l/ASL-Sign-Language-Translator
    cd "ASL Complete GITHUB Ready"
    ```

2.  **Install Dependencies**
    ```bash
    pip install flask flask_cors opencv-python mediapipe numpy joblib textblob requests
    ```

3.  **Verify Files**
    Ensure the following files are present in the root directory:
    *   `ASl Signvision Fullapp .py`
    *   `knn_asl_model.pkl`
    *   `ASL Avatar.gif`

---

## ▶️ How to Run

1.  **Configure API Key**
    It is recommended to use an environment variable for security.
    *   **Option A (Environment Variable):**
        ```bash
        # Windows CMD
        set OPENROUTER_API_KEY=your_actual_api_key
        ```
    *   **Option B (Direct Edit):**
        Open `ASl Signvision Fullapp .py` and paste your key into the `API_KEY` variable.

2.  **Launch the Application**
    ```bash
    python "ASl Signvision Fullapp .py"
    ```

3.  **Access the Interface**
    Open your web browser and navigate to:
    `http://127.0.0.1:5000`

---

## 🔑 Getting Your API Key

This project uses **OpenRouter** (DeepSeek) for AI responses.

1.  Sign up at [OpenRouter.ai](https://openrouter.ai/).
2.  Navigate to your dashboard to generate a new API Key.
3.  Copy the key and configure it as described in the [How to Run](#-how-to-run) section.

---

## 🖐️ How to Use

1.  **Home Page:** Click **"Try Now"** to enter the translation interface.
2.  **Start Recognition:** Click **"Start Webcam"** to begin.
3.  **Gesture Input:** Perform ASL signs in front of the camera. The app will detect them and form words.
4.  **Edit & Correct:** Use the on-screen **Space** and **Backspace** controls to format your sentence.
5.  **AI Response:** Click **"Submit & Search"** to send your signed sentence to the AI and receive a smart reply.

---

## 🛠️ Customization & Tips

*   **Model Training:** Update `knn_asl_model.pkl` by running `ASL Training Code.ipynb` with your own dataset.
*   **UI Branding:** You can modify the HTML/CSS directly inside the `HTML_CONTENT` variable within the main Python script.
*   **Change Avatar:** Replace `ASL Avatar.gif` with any GIF of your choice (ensure the filename matches or update the code).

---

## 🩺 Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **Webcam not detected** | Check permissions and close other apps using the camera. |
| **Model error** | Ensure `knn_asl_model.pkl` exists in the root directory. |
| **401 Unauthorized** | Verify your OpenRouter API Key is correct. |
| **CORS / Browser Issues** | Try using Incognito mode or clearing your browser cache. |

---

## 📄 License

This project is intended for **academic and personal use only**. Please refer to individual file headers for third-party library licenses.

---

**Made with ❤️ by Faizan Rasool and Anoosha Alina.**
