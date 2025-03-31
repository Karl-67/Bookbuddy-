# BookBuddy: Your AI Reading Companion

BookBuddy is an AI-powered assistant that helps users understand and pronounce complex book content. It allows users to **ask spoken questions** about any passage and returns **clear, easy-to-read explanations**. Additionally, BookBuddy can detect mispronunciations and hesitation in speech to offer **personalized pronunciation feedback**.

---

## 🎯 Features

- **Speech-to-Text Query**: Converts spoken input into text using automatic speech recognition.
- **Text Explanation**: Uses a Large Language Model (LLM) to break down complex passages into simple explanations.
- **Pronunciation Feedback**: Detects mispronunciations or hesitation and offers suggestions for improvement.
- **Interactive Web Interface**: Easy-to-use web UI for students, educators, and readers.

---

## 🧠 Project Structure

```bash
BookBuddy/
├── backend/               # FastAPI backend
├── frontend/              # Vite + React frontend
├── requirements.txt       # Backend dependencies
├── README.md              # Project documentation
```

## ⚙️ Backend Setup

1. Install Python 3.12 or higher
2. Navigate to the backend folder
3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the virtual environment

```bash
source .venv/bin/activate # For macos
.venv\Scripts\activate # For windows
```

5. Install the required packages

```bash
pip install -r requirements.txt
```

## 🌐 Frontend Setup

1. Navigate to the frontend folder

```bash
cd frontend
```

2.Instal dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev #Using Vite
```

## Contributors:

-[Karl Gerges](https://github.com/Karl-67)