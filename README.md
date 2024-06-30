# Chat with Documents Application

This project is a web application that allows users to upload PDF files and chat with an AI assistant to ask questions about the content of the uploaded documents. The application also integrates with Google Calendar to automatically save deadlines extracted from the documents.

## Tech Stack

- **Frontend**
  - React
  - Ant Design

- **Backend**
  - FastAPI
  - Python
  - HuggingFace
  - Qdrant
  - Google Calendar API
  - OpenAI API
  - Llama2

## Features

- Upload PDF files and parse their content.
- Ask questions about the content of the uploaded documents.
- Choose between ChatGPT and Llama2 for answering questions.
- Automatically extract deadlines from documents and save them to Google Calendar.
- Stream responses from ChatGPT for real-time interaction.

## Setup and Installation

### Prerequisites

- Python 3.8+
- Node.js 14+
- Qdrant server running locally or accessible remotely
- Google Cloud Project with Calendar API enabled
- Openai API
- Llama2

### Backend Setup

Create and activate a virtual environment:

bash
python -m venv venv
source venv/bin/activate
Install the dependencies:

bash
pip install -r requirements.txt
Create a .env file and add your configuration settings:


Run the FastAPI server:

bash
uvicorn app.main:app --reload
Frontend Setup
Navigate to the frontend directory:

bash
cd ../frontend
Install the dependencies:

bash
npm install
Start the React development server:

bash
npm start
Usage
Open the application in your web browser:


Ask questions about the content of the uploaded document by typing in the input field and selecting the AI model (ChatGPT or Llama2).

View the responses from the AI assistant in the chat interface.


