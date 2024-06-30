import React, { useState, useEffect, useRef } from 'react';
import { Input, List, Button, Row, Col, Select } from 'antd';
import Message from './Message';
import './styles.css';

const { TextArea } = Input;
const { Option } = Select;

interface Message {
  text: string;
  isUser: boolean;
}

const ChatComponent: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState<boolean>(false);
  const [modelChoice, setModelChoice] = useState<string>('gpt');
  const chatEndRef = useRef<HTMLDivElement>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value);
  };

  const handleModelChange = (value: string) => {
    setModelChoice(value);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = async () => {
    if (inputValue.trim() === '') return;
    const userMessage = { text: inputValue, isUser: true };
    setMessages((prevMessages) => [...prevMessages, userMessage, { text: 'Loading...', isUser: false }]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/v1/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: inputValue, model_choice: modelChoice }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch response from the server.');
      }

      if (modelChoice === 'gpt') {
        const reader = response.body?.getReader();
        if (reader) {
          const decoder = new TextDecoder('utf-8');
          let result = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            result += decoder.decode(value);
            setMessages((prevMessages) => [
              ...prevMessages.slice(0, -1),
              { text: result, isUser: false },
            ]);
          }
        }
      } else {
        const data = await response.json();
        const answer = data.answer;
        setMessages((prevMessages) => [
          ...prevMessages.slice(0, -1),
          { text: answer, isUser: false },
        ]);
      }
    } catch (error) {
      console.error('Error fetching and streaming response:', error);
      setMessages((prevMessages) => [
        ...prevMessages.slice(0, -1),
        { text: 'Error: Failed to fetch response from the server.', isUser: false },
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div className="chat-container">
      <List
        itemLayout="horizontal"
        size="large"
        dataSource={messages}
        renderItem={(item) => (
          <List.Item className={item.isUser ? 'user-message' : 'ai-message'}>
            <Message text={item.text} isUser={item.isUser} />
          </List.Item>
        )}
      />
      <div ref={chatEndRef} />
      <Row gutter={16} className="input-row">
        <Col flex="auto">
          <Select
            defaultValue={modelChoice}
            onChange={handleModelChange}
            className="model-select"
          >
            <Option value="gpt">ChatGPT</Option>
            <Option value="llama2">Llama2</Option>
          </Select>
          <div className="input-container">
            <TextArea
              className="message-input"
              value={inputValue}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask your question here..."
              autoSize={{ minRows: 1, maxRows: 6 }}
            />
            <Button
              type="primary"
              onClick={handleSubmit}
              loading={loading}
              className="send-button"
            >
              Send
            </Button>
          </div>
        </Col>
      </Row>
    </div>
  );
};

export default ChatComponent;
