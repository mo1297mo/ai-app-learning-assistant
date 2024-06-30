import React from 'react';
import { Avatar, Typography, Space } from 'antd';
import './styles.css';

const { Text } = Typography;

interface MessageProps {
  text: string;
  isUser: boolean;
}

const Message: React.FC<MessageProps> = ({ text, isUser }) => {
  return (
    <Space direction="horizontal" size={4} className="message-container">
      <Avatar
        size={40}
        className={isUser ? 'user-avatar' : 'ai-avatar'}
      >
        {isUser ? 'U' : 'AI'}
      </Avatar>
      <Text className="message-text">{text}</Text>
    </Space>
  );
};

export default Message;
