import React from 'react';
import type { ChatMessage as ChatMessageType } from '../../types/chat';
import ChatMessage from './ChatMessage';

interface ChatHistoryProps {
  messages: ChatMessageType[];
  currentChatId: string | null;
}

const ChatHistory: React.FC<ChatHistoryProps> = ({ messages, currentChatId }) => (
  <div className="chat-history">
    {messages.map((message) => (
      <ChatMessage
        key={message.id}
        message={message}
        isCurrent={message.id === currentChatId}
      />
    ))}
  </div>
);

export default ChatHistory;
