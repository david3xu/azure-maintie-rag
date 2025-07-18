import React from 'react';
import type { ChatMessage as ChatMessageType } from '../../types/chat';
import { formatResponse, formatTimestamp } from '../../utils/formatters';

interface ChatMessageProps {
  message: ChatMessageType;
  isCurrent: boolean;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, isCurrent }) => (
  <div className={`chat-message${isCurrent ? ' current' : ''}`}>
    <div className="chat-message-query">Q: {message.query}</div>
    <div className="chat-message-timestamp">{formatTimestamp(message.timestamp)}</div>
    {message.error ? (
      <div className="chat-message-error">❌ Error: {message.error}</div>
    ) : message.isStreaming && !message.response ? (
      <div className="chat-message-loading">🔄 Processing...</div>
    ) : message.response ? (
      <div>
        <div className="chat-message-response">
          📝 {formatResponse(message.response.generated_response)}
        </div>
        <div className="chat-message-metadata">
          <span className="confidence-badge">
            🎯 {Math.round(message.response.confidence_score * 100)}%
          </span>
          <span className="processing-time-badge">
            ⏱️ {(message.response.processing_time || 0).toFixed(1)}s
          </span>
        </div>
      </div>
    ) : (
      <div className="chat-message-loading">⏳ Waiting for response...</div>
    )}
  </div>
);

export default ChatMessage;
