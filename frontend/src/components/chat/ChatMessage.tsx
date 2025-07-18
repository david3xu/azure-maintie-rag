import React from 'react';
import type { ChatMessage as ChatMessageType } from '../../types/chat';

interface ChatMessageProps {
  message: ChatMessageType;
  isCurrent: boolean;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, isCurrent }) => (
  <div className={`chat-message ${isCurrent ? 'current' : ''}`}>
    <div className="chat-message-query">
      Q: {message.query}
    </div>
    <div className="chat-message-timestamp">
      {message.timestamp.toLocaleTimeString()}
    </div>
    {message.error ? (
      <div className="chat-message-error">
        ❌ Error: {message.error}
      </div>
    ) : message.isStreaming && !message.response ? (
      <div className="chat-message-loading">
        🔄 Processing...
      </div>
    ) : message.response ? (
      <div className="chat-message-response">
        <p>{message.response.generated_response}</p>
        {message.response.sources?.length > 0 && (
          <div className="response-sources">
            <strong>Sources:</strong> {message.response.sources.join(', ')}
          </div>
        )}
      </div>
    ) : null}
  </div>
);

export default ChatMessage;
