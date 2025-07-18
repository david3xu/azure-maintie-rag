import { useState } from 'react';
import type { ChatMessage } from '../types/chat';

export function useChat() {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);

  return {
    chatHistory,
    setChatHistory,
    currentChatId,
    setCurrentChatId,
  };
}
