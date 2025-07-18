import { useState } from 'react';

export function useWorkflow() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [queryId, setQueryId] = useState<string | null>(null);
  const [viewLayer, setViewLayer] = useState<1 | 2 | 3>(1);
  const [showWorkflow, setShowWorkflow] = useState(true);

  return {
    isStreaming,
    setIsStreaming,
    queryId,
    setQueryId,
    viewLayer,
    setViewLayer,
    showWorkflow,
    setShowWorkflow,
  };
}
