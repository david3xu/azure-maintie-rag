import { useState } from 'react';

export const useWorkflow = () => {
  const [queryId, setQueryId] = useState<string | null>(null);

  return {
    queryId,
    setQueryId
  };
};
