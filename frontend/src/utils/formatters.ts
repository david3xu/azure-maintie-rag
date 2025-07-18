export function formatTimestamp(date: Date): string {
  return date.toLocaleTimeString();
}

export function formatResponse(response: string | object): string {
  if (typeof response === 'string') return response;
  return JSON.stringify(response, null, 2);
}
