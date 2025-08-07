# Azure Universal RAG - Frontend Guide

**Real React + TypeScript Frontend Implementation**

Frontend development guide for the Azure Universal RAG system based on the actual React implementation with real component structure and API integration.

## 🔍 Actual Frontend Implementation

Based on verified frontend directory structure:

### **Technology Stack (Real Versions)**
- **React 19.1.0** with modern hooks and concurrent features
- **TypeScript 5.8.3** for full type safety
- **Vite 7.0.4** for fast development and building
- **Axios 1.10.0** for HTTP client and API communication
- **ESLint 9.30.1** with React-specific rules

### **Project Structure (Actual)**
```
frontend/
├── src/
│   ├── components/              # Real React components
│   │   ├── chat/               # Chat interface components
│   │   │   ├── ChatHistory.tsx     # Chat message history
│   │   │   ├── ChatMessage.tsx     # Individual chat messages  
│   │   │   └── QueryForm.tsx       # Query input form
│   │   ├── domain/             # Domain-specific components
│   │   │   └── DomainSelector.tsx  # Domain selection interface
│   │   ├── shared/             # Shared UI components
│   │   │   └── Layout.tsx          # Main layout component
│   │   └── workflow/           # Workflow visualization
│   │       ├── WorkflowPanel.tsx   # Main workflow panel
│   │       ├── WorkflowProgress.tsx # Progress indicators
│   │       ├── WorkflowProgress.css # Progress styling
│   │       └── WorkflowStepCard.tsx # Individual workflow steps
│   ├── hooks/                   # Custom React hooks
│   │   ├── useChat.ts              # Chat functionality hook
│   │   ├── useUniversalRAG.ts      # Universal RAG API integration
│   │   ├── useWorkflow.ts          # Workflow state management
│   │   └── useWorkflowStream.ts    # Streaming workflow updates
│   ├── services/               # API service layer
│   │   ├── api.ts                  # Base API configuration
│   │   ├── streaming.ts            # Server-sent events handling
│   │   └── universal-rag.ts        # Universal RAG service
│   ├── types/                  # TypeScript definitions
│   │   ├── api.ts                  # API response types
│   │   ├── chat.ts                 # Chat message types
│   │   ├── domain.ts               # Domain-related types
│   │   ├── workflow.ts             # Workflow state types
│   │   └── workflow-events.ts      # Workflow event types
│   └── utils/                  # Utility functions
│       ├── api-config.ts           # API configuration
│       ├── constants.ts            # Application constants
│       ├── formatters.ts           # Data formatting utilities
│       └── validators.ts           # Input validation
└── package.json                # Dependencies (React 19.1.0, TypeScript 5.8.3)
```

## 🚀 Quick Start

### **Development Setup**
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (React 19.1.0, TypeScript 5.8.3)
npm install

# Start development server (localhost:5174)
npm run dev

# Build for production
npm run build

# Run linting (ESLint 9.30.1)
npm run lint

# Preview production build
npm run preview
```

### **Development Workflow**
```bash
# Backend must be running first (port 8000)
cd /path/to/azure-maintie-rag
uvicorn api.main:app --reload --port 8000

# Start frontend development server (port 5174)  
cd frontend
npm run dev

# Access the application
open http://localhost:5174
```

## 🔧 Real Implementation Details

### **Custom Hooks (Actual Implementation)**

**useUniversalRAG Hook:**
```typescript
// Real hook from src/hooks/useUniversalRAG.ts
import { useState } from 'react';
import { postUniversalQuery } from '../services/api';

export const useUniversalRAG = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const runUniversalRAG = async (request: { query: string; domain: string }) => {
    setLoading(true);
    setError(null);
    try {
      const response = await postUniversalQuery(request.query, request.domain);
      setResult(response);
      return response;
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { loading, result, error, runUniversalRAG };
};
```

**Other Custom Hooks:**
- **useChat.ts**: Chat functionality with message history
- **useWorkflow.ts**: Workflow state management
- **useWorkflowStream.ts**: Real-time streaming updates via Server-Sent Events

### **Component Architecture (Real Components)**

**Chat Interface Components:**
```typescript
// src/components/chat/
- ChatHistory.tsx      // Displays conversation history
- ChatMessage.tsx      // Individual message rendering
- QueryForm.tsx        // User input form with validation
```

**Domain Selection:**
```typescript  
// src/components/domain/
- DomainSelector.tsx   // Domain selection interface
```

**Workflow Visualization:**
```typescript
// src/components/workflow/
- WorkflowPanel.tsx    // Main workflow visualization panel
- WorkflowProgress.tsx // Progress indicators for multi-agent processing
- WorkflowStepCard.tsx // Individual workflow step visualization
- WorkflowProgress.css // Styling for progress components
```

**Shared Components:**
```typescript
// src/components/shared/
- Layout.tsx          // Main application layout structure
```

### **API Integration (Real Services)**

**API Service Layer:**
```typescript
// src/services/api.ts - Base API configuration
// Connects to FastAPI backend at localhost:8000

// src/services/universal-rag.ts - Universal RAG integration
// Interfaces with the real Universal Search Agent

// src/services/streaming.ts - Server-Sent Events
// Real-time workflow progress updates
```

**TypeScript Types:**
```typescript  
// src/types/ - Complete type definitions
- api.ts              // API request/response types
- chat.ts             // Chat message and conversation types
- domain.ts           // Domain analysis and selection types
- workflow.ts         // Workflow state and progress types  
- workflow-events.ts  // Real-time event types
```

## 📊 API Integration Patterns

### **Universal RAG Integration**

Based on the real `useUniversalRAG` hook implementation:

```typescript
// Query submission to Universal Search Agent
const response = await postUniversalQuery(query, domain);

// Response structure matches the backend API
interface UniversalRAGResponse {
  results: SearchResult[];
  domain_detected?: string;
  processing_time?: number;
  // Additional response fields from Universal Search Agent
}
```

### **Backend API Endpoints**

Frontend integrates with real FastAPI endpoints:
```typescript
// Base URL configuration (src/utils/api-config.ts)
const API_BASE_URL = 'http://localhost:8000';

// Endpoint integration
POST /api/v1/search          // Universal search endpoint
GET  /health                 // Health check endpoint
GET  /                       // API root information
```

### **Real-Time Features**

**Streaming Updates:**
```typescript
// src/services/streaming.ts
// Server-Sent Events integration for real-time workflow progress
// Connects to backend streaming endpoints for live updates
```

## 🎨 UI/UX Implementation

### **Progressive Disclosure Pattern**

The frontend implements progressive information disclosure:

1. **Basic User Layer**: Simple query interface with clean results
2. **Technical Layer**: Workflow progress and technical details  
3. **Diagnostic Layer**: Full system diagnostics and performance metrics

### **Responsive Design**

- Mobile-first responsive design
- Component-based architecture for reusability
- CSS modules for styling isolation
- TypeScript for development-time safety

## 🧪 Development Testing

### **Component Testing**
```bash
# Lint TypeScript and React components
npm run lint

# Type checking
npx tsc --noEmit

# Development mode with hot reload
npm run dev
```

### **Integration Testing**
```bash
# Test with real backend (ensure backend is running)
cd ../  # Back to project root
uvicorn api.main:app --reload --port 8000 &

# Start frontend
cd frontend
npm run dev

# Test real API integration
curl http://localhost:8000/health
```

### **Production Build Testing**
```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## 🔧 Configuration

### **Environment Configuration**
```typescript
// src/utils/api-config.ts
// API endpoint configuration
export const API_BASE_URL = process.env.NODE_ENV === 'development' 
  ? 'http://localhost:8000'
  : '/api';  // Production API path
```

### **TypeScript Configuration**
```json
// tsconfig.json - Actual configuration
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedSideEffectImports": true
  }
}
```

### **Vite Configuration**
```typescript
// vite.config.ts - Real Vite setup
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,  // Development server port
    proxy: {
      '/api': 'http://localhost:8000'  // Proxy API calls to backend
    }
  }
})
```

## 📈 Performance Optimization

### **Build Optimization**
- **Vite 7.0.4**: Fast development builds and optimized production bundles
- **TypeScript 5.8.3**: Compile-time optimization and type checking
- **React 19.1.0**: Latest React features and performance improvements

### **Development Performance**
- Hot Module Replacement (HMR) via Vite
- Fast refresh for React components
- TypeScript incremental compilation

## 🛠️ Development Tools

### **Code Quality**
```json
// .eslint.config.js - Real ESLint configuration
export default tseslint.config(
  { ignores: ['dist'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
    },
  },
)
```

### **Available Scripts**
```bash
npm run dev      # Start development server (Vite)
npm run build    # Build for production
npm run lint     # Run ESLint with React rules
npm run preview  # Preview production build locally
```

## 🎯 Success Indicators

Your frontend development environment is ready when:

- **React 19.1.0**: Latest React version with concurrent features
- **TypeScript 5.8.3**: Full type safety across components and services  
- **Vite 7.0.4**: Fast development server and optimized builds
- **API Integration**: Real connection to FastAPI backend on port 8000
- **Component Architecture**: Organized component structure with chat, domain, workflow, and shared components
- **Custom Hooks**: Real implementation of useUniversalRAG, useChat, useWorkflow
- **Type Safety**: Complete TypeScript definitions for all API interfaces
- **Development Tools**: ESLint, hot reload, and production build working

This frontend represents a **real, production-ready React application** with genuine Azure Universal RAG integration, comprehensive TypeScript types, and modern development tooling.