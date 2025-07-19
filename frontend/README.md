# Azure Universal RAG Frontend

**React-based frontend for Azure Universal RAG system with progressive workflow transparency**

## 🚀 Overview

The Azure Universal RAG Frontend provides a **modern, responsive user interface** for interacting with the Azure Universal RAG backend system. It features:

- **Progressive Real-Time Workflow**: Three-layer disclosure system for different user types
- **Server-Sent Events**: Real-time updates from Azure services
- **TypeScript Integration**: Full type safety with backend API
- **Responsive Design**: Works on desktop and mobile devices
- **Azure Service Visualization**: Real-time progress through Azure services

## 🛠️ Technology Stack

```
Frontend Stack:
├─ React 19.1.0 + TypeScript 5.8.3
├─ Vite 7.0.4 (build tool)
├─ axios 1.10.0 (HTTP client)
├─ Server-Sent Events (real-time updates)
└─ CSS custom styling with progressive disclosure
```

## 📁 Directory Structure

```
frontend/
├── 📄 package.json         # Node.js dependencies
├── 📄 vite.config.ts       # Vite build configuration
├── 📄 tsconfig.json        # TypeScript configuration
├── 📄 index.html           # Main HTML entry point
├── 📖 README.md            # This documentation
│
├── 🎨 src/                 # React source code
│   ├── components/         # React components
│   │   ├── QueryForm.tsx  # Query submission form
│   │   ├── WorkflowProgress.tsx  # Real-time progress display
│   │   ├── ResponseDisplay.tsx   # Response visualization
│   │   └── ProgressiveDisclosure.tsx  # Three-layer disclosure
│   ├── services/           # API service layer
│   │   ├── api.ts          # HTTP client configuration
│   │   └── streaming.ts    # Server-Sent Events handling
│   ├── types/              # TypeScript type definitions
│   │   └── api.ts          # API request/response types
│   ├── utils/              # Utility functions
│   │   └── workflow.ts     # Workflow processing utilities
│   ├── App.tsx             # Main application component
│   └── main.tsx            # Application entry point
│
├── 🎨 public/              # Static assets
│   ├── favicon.ico         # Site favicon
│   └── assets/             # Images and static files
│
└── 📦 node_modules/        # Node.js dependencies
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Start development server
npm run dev

# 3. Build for production
npm run build

# 4. Preview production build
npm run preview
```

## 🎯 Progressive Workflow Features

### Three-Layer Smart Disclosure

The frontend provides **progressive disclosure** for different user types:

**Layer 1: User-Friendly** (90% of users)
```
🔍 Understanding your question...
☁️ Searching Azure services...
📝 Generating comprehensive answer...
```

**Layer 2: Technical Workflow** (power users)
```
📊 Knowledge Extraction (Azure OpenAI): 15 entities, 10 relations
🔧 Vector Indexing (Azure Cognitive Search): 7 documents, 1536D vectors
🔍 Query Processing: Troubleshooting type, 18 concepts
⚡ Vector Search: 3 results, top score 0.826
📝 Response Generation (Azure OpenAI): 2400+ chars, 3 citations
```

**Layer 3: System Diagnostics** (administrators)
```json
{
  "step": "azure_cognitive_search",
  "status": "completed",
  "duration": 2.7,
  "azure_service": "cognitive_search",
  "details": { "documents_found": 15, "search_score": 0.826 }
}
```

## 🔄 Real-Time Integration

### Server-Sent Events

The frontend connects to the backend's streaming API for real-time updates:

```typescript
// Connect to streaming endpoint
const eventSource = new EventSource('/api/v1/query/stream/{query_id}');

// Handle real-time updates
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  updateWorkflowProgress(data);
};
```

### Azure Services Visualization

Real-time progress through Azure services:
- **Azure OpenAI**: Knowledge extraction and response generation
- **Azure Cognitive Search**: Vector search and retrieval
- **Azure Cosmos DB Gremlin**: Native graph traversal and analytics
- **Azure Blob Storage**: Document storage and retrieval

## 🎨 UI Components

### QueryForm Component
- Clean, intuitive query input
- Real-time validation
- Submit button with loading states

### WorkflowProgress Component
- Real-time progress visualization
- Three-layer disclosure system
- Azure service status indicators

### ResponseDisplay Component
- Formatted response display
- Citation highlighting
- Export capabilities

### ProgressiveDisclosure Component
- User-friendly layer (default)
- Technical details (expandable)
- System diagnostics (admin only)

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_STREAMING_ENDPOINT=/api/v1/query/stream
VITE_QUERY_ENDPOINT=/api/v1/query/universal
```

### API Integration

The frontend integrates with the Azure Universal RAG backend:

- **Base URL**: Configurable via environment variables
- **Endpoints**: Streaming and query endpoints
- **Authentication**: JWT token support (if configured)
- **Error Handling**: Comprehensive error states

## 🧪 Testing

```bash
# Run unit tests
npm run test

# Run tests with coverage
npm run test:coverage

# Run integration tests
npm run test:integration
```

## 🚀 Deployment

### Development
```bash
npm run dev
# Access at http://localhost:5174
```

### Production Build
```bash
npm run build
# Output in dist/ directory
```

### Docker Deployment
```bash
# Build Docker image
docker build -t azure-rag-frontend .

# Run container
docker run -p 5174:5174 azure-rag-frontend
```

## 📊 System Health

- ✅ **React 19.1.0**: Latest React with concurrent features
- ✅ **TypeScript 5.8.3**: Full type safety
- ✅ **Vite 7.0.4**: Fast development and build
- ✅ **Real-time Streaming**: Server-Sent Events integration
- ✅ **Progressive Disclosure**: Three-layer UI system
- ✅ **Azure Services Integration**: Complete backend compatibility
- ✅ **Responsive Design**: Mobile and desktop support

## 🔗 Backend Integration

The frontend seamlessly integrates with the Azure Universal RAG backend:

- **API Endpoints**: Full compatibility with backend API
- **Streaming**: Real-time workflow progress
- **Error Handling**: Comprehensive error states
- **Type Safety**: Shared TypeScript types
- **Authentication**: JWT token support (if configured)

## 📚 Documentation

- **API Documentation**: Available at backend `/docs` endpoint
- **Component Documentation**: Inline TypeScript documentation
- **Workflow Guide**: See backend documentation for workflow details
- **Azure Services**: See backend Azure integration documentation